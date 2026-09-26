---
title: 首请求等了 47 秒：沿着 Triton 缓存追一次 vLLM Ascend 的 JIT 卡顿
description: 从客户反馈的冷启动首请求慢出发，用流式时间线、编译栈和缓存 key 定位 DSpark 推理链上的三处预热缺口。
pubDate: 2026-09-26
category: ai-infra
tags:
  - vLLM
  - vLLM Ascend
  - DSpark
  - Triton
  - 性能排查
draft: false
---

最近有客户反馈：推理服务启动成功后，第一个请求明显比后面的请求慢。我以前也见过这种现象，但一直把它笼统地归为“冷启动”，没有追问到底在等什么。这次决定沿着一次真实请求查下去。

搜索 GitHub 时，我看到 [vLLM Ascend #7193](https://github.com/vllm-project/vllm-ascend/issues/7193)：有人报告服务启动后，首次请求才触发 Bisheng 编译，导致 TTFT 升高，并询问能否在服务就绪前完成预热。这个 issue 创建于 **2026 年 3 月 12 日**，写作时仍是 open。它描述的是同一类现象；本文分析的是一套启用了 Model Runner V2 和 DSpark 的环境，不能把两者直接当成同一处代码问题。

在这套环境里，“第一次慢”其实分成两段：**首个流式 token 前静默约 47 秒；首 token 已经发出后，流里又断续停顿约 40 秒。** 最终定位到三处相互叠加的问题：Ascend worker 少了一次 V2 预热调用、预热只走了采样参数的一种 dtype 路径，以及 DFlash kernel 的一个无用编译期参数让缓存按输入长度分裂。

![首请求与修复后的流式时间线](/images/posts/dspark-first-request-jit/01-timeline.svg)

> 文中的数据来自一组本地冷缓存对照实验，不代表其他模型、硬件或版本的性能承诺。模型名称、客户环境信息和内部部署细节已省略。源码行为锚定实验时的 vLLM `ced6857` 与 vLLM Ascend `39fef3f8f`；后文所说的修复为本地验证组合，尚不表示已合入上游。

## 先把“首响应”量准

服务走 OpenAI 兼容的流式接口，模型会先输出 `reasoning`，再输出正文。如果探针只认 `content`，它记录的是“首个正文 chunk”，会把已经发出的思考内容也算进等待时间。早期探针就是这样得到约 **90 秒** 的读数。

我把探针改成三个时刻：第一条流式数据、首个可见 token（包括 `reasoning`）、首个正文。下表是同一套最终代码上的冷缓存对照：关闭 JIT 预热的是 Run G，开启完整预热的是 Run F；两轮都保留了 DFlash 参数修复。

| 时刻 | 关闭预热：Run G | 完整预热：Run F |
|---|---:|---:|
| 首条数据 / 首个可见 token | 46.68 秒 | 0.505 秒 |
| 首个正文 | 87.02 秒 | 0.683 秒 |
| 整个请求完成 | 88.07 秒 | 1.713 秒 |
| 请求期间新增 Triton kernel | 16 个 | 0 个 |

这组对照回答了两个问题。首 token 前的 46.68 秒发生在模型产生任何可见输出之前，不能归咎于“模型在思考”。首 token 到正文之间还有约 40 秒，说明编译不只堵住了 prefill，也在 decode 途中打断了流。完整预热后，同一个提示词的这段间隔约为 0.18 秒。

还有一个简单对照：沿用已编译过的磁盘缓存重启服务，首请求的旧口径读数为 1.25 秒。它进一步支持“等待主要来自首次编译”，但这个早期读数和上表的首 token 口径不同，不能直接拿来计算改善比例。

## 请求期间到底在做什么

我把 Triton 缓存当成一份编译时间线：启动完成时记录目录和 kernel 名，请求结束后再做差；同时每隔两秒抓一次 worker 栈。慢请求期间，栈反复停在 Triton 编译器调用 Bisheng 的路径上。Run G 的请求窗口多出 **32 个顶层缓存目录，对应 16 个 kernel**；另一半是 launcher 等伴生目录，不能把 32 误写成 32 次 kernel 编译。

更有说服力的是跨轮次的 key 比较：Run G 请求时新增的 32 个目录，全部包含在 Run F 服务就绪前已经生成的 91 个目录里。也就是说，完整预热确实提前覆盖了这次请求会使用的特化，而不是碰巧让一个请求变快。

接下来要解释：为什么服务启动时已有平台级 warmup 和图捕获，真实请求仍会编译？

## 第一处：V2 的预热没有接到 Ascend worker

实验版本的 vLLM GPU worker 在 `compile_or_warm_up_model` 中，针对 Model Runner V2 调用 `warmup_kernels`。它构造 dummy 请求，让输入准备、prefill、decode 和 speculative verify 提前经过相关 kernel。对应的 Ascend worker 当时没有接入这条调用，平台级 warmup 也未覆盖完整的 V2 请求路径。

补上调用后，早期探针所测的首个正文从 **90.19 秒降到 28.46 秒**；请求窗口新增 kernel 从 17 个降到 4 个。这说明大部分编译已经被移到启动阶段，但四个 verify kernel 仍在真实请求中重编。

![两种采样参数让预热和真实请求进入不同的 dtype 特化](/images/posts/dspark-first-request-jit/02-dtype-paths.svg)

## 第二处：预热走 f32，默认请求走 bf16

上游预热使用 `SamplingParams.for_sampler_warmup()`。这组参数刻意打开 temperature、top-p、penalty 等采样处理；`apply_sampling_params` 因此把 logits 升到 **f32**，再送进后续 verify kernel。

我的请求只指定输出长度，没有这些采样处理，logits 保持模型输出的 **bf16**。Triton 会按 dtype 特化，同名 kernel 的 f32 和 bf16 版本有不同缓存 key。于是那四个 verify kernel 虽然“已经预热过”，真实请求仍要重新编译。

这套 DSpark Ascend 路径还有个细节：当 draft logits 为空时，Ascend 的 rejection sampling 实现会从 target logits 创建一个 dummy tensor，它继承 target 的 dtype。因此 target 侧的 f32 / bf16 差异会沿 verify 链传下去。这个结论限定于此次检查的 Ascend 路径；不能直接推到所有上游 GPU 实现。

本地修复是在现有预热之外，再用一组普通采样参数跑一次。第二轮产生的四个 kernel key，与先前真实请求中新增的四个 key 对得上。请求期间只剩一个新的 DFlash kernel，旧口径首正文约 **4.00 秒**。第二轮预热会增加启动耗时；实验中这一轮比只做第一轮多约 57 秒，所以“请求更快”不是免费的。

## 第三处：一个没被使用的 `constexpr`

剩下的 DFlash 输入准备 kernel 更奇怪：启动预热编过两个 key，真实请求又编出第三个。三份缓存产物的 TTIR 相同，最终 NPU 二进制也相同。编译了三次，却得到相同的代码。

![DFlash 无用 constexpr 如何制造不同的缓存 key](/images/posts/dspark-first-request-jit/03-cache-keys.svg)

检查调用侧发现，上游按当前查询长度计算 `BLOCK_SIZE`，再作为编译期参数传入；Ascend 版 kernel 的签名接收了 `BLOCK_SIZE: tl.constexpr`，内核体却没有读取它，实际计算使用的是另一个运行时的 `block_size`。于是输入长度落入不同的 2 的幂次桶时，`BLOCK_SIZE` 的值可能改变缓存 key，却不改变生成的内核代码。

这里要分清证据和解释：**三份 TTIR 与最终二进制相同**可以从缓存产物直接验证；“差异由无用的 `BLOCK_SIZE` 造成”是结合源码、其他参数检查和修复对照得到的归因。缓存记录没有保存每个 key 对应的具体 `BLOCK_SIZE` 值，不能把三个桶的数值写成实测结果。

第一版修复曾把这个参数固定为 256，验证了请求期间不再重编。但固定一个内核并未使用的“假值”会留下隐患：以后如果内核开始读取它，固定值可能与调用侧计算的 grid 不一致。最终改为从 Ascend kernel 签名删除该参数，并在适配层丢弃上游传来的同名 kwarg。较长的新提示词也没有再触发编译。

## 把三段证据连起来

| 问题 | 请求期表现 | 本地处理 |
|---|---|---|
| Ascend worker 未调用 V2 kernel 预热 | 整条请求路径首次编译 | 接入上游公开的 `warmup_kernels`，遵守 JIT 预热开关 |
| 预热只走复杂采样参数 | 四个 verify kernel 以 bf16 特化重编 | 再覆盖普通采样参数路径 |
| DFlash 保留无用的编译期参数 | 新长度桶可能生成新 key | 删除参数，适配层丢弃上游 kwarg |

完整修复组合下，冷缓存服务在就绪前完成相关编译；三个顺序请求均没有生成新 kernel key，其中第三个请求使用了较长的新提示词。首个流式 token 从对照轮的 **46.68 秒降至 0.505 秒**。这项收益以更长的启动过程为代价：在本次对照中，完整预热轮从启动到就绪约 17.6 分钟，关闭预热轮约 12.5 分钟。就绪时间还包含模型加载和其他工作，不能把两者之差全部算成新增 JIT 成本。

这次排查给我留下最有用的方法是：**先明确客户端看到的三个时刻，再把 worker 栈与启动前后 Triton key 集合对齐。** 如果只盯着一个“90 秒 TTFT”，容易把 prefill 阻塞、流内停顿和正常思考时间混在一起。修复也不能停留在“把见过的长度都预热一遍”；当 key 的变化不再对应代码变化时，更应该检查特化参数本身。

> 范围说明：上述结果只验证了本地实验环境和这组请求。普通采样参数的第二轮预热目前依赖参数注入，仍需在上游提供正式入口；相关 issue 和 PR 草案尚不能视作已合入修复。持久化 Triton 缓存可降低重启后的重复编译，但不能代替对首次特化路径的覆盖。
