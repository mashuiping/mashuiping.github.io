---
title: 从 47s 到 0.6s，vLLM Ascend 首请求优化
description: 为什么预热之后，首个请求仍会触发 JIT? 本文用缓存 key 和请求时间线追踪
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

最近有客户反馈：推理服务启动成功后，第一个有明显卡顿，这个现象在 vLLM Ascend 很常见，因为第二个请求就正常了，所以之前并没有特别在意。这次来深入分析一下原因。

先搜一下 GitHub 有没有类似的问题，[vLLM Ascend #7193](https://github.com/vllm-project/vllm-ascend/issues/7193)：反馈了首请求 TTFT 很高，说是首次请求触发 Bisheng 编译，导致 TTFT 升高，并询问能否在服务就绪前完成预热。这个 issue 创建于 **2026 年 3 月 12 日**，目前仍是 open。

## 第一次复现：90 秒到底量的是什么

我用冷 Triton 缓存启动服务，连续发三个请求：第一个请求、相同内容的第二个请求、换一段内容的第三个请求。第一版探针测到的首响应分别是 **90.19 秒、0.95 秒、0.74 秒**。换了内容依然快，问题大概率只发生在服务启动后的第一次执行。

这里先交代探针的一个错误：它只把流式返回里的 `content` 当作输出，没有识别 `reasoning`。所以这三个数测的是**首个正文 chunk**，不是首个 token。后面会用修正后的探针把 90 秒拆开；在此之前，所有中间实验都沿用旧口径，数值可以互相比较。

服务启动时已经做过一些 kernel warmup 和图捕获，但请求期间抓到的 worker 栈一直停在 Triton 调用 Bisheng 编译的路径上。再看 Triton 缓存：服务就绪时记一次 key，请求结束后再记一次，首请求期间新增了 **17 个 kernel key**，涉及输入准备、KV 写入、推测解码验证和 DFlash 输入准备。编译并没有在服务就绪前做完。

我还保留磁盘缓存重启了一次。首请求按同一旧口径只用了 **1.25 秒**。这不能精确算出 90 秒里每一秒的去向，但和编译栈、新增 key 放在一起，足以把排查重点放到首次 JIT 上。

## 补上 V2 预热，剩下四个 kernel

顺着启动流程查代码，实验版本的 vLLM GPU worker 会在 `compile_or_warm_up_model` 中调用 `warmup_kernels`，让 Model Runner V2 的相关路径提前执行。Ascend worker 当时没有接上这次调用。平台已有的 warmup 覆盖不到完整的 V2 请求路径。

给 Ascend worker 补上调用后，我重新清空 Triton 缓存、启动服务，再跑相同探针。首个正文从 **90.19 秒降到 28.46 秒**，请求期间新增的 kernel 从 **17 个降到 4 个**。这四个都在推测解码的 verify 链上。

这时不能再说“没有预热”。缓存里已经有这些 kernel 的编译记录，只是预热产生的 key 和真实请求产生的 key 不同。

## 为什么预热过，verify 还要重编

对照两边的输入，区别出在采样参数。`warmup_kernels` 使用 `SamplingParams.for_sampler_warmup()`，会打开 temperature、top-p、penalty 等处理。`apply_sampling_params` 因此把 logits 转成 **f32**。我发的请求只设置输出长度，不需要这些处理，logits 保持模型输出的 **bf16**。

Triton 按 dtype 编译不同的特化。预热编出了 f32 版本，真实请求第一次走 bf16 时，四个 verify kernel 仍要编译。Ascend 这条 rejection sampling 路径还会在 draft logits 为空时，从 target logits 创建一个同 dtype 的 dummy tensor，于是 dtype 的差异会继续传到后面的 kernel。

![预热与普通请求进入不同的 dtype 特化](/images/posts/dspark-first-request-jit/02-dtype-paths.svg)

我加了第二轮预热，让它使用普通采样参数。新一轮预热生成的四个 key，和上一步真实请求期间新增的四个 key 一一对应。再做冷缓存实验，首个正文降到 **4.00 秒**；请求期间只剩 **1 个**新 kernel。第二轮预热把时间挪到了启动阶段，这一轮实验的启动过程比只做第一轮预热多约 57 秒。

## 最后一个新 key：DFlash 的无用参数

剩下的是 DFlash 输入准备 kernel。预热期间它已经生成两个不同的缓存 key，首请求又生成第三个。我把三份产物拿出来比，TTIR 相同，最终的 NPU 二进制也相同：三次编译得到了一样的代码。

调用侧会根据当前查询长度计算 `BLOCK_SIZE`，作为编译期参数传入。Ascend kernel 的签名接收了 `BLOCK_SIZE: tl.constexpr`，内核体却没有使用它；实际计算用的是另一个运行时参数 `block_size`。这样，长度变化可能让 `BLOCK_SIZE` 取不同的值、生成不同 key，却不改变内核代码。

![不同缓存 key 对应相同的 DFlash 编译产物](/images/posts/dspark-first-request-jit/03-cache-keys.svg)

三份产物相同是直接观察到的。把 key 的差异归因于这个无用参数，还用到了源码检查和下面的修复对照；缓存产物没有记录每个 key 对应的具体 `BLOCK_SIZE` 值。

我先试过把 `BLOCK_SIZE` 固定为 256。请求不再重编，但这个值只是为了避免缓存分裂：如果以后内核开始使用它，固定值可能与调用侧计算的 grid 不一致。最终做法是从 Ascend kernel 签名中删掉这个参数，并在适配层丢弃上游传来的同名 kwarg。冷缓存下再次测试，使用较长的新提示词也没有触发新的编译。

## 最后重测：把 90 秒拆成两段

前面的 90.19、28.46、4.00 秒都来自那个只认 `content` 的旧探针。修完代码后，我重新写了探针，分别记录第一条流式数据、首个可见 token（包括 `reasoning`）、首个正文。然后在同一套修复代码上做两轮冷缓存对照：一轮关闭 JIT 预热，一轮开启完整预热；两轮都保留 DFlash 参数修复。

| 请求发出后 | 关闭预热 | 完整预热 |
|---|---:|---:|
| 首个可见 token | 46.68 秒 | 0.505 秒 |
| 首个正文 | 87.02 秒 | 0.683 秒 |
| 请求结束 | 88.07 秒 | 1.713 秒 |
| 请求期间新增 kernel | 16 个 | 0 个 |

![关闭和开启预热时的流式输出时间线](/images/posts/dspark-first-request-jit/01-timeline.svg)

现在能看清楚那次“约 90 秒”的等待了：前 **46.68 秒**，服务器没有发出可见 token；随后虽然开始输出 `reasoning`，到首个正文之间又过了约 **40 秒**。编译产物的时间戳也散布在这两段里，说明后半段的流式停顿同样有请求期间的 JIT。完整预热后，同一提示词从首 token 到首正文只隔了约 **0.18 秒**。

这轮关闭预热的请求期间新增了 **32 个 Triton 顶层缓存目录**，其中 **16 个对应 kernel**，其余是 launcher 等伴生目录。完整预热轮在服务就绪前生成的目录集合，包含了关闭预热轮请求期间新增的全部 32 个目录；完整预热轮的三个顺序请求都没有新 kernel key。这比单看耗时更能说明：本次请求需要的特化，已经被提前编好。

最终这套修复让首个可见 token 从 **46.68 秒变成 0.505 秒**，代价是更久的启动过程。本次对照中，完整预热轮约 17.6 分钟就绪，关闭预热轮约 12.5 分钟就绪；就绪时间还包括模型加载等工作，不能把差值全算作新增编译成本。

这次查下来，三处问题是按顺序露出来的：先补上 Ascend worker 缺失的 V2 预热，才看见预热参数和普通请求之间的 dtype 差异；补齐 bf16 路径后，又剩下 DFlash 的无用 `constexpr`。这些是本地实验中的代码和结果，尚不表示修复已经合入上游。实验所用源码版本为 vLLM `ced6857`、vLLM Ascend `39fef3f8f`；其他模型、采样配置和版本还需要各自验证。
