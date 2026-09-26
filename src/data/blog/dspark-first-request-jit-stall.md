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

## 先看首次请求在做什么

我用冷 Triton 缓存启动服务，连续发三个请求：第一个请求、相同内容的第二个请求、换一段内容的第三个请求。第一个明显慢，后两个都很快。新内容也没有复现卡顿，说明问题发生在服务启动后的第一次执行，而不是某段提示词本身。

请求期间抓到的 worker 栈反复停在 Triton 调用 Bisheng 编译的路径上。我在服务就绪时记录一次 Triton 缓存，请求结束后再记录一次，发现首请求期间新增了 **17 个 kernel key**，涉及输入准备、KV 写入、推测解码验证和 DFlash 输入准备。保留磁盘缓存重启后，首请求也不再卡顿。那问题就转为：为什么这些 kernel 没有在服务就绪前编译好？

## 补上 V2 预热，剩下四个 kernel

顺着启动流程查代码，实验版本的 [vLLM GPU worker](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu_worker.py) 在 `compile_or_warm_up_model` 中有这段调用：

```python
if self.use_v2_model_runner:
    warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
```

`warmup_kernels` 会构造请求，提前走 V2 的 prefill、decode 和采样路径。而实验版本的 [Ascend worker](https://github.com/vllm-project/vllm-ascend/blob/39fef3f8fe71e71a8b7a684ae0f62b85ff18661f/vllm_ascend/worker/worker.py) 在对应位置只有 ATB warmup，没有这次调用。平台已有的 warmup 覆盖不到完整的 V2 请求路径。我在 Ascend worker 的 `compile_or_warm_up_model` 中补上：

```python
if self.use_v2_model_runner:
    from vllm.v1.worker.gpu.warmup import warmup_kernels

    warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
```

这是本地补丁的关键几行；最终版本还按配置项 `enable_jit_warmup` 控制是否执行。

给 Ascend worker 补上调用后，我重新清空 Triton 缓存、启动服务，再跑相同请求。请求期间新增的 kernel 从 **17 个降到 4 个**。这四个都在推测解码的 verify 链上。

现在已经调用了 vLLM 的 `warmup_kernels`，为什么还有四个 kernel 漏掉？缓存里能找到它们的预热记录，只是预热产生的 key 和真实请求产生的 key 不同。

## 为什么预热过，verify 还要重编

看 [warmup.py](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu/warmup.py)，dummy 请求的采样参数来自 `SamplingParams.for_sampler_warmup()`。它会打开 temperature、top-p、penalty 等处理。[sampler.py](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu/sample/sampler.py) 在需要处理 logits 时会转成 f32，否则原样返回：

```python
if not np.any(self.needs_logits_processing[idx_mapping_np]):
    return logits
logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
```

预热请求走 **f32**；我发的请求只设置输出长度，logits 保持模型输出的 **bf16**。

Triton 按 dtype 编译不同的特化。预热编出了 f32 版本，真实请求第一次走 bf16 时，四个 verify kernel 仍要编译。Ascend 这条 rejection sampling 路径还会在 draft logits 为空时，从 target logits 创建一个同 dtype 的 dummy tensor，于是 dtype 的差异会继续传到后面的 kernel。

![预热与普通请求进入不同的 dtype 特化](/images/posts/dspark-first-request-jit/02-dtype-paths.svg)

为覆盖普通请求，我在第一轮预热后再调用一次 `warmup_kernels`，临时把它使用的参数工厂改为普通采样参数：

```python
from vllm import SamplingParams

if self.model_runner.num_speculative_steps > 0:
    original = SamplingParams.__dict__["for_sampler_warmup"]
    SamplingParams.for_sampler_warmup = staticmethod(
        lambda: SamplingParams(max_tokens=2)
    )
    try:
        warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
    finally:
        SamplingParams.for_sampler_warmup = original
```

这段替换只用于第二轮，并在 `finally` 中恢复。第二轮生成的四个 key，和上一步真实请求期间新增的四个 key 一一对应。再做冷缓存实验，请求期间只剩 **1 个**新 kernel。多一轮预热也会增加启动开销，但两次启动到就绪的总耗时差不能直接算作这段代码的耗时。

这只是暂时绕过上游预热只使用一组采样参数的问题。[vLLM issue #54455](https://github.com/vllm-project/vllm/issues/54455) 在 Intel XPU + MTP 上报告了相同的原生 dtype 路径漏预热；关联的 [PR #54630](https://github.com/vllm-project/vllm/pull/54630) 改为预热多组采样参数。等 PR 合入、升级到包含修复的 vLLM 版本，并在 Ascend + DSpark 冷缓存下确认这些 bf16 verify key 已经于启动期生成，就可以删掉这里的第二轮预热和参数工厂替换。

## 最后一个新 key：DFlash 的无用参数

剩下的是 DFlash 输入准备 kernel。预热期间它已经生成两个不同的缓存 key，首请求又生成第三个。我把三份产物拿出来比，TTIR 相同，最终的 NPU 二进制也相同：三次编译得到了一样的代码。

调用侧会根据当前查询长度计算 `BLOCK_SIZE`，作为编译期参数传入。[Ascend kernel](https://github.com/vllm-project/vllm-ascend/blob/39fef3f8fe71e71a8b7a684ae0f62b85ff18661f/vllm_ascend/worker/v2/spec_decode/dflash/speculator.py) 的签名接收了 `BLOCK_SIZE: tl.constexpr`，但计算中用的是另一个运行时参数 `block_size`：

```python
# 参数列表节选
block_size,
BLOCK_SIZE: tl.constexpr,

# kernel 内部
ctx_block_num = ctx_pos // (block_size * CP_SIZE)
```

内核体没有使用大写的 `BLOCK_SIZE`。上游那颗内核把它当作一块 token 的宽度：块数是查询跨度除以 `BLOCK_SIZE` 向上取整，每个 program 用 `j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` 一次处理这一块，补 padding 也按这个宽度步进。Ascend 这份实现的注释写着保留标量循环，`block_idx > 0` 直接返回，第 0 块里用 `for j in range(0, num_ctx)` 逐个 token 扫。小写 `block_size` 两边都用来算 slot，那是 KV cache 的块大小。所以查询长度变了，`BLOCK_SIZE` 可以换成另一个 2 的幂、多出一个 key，编出来的代码还是同一份。

![不同缓存 key 对应相同的 DFlash 编译产物](/images/posts/dspark-first-request-jit/03-cache-keys.svg)

三份产物相同是直接观察到的。把 key 的差异归因于这个无用参数，还用到了源码检查和下面的修复对照；缓存产物没有记录每个 key 对应的具体 `BLOCK_SIZE` 值。

我先试过把 `BLOCK_SIZE` 固定为 256。请求不再重编，但如果以后内核开始使用它，固定值可能与调用侧计算的 grid 不一致。最终做法是从 Ascend kernel 签名中删掉这个参数，并在适配层丢弃上游传来的同名 kwarg。冷缓存下再次测试，使用较长的新提示词也没有触发新的编译。

## 用首个可见 token 重新对照

完成修复后，我用流式输出里的首个可见 token 计时，`reasoning` 也算在内。在同一套代码上做两轮冷缓存对照：一轮关闭 JIT 预热，一轮开启完整预热；两轮都保留 DFlash 参数修复。

| 请求发出后 | 关闭预热 | 完整预热 |
|---|---:|---:|
| 首个可见 token | 46.68 秒 | 约 0.6 秒 |
| 请求期间新增 kernel | 16 个 | 0 个 |

中间几轮只记录了请求结束的时间，没有首个可见 token 的时间。请求总耗时还包括首 token 后的 decode，而这段过程也会触发 JIT；其中一轮生成的 token 数也不同。因此，不能从这些总耗时还原每一步让首 token 提前了多少秒或多少倍。能逐轮核对的是首请求期间新增的 kernel 数：补 V2 预热后从 17 个降到 4 个；补普通参数预热后剩 1 个；处理 DFlash 的多余特化后为 0。

![每轮修复后首请求期间新增的 kernel 数](/images/posts/dspark-first-request-jit/01-timeline.svg)

第三步先用固定 `BLOCK_SIZE` 的版本验证，最终代码改为删除无用参数。图中的 17→4→1→0 是这些逐轮实验的 key 计数；上表的首 token 数据来自最后单独做的同代码对照。

关闭预热时，首个 token 前一直没有可见输出；首个 token 发出后，流里仍有断续停顿。编译产物的时间戳分布在这两个阶段，说明首次 JIT 也发生在 decode 过程中。完整预热后，请求期间没有新增 kernel key。

这轮关闭预热的请求期间新增了 **32 个 Triton 顶层缓存目录**，其中 **16 个对应 kernel**，其余是 launcher 等伴生目录。完整预热轮在服务就绪前生成的目录集合，包含了关闭预热轮请求期间新增的全部 32 个目录；完整预热轮的三个顺序请求都没有新 kernel key。前面第一次复现时看到的 17 个新 kernel，来自尚未修复 DFlash 参数的代码，所以与这里的 16 个并不冲突。

首个可见 token 从 **46.68 秒变成约 0.6 秒**，代价是更久的启动过程。本次对照中，完整预热轮约 17.6 分钟就绪，关闭预热轮约 12.5 分钟就绪；就绪时间还包括模型加载等工作，不能把差值全算作新增编译成本。

这次查下来，三处问题是按顺序露出来的：先补上 Ascend worker 缺失的 V2 预热，才看见预热参数和普通请求之间的 dtype 差异；补齐 bf16 路径后，又剩下 DFlash 的无用 `constexpr`。

## 换成其他模型会怎样

第一处缺失的调用在 Ascend worker，而不在 DeepSeek V4 的模型代码里。其他模型如果也走这套 Model Runner V2，接入 `warmup_kernels` 同样能提前执行一部分通用路径。但是否还有首请求 JIT，要看它实际用到哪些 kernel、shape 和采样配置；本次的延迟数字不能直接搬过去。

第二处漏编译和模型名称也没有直接关系，条件是**预热参数触发 logits 转 f32，而普通请求保留原生 dtype，后续又有按 dtype 特化的 kernel**。这次的四个 verify kernel 属于 DSpark 推测解码。没有推测解码的模型不会因为这四个 kernel 受益；其他推测解码方法是否走同一条 verify 路径，需要按各自的缓存 key 检查。第二轮预热代码也只在 `num_speculative_steps > 0` 时执行。

第三处是 Ascend DFlash 输入准备 kernel 的问题。只有实际调用这个 kernel 的推测解码路径会受到那个无用 `BLOCK_SIZE` 的影响；普通 decode 或使用其他 draft 方法的模型，不能据此推断有相同的重复编译。

## 结语

这次排查先看首请求卡住时的 worker 栈，确认在 Bisheng 编译；再对比服务就绪和请求结束时的 Triton 缓存，找出请求期间新增的 kernel。沿着这些 key 回查代码，先补 Ascend worker 缺失的 V2 预热，再补普通采样参数的 dtype 路径，最后删掉 DFlash 没有用到的 `BLOCK_SIZE`。每一步都清空缓存重跑，新 key 从 17 个降到 4 个、1 个，最后为 0。

最终在同一套代码上做冷缓存对照，首个可见 token 从 46.68 秒降到约 0.6 秒。编译提前到了启动阶段，首请求期间也不再新增 kernel。针对这几处问题，我已经整理好 vLLM 和 vLLM Ascend 的 PR 草稿，准备提交。
