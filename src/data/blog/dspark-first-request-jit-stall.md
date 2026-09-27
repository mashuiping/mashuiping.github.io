---
title: 从 47 秒到 0.5 秒：vLLM Ascend 首请求的 JIT 卡顿
description: 服务就绪后第一个请求要等一分半，第二个就正常了。顺着首请求期间多编译的 17 个 Triton kernel，一个个往回查。
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

最近有客户反馈：推理服务启动成功后，第一个请求明显卡顿。这个现象在 vLLM Ascend 上很常见，第二个请求就正常了，所以之前一直没太在意。

GitHub 上也有人报过。[vLLM Ascend #7193](https://github.com/vllm-project/vllm-ascend/issues/7193) 说首请求 TTFT 很高，原因是第一次请求触发 Bisheng 编译，问能不能在服务就绪前把预热做完。这个 issue 3 月开的，到现在还是 open。

我先在自己的环境复现。冷 Triton 缓存启动服务，连发三个请求：第一个，内容相同的第二个，换一段内容的第三个。第一个请求等到正文出来用了 90.19 秒，第二个 0.95 秒，第三个 0.74 秒。换了内容照样快，慢的不是哪段提示词，是服务起来以后的第一次执行。

> 硬件是 Ascend 910B2 × 8，CANN 9.1、torch_npu 2.10.0.post4、triton-ascend 3.2.0。模型 DeepSeek-V4，TP8 + EP，DSpark 推测解码 `num_speculative_tokens=7`，Model Runner V2，ACL graph。vLLM `ced6857`，vLLM Ascend `39fef3f8f`。

## 首请求期间多编译了什么

请求跑着的时候，我每 2 秒对 Worker 抓一次 py-spy。栈反复停在这里：

```text
linalg_to_bin_enable_npu_compile_A2_A3  (triton/backends/ascend/compiler.py:877)
compile                                  (triton/compiler/compiler.py:289)
```

Worker 在等 Bisheng 编译。要知道编的是哪些 kernel，直接看 Triton 缓存目录就行：目录名是 cache key，目录里的 json 写着 kernel 名，mtime 就是编译完成的时刻。服务就绪时记一份目录清单，请求结束后再记一份，两份一减，就是请求期间才编译的 kernel。

首请求期间多了 17 个，输入准备、KV 写入、推测解码的 verify 链、DFlash 输入准备都有。Bisheng 编一个 kernel 要 3 到 7 秒，17 个加起来和 90 秒是一个量级。保留缓存目录再重启一次，首请求降到 1.25 秒。

启动时平台自己其实也有一段 `kernel_warmup`。它在启动期多编了 12 个 decode 相关的 kernel，但首请求还是 94.30 秒，覆盖不到这条路径。

那就要看这 17 个 kernel 为什么没在服务就绪前编好。

## 把 V2 预热接到 Ascend worker

顺着启动流程看代码。实验版本的 [vLLM GPU worker](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu_worker.py) 在 `compile_or_warm_up_model` 里有这么一段：

```python
if self.use_v2_model_runner:
    warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
```

`warmup_kernels` 会构造 dummy 请求，把 V2 的 prefill、decode、推测解码 verify 和采样都提前跑一遍。[Ascend worker](https://github.com/vllm-project/vllm-ascend/blob/39fef3f8fe71e71a8b7a684ae0f62b85ff18661f/vllm_ascend/worker/worker.py) 的同一个位置只有 ATB warmup，这次调用没有搬过来。我在 Ascend worker 的 `compile_or_warm_up_model` 里补上：

```python
if self.use_v2_model_runner:
    if self.vllm_config.kernel_config.enable_jit_warmup:
        from vllm.v1.worker.gpu.warmup import warmup_kernels

        logger.info("Warming up V2 prefill/decode Triton kernels.")
        warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
```

里面那层 `enable_jit_warmup` 是新一点的 vLLM 主干加的，用户可以借它关掉预热，`enforce_eager` 也会把它关掉。这段改动提成了 [vLLM Ascend #17556](https://github.com/vllm-project/vllm-ascend/pull/17556)。

清缓存重启，首请求等到正文用了 28.46 秒，请求期间新编的 kernel 从 17 个降到 4 个。剩下 4 个全在 verify 链上：`_update_stats`、`_probabilistic_rejection_kernel`、`_resample_kernel`、`_finalize`。

## 预热跑过了，verify 为什么还要编

我先怀疑的是 `warmup_kernels` 这层包装。它在预热期间会临时拿掉 `model_runner.adaptive_verification`，关掉 `rejection_sampler.enable_adaptive_verification`，`finally` 里再恢复。看上去预热走的 verify 路径可能和真实请求不一样，我为此还绕开包装，直接调了私有的 `_warmup_kernels`。后来翻部署配置，speculative config 只有 `num_speculative_tokens` 和 `method` 两项，adaptive 本来就没开。上游 `__call__` 每次都会进 `_verify_in_chunks`，里面有一句 `assert not self.enable_adaptive_verification`，要是真开着，服务早就断言失败了。这两个开关在这个部署里什么也没改，私有调用后来也改回了公开的 `warmup_kernels`。

接着对启动期和请求期的 key 清单。`_probabilistic_rejection_kernel` 只出现在请求期，像是预热整个漏掉了它。可预热明明跑了 verify，这对不上。回去一个个看目录，有一个叫 `-UPnN_YtRp8DY2EmgCU-jHMAK3nIOXUEF0GX7UIbbUE`。Triton 的 key 是 URL-safe base64，可以用 `-` 开头，我前面用 shell 命令批量读目录，它被当成命令行选项吞掉了，也没报错。改用 Python 遍历、读每个目录 json 里的 `name`，启动期是编过 `_probabilistic_rejection_kernel` 的，只是另一个特化。

请求期这 4 次编译也没有生成新的 launcher 目录。launcher 的 key 只看签名，不看 dtype 和常量值。签名一样，特化不一样，差别在输入上。

对照两边的输入，分岔在采样参数。dummy 请求的参数来自 `SamplingParams.for_sampler_warmup()`，这是一套故意开满的配置：`temperature=0.9`、`top_p=0.9`、`top_k=50`、`min_p=0.1`，再加 penalty、logit_bias、bad_words。[sampler.py](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu/sample/sampler.py) 只要有一项需要处理 logits，就先转成 f32，否则原样返回：

```python
if not np.any(self.needs_logits_processing[idx_mapping_np]):
    return logits
logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
```

我的请求只设了 `max_tokens`，temperature 是默认的 1.0，logits 保持模型输出的 bf16。预热编的是 f32 版本，真实请求第一次走 bf16，4 个 verify kernel 就得重编。

一个 dtype 能带动整条链，是因为 DSpark 默认 greedy 起草，不分配 draft logits。Ascend 的 rejection sampling 碰到 draft logits 为空，会用 `target_logits.new_empty(1, 1, 1)` 建一个 dummy，dtype 跟着 target 走。上游 vLLM 这里直接接受 `None`，不建 dummy，受影响的 kernel 不一定是这 4 个。

![预热走 f32，普通请求走 bf16，verify 链上的 4 个 kernel 因此各有一套特化](/images/posts/dspark-first-request-jit/01-dtype-paths.png)

*图 1：左边是 `for_sampler_warmup()` 的预热请求，右边是只设了 `max_tokens` 的普通请求。两边在 `needs_logits_processing` 分开，之后每一步都一样，只有 dtype 不同。*

上游 `warmup_kernels` 没有暴露采样参数，我只能在第一轮预热之后再跑一轮，临时把参数工厂换成普通参数：

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

第二轮预热生成的 4 个 key，和上一轮请求期多出来的 4 个逐个相同：`P3QIW1Rw…`、`gL4uyYUi…`、`Im_46A7…`、`4CBjpahY…`。Triton 的 key 是输入的确定性哈希，两次运行 key 相同，特化就相同。冷缓存再跑，首请求等到正文 4.00 秒，请求期只剩 1 个新 kernel。启动比只做第一轮多了 57 秒。

这段替换是临时的。[vLLM #54455](https://github.com/vllm-project/vllm/issues/54455) 在 Intel XPU + MTP 上报了同样的问题，原生 dtype 的路径没被预热到；关联的 [PR #54630](https://github.com/vllm-project/vllm/pull/54630) 改成预热多组采样参数。等它合入、升级之后，在 Ascend + DSpark 冷缓存下确认 bf16 的 verify key 已经在启动期生成，这段就可以删了。

## 剩下的 DFlash kernel

最后一个是 `_prepare_dflash_inputs_kernel_ascend`。它在启动预热里已经编过两次，两个不同的 key，首请求又编了第三个。三个目录我都拷出来比了：

```text
ttir   md5  93413904e5320eddf3a7c16d107fec41   三份相同
npubin md5  7bf9a7adfc95e04c0070af77f5e1cd59   三份相同
options json 去掉 hash 字段后逐字节相同
```

编了三次，产物一模一样。

![DFlash 输入准备 kernel 的三个缓存 key，TTIR 和 NPU 二进制的 md5 都相同](/images/posts/dspark-first-request-jit/02-dflash-keys.png)

*图 2：前两个 key 来自启动预热，第三个来自首个请求。三份产物的 TTIR 和最终 npubin 字节级相同。*

Triton 的 key 由源码哈希、签名和特化、constexpr 的值、编译选项组成。源码和签名变了，TTIR 会跟着变；选项已经比过是一样的。剩下能怀疑的，是一个传了值却没被用到的 constexpr：值进了 key，不进 IR。

launch 一共传 5 个 constexpr。`SAMPLE_FROM_ANCHOR` 在 DFlash 里恒为 False，`PAD_SLOT_ID`、`CP_SIZE`、`CP_INTERLEAVE` 是部署常量，剩下 `BLOCK_SIZE`。[Ascend kernel](https://github.com/vllm-project/vllm-ascend/blob/39fef3f8fe71e71a8b7a684ae0f62b85ff18661f/vllm_ascend/worker/v2/spec_decode/dflash/speculator.py) 的签名收了它，kernel 体里却搜不到：

```python
# 参数列表节选
block_size,
BLOCK_SIZE: tl.constexpr,

# kernel 内部
ctx_block_num = ctx_pos // (block_size * CP_SIZE)
```

算 slot 用的是小写的 `block_size`，那是 KV cache 的块大小。大写的 `BLOCK_SIZE` 由上游调用侧按每一步的查询长度算出来：

```python
max_tokens_per_req = max_target_query_len + num_query_per_req
BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
_prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](..., BLOCK_SIZE=BLOCK_SIZE)
```

上游那颗 kernel 把 `BLOCK_SIZE` 当成一块 token 的宽度，每个 program 用 `tl.arange(0, BLOCK_SIZE)` 一次处理一块。Ascend 这份保留了标量循环，`block_idx > 0` 直接返回，第 0 块里用 `for j in range(0, num_ctx)` 逐个 token 扫。预热的几步和真实请求的查询长度不一样，`BLOCK_SIZE` 就换成另一个 2 的幂，多出一个 key，编出来还是同一份代码。缓存里不记 constexpr 的值，三个 key 各对应哪个 `BLOCK_SIZE` 我看不到，这个归因靠的是源码和下面的修复对照。

这比多编一次要麻烦。服务上线以后，每碰到一个没见过的长度档，这一档的第一个请求都要再编 3 到 4 秒。

我先在适配层把 `BLOCK_SIZE` 钉成 256：

```python
class _PinnedBlockSizeDFlashKernel:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs["BLOCK_SIZE"] = 256
            return _prepare_dflash_inputs_kernel_ascend[grid](*args, **kwargs)
        return launch
```

冷缓存重跑，首请求等到正文 1.18 秒，请求期没有新 kernel。第三个请求特意换成 122 token 的提示词，落在一个没编过的长度档，也没有重编。

钉成 256 能用，是因为 kernel 现在不读它。哪天有人照上游把这颗 kernel 向量化了，调用侧还按自己算的 `BLOCK_SIZE` 开 grid，两边就会对不上，也不会报错。最后改成从 Ascend kernel 签名里删掉这个参数，适配层把上游传进来的同名 kwarg 丢掉：

```python
class _NoBlockSizeDFlashKernel:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.pop("BLOCK_SIZE", None)
            return _prepare_dflash_inputs_kernel_ascend[grid](*args, **kwargs)
        return launch

dflash_speculator._prepare_dflash_inputs_kernel = _NoBlockSizeDFlashKernel()
```

不加这层、只删签名的话，上游传的 `BLOCK_SIZE=` 会让 Triton 报未知参数。这个修复提成了 [vLLM Ascend #17554](https://github.com/vllm-project/vllm-ascend/pull/17554)，已有的 `test_prepare_dflash_inputs.py` 直接调 Ascend kernel，launch 里的 `BLOCK_SIZE` 一并去掉。

![四轮冷缓存实验中，首请求期间新编译的 kernel 数从 17 降到 4、1、0](/images/posts/dspark-first-request-jit/03-key-count.png)

*图 3：每一轮都是清空缓存重启后发同样三个请求。柱子上方的时间按首个含正文的 chunk 计。*

## 换成首个可见 token 再测

前面的时间都是等到第一个正文 chunk。DeepSeek-V4 会先思考，思考内容走 `reasoning` 字段，同样一段段流式下发，对用户来说那已经是首 token 了。按正文计时，90 秒里可能混着思考。

我把探针改成记三个时刻：首条数据行、首个可见 token（算上 `reasoning`）、首个正文。然后在最终代码上做两轮冷缓存对照，一轮用 `--kernel-config '{"enable_jit_warmup": false}'` 关掉预热，一轮开着，两轮都带着 DFlash 的修复。首条数据行和首个可见 token 在两轮里都一样，表里只列后者：

| 首个请求 | 关闭预热 | 完整预热 |
|---|---:|---:|
| 首个可见 token | 46.68 秒 | 0.505 秒 |
| 首个正文 | 87.02 秒 | 0.683 秒 |
| 请求结束 | 88.07 秒 | 1.71 秒 |
| 请求期间新编译的 kernel | 16 个 | 0 个 |

关预热那轮，首个 token 46.68 秒就出来了，正文却要到 87 秒，中间只来了 5 个 reasoning chunk。开预热那轮，同一段思考只用了 0.18 秒，这 40 秒不是在想。我把请求期新建的 32 个缓存目录按 mtime 排到请求的时间线上：

![关闭预热时，请求期的缓存目录一部分在首个 token 之前创建，另一部分在首个 token 和首个正文之间](/images/posts/dspark-first-request-jit/04-first-request-timeline.png)

*图 4：上面一行是关闭预热，红线是每个缓存目录的创建时刻，精度到秒。下面一行是完整预热，请求期间没有新目录。*

19 个目录在首个 token 之前，13 个在之后，最后一个几乎和正文同时出现。verify 和采样链上有一部分 kernel 要等第一个 decode step 才会第一次调用，编译就被拆成了两段：出字之前静默约 47 秒，出字之后又断断续续卡了约 40 秒。一开始量到的 90 秒，是这两段加在一起。

每个 kernel 编译会生成一个 kernel 目录和一个 launcher 目录，32 个目录就是 16 个 kernel。第一次复现时是 17 个，那时 DFlash 还没修，多出来的就是它。开预热那轮在服务就绪时已经有 91 个目录，关预热那轮请求期多出来的 32 个全在里面。

预热的代价在启动上。开预热那轮 17.6 分钟就绪，关预热 12.5 分钟。就绪时间里还有模型加载，差值不全是编译。

首个可见 token 从 46.68 秒降到 0.505 秒。我只在 DeepSeek-V4 + DSpark 这一套配置上测过。V2 预热和 DFlash 两处已经提了 [#17556](https://github.com/vllm-project/vllm-ascend/pull/17556) 和 [#17554](https://github.com/vllm-project/vllm-ascend/pull/17554)；普通参数的第二轮预热还留在本地，等上游 [PR #54630](https://github.com/vllm-project/vllm/pull/54630) 的结果再定。
