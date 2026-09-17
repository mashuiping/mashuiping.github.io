---
title: 从 33 到 47 tok/s：GLM-5.3-Flash 在昇腾 NPU 上的解码性能优化实录
description: 单流解码（数数，spec=2）从 33 到 47 tok/s。eager 路径上的 .nonzero() 每次触发全设备同步，一行掩码写入补丁拿掉了它。
pubDate: 2026-09-12
category: ai-infra
tags:
  - vLLM
  - vLLM Ascend
  - GLM-5.3-Flash
  - NPU
  - 性能优化
draft: false
---

> 机器：一台 Atlas 800I A2（8×910B2，TP=8/EP）
>
> 镜像：`vllm-ascend:glm-5.3-flash`
>
> 模型：GLM-5.3-Flash W8A8 + MTP 投机解码，参考[官方部署文档](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/tutorials/models/GLM5.3-Flash.html#52-multi-node-deployment)
>
> 结果：单流解码（数数，spec=2）33 → 47 tok/s（+42%）。代码改了一行。

## 尝鲜

GLM-5.3-Flash 又名牛来模型。GLM-5.3 只是在 GLM-5.2 上做了后训练和数据增强，Flash 换了新架构。官方 benchmark 打过 DeepSeek V4 Flash 0731（后面简称 DeepSeek Flash 0731），又原生支持多模态。我想用它替换手头这套 DeepSeek Flash 0731，私有化部署，处理一些小型内部排障。

社区支持还不完善。按官方文档指定的镜像和模型，在这台 8 卡机器上跑了一遍，慢得离谱：DeepSeek Flash 0731 单流 decode 能到 90tok/s+，GLM-5.3-Flash 只有 30 tok/s 左右。

慢可能出在请求排队、模型前向，或者解码循环被卡住。排查分三层，每层只答一个问题：

| 层 | 工具 | 回答的问题 |
|---|---|---|
| 1. 服务端日志 | grep | 慢在哪个环节？投机解码效率如何？ |
| 2. CPU 火焰图 | py-spy | host 侧 CPU 忙在哪段代码？ |
| 3. NPU 级 trace | CANN msprof | host/device 时序上，同步点在哪、空隙多大？ |

## 三个前提

投机解码（speculative decoding）。大模型逐 token 生成，每步只出一个字，NPU 大量算力在空转。投机解码用一个小的 draft 模型连猜 N 个 token，再让大模型一次 verify，猜对的部分一并收下。一个周期的产出是接受的 token 数，成本是 N 步 draft 加 1 次 verify。有效吞吐 ≈ 接受长度 ÷ 周期时间。

图捕获（ACL Graph）与动态形状。NPU 上的图捕获（类似 CUDA Graph）把一整个解码步录成固定形状的回放块，跳过 Python 调度和算子下发，解码会快很多。图要求形状固定。输出形状取决于数据内容的算子进不了图。

`.nonzero()` 为什么会同步。`nonzero` 的输出行数取决于张量内容。要给结果分配内存，框架得先知道有几行。这个数字在设备上，所以必须做一次设备到主机的回传。回传会让整条执行流停下来等：之前排队的 kernel 全部跑完才能继续。CUDA 上 `nonzero` 同样强制同步；这场排查里，它被放到了每个 draft 步里，所以特别疼。

## 日志

服务端每 10 秒打印两类日志：

```bash
# 引擎吞吐
grep "loggers.py" /root/vllm.log | tail -5
# 投机解码效率：接受长度 + 逐位接受率
grep -iE "Mean acceptance" /root/vllm.log | tail -5
```

看到两条：

- `Avg generation throughput` 长期在 24-29，和刚跑起来看到的 20 到 30 对得上。请求没有堆着，慢在解码路径。
- spec=5 的逐位接受率是 `0.62 / 0.43 / 0.24 / 0.05 / 0.00`。draft 的第 4、5 步几乎全被拒，白烧。

第二条当场就能改：把 `num_speculative_tokens` 从 5 降到 2，编程负载从 35.5 涨到 39.4-41.1 tok/s。这是编程负载，后面 A/B 用的数数基线是 33。

吞吐低的具体位置日志看不出来，上 profiler。

## 火焰图

用 py-spy 采引擎进程。vLLM v1 是多进程，要采 EngineCore，别采 APIServer。

| 进程 | 职责 | 识别方式 |
|---|---|---|
| APIServer | fastapi 入口、请求解析 | 日志前缀 `(APIServer pid=xxx)` |
| EngineCore | 调度器 + 模型 forward，真正干活 | torch trace 里的 pid；APIServer 的子进程 |

采错了对象，火焰图上只有一片空闲的事件循环，会以为 CPU 没瓶颈。先 `dump` 看栈，确认栈里是模型代码再录：

```bash
pip install py-spy
ps -ef | grep -v grep | grep -iE "vllm|EngineCore"
py-spy dump --pid <EngineCore_PID>     # 栈里应出现 model forward 代码
# 负载运行中录制（录的是干活那一刻）
py-spy record --pid <PID> --duration 21 --rate 100 -o engine.json --format speedscope
py-spy record --pid <PID> --duration 30 --rate 100 -o engine.svg  --format flamegraph
```

![修复前 EngineCore 的 py-spy 时间线](/images/posts/glm-5.3-flash/py-spy-before.png)

*图 1：修复前 EngineCore。`indexer_kpool_mla_forward` 和 `_scatter_paged_cache` 在 draft 循环里反复出现。*

CPU 的大把时间花在 indexer forward 相关的 Python 调度上，栈里密密麻麻全是细碎算子。py-spy 只能看 host。这些算子在设备上干了什么、有没有空转，要靠 NPU 级 trace。

## msprof

vLLM 内置了 profile 开关，底层走 torch_npu profiler 生成 CANN msprof 数据。启动时加一行配置：

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/prof_traces","torch_profiler_with_stack":false}'
```

`with_stack=false` 是因为带栈的 trace 体积和开销都会暴涨，这里只要时序和算子统计。这个配置闲置时零开销，随时可以开窗采样：

```bash
# pod 内保持解码负载运行中
python3 /tmp/bench_decode.py direct 1 1024
curl -X POST http://127.0.0.1:8000/start_profile   # 开窗
# 等 10~20 秒（本次窗口 10.26s ≈ 310 个解码周期）
curl -X POST http://127.0.0.1:8000/stop_profile    # 关窗，等落盘
```

TP=8 会给每个 rank 各生成一份目录，每份三件套：`op_statistic.csv`（算子统计）、`trace_view.json`（chrome timeline，含 host 侧 API 调用）、`kernel_details.csv`（kernel 明细）。8 份内容在 host 侧等价，取 rank0 分析即可。

从 op_statistic.csv 对比算子：

```bash
grep -iE "nonzero|synchronize|kpool" traces/before/op_statistic.csv
# 修复前：NonZero  MIX_AIV  620 次 / device 总时长 3114µs（avg 5.0µs）
# 修复后：NonZero  整行消失
```

NonZero 在设备上只跑了 5 微秒。打开 trace_view.json 看 host 侧：

| 指标（10.26s 窗口） | 修复前 | 修复后 |
|---|---|---|
| `aclnnNonzero`（host API self 耗时） | 620 次 / 2024ms（avg 3.3ms） | 0 |
| `aclrtSynchronizeStream` | 与 NonZero 配对 620 次 / 1879ms | 配对项归零（另有 138 次其它固定同步点） |
| `vllm::indexer_kpool_mla_forward` host 侧 | 4359ms / 42.5% | 正常水位 |
| 全窗口 kernel launch | 84,568 次（avg 34.4µs） | — |
| NPU Computing 占比 | 36.5% | — |
| AICore 利用率 | 0% | — |

设备上 5µs 的算子，host 侧耗了 3.3ms，大约 650 倍（2024ms / 620 ≈ 3.3ms，相对设备 5.0µs）。`aclnnNonzero` 和与它配对的 `aclrtSynchronizeStream` 都是 620 次：每一次 `.nonzero()`，就触发一次全设备同步。

![修复前 msprof timeline](/images/posts/glm-5.3-flash/msprof-before.png)

*图 2：修复前。host 上 `aclnnNonzero` 后面紧跟 `aclrtSynchronizeStream`，device stream 上是一段空白。*

![修复后 msprof timeline](/images/posts/glm-5.3-flash/msprof-after.png)

*图 3：修复后。`indexer_kpool_mla_forward` 变成一串短条，device 上 kernel 排密了。*

拼起来是：

`.nonzero()` 动态形状 → 每次触发全设备同步 → draft 步无法进图、全程 eager → Python 算子风暴（8.4 万次 launch）→ NPU 大部分时间在等 host → Computing 36.5%、AICore 0%

CPU 忙、NPU 闲，对应的是同一段路径。去代码里找这个 `.nonzero()`。

## 根因

定位到的代码在 `vllm_ascend/attention/indexer_kpool_mla_v1.py`。GLM-5.3 用稀疏 indexer 注意力，draft token 被接受后，要把它的 indexer 状态写回 paged cache。这个散写由 `_scatter_paged_cache` 完成，每个 draft 步、每一层都要调一次（三种 cache 角色各一次）。

这个函数里有两个平行实现：

```python
@staticmethod
def _scatter_paged_cache(cache, slots, values, block_size):
    if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
        # ---- 分支 A：图模式。静态形状的掩码写入 ----
        values = values.reshape(values.shape[0], *cache.shape[2:])
        valid = (slots >= 0) & (slots < cache.shape[0] * block_size)
        safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
        block_ids = torch.div(safe_slots, block_size, rounding_mode="floor")
        block_offsets = torch.remainder(safe_slots, block_size)
        row_mask = valid.view(-1, *([1] * (values.ndim - 1)))
        row_zero = cache[0, 0].clone()
        safe_values = torch.where(row_mask, values, row_zero.unsqueeze(0))
        # ...（slot (0,0) 的恢复逻辑，见下文）
        cache[block_ids, block_offsets] = safe_values
        cache[0, 0].copy_(expected_zero)
        return

    # ---- 分支 B：eager 模式。基于索引的散写 ----
    valid_rows = (
        (slots >= 0) & (slots < cache.shape[0] * block_size)
    ).nonzero().flatten()                     # ← 罪魁祸首
    if valid_rows.numel() == 0:
        return
    valid_slots = slots[valid_rows]
    block_ids = torch.div(valid_slots, block_size, rounding_mode="floor")
    block_offsets = torch.remainder(valid_slots, block_size)
    indices = torch.stack([block_ids, block_offsets], dim=-1)
    torch_npu.npu_scatter_nd_update_(
        cache, indices,
        values.reshape(values.shape[0], *cache.shape[2:])[valid_rows],
    )
```

它把 `values`（本批要写入的缓存行）按 `slots`（目标槽位）散写到 paged cache 里，跳过无效槽位（负数或越界，例如滑动窗口已淘汰的 token）。

两个分支当初都有理由。分支 A 给图捕获用：图要求静态形状，不能有数据依赖的动态输出，所以用掩码写法，无效行不删除，而是重定向到槽位 (0,0)，最后再恢复原值。那个 `row_zero` dance 处理的是边界：无效行会污染 (0,0)，但 (0,0) 本身又可能是合法目标。分支 B 给 eager 用：先 `nonzero` 求出有效行，再按索引 scatter。语义清楚，在 CPU/GPU 的 eager 环境里也常见。

出事的是组合方式。`cudagraph_runtime_mode` 逐次调用判定：target 的 verify 步走 FULL 图，进分支 A；draft 被强制 eager（`llm_base_proposer.py` 里对 GLM 的硬编码，图捕获还没接上），每层每个 draft 步都掉进分支 B。

eager 路径上，每个 draft 步要对三种 cache 角色各做一次 scatter，每次都是 `.nonzero()` 加一次全设备同步。这次 10.26s 窗口记到 620 次 NonZero、约 310 个解码周期，平均每周期 2 次。按「3 角色 × spec 步数」推出来的次数和 620 对不上，这里只报测量值：620 次调用，和 620 次配对同步。

让 eager 路径也走掩码分支，一行：

```diff
-        if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
+        if True:  # PERF FIX: masked scatter avoids eager-path .nonzero() device sync
```

两个分支的契约一样：有效槽位写入对应值，其余原样保留。掩码版靠无效行指向 (0,0) 再事后恢复，索引版靠直接不散写。分支 A 全是 `torch.where / div / remainder / 下标赋值`，eager 下跑和图里跑没有区别。修完之后两条路径走同一份实现。视觉问答 3/3、85k 长上下文输出正确、缓存前缀多轮语义无损、draft 接受率不降。

留 `if True:` 而不是删掉 else，是为了最小 diff：一行可以秒级翻转做 A/B（后来 TTFT 对照实验靠的就是它），review 面积也小。上游合入时应该收成单一实现。

同日受控 A/B，数数，spec=2：

| 指标 | 原版（nonzero） | 补丁版（masked） |
|---|---|---|
| 解码吞吐 | 33 tok/s | 47 tok/s（+42%） |
| 76k 冷预填充 TTFT（n=3/4 均值） | 29.1s | 28.9s（无回退） |
| trace 中 NonZero/同步 | 620 次 | 0 |

## 验证

视觉问答 3/3 正确，85k token 上下文回答正常。后来开过 prefix caching 做回归：缓存前缀加新问题时，能逐字引用 84k 前的首句，缓存块注意力是完好的。76k 冷 TTFT，补丁版 28.9s，原版 29.1s，在噪声里。修复后 NonZero 归零，与它配对的 `aclrtSynchronizeStream` 也归零。归因只看同一天、只动一个变量的 A/B；跨天的数字受环境漂移影响，只做参考。

补丁打上之后，又开过 prefix caching 和 `fuse_allreduce_rms`。同日只拨这两个开关：prefix caching 大约吃掉 9% 到 11% 解码吞吐，fuse 大约 -7%。prefix caching 换来 agent 多轮 TTFT 从 41.7s 降到 1.6s（26×），留着；fuse 没收益，撤了。
