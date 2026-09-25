---
title: 5.5 秒全场冻结：一起 vLLM-Ascend Triton JIT 编译风暴的排查与修复
description: v1 吞吐只有 v2 的一半，decode 单步却还是大约 52ms。顺着这个缝往下看，停顿来自 fused slot mapping 的 Triton JIT。
pubDate: 2026-09-25
category: ai-infra
tags:
  - vLLM
  - vLLM Ascend
  - Triton
  - 性能排查
  - JIT
draft: false
---

dspark v1/v2 Model Runner 的对比报告出来了。同一负载，200 个 128→128、concurrency 64、graph 模式，k0 上 v1 是 409 tok/s，v2 是 818 tok/s。报告里有一句话差点就这么写下去：v2 相比 v1 提升 2 倍。v1 是老 runner，v2 是重写版，新版本快一倍，好像很正常。

有一个细节对不上。v1 的逐 token 延迟主体和 v2 一样，decode FULL step 大约 52ms。整体慢一倍，单步却一样。慢的不是每一步，是某些时刻突然停住。

k5、k7 这两档的 graph b64，v1 和 v2 也有差距，只是没这么大：557 对 802，553 对 787。

> 硬件是 Ascend 910B2 × 8，TP8 + EP，CANN 9.1、torch_npu 2.10.0.post4、vLLM 0.30.0，模型是 DeepSeek-V4-Flash。现场树在 vLLM-Ascend `39fef3f8f`，补丁后来迁到 `bea70ab8f`。

## 把「慢」拆成「停」

我按 `itls` 数大于 1 秒的间隔。v1 慢的那几轮有 211 个，而且全挤在 5.4–5.8 秒这一条窄带里。一次停顿会同时落在当时在跑的全部 64–89 个请求上。停顿只出现在波次 2 以后，新请求的 prefill 和老请求的 decode 混在同一步的时候。第一波纯 prefill 没有出现过。v2 的全部配置是 0 个。b1、b16 的 v1 也接近 0。同配置换一次进程，有的实例 259 个停顿，有的实例 0 个。

我当时用 bench 结束时间减去「剩下的 token 数 × 平均步长」去对日志。请求什么时候进来、TTFT 多长、长间隔自己多长，这个算法都没算，时间对不齐。JSON 里有 `start_times`、`ttfts` 和逐请求 `itls`，从每个请求的首 token 时刻把间隔累加上去：

```python
origin = min(result["start_times"])
for request_id, (start, ttft, itls) in enumerate(zip(
    result["start_times"], result["ttfts"], result["itls"]
)):
    cursor = start - origin + ttft
    for gap in itls:
        if gap > 1.0:
            print(request_id, cursor, cursor + gap)
        cursor += gap
```

这些长间隔在时间上叠在一起，是共享执行路径上的一次阻塞。109 条记录是 109 个请求各自记了一条长 ITL，不是编译了 109 次。

v1 在混排准入的时候会突然全场停大约 5.5 秒，换一次进程就可能不出现，v2 没有。停的那 5.5 秒里进程在干什么，静态日志里没有。

## 六条对不上的假设

停顿窗口里，CANN plog、atrace、vllm 服务日志都是零行。当时还能做的静态假设，逐条对过：

| 假设 | 排除依据 |
|---|---|
| CFS CPU 限流 | 当时看的 cgroup `nr_throttled=0`，配额没顶到 |
| cudagraph capture sizes 走了默认路径 | 启动命令已经显式注入了捕获列表 |
| v1 没把图加载起来 | v1/v2 都加载了同型编译产物，只是配置 hash 不同 |
| 指标收集器在捣乱 | 每轮 v1/v2 都有同型 `collect.py`，v2 不停 |
| NPU / 驱动在停顿窗里报了错 | 窗内设备日志没有新增行 |
| 调度器里有个 Python 二次方循环 | 64×2048×40µs ≈ 5.2s，和停顿时长贴得很近 |

最后这一条当时最像，时长几乎对得上。

设备日志没有新增，只能说明这段时间里日志是空的，不能单独证明驱动没问题。`nr_throttled=0` 也只覆盖查过的那段 cgroup。六条都对不上。停顿发生的当下，把栈抓下来。

## 停顿当下的栈

抓栈必须 `--nonblocking`。默认模式会暂停目标进程，观测自己就能造出一次停顿。非阻塞也会漏采样，所以不能只抓一张栈，要在 bench 跑着的时候连续抓。

引擎就绪后先打一发 64×128 的 warmup，然后每轮 bench 配大约 100 秒的采样，连跑 5 轮。采样每 0.5 秒一轮，记四样：`npu-smi` 的芯片行、API / EngineCore / Worker 的 `ps`、EngineCore 的线程 `top`，以及两边的 py-spy：

```bash
ts=$(date +%H:%M:%S.%3N)
npu-smi info | grep -E "^\| [0-9] +910" >> watcher.log
ps -o pid,pcpu,stat,wchan:32 -p "$API_PID,$EC_PID,$WORKER_PID" >> watcher.log
py-spy dump --nonblocking -p "$EC_PID" > "dump_${ts}_EC.txt"
py-spy dump --nonblocking -p "$WORKER_PID" > "dump_${ts}_W0.txt"
sleep 0.5
```

一轮结束之后，在 Worker 的 dump 里找编译栈。慢轮会反复出现 `linalg_to_bin_enable_npu_compile`，干净轮是 0：

```bash
grep -l linalg_to_bin_enable_npu_compile dump_*_W0.txt
```

| 轮次 | 吞吐（tok/s） | ITL > 1s | 最大 ITL（秒） |
|---|---:|---:|---:|
| R1 | 823.83 | 0 | 0.393 |
| R2 | 489.51 | 109 | 5.758 |
| R3 | 822.78 | 0 | 0.402 |
| R4 | 491.50 | 106 | 5.704 |
| R5 | 700.79 | 33 | 5.797 |

![原节点连续五轮吞吐，R1、R3 约 823 tok/s，出现长间隔的轮次明显下降](/images/posts/vllm-ascend-jit-stall/01-reproduction.png)

*图 1：直接重算原节点的五份 bench JSON。颜色区分是否出现超过 1 秒的 ITL。*

干净轮 823.83 tok/s，和报告里 v2 的 818 贴在一起。这一轮的差距主要是停顿。k5、k7 没有按同一套再比过。

![R2 的 ITL 分布及长尾放大图，109 个超过一秒的间隔聚集在约五点五秒](/images/posts/vllm-ascend-jit-stall/02-itl-distribution.png)

*图 2：左侧包含所有间隔，纵轴取对数；右侧只画超过 1 秒的部分。两幅图都来自 R2。*

![按 start_times、TTFT 和实际 ITL 重建的请求长间隔时间轴，可见多个请求的间隔重叠](/images/posts/vllm-ascend-jit-stall/03-overlap.png)

*图 3：每条红线是一个请求的长 ITL。横轴是相对时间。红线里还含着正常执行和传输，不能当成编译的精确起止。*

R2 的 `dump_08:14:47.364_W0.txt` 里，Worker 主线程是这样（只留关键帧，从栈顶往下）：

```text
select                         selectors.py
_communicate / communicate     subprocess.py
run                            subprocess.py
linalg_to_bin_enable_npu_compile_A2_A3
compile                        triton/compiler/compiler.py
_do_compile / run              triton/runtime/jit.py
compute_slot_mapping_fused_groups
compute_slot_mapping            worker/block_table.py
_prepare_inputs                 worker/model_runner_v1.py
execute_model                   worker/model_runner_v1.py
```

![从 MRV1 输入准备到 Triton JIT 再到 subprocess 等待的简化调用链](/images/posts/vllm-ascend-jit-stall/04-compile-stack.png)

*图 4：按真实栈画的调用关系，不是火焰图。框的大小不表示耗时。*

栈顶的 `select` 在等子进程结束。顺着栈往下，等待来自昇腾编译后端，触发编译的是 slot mapping。

每步都要算的 slot mapping，命中了一个没编译过的 Triton kernel 变体，当场起子进程编译，当时在跑的请求一起停住。同一时刻还有一个启动不到 10 秒的 `bisheng` 子进程，和停顿时长对得上。慢轮的采样里这条编译栈反复出现，干净轮的采样里是 0。

节点上这个 kernel 累计有 135 个不同编译 key。key 含源码 hash，可能混了历次构建，不能当成这一轮编译了 135 次。能对上停顿时刻的是其中一些 key 目录的 mtime。

## 一个地址计算，为什么要这么多变体

slot mapping 把 token 的逻辑位置换成 KV cache 里的物理槽。普通 paged 组就是一次除法和一次乘加：

```text
block_size = 16
position = 35
block_table[request, 2] = 100

page_index  = 35 // 16 = 2
slot_offset = 35 - 2 × 16 = 3
slot_id     = 100 × 16 + 3 = 1603
```

多组 KV cache 各自算一遍。融合之后一次 launch 处理多个组，Host 提交次数下去了。引入这版融合的是 `e5b903620`（#15289）。方向后来看是对的，但签名里放了两个随 step 变的 `tl.constexpr`：

```python
@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_fused_groups_adaptive_kernel(
    NUM_REQS: tl.constexpr,              # 这一步的请求数
    SMALL_TILE_BLOCK_SIZE: tl.constexpr, # next_pow2(tokens_per_req)
    ...
):
    ...
```

tile 先按平均每请求 token 数向上取整，再取二次幂，卡在最小和最大之间。`NUM_REQS` 就是当前 step 的请求数。两个 step 都是 512 个 token，一个 32 个请求、一个 33 个请求，tile 可以都是 16，key 仍然不同。

波次准入的时候，scheduler 把新的 prefill 和还在跑的 decode 塞进同一步，`num_reqs` 一直在变。磁盘上的缓存只覆盖见过的组合。重启不一定清空缓存，这轮走没走过那些形状，决定了热路径上会不会再编译一次。b1 少，是因为 `num_reqs=1` 时 key 没几个。b64 的混排组合多。v2 用的是另一套 constexpr 全固定的 kernel。第一波纯 prefill 的 key，启动 warmup 刚好盖住了。

和上面的栈是同一条调用：`compute_slot_mapping.py` 里 fused kernel 第一次编译，调用方是 `MultiGroupBlockTable.compute_slot_mapping`，再往上是 `MRV1._prepare_inputs`，每个 step 都要过。

## 关掉融合，再改内核

栈只说明停的时候在编译。把融合关掉，看停顿还在不在。

### 短路融合分派

把 `_can_fuse_slot_mapping` 改成 `False`，融合分派短路，改走逐组计算，其余不动，跑 6 轮：没有超过 1 秒的 ITL，吞吐回到大约 818–824 tok/s。停顿来自 fused slot-mapping 这条路径。

fused 比逐组 fallback 在 host 侧快 1.31–1.62 倍，一次 launch 对三次。要改的是编译 key 的数量，融合留着。

接着在原节点上做了一次预热：block table 初始化完成之后把 tile 档位预热掉，5 轮全干净，824–826 tok/s。预热挂在建组之后，这一步是有效的。后面有一版把同一个调用挂进 `profile_run`，block table 还没建好，函数直接返回。

### 换到上游 main

原节点的树上有本地改动，后面的对比换到另一台机器上的上游 main。模型还是 DeepSeek-V4-Flash 这一系，路径不同。原节点的 823 不能当成这台的基线。这台自己的干净轮是 761.68 tok/s。

之后的对比都走同一份 `e2e_run.sh`：自己拉起引擎、等就绪、warmup、跑 N 轮、再拆掉。未修复和打过补丁各跑一次，只换 `PHASE`。启动参数是 TP8、EP、`FULL_DECODE_ONLY`，capture sizes 显式写成 14 档。拆引擎时要等 HBM 放掉，不然下一轮启动会 OOM，那是假阳性。

### 未修复的五轮

同节点、同模型、同负载，上游 main 跑了 5 轮：

| 轮次 | 吞吐（tok/s） | ITL > 1s | 备注 |
|---|---:|---:|---|
| R1 | 261.68 | 284 | 最长 6.3s |
| R2 | 761.68 | 0 | 干净轮 |
| R3 | 373.20 | 160 | |
| R4 | 294.73 | 253 | |
| R5 | 371.78 | 170 | |

五轮吞吐的算术平均大约 412.61，和干净轮差约 1.85 倍。这是五轮各自吞吐的平均，不是总 token 除以总耗时。

对最差那轮做形状拆开：

```text
>1s: 284    0.5-1s: 0    median=58.6ms    max=6261ms
stall bands: 6s x253, 6.5s x31
affected requests: 184/200
first-stall step range: [0, 126]
```

窄带，不是每步都变慢。184/200 个请求被打中，是同一步一起停。第一波没有，停在后面的波次。和原节点上看到的是同一种停。

![对照节点上未修复五轮、温缓存修复六轮和冷缓存修复两轮的输出吞吐](/images/posts/vllm-ascend-jit-stall/06-throughput.png)

*图 5：虚线是这台机器未修复时的干净轮 761.68。温缓存和冷缓存是同一套内核修复。温缓存启动时磁盘里已经有编译结果，预热没跑成。冷缓存从空目录启动，预热挂在 block table 建好之后。*

### 第一版内核：`tl.num_programs` 把内核做慢了

这里的「第一版」是内核改法，不是前面的 Model Runner v1。`NUM_REQS` 从签名里拿掉，用 2D grid，在 kernel 里用 `tl.num_programs(1)` 把请求数算回来。key 固定了，代码也更短。动手前先用 `smoke3d.py` 确认 triton-ascend 上 3D grid、`program_id(2)` 和 `num_programs` 的语义。

然后是几轮微基准：fused 对逐组、三臂对比、老内核对 main 的逐因素、取模换成乘减。中途 `debug_circular.py` 发现 circular tail 的处理有问题，补了 `pos >= 0` 的 mask 和 clamp。这两处在后面的二分里把内核拖慢了。

E2E 过了：6/6 没有长停顿，吞吐 751–757 tok/s。事件计时里有些形状慢了 10–21%。同一份二进制、同一个 1×4096，两次是 86µs 和 189µs。单次采样对不上。

### 事件计时对不上，改用 msprof

每个形状先在剖面窗口外跑 5 次，把编译和预热排出去，再采 20 次：

```python
for _ in range(5):
    fn()
torch.npu.synchronize()
prof.start()
for _ in range(20):
    fn()
torch.npu.synchronize()
prof.stop()
```

`prof` 是 `torch_npu.profiler.profile`，活动开 CPU 和 NPU，`ProfilerLevel.Level1`，`aic_metrics` 用 `PipeUtilization`。停下来之后还没有可读的表，要离线导出：

```python
from torch_npu.profiler.profiler import analyse
analyse(session_dir, export_type="text")
```

产物是 `ASCEND_PROFILER_OUTPUT/kernel_details.csv`。对应命令：

```bash
python3 slot_mapping_msprof_bench.py --label main --paths fused --calls 20 --out msprof_main.json
python3 slot_mapping_msprof_bench.py --label patched --paths per_group,fused --calls 20 --out msprof_patched.json
```

Duration 把这 20 次取平均，用来看量级。它还是会双峰：第一版内核两次采集是 72.64µs 和 189.37µs。下面表里 1×4096 的 189.37µs 是慢的那次，大约是 main 的 9 倍。表后面那张图不看 Duration，只加 `aiv_time`。

| 形状 (reqs×ctx) | main fused | 第一版（2D grid） | per_group |
|---|---:|---:|---:|
| 1×4096 | 21.09 | 189.37 | 16.62 |
| 1×128 | 44.35 | 157.15 | 14.91 |
| 2×2048 | 44.48 | 189.05 | 19.55 |
| 8×512 | 25.83 | 98.37 | 16.01 |
| 8×16 | 6.68 | 8.80 | 15.78 |
| 64×1 | 17.69 | 24.36 | 34.65 |
| 64×128 | 38.82 | 131.70 | 33.18 |

单位 µs，Duration 的 20 次均值。`num_reqs≤2` 的形状，main 走的是旧版 fused kernel；`num_reqs≥8` 才是 #15289 的 adaptive 版，也就是这次触发编译的内核。

E2E 看不出这个回退（751–757 对干净轮 762），slot mapping 在整步里占比小。把二十来微秒的内核做成一百多微秒，这一版不能留。

![统一 aiv_time 口径后，第一版修复在多个形状上明显慢于原版](/images/posts/vllm-ascend-jit-stall/07-first-fix-cost.png)

*图 6：只留 slot_mapping 内核，`aiv_time` 求和再除以 20。1×4096 从 16.29µs 到 42.18µs，2×2048 从 15.51µs 到 55.88µs。这张图不是上表的 Duration。*

### 网格没问题，慢在循环体

189µs 那个 Duration 慢峰里，哪一段在拖时间，做了三级二分。

第一级只改网格，循环体相同。P0 是第一版原样，P1 是 3D grid 加运行时传参，P2 是 main 那种 1D grid 加运行时传参：

| probe（Duration，µs） | 1×4096 | 2×2048 |
|---|---:|---:|
| P0（第一版原样） | 86.13 | 189.06 |
| P1（3D grid） | 189.39 | 189.14 |
| P2（1D grid） | 190.45 | 190.27 |

1D 和 3D 只差维度声明，都钉在大约 190µs。网格不是原因。P0 又出现 86 和 189 的双峰。

第二级以 main 的循环体为底（p3，aiv 11.13µs），一次加回一个第一版引进的东西。口径换成 `aiv_time`，单位 µs，形状 1×4096：

| 变体 | aiv_time | 相对 p3 |
|---|---:|---:|
| p3：main 循环体，装在 2D grid 里 | 11.13 | — |
| p4：加上 `pos>=0` mask | 24.76 | +13.63 |
| p5：加上 clamp | 20.42 | +9.29 |
| p6：取模换成乘减，不带 mask/clamp | 1.53 | −9.60 |
| p1：完整的第一版 | 16.21 | +5.08 |

mask 和 clamp 是 `debug_circular.py` 为 circular tail 补的。main 自己也一直在跑同类代码：`is_circular_ptr` 总是传进去，全组都不是 circular 也在走。这几行只对 1×4096 这一档，不能加到别的形状上。

第三级单独看 `tl.num_programs(1)`。910B2 上每次执行大约多 27µs。

所以网格可以改成多维，`num_reqs` 用 `do_not_specialize` 传，不用 `tl.num_programs(1)`。circular 的 mask 和 clamp 整段放进 `HAS_CIRCULAR` 编译期分支，非 circular 的编译产物里没有这两行。

### 第二版内核：`do_not_specialize` 和编译期分支

三处改动：

1. `NUM_REQS` 从 constexpr 签名里删掉。`num_reqs` 和 `parallel_tiles` 放进 `do_not_specialize`，同时去掉它们的 `tl.constexpr`。不是在旧签名上加一个装饰器参数就完了。
2. grid 改成 `(group_count, num_reqs + 1)`，parallel 再加一维 `parallel_tiles`。多出来的那个程序写 PAD。上面的探针已经说明网格维度本身不慢。
3. `HAS_CIRCULAR` 是一个布尔 constexpr。circular 的计算只留在 `True` 的产物里。全组都不是 circular 时，`is_circular_ptr=None`，相关处理在编译期拿掉。`#16537` 的 `%` 换成 `pos - page_idx * block_size`。

`do_not_specialize` 只负责这个参数不参与特化。tile 档位还在：adaptive 六档 16、32、64、128、256、512，再加上 parallel 的 1024。固定源码、设备、dtype、block 布局和 `HAS_CIRCULAR` 之后，这套配置实测 7 个 key。别的编译参数一变，还会多。旧的 key 集合也不是没有上界，它被 `max_num_seqs` 和 tile 档位数卡住；热路径上追不上，是因为新组合在请求跑起来之后才第一次出现。

![修复前请求数与 tile 共同参与特化，修复后固定配置下只需覆盖六档 adaptive 加一个 parallel 变体](/images/posts/vllm-ascend-jit-stall/05-key-space.png)

*图 7：特化维度的示意。7 个变体有「这套配置不变」这个前提。*

Duration 的 20 次均值（µs）：

| 形状 | 第二版，无 circular | 第二版，有 circular | main |
|---|---:|---:|---:|
| 1×4096 | 4.87 | 34.27 | 21.09 |
| 1×128 | 5.40 | 71.67 | 44.35 |
| 2×2048 | 5.91 | 72.06 | 44.48 |
| 8×512 | 5.93 | 39.70 | 25.83 |
| 8×16 | 5.35 | 7.27 | 6.68 |
| 64×1 | 13.01 | 20.49 | 17.69 |
| 64×128 | 12.29 | 58.15 | 38.82 |

无 circular 这列，主力形状大约快 4 到 8 倍。8×16 只有 1.25 倍，绝对值本来就几微秒。快在两处：编译期拿掉了 circular 的 mask 和 clamp（main 上 `is_circular_ptr` 总会传入）；`%` 换成乘减，1×4096 的降幅里这一项大约 9.6µs。

这次 DeepSeek-V4-Flash 的 E2E 走无 circular。`deepseek_v4` 的 compressor 是 `AscendSlidingWindowMLASpec`，不是 circular。`deepseek_v41` 才是 `CircularBufferSpec`。GLM-5-Next 有 circular 的 indexer tail 组，大约比 main 慢 1.05–1.61 倍，每次多 1–7µs。含 circular 的形状不能套上面那组倍数。

![最终修复的 circular 与非 circular 变体，与原版统一 aiv_time 对比](/images/posts/vllm-ascend-jit-stall/08-final-kernel-cost.png)

*图 8：`aiv_time`，不是上表的 Duration。无 circular 在这些形状上大约是原版的 0.10–0.76 倍；含 circular 大约是 1.05–1.61 倍。*

| 形状 | 原版（µs） | 含 circular（µs） | 无 circular（µs） |
|---|---:|---:|---:|
| 1×4096 | 16.29 | 17.17 | 2.54 |
| 1×128 | 9.94 | 16.04 | 1.32 |
| 2×2048 | 15.51 | 21.16 | 1.57 |
| 8×512 | 9.68 | 15.01 | 1.86 |
| 8×16 | 2.09 | 2.26 | 1.59 |
| 64×1 | 6.39 | 7.19 | 4.60 |
| 64×128 | 14.04 | 20.25 | 4.46 |

两列的 key 都封顶，6 档 tile 加 1 个 parallel，再按 circular 与否各一份。第二版跑了 6 轮，6/6 没有长停顿。预热的挂点还是错的。

负位置是另造的测试里看到的。混合布局里，本该 PAD 的位置可能算出一个普通槽位。查过的 MRV1 路径上 positions 不是负的，没有造成这次线上停顿。

### 预热挂在 `profile_run`，没有执行到

原节点那次预热直接挂在 block table 初始化之后，所以有效。第二版把同一个函数挂进 `profile_run`。`profile_run` 在 `worker.py` 第 572 行，`initialize_kv_cache` 在第 1157 行。预热跑的时候 InputBatch 还是单组，`_can_fuse_slot_mapping` 为假，函数直接返回，碰不到 fused 路径。

```bash
grep -n "prewarm_fused_slot_mapping_kernels" vllm_ascend/worker/worker.py
grep -n "def profile_run\|def initialize_kv_cache" vllm_ascend/worker/worker.py
```

所以图 5 的温缓存是干净的：磁盘上已经有 key。空缓存的新实例不会被这个挂点保护。

冷缓存把 `slot_mapping_triton_warmup` 登记进已有的 kernel warmup，和 rms、penalties、indexer 同一条通道。顺序是：

```text
profile_run：探测显存
  → initialize_kv_cache：建好多组 block table
  → kernel_warmup：预热 selector 够得着的 slot-mapping 变体
  → graph capture
  → 服务就绪
```

测试除了查输出，还查预热之后再扫形状不会增加 key，以及生产用的 `MultiGroupBlockTable` 真的会走进去。fused 内核 29 passed，warmup 注册 21 passed。

### 冷缓存：key 停在 7

把 `TRITON_CACHE_DIR` 指到空目录，启动过程中数 key：

```bash
CACHE=$EXP/triton-cache
rm -rf "$CACHE"
export TRITON_CACHE_DIR="$CACHE"
# 每个 json 目录算一个 key，名字里带 compute_slot_mapping_fused_groups
```

`STATUS` 里：

```text
READY keys_at_ready=7
slot_mapping Triton warmup complete in 39.671s.
WARMUP_DONE keys=7
ROUND1_DONE keys=7
ROUND2_DONE keys=7
ALL_DONE ... final_keys=7
```

warmup 那行写在 READY 下面，日志时间其实更早。`WARMUP_DONE` 是后面那轮请求预热结束。39.671 秒是一条 rank 日志里的内核预热耗时，不是引擎从拉起到就绪的总时间。

![冷缓存启动后，READY、请求预热、两轮 benchmark 与退出阶段的 key 数均为七](/images/posts/vllm-ascend-jit-stall/09-cold-cache.png)

*图 9：五个阶段的 key 数。*

两轮 bench 是 754.69 和 756.44 tok/s，没有超过 1 秒的 ITL，key 也没有再涨。7 个 key 是这套配置的实测。未修复时，同样负载下 key 的上界大约是 `max_num_seqs` 乘 6 档 tile。

![未修复版本五轮的长间隔计数为 284、0、160、253、170，修复后的八轮均为零](/images/posts/vllm-ascend-jit-stall/10-long-intervals.png)

*图 10：和图 5 同一组结果。数的是逐请求长间隔，不是独立的编译次数。*

## 相关

- 问题：[vllm-ascend#17527](https://github.com/vllm-project/vllm-ascend/issues/17527)
- 修复：[vllm-ascend#17529](https://github.com/vllm-project/vllm-ascend/pull/17529)
