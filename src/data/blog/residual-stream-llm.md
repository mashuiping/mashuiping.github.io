---
title: Residual Stream 全解析：LLM 主干里那条不断被改写的“残差”
description: 从一段 pre-norm Transformer 代码出发，讲清 residual stream 到底是什么、为什么 hidden_states 和 residual 要分开维护、再到 vLLM-Ascend 里的 7 类推理优化落点。
pubDate: 2026-08-19
updatedDate: 2026-08-19
category: ai-infra
tags:
  - LLM Inference
  - vLLM
  - vLLM-Ascend
  - Transformer
  - 推理优化
draft: false
---

> 读 DeepSeek V4 主干时，反复看到一种写法：`hidden_states, residual = layer(...)`。本文想回答两个问题：`residual` 到底是什么？`hidden_states` 和 `residual` 同时维护，是为了改 Transformer 的数学结构，还是为了别的？

```python
for layer in self.layers[start:end]:
    hidden_states, residual = layer(
        positions, hidden_states, residual,
        llama_4_scaling, input_ids=input_ids,
    )
```

我一开始把 `residual` 理解成"最一开始 embedding 那个东西，然后每层都加它"。代码一跑就发现不对。

本文记录一下到底发生了什么。

---

## 一、先把最常见的误解破掉

`residual` 是一段会随层数不断被改写的主干。

它通常叫做 **residual stream**。每一层 `layer()` 都会做一次"加法"，把当前分支（Attn 或 MoE）的输出并入这条主干。所以每经过一层，residual 都会被改写一次，再传给下一层。

哪怕只看最朴素的 pre-norm Transformer：

```python
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

也能直接看到：每一次 `+` 之后，`x` 都已经变了。下一层拿到的是新值。

---

## 二、一个最朴素的例子

设 embedding 是 `E`，第 1 层 Attn 输出 $A_1$，第 1 层 MoE 输出 $M_1$，依此类推。

$$
\begin{aligned}
x_0 &= E \\
x_1 &= x_0 + A_1 = E + A_1 \\
x_2 &= x_1 + M_1 = E + A_1 + M_1 \\
x_3 &= x_2 + A_2 = E + A_1 + M_1 + A_2 \\
x_4 &= x_3 + M_2 = E + A_1 + M_1 + A_2 + M_2
\end{aligned}
$$

画成图：

```
E ──► +A1 ──► +M1 ──► +A2 ──► +M2 ──► ... ──► final hidden
```

最关键的点：**Attn 加的是它进入分支前那一刻的主干状态。**

"每一层都加同一个 residual" 这种直觉是错的。每一层加的是"那个时刻的 residual stream 状态"。

---

## 三、实战代码里为什么会同时有 `hidden_states` 和 `residual`？

理论写法其实只有一个 `x`：

```python
x = x + attn(norm(x))
x = x + moe(norm(x))
```

但高性能实现里几乎都把这一行拆成两个 tensor 同时传来传去。原因基本就一个：**RMSNorm + Residual Add 经常被 fuse 进同一个 kernel**。

读代码的时候可以先这样建立心理模型：

| 变量 | 含义 |
|---|---|
| `residual` | 真正在累积的主干 |
| `hidden_states` | 当前分支算出来的 update，或者 Norm 之后的中间值 |

---

## 四、fused residual + RMSNorm 长得什么样

一段典型 layer forward：

```python
hidden_states, residual = self.input_layernorm(hidden_states, residual)
hidden_states = self.self_attn(hidden_states)

hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.moe(hidden_states)
```

这里看不到显式的 `hidden_states = hidden_states + residual`，因为它**被塞进 norm 内部**。fused kernel 通常等价于：

```python
residual      = residual + hidden_states   # 先加
hidden_states = RMSNorm(residual)          # 再 norm
return hidden_states, residual
```

源码里"加 residual"那一行不一定明显，但语义上那一步一定存在，只是被 RMSNorm 吃掉了。

---

## 五、一个完整的 trace

设 embedding 是 $E$，第 1 层 Attn 输出 $A_1$，第 1 层 MoE 输出 $M_1$。

**进 Attn 之前：**

```
residual      = E
hidden_states = RMSNorm(E)
```

**Attn 算完：**

```
hidden_states = A1
residual      = E
```

**post_attention_layernorm（fused add + norm）：**

```
residual      = E + A1
hidden_states = RMSNorm(E + A1)
```

**MoE 算完：**

```
hidden_states = M1
residual      = E + A1
```

**进第 2 层之前的 norm（fused add + norm）：**

```
residual      = E + A1 + M1
hidden_states = RMSNorm(E + A1 + M1)
```

**第 2 层 Attn：**

```
hidden_states = A2
residual      = E + A1 + M1
```

**进下一段 norm：**

```
residual      = E + A1 + M1 + A2
```

如此循环。

如果有人问"residual 什么时候被更新"，准确答案是：

> 更新 residual 的是**紧邻它们的 fused RMSNorm**，不是 Attn / MoE 本身。

Attn / MoE 做的事情是算出 `hidden_states`，然后 norm 那一步负责把它并入 `residual`。

---

## 六、DeepSeek V4 这种 HC 模型里的特殊性

V4 主干里有一段特别值得注意：

```python
hidden_states = self.embed_input_ids(input_ids)            # [T, H]
hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
# -> [T, C, H]
```

embedding 被 `repeat` 成了 $C$ 份（HC 多流）：

```
token t
  stream 0  ── H dims
  stream 1  ── H dims
  stream 2  ── H dims
  stream 3  ── H dims
```

初始每条 stream 都是 $E$，但经过各 layer 之后，它们会走出不同的轨迹：

```
stream 0:  E → E + A0     → E + A0 + M0     → ...
stream 1:  E → E + A1     → E + A1 + M1     → ...
stream 2:  E → E + A2     → E + A2 + M2     → ...
stream 3:  E → E + A3     → E + A3 + M3     → ...
```

每条 stream 都有自己独立的 residual stream。`residual` 的 shape 通常也是 $[T, C, H]$。最后才由 `hc_head` 把 $C$ 条 stream 压回 $[T, H]$。

这也解释了那行注释：

```python
# 这里把 pre-hc_head 的多流残差 stash 进 _mtp_hidden_buffer
# → MTP / speculative 路径可能用到
```

MTP / speculative 路径需要的，是 `hc_head` 之前那个 $[T, C, H]$ 的多流 residual（已经被压成 $[T, H]$ 的版本反而不能用）。所以这条 residual 的"中间形态"是有语义的，不能随便丢掉。

---

## 七、读这种代码时的几个反直觉

下次再看到 `hidden_states, residual = layer(...)` 的时候可以记一下：

- **不要假设 `residual` 一直等于 layer 输入。** 它的值在每个 fused norm 里被改写。
- **`hidden_states` 和 `residual` 是同一条流水线上的两个寄存器。** `residual` 是累积量，`hidden_states` 是当前分支的 update / norm 之后的中间值。
- **Attn / MoE 本身不做 residual add。** 它们只算 `hidden_states`；add 那一手藏在紧邻的 fused RMSNorm 里。
- **如果某条路需要 pre-head 的多流状态，就不能依赖 `hc_head` 之后的 `hidden_states`，得自己 stash 一份 residual。** 这就是 `_mtp_hidden_buffer` 这类 buffer 出现的原因。

---

## 八、这些优化在 vllm-ascend 里长什么样

把 `hidden_states` 和 `residual` 拆开维护，能拿到 vllm-ascend 在推理侧的 7 类好处，Transformer 的数学结构本身没变。下面每一条都对应到仓库里的具体实现。

### 8.1 Fused `residual add + RMSNorm`：省 HBM 读写

入口是 `AscendRMSNorm.forward_oot`（`vllm_ascend/ops/layernorm.py:63-86`），同时调 `npu_add_rms_norm` 或 `npu_add_rms_norm_bias`：

```python
residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
x, _, residual = torch_npu.npu_add_rms_norm(
    x, residual, self.weight, self.variance_epsilon
)
```

一次 kernel 内同时返回 norm 输出和更新后的 residual，避免朴素实现里"写一次 `x + attn_out`，再读一次做 norm"的中间 HBM 往返。Gated 版本（Qwen3-Next / GDN）走同文件 `:160-199` 的 `AscendRMSNormGated` / `LayerNormFn`，把 `norm * silu(z)` 也一起 fuse。

### 8.2 Residual 用更高精度累加

朴素预想是 `residual.float(); residual += hidden_states.float()` 这种显式 cast。vllm-ascend 不这么做，是通过：

- 上层 vllm config 决定 `residual` / `hidden_states` 的 dtype（`vllm_ascend/ops/layernorm.py:38` 的 `factory_kwargs`）。
- `enable_custom_op()`（`vllm_ascend/utils.py`）在 `npu_add_rms_norm`（不带 bias）和 `npu_add_rms_norm_bias`（带 bias）之间切换。
- CANN 内部按 op 约定做高精度累加，不在 Python 侧额外 cast。

DSV4 主干里那行 `residual = hidden_states.clone()`（`vllm_ascend/models/deepseek_v4.py:1004`）是为了给 `hc_post` 留一份"还没被 fused norm 改写"的 residual，跟升精度无关。

### 8.3 Residual BF16，Attn / MoE 走量化（高精度主干 + 量化分支）

这是仓库里最显式的一类，集中在 `compilation/passes/norm_quant_fusion_pass.py`：

| Pattern | 替换 | 用途 |
|---|---|---|
| `AddRMSNormQuantPattern`（`:45`） | `npu_add_rms_norm` + `quantize` → `npu_add_rms_norm_quant` | W8A8 静态量化 |
| `AddRMSNormQuantPatternWithBias`（`:89`） | 同上，加 bias | bias-aware 量化 |
| `AddRMSNormQuantSPPattern*`（`:152` / `:214`） | 在融合后再走 `maybe_all_gather_and_maybe_unpad` | Sequence Parallel 下的 fusion |
| `AddRMSNormDynamicQuantPattern*`（`:279` / `:322`） | 动态量化版 | 运行时算 scale |
| `AddRMSNormDynamicMXQuantPattern*`（`:477` / `:524`） | MX FP8 动态量化 | A5 设备 |
| `RMSNormDynamicMXQuantPattern*`（`:571` / `:611`） | 纯 RMSNorm + MX quant | 不带 residual add |

`residual` 的 dtype 仍由调用方决定（一般 BF16），而 norm 输出直接以 FP8/INT8 形式进 Attn；这就是"residual 高精度主干 + 低精度计算分支"。`AddRMSNormQuantFusionPass.__init__`（`:680`）在开头还显式判了 W4A4 int4 方案并禁用 fusion（`_model_uses_w4a4_quant` 在 `:654`）。

### 8.4 MoE 的 dispatch / All-to-All：residual 留在本地

DSV4 的 MoE 入口在 `DeepseekV4MoE.forward`（`vllm_ascend/models/deepseek_v4.py:459-524`）：

```python
if self.is_sequence_parallel:
    hidden_states = sequence_parallel_chunk(hidden_states)        # :475

fused_moe_out = self.experts(
    hidden_states=hidden_states, ...                              # :479
)
...
if self.is_sequence_parallel:
    final_hidden_states = tensor_model_parallel_all_gather(...)
```

experts 只接收 fused norm 输出的 `hidden_states`（branch 的 delta），residual 完全不进 experts。dispatch / all-to-all / combine 整条流水线由 `vllm_ascend/ops/fused_moe/prepare_finalize.py` 负责，策略在文件 `:372` 附近：

> `TP AG → Attn → TP RS → TP AG → DP AG → MoE → DP RS → TP RS`  
> 进一步把 `TP AG + DP AG` 合成 EP All-Gather，`TP RS + DP RS` 合成 EP Reduce-Scatter。

整条搬运链只动 `hidden_states` 和 expert 输出，residual 始终留在原 rank 等待下一层 fused norm。

### 8.5 Tensor Parallel 通信 overlap

主战场在 `compilation/passes/sequence_parallelism.py`：

- `_maybe_all_reduce_search_pattern`（`:56-91`）显式建模 `AllReduce → maybe_chunk_residual → AddRMSNormBias` 的整段：
  - `maybe_all_reduce` 的结果 `alias` 同时被两路消费：一路走 `maybe_chunk_residual` 拿来做 residual，另一路直接做 norm input；
  - `npu_add_rms_norm_bias` 在 pattern 里标 `_users=2`，表示 norm 输出既给当前 branch 也给下一 branch；
  - 返回 `MultiOutputPattern([output, residual])`，两个 tensor 都要保留。
- `MiddleAllReduceRMSNormPattern`（`:112-151`）是关键替换：把 `all_reduce → AddRMSNormBias → next` 换成 `reduce_scatter → AddRMSNormBias → all_gather`。`reduce_scatter` 完成后立刻进入 fused norm，norm 完再 `all_gather` 出去；通信和计算在时间轴上重叠，不严格等同步。

MoE 路径对应的通信 overlap 在 `compilation/passes/sequence_parallelism_moe.py`，跟 fused_moe prepare/finalize 配合。

### 8.6 避免冗余 clone

仓库默认策略是"从一开始就把 residual 当一等公民"：

- `vllm_ascend/ops/layernorm.py:71` `residual = torch.ops.vllm.maybe_chunk_residual(x, residual)` — 在 fused norm 入口按需 chunk，不预先 clone。
- `vllm_ascend/compilation/passes/sequence_parallelism.py:66-72` 在 pattern 里用 `aten.alias.default` 表示"同一份 tensor 的多用户"，避免被调度器去 clone。
- `vllm_ascend/models/deepseek_v4.py:1004` `residual = hidden_states.clone()` 是 DSV4 唯一一处显式 clone，理由是 HC block 入口要拿一份"还没被 fused norm 改写"的 residual 给 `hc_post` 消费——这是必要的 clone。

### 8.7 MTP / Speculative 复用 pre-hc_head 的多流 residual

DSV4 main model 在 `hc_head` 之前把 $[T, C, H]$ 多流残差 stash 进 `_mtp_hidden_buffer`（`vllm_ascend/models/deepseek_v4.py:1318` 附近，`get_mtp_target_hidden_states`）。直接消费方：

- `vllm_ascend/models/deepseek_v4_mtp.py:108-136` `DeepSeekMultiTokenPredictorLayer.forward`：MTP 拿到的 `previous_hidden_states` 就是上一步 stash 的 pre-hc_head 多流 residual；`hnorm` 之后再 `e_proj + h_proj` 注入新 hidden。
- `vllm_ascend/models/deepseek_v4_mtp.py:175-205` MTP 的 `compute_logits`：从 pre-hc_head 状态走 `hc_head` 再 `logits_processor`。
- `vllm_ascend/models/deepseek_v4_dspark.py` DSpark proposer 走类似路径，多了一层"从 main model 抽 pre-hc_head residual"的接口。
- `vllm_ascend/spec_decode/extract_hidden_states_proposer.py` 名字直接写明"抽取 hidden states"做 proposer，对应 `_mtp_hidden_buffer` 这类 stash 机制。

### 8.8 一张总表

| 优化模式 | 关键实现 | 文件 : 行 |
|---|---|---|
| Fused add + RMSNorm | `AscendRMSNorm.forward_oot` | `vllm_ascend/ops/layernorm.py:63` |
| Gated norm + silu | `AscendRMSNormGated` / `LayerNormFn` | `vllm_ascend/ops/layernorm.py:160` |
| Residual 精度控制 | dtype 由 vllm config + `enable_custom_op()` 切换算子 | `vllm_ascend/ops/layernorm.py:38` / `vllm_ascend/utils.py` |
| Residual + quant fusion | `AddRMSNorm*Quant*Pattern` | `vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:29-651` |
| MoE dispatch / all-to-all | `prepare_finalize` + `DeepseekV4MoE.forward` | `vllm_ascend/ops/fused_moe/prepare_finalize.py:372` / `vllm_ascend/models/deepseek_v4.py:459` |
| TP/EP 通信 overlap | `MiddleAllReduceRMSNormPattern` | `vllm_ascend/compilation/passes/sequence_parallelism.py:56-151` |
| MoE 通信 overlap | SP + MoE fusion pass | `vllm_ascend/compilation/passes/sequence_parallelism_moe.py` |
| 避免冗余 clone | `maybe_chunk_residual` + `aten.alias` + HC 入口 clone | `vllm_ascend/ops/layernorm.py:71` / `vllm_ascend/compilation/passes/sequence_parallelism.py:66` / `vllm_ascend/models/deepseek_v4.py:1004` |
| MTP/DSpark 复用 | MTP layer + DSpark proposer + extract hidden proposer | `vllm_ascend/models/deepseek_v4_mtp.py:108` / `vllm_ascend/models/deepseek_v4_dspark.py` / `vllm_ascend/spec_decode/extract_hidden_states_proposer.py` |

下次再看到 `hidden_states, residual = layer(...)` 的时候，可以拿这张表对照：每条边界基本都是上面某一个优化在落地。

---

## 九、相关文章

- [Speculative Decoding 全解析：从概率质量到 vLLM / Ascend 一次 Decode](/blog/speculative-decoding-rejection-sampling/) — 讲 `residual` 之外的另一条"残差"：rejection sampling 里把 $\max(0, q-p)$ 加回到分布上的"残差采样"。
- [CUDA Kernel Launch 全解析](/blog/cuda-kernel-launch/) — 讲 launch 一个 kernel 的成本，本文 8.1 提到的 HBM 读写次数会直接影响 kernel 之间的开销。
