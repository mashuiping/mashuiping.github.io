---
title: DeepSeek V4 在 vLLM Ascend 中的推理链路：mHC、DSA 与三类 KV Cache
description: 沿一次 token forward 追踪 DeepSeek V4 在 vLLM Ascend 中的执行路径，分析 mHC 多流残差、DSA 稀疏注意力以及三类 KV Cache 的串联方式。
pubDate: 2026-08-20
updatedDate: 2026-08-25
category: ai-infra
tags:
  - DeepSeek V4
  - vLLM
  - vLLM-Ascend
  - NPU
  - LLM Inference
draft: false
---

DeepSeek V4 接入 vLLM Ascend 后，模型主干出现了两项区别于常规 Transformer 的实现。

第一项是 hidden states 的形状变化：embedding 输出为 `[T, H]`，进入模型后扩展为 `[T, C, H]`，在 Attention 和 MoE 前恢复到 `[T, H]`。第二项是 KV Cache 的组织方式：一次 Attention 会同时访问 Compressor、Indexer 和 SWA 对应的多组状态。

本文沿一次 token forward 分析三个部分：

- mHC 在多流主干和单流子层之间的形状转换；
- Indexer、Compressor 和 SWA 三类状态在 DSA 中的作用；
- Python 模型层、vLLM Attention 接口和 Ascend 后端的职责边界。

代码基于 vLLM Ascend `main @ b4b04c5eb`。本文只讨论当前仓库里的推理路径，不展开训练、反向传播和算子内部实现。

## 一、整体执行链路

设一次 forward 包含 `T` 个 token，hidden size 为 `H`，`hc_mult=C`。模型主干的数据流如下：

```text
input_ids [T]
  │
  ▼
embedding [T, H]
  │  unsqueeze(1).repeat(1, C, 1)
  ▼
多流残差 [T, C, H]
  │
  │  每个 Decoder Layer：
  │    hc_pre   [T, C, H] → [T, H]
  │    Attention（DSA）
  │    hc_post  [T, H] → [T, C, H]
  │
  │    hc_pre   [T, C, H] → [T, H]
  │    MoE
  │    hc_post  [T, H] → [T, C, H]
  ▼
最后一层输出 [T, C, H]
  │  hc_head
  ▼
[T, H] → RMSNorm → compute_logits → logits [T, V]
```

入口是 `AscendDeepseekV4ForCausalLM.forward`。它只调用 `self.model(...)` 并返回 hidden states；logits 由单独的 `compute_logits` 计算。因此，调试时停在模型 `forward` 的返回位置，看不到 logits 是正常的。

hidden states 的形状转换发生在 `DeepseekV4Model.forward`：

```python
if get_pp_group().is_first_rank:
    hidden_states = hidden_states.unsqueeze(1).repeat(
        1, self.hc_mult, 1
    )

for layer in islice(self.layers, self.start_layer, self.end_layer):
    hidden_states, residual = layer(
        positions,
        hidden_states,
        residual,
        llama_4_scaling,
        input_ids=input_ids,
    )

hidden_states = self.hc_head(
    hidden_states,
    self.hc_head_fn,
    self.hc_head_scale,
    self.hc_head_base,
)
```

层与层之间传递 `[T, C, H]` 多流残差，Attention 和 MoE 处理 `[T, H]` 单流表示，`hc_pre` 和 `hc_post` 负责两种形状之间的映射。

## 二、mHC：主干是多流，子层仍是单流

mHC 是 Manifold-Constrained Hyper-Connections。普通 Transformer 的残差主干只有一条：

$$
x_{l+1}=x_l+F(x_l)
$$

在当前实现里，mHC 把主干扩成 `C` 条流，层间状态形状为 `[T, C, H]`。为复用单流 Attention 和 MoE，代码在每个子层前后增加形状映射：

```text
多流 residual ──┬── hc_pre ──► 单流 x ──► Attention / MoE ──┐
                │                                           │
                └──────────────── hc_post ◄─────────────────┘
                                   │
                                   ▼
                              新的多流 residual
```

### `hc_pre`：多流聚合为单流

`DeepseekV2DecoderLayer.__init__` 根据 `hc_mult` 建立 Attention 和 FFN 各自的 mHC 参数：

```python
self.hc_mult = hc_mult = config.hc_mult
mix_hc = (2 + hc_mult) * hc_mult
hc_dim = hc_mult * config.hidden_size

self.hc_attn_fn = nn.Parameter(
    torch.empty(mix_hc, hc_dim, dtype=torch.float32)
)
self.hc_attn_base = nn.Parameter(
    torch.empty(mix_hc, dtype=torch.float32)
)
self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
```

当 `C=4` 时，`mix_hc=(2+4)×4=24`。这 24 个值对应三组映射参数：

| 映射 | 数量 | 作用 |
|---|---:|---|
| pre | `C` | 把 `C` 条流加权合成一条子层输入 |
| post | `C` | 把子层输出写回 `C` 条流 |
| comb | `C×C` | 让原有的 `C` 条残差流彼此混合 |

`hc_pre` 调用 `npu_hc_pre_v2`，返回单流 hidden states，以及稍后 `hc_post` 要用的 `post` 和 `comb`：

```python
hidden_states, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
    x,
    hc_fn,
    hc_scale,
    hc_base,
    self.hc_mult,
    self.hc_sinkhorn_iters,
    self.norm_eps,
    self.hc_eps,
)
```

Python 侧将 sigmoid、Sinkhorn 所需的参数和配置交给 NPU 自定义算子。当前代码可以确认输入输出关系与算子边界；内部数值过程还需要结合 CANN 实现验证。

### `hc_post`：子层输出写回多流

子层算完得到 `[T, H]`，`hc_post` 同时接收三样东西：子层输出、进入子层前保存的多流 residual、`hc_pre` 生成的 `post/comb`。

```python
hidden_states = torch.ops._C_ascend.npu_hc_post(
    x.unsqueeze(0),
    residual.unsqueeze(0),
    post.unsqueeze(0),
    comb.unsqueeze(0),
).squeeze(0)
```

`post` 决定子层结果如何写回，`comb` 决定原有多流 residual 如何重新混合，两部分共同生成下一份 `[T, C, H]` 状态。

### 一层里会走两遍 mHC

Decoder Layer 的执行顺序很规整：

```python
# Attention 半层
residual = hidden_states.clone()
hidden_states, post, comb = self.hc_pre(...)
hidden_states = self.input_layernorm(hidden_states)
attn_kwargs = {
    "positions": positions,
    "hidden_states": hidden_states,
    "llama_4_scaling": llama_4_scaling,
}
hidden_states = self.self_attn(**attn_kwargs)
hidden_states = self.hc_post(hidden_states, residual, post, comb)

# MoE 半层
residual = hidden_states.clone()
hidden_states, post, comb = self.hc_pre(...)
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states, input_ids)
hidden_states = self.hc_post(hidden_states, residual, post, comb)
```

因此每层各调用两次 `hc_pre` 和 `hc_post`。所有层结束后，`hc_head` 再把 `[T, C, H]` 收回 `[T, H]`，交给最终 RMSNorm 和 LM Head。

## 三、DSA 的调用边界与状态组成

从 Decoder Layer 进入 `self.self_attn` 后，形状已经回到 `[T, H]`。`DeepseekV4Attention` 会把 DSA 所需模块打包成 `DSAModules`，交给 `AscendDeepseekSparseAttention`：

```text
DeepseekV4Attention
  └─ AscendDeepseekSparseAttention
       └─ DSAAttention
            └─ AscendDSAImpl / AscendSFAImpl
```

`AscendDeepseekSparseAttention.forward` 自己没有写 Attention 数学，它分配输出 tensor，然后调用自定义 op：

```python
torch.ops.vllm.dsa_forward(
    hidden_states,
    need_gather_q_kv,
    output,
    self.prefix,
)
```

`dsa_forward` 从 vLLM 的 `ForwardContext` 取 metadata，拼好当前层的 KV Cache，再把工作交给实际 backend：

```python
kv_cache = _build_kv_cache(self, forward_context)
self.dsa_attn.impl.forward(
    self.dsa_attn.layer_name,
    hidden_states,
    kv_cache,
    attn_metadata,
    need_gather_q_kv,
    output,
)
```

这层封装沿用 vLLM 的 Attention 调度和 cache 管理，同时将完整 DSA 路径纳入 ACL Graph 可捕获的自定义 op 边界。

### DSA backend 使用的多类 cache

`_build_kv_cache` 是理解 DSA 数据面的关键。非 A5 设备最终传给 backend 的是一个 6 元组；A5 还会增加 `indexer_full_cache`：

| 顺序 | Cache | 保存什么 | 谁使用 |
|---:|---|---|---|
| 1 | `compress_kv_cache` | 主 Attention 的压缩 KV | DSA/SFA Attention |
| 2 | `swa_kv_cache` | 滑动窗口内的近期 KV | SWA 路径 |
| 3 | `state_cache` | 主 Compressor 的状态 | KV 压缩 |
| 4 | `indexer_state_cache` | Indexer Compressor 的状态 | 稀疏索引构建 |
| 5 | `indexer_k_cache` | Indexer 使用的历史 K | top-k 打分 |
| 6 | `indexer_scale_cache` | Indexer K 的量化 scale | top-k 打分/反量化 |
| 7（A5） | `indexer_full_cache` | A5 路径额外展开的 Indexer cache | A5 backend |

Indexer 也需要持久 cache。它要对当前 query 和历史 token 重新打分；保留历史 K 后，下一次 decode 才能构造候选集合。

## 四、Indexer、Compressor 与 SWA 的职责划分

三类状态分别对应候选选择、历史压缩和近期窗口。

### Indexer：选择历史 token

`Indexer` 里有 query 投影 `wq_b`、打分权重投影 `weights_proj`、Indexer K Cache，以及可选的 Compressor：

```python
self.wq_b = ReplicatedLinear(
    self.q_lora_rank,
    self.n_heads * self.head_dim,
    ...
)
self.weights_proj = ReplicatedLinear(
    config.hidden_size,
    self.n_heads,
    ...
)

k_dtype = (
    torch.float8_e4m3fn
    if get_ascend_device_type() == AscendDeviceType.A5
    else torch.int8
)
```

当前实现里，A5 的 Indexer K 使用 `float8_e4m3fn`，其他设备走 `int8`。`AscendDeepseekV4IndexerCache.get_kv_cache_spec` 负责把 cache 的 block size、head size、dtype、压缩比例和 scale 信息注册给 vLLM 的 KV Cache 管理层。

一次 decode 先使用当前 query 和历史 Indexer K 计算分数，再选出 `index_topk` 个位置供稀疏 Attention 使用。top-k 结果只属于当前 step，历史 K 则跨 step 保存在 cache 中。

### Compressor：压缩历史 KV

Indexer 负责选择位置，Compressor 用于降低历史状态的存储开销。当前 `Compressor` 支持的 `compress_ratio` 为 `4` 和 `128`；其他值会直接抛出 `ValueError`。

它包含 `wkv`、`wgate`、可学习参数 `ape` 和 `state_cache`。当 `compress_ratio == 4` 时还会打开 overlap 布局：

```python
self.overlap = compress_ratio == 4
self.coff = 1 + self.overlap
self.ape = nn.Parameter(
    torch.empty(
        compress_ratio,
        self.coff * self.head_dim,
        dtype=torch.float32,
    )
)
```

仓库中还提供了 Hadamard 参考实现：输入先补到 2 的幂维度，乘 Hadamard 矩阵，缩放后再裁回原维度。正交旋转负责重新分布数值，有损过程还包括后续压缩与量化。缺少端到端测量时，无法根据这段参考代码判断具体的精度或吞吐收益。

`dsa_v1.py` 还创建了独立的 NPU stream，用来重叠部分 Hadamard 计算和 KV 搬运。这是执行层优化，不改变上面的 cache 语义。

### SWA：保留近期窗口

top-k 按分数选择历史位置，无法保证近期 token 入选。SWA 单独保存近期窗口，使 backend 能把滑动窗口位置和稀疏索引一起纳入 Attention。

Ascend 侧用 `AscendDeepseekV4SWACache` 注册这块 cache，底层 spec 是 `AscendSlidingWindowMLASpec`。它在上游 `SlidingWindowMLASpec` 之外带有 Ascend 所需的 dtype、模型版本、对齐和页面 padding 信息。

DSA 的可访问集合可以表示为：

```text
当前 query 可访问的位置
  = Indexer 选出的 top-k 历史位置
  + SWA 覆盖的近期窗口
```

实际 backend 还要处理重复位置、block 对齐、不同请求的长度和 metadata，不能简单把两边的数量直接相加。

## 五、SFA 与 DSA backend 的选择

仓库里同时存在 `attention/sfa_v1.py` 和 `attention/dsa_v1.py`。两者都服务 DeepSeek V4 稀疏 Attention，但执行组织和适用路径不同。

- `AscendSFAImpl` 继承 `MLAAttentionImpl`，包含 Indexer 选择、稀疏 Flash Attention、KV offload 等路径。
- `AscendDSAImpl` 继承 `AttentionImplBase`，包含 Compressor、SWA metadata、DSpark/MTP 相关处理以及多 stream 流程。

模型层通过 `DSAAttention`、backend 选择和 `ForwardContext` 将具体实现交给运行时配置。调试请求时，应检查实际注册的 backend 和 `self.dsa_attn.impl` 类型。

`sfa_v1.py` 仅在 `sparse_kv_offload_config.enabled` 时加载 KV offload 对应的 metadata builder 和 impl，因此 KV offload 属于可选路径。

## 六、单 Token 执行示例

取 `T=1`、`C=4`，一次 forward 的控制流和数据流如下：

| 阶段 | 主要对象 | 输入 → 输出 | 状态变化 |
|---|---|---|---|
| Embedding | `DeepseekV4Model` | `[1] → [1,H]` | 无 KV 写入 |
| 扩流 | `DeepseekV4Model.forward` | `[1,H] → [1,4,H]` | 建立多流 residual |
| Attention 前 | `hc_pre` | `[1,4,H] → [1,H]` | 生成 `post/comb` |
| 稀疏 Attention | `AscendDeepseekSparseAttention` | `[1,H] → [1,H]` | 更新 Compressor、Indexer、SWA 相关 cache |
| Attention 后 | `hc_post` | `[1,H] → [1,4,H]` | 子层输出写回多流 |
| MoE 前后 | `hc_pre` / `hc_post` | 多流 → 单流 → 多流 | 更新多流 residual |
| 模型尾部 | `hc_head` | `[1,4,H] → [1,H]` | 多流收回单流 |
| 输出头 | `compute_logits` | `[1,H] → [1,V]` | 生成 logits |

`DeepseekV4Model.forward` 会在 `hc_head` 之前把展平的多流状态写入 `_mtp_hidden_buffer`。MTP draft model 使用 pre-`hc_head` 状态，主模型输出则是收流并 Norm 后的 hidden states。启用 FlashComm1 时，代码先执行 tensor-parallel all-gather，再移除 padding 并写入 buffer。

## 七、源码定位与调试入口

继续跟踪当前实现时，可以按调用顺序检查以下位置：

1. `vllm_ascend/models/deepseek_v4.py`
   - `DeepseekV4Model.forward`
   - `DeepseekV2DecoderLayer.forward`
   - `hc_pre`、`hc_post`、`hc_head`
   - `Indexer`、`Compressor`
2. `vllm_ascend/ops/dsa.py`
   - `AscendDeepseekSparseAttention.forward`
   - `dsa_forward`
   - `_build_kv_cache`
3. `vllm_ascend/attention/sfa_v1.py`
   - `AscendSFAImpl`
   - Indexer 选择与 sparse KV offload 分支
4. `vllm_ascend/attention/dsa_v1.py`
   - `AscendDSAImpl`
   - Compressor、SWA metadata 和多 stream 流程
5. `vllm_ascend/core/kv_cache_interface.py`
   - `AscendSlidingWindowMLASpec`

## 八、总结

mHC 管理层间表示：主干保留 `[T, C, H]` 多流 residual，Attention 和 MoE 在 `[T, H]` 单流上执行。DSA 根据 Indexer 结果与 SWA 窗口确定当前 query 可访问的历史位置，Compressor、Indexer Cache 和 SWA Cache 保存各自所需的跨 step 状态。模型层负责组织模块和张量，vLLM Attention 接口提供调度与 cache 管理，Ascend backend 完成设备侧执行。
