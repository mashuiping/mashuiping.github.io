---
title: 从张量形状理解 DFlash 的 Block-Parallel Forward
description: 从 B、S、H、V 四个维度分析 DFlash 如何在一次 Transformer forward 中并行预测一个 token block，以及它与 Attention、Batch 和 Speculative Decoding 的关系。
pubDate: 2026-08-21
updatedDate: 2026-08-26
category: ai-infra
tags:
  - DFlash
  - Transformer
  - Speculative Decoding
  - LLM Inference
  - Attention
draft: false
---

DFlash 可以在一次 draft forward 中预测一个 token block。以 block size 16 为例，模型会同时处理同一条序列中的 16 个未来位置。这里扩展的是 sequence 维，batch size 不变。

两种 shape 表达的含义不同：

```text
[16, T]  = 16 条相互独立的序列
[B, 16]  = B 条序列，每条包含 16 个待预测位置
```

本文从 Transformer 的张量形状出发，沿 `input_ids → embedding → Q/K/V → hidden states → logits → token block` 分析 DFlash 的 block-parallel forward。为便于说明，文中的 Q/K/V 采用标准多头 Attention 表示；具体模型可能使用 GQA、不同的 mask 或压缩后的 context KV，但 sequence 维上的关系不变。

## 一、Transformer 已具备位置级并行能力

先约定四个维度：

| 符号 | 含义 | 示例 |
|---|---|---:|
| $B$ | batch size | 2 |
| $T$ | sequence length | 128 |
| $H$ | hidden size | 4096 |
| $V$ | vocabulary size | 150000 |

输入 token IDs 的 shape 为：

```text
input_ids: [B, T] = [2, 128]
```

Embedding 将每个 token 映射成一个 $H$ 维向量：

```text
[B, T]
   ↓ Embedding
[B, T, H] = [2, 128, 4096]
```

Transformer layers 保持 `[B, T, H]` 的主 shape，LM Head 再将 hidden size 投影到词表维度：

```text
[B, T, H]
   ↓ Linear(H → V)
[B, T, V] = [2, 128, 150000]
```

一次 forward 会为 $T$ 个位置分别计算 vocabulary logits。GPU 上执行的是批量矩阵运算，位置维本来就是并行的。

自回归生成的串行性来自输入依赖。第 $N+1$ 个位置需要以前面已经确定的 token 为条件：

$$
P(x_2\mid x_1),\quad
P(x_3\mid x_1,x_2),\quad
P(x_4\mid x_1,x_2,x_3)
$$

标准 GPT 生成下一个 token 后，才能把该 token 拼回输入并继续计算。Transformer 能并行处理已有位置，但未来位置的离散输入尚未确定。

## 二、DFlash 的输入组织：Anchor 与 MASK Block

DFlash 为未来位置准备一组特殊 token。假设当前 context 为“我 喜欢 吃”，block size 为 16，draft 输入可以表示为：

```text
吃  MASK  MASK  MASK  MASK  ...  MASK
↑
anchor
```

对应 shape 为：

```text
B = 1
S = block_size = 16

draft_input_ids: [B, S] = [1, 16]
```

`<MASK>` 是 vocabulary 中的特殊 token，拥有独立 token ID 和 embedding。相同的 MASK embedding 进入不同位置后，还会叠加不同的位置信息：

```text
MASK 0  → position 129
MASK 1  → position 130
MASK 2  → position 131
...
MASK 15 → position 144
```

RoPE 等位置编码使 16 个 MASK position 具有不同的序列坐标。Embedding 后得到：

```text
draft_input_ids [B, S]
        ↓
draft_embeddings [B, S, H]
```

当 `B=2`、`S=16`、`H=4096` 时，shape 为 `[2, 16, 4096]`。这里的 2 表示两条请求，16 表示每条请求内部的待预测位置。

| 维度 | 运行时含义 | Attention 是否跨越该维度 |
|---|---|---|
| $B$ | 相互独立的请求或序列 | 否 |
| $S$ | 同一序列中的 token positions | 是，受 attention mask 约束 |

把 batch size 设置为 16 会创建 16 条相互隔离的序列，无法表达一个长度为 16 的未来 block。

## 三、从 Q/K/V Shape 看 Block 并行

设：

```text
B          = 2
S          = 16
H          = 4096
num_heads  = 32
head_dim   = 128
```

因为 $4096=32\times128$，draft embeddings 经过 Q projection 后可以整理为：

```text
X: [2, 16, 4096]
        ↓ Q projection
   [2, 16, 32, 128]
        ↓ transpose
Q: [2, 32, 16, 128]
```

标准表示为 `[B, heads, sequence, head_dim]`。其中 `sequence=16`，对应 16 个并行 query positions。

Attention score 来自 $QK^\top$：

```text
Q:      [B, heads, Tq, d]
Kᵀ:     [B, heads, d, Tk]
Q @ Kᵀ: [B, heads, Tq, Tk]
```

若当前 context length 为 128，draft block size 为 16，可以用下面的简化 shape 表示：

```text
Q:   [B, heads,    16, d]
K/V: [B, kv_heads, 144, d]
```

真实模型可能使用 GQA，因此 `kv_heads` 可以小于 `heads`。逻辑上，16 个 query 会同时对可见的 context 和 draft block 位置计算 Attention：

```text
attention_scores: [B, heads, 16, 144]
```

对应公式为：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
$$

以 `B=1`、`heads=32`、`d=128` 为例：

```text
Q:  [1, 32, 16, 128]
Kᵀ: [1, 32, 128, 144]

Q @ Kᵀ
  → [1, 32, 16, 144]

softmax(scores) @ V
  → [1, 32, 16, 128]

concat heads
  → [1, 16, 4096]
```

经过一层 Attention 后，16 个位置仍然保留：

```text
[B, 16, H] → [B, 16, H]
```

Attention 生成的是各位置的 hidden representations。离散 token 要到 LM Head 和 sampling 阶段才确定。

## 四、Target Context 如何进入 Draft Forward

仅使用 MASK embedding 提供的信息有限。DFlash 会利用 target model 已经计算出的 context hidden states，帮助 draft model 预测后续 block。

假设 context length 为 128：

```text
target_hidden:   [B, 128, H]
draft_embedding: [B,  16, H]
```

在简化的 Attention 视角中：

```text
Q = Wq(draft_embedding)

K = concat(K_context, K_draft)
V = concat(V_context, V_draft)
```

target model 提供的 hidden states 经过 draft model 所需的投影或预计算，形成 draft context KV；draft block 则提供本轮 query 以及相应的 block KV。最终可见范围还取决于具体 attention mask。

这部分实现带来两个边界：

- target hidden states 与 draft context KV 是不同的 tensor，后者需要经过 draft 模型的投影；
- context length 与 block size 共同决定 $T_k$，每个 query 实际可见的位置还要由 attention metadata 和 mask 限定。

因此 `[B, heads, 16, 144]` 是用于说明 shape 的简化结果，不能替代具体 backend 的 mask 与 KV layout。

## 五、Hidden States 到 Token Block

若 Transformer 最终输出：

```text
hidden_states: [B, 16, H] = [1, 16, 4096]
```

LM Head 将每个位置独立映射到 vocabulary 维：

```text
[1, 16, 4096]
        @
[4096, 150000]
        ↓
[1, 16, 150000]
```

因此：

```text
logits: [B, 16, V]
```

执行 argmax 或 sampling 后得到：

```python
pred = logits.argmax(dim=-1)
```

```text
pred: [B, 16]
```

这一步产生整个 block 的离散候选 token。

### Block 内位置如何建立依赖

同一个 forward 中，后续位置看不到前面位置最终采样出的离散 token。输入仍然是 anchor 与 MASK block：

```text
MASK0  ─┐
MASK1  ─┤
MASK2  ─┤
...     ├──→ Transformer → h0 h1 ... h15 → LM Head → token block
MASK15 ─┘
```

block 内的依赖通过连续 hidden representations 传递。每一层 Attention 都会更新各位置的表示，最终一层再将这些表示交给 LM Head：

```text
Layer 1     → 初始语义结构
Layer 2     → 位置间继续交互
...
Final Layer → LM Head → 离散 token
```

这种并行方式减少了 draft 阶段的串行依赖，同时也弱化了“前一个离散 token 已经确定”这一条件。模型训练方式、attention mask 和 block 内协调能力会直接影响候选质量。

## 六、DFlash 在 Speculative Decoding 中的职责

传统 Speculative Decoding 的 draft model 通常按自回归方式生成多个候选，再由 target model 一次验证：

```text
Draft Model:
token1 → token2 → token3 → ... → token16
                              ↓
Target Model:             并行验证
```

DFlash 将 draft 阶段也改成 block-parallel forward：

```text
Draft Model:  [token1 token2 token3 ... token16]
                           ↑
                      一次 draft forward
                           ↓
Target Model:         并行验证候选
```

Target model 只接受从开头连续通过验证的 prefix。假设 DFlash 提出：

```text
相 对 论 ， 并 对 现 代 物 理 学 产 生 了
```

验证结果为：

```text
相 ✓
对 ✓
论 ✓
， ✓
并 ✓
对 ✓
现 ✓
代 ✓
物 ✓
理 ✓
学 ✓
产 ×
```

本轮可以接受“相对论，并对现代物理学”这一连续 prefix，后续候选被拒绝。最终输出仍由 target verification 决定，draft block 无需全部命中。

DFlash 的性能收益取决于多个因素：

- draft forward 的成本；
- block size；
- target verification 的成本；
- 每轮连续接受的 token 数；
- 额外输入准备、KV 管理和采样开销。

“一次预测 16 个位置”描述的是执行方式，不代表每轮都能推进 16 个 token，也不能直接推出端到端吞吐提升。

## 七、完整 Shape 示例

取以下参数：

| 参数 | 值 |
|---|---:|
| Batch size $B$ | 1 |
| Context length $C$ | 128 |
| Block size $S$ | 16 |
| Hidden size $H$ | 4096 |
| Attention heads | 32 |
| Head dimension $d$ | 128 |
| Vocabulary size $V$ | 150000 |

完整数据流为：

```text
Target context IDs
[1, 128]
    ↓ Target Transformer
Target hidden
[1, 128, 4096]
    ↓ Draft context projection / KV precompute
Draft context KV

Draft block IDs
[anchor, MASK, ..., MASK]
[1, 16]
    ↓ Embedding
[1, 16, 4096]

Q
[1, 32, 16, 128]

K/V（context + draft block）
[1, kv_heads, 144, 128]

Q @ Kᵀ
[1, 32, 16, 144]

Attention output
[1, 32, 16, 128]
    ↓ concat heads
[1, 16, 4096]
    ↓ Transformer layers
[1, 16, 4096]
    ↓ LM Head
[1, 16, 150000]
    ↓ argmax / sampling
[1, 16]
```

这条 shape 链说明了一次 block forward 的数据规模。具体实现还会将 batch 和 token 维展平，使用 paged KV Cache，并按 backend 要求构造 attention metadata；调试时需要结合实际 tensor layout 还原逻辑 shape。

## 八、源码分析与调试检查项

跟踪 DFlash 实现时，可以围绕以下 tensor 建立 shape 账本：

```text
Draft token IDs
[B, S]

Draft embeddings
[B, S, H]

Q
[B, heads, S, d]

K/V
[B, kv_heads, context_len + S, d]

Hidden states
[B, S, H]

Logits
[B, S, V]

Selected tokens
[B, S]
```

除了 shape，还要记录以下状态：

| 检查项 | 作用 |
|---|---|
| `positions` | 确认 16 个 query 对应连续且正确的位置 |
| `query_start_loc` | 区分 batch 中各请求的 token 区间 |
| block table / slot mapping | 确认 context KV 与 query KV 的物理地址 |
| attention mask / causal 配置 | 确认每个 query 的可见范围 |
| sample indices | 确认哪些 hidden rows 进入 LM Head 或采样 |
| rejected token count | 确认下一轮 positions、长度和 KV 状态回滚 |

排查输出差异时，建议沿下面的顺序比较：

```text
draft input_ids / positions
  → context/query slot mappings
  → attention metadata
  → draft hidden states
  → logits [B, S, V]
  → sampled token block [B, S]
  → target verification result
```

如果 `[B, S, H]` 已经出现差异，继续比较最终 token 只能看到差异传播；应回到该层的输入、mask 和 KV 状态定位首次分叉。

## 九、总结

DFlash 将同一条序列中的 $S$ 个未来位置组织成 `[B, S]` 的 draft block。Transformer 在一次 forward 中并行计算这些位置的 hidden representations，LM Head 将 `[B, S, H]` 映射为 `[B, S, V]`，再生成 `[B, S]` 的候选 token。

batch 维负责隔离独立序列，sequence 维承载同一序列内的 block positions。block 内位置通过多层 hidden representations 交换信息，无法直接使用彼此最终采样出的离散 token。DFlash 将这批 token 交给 target model 验证，实际推进长度由连续接受的 prefix 决定。
