---
title: 从 Transformer Shape 理解 DFlash：为什么一次 Forward 能生成一批 Token？
description: 从 B、T、H、V 四个维度出发，解释 DFlash 为什么能在一次 Transformer forward 中并行预测一个 token block，以及它和 batch、Attention、Speculative Decoding 的关系。
pubDate: 2026-08-21
updatedDate: 2026-08-21
category: ai-infra
tags:
  - DFlash
  - Transformer
  - Speculative Decoding
  - LLM Inference
  - Attention
draft: false
---

> 大语言模型不是一个 token 一个 token 地生成吗？DFlash 所谓“一次生成 16 个 token”到底是什么意思？
>
> 关键不在 batch 维，而在 sequence / token position 维：这 16 个 token 是同一条序列里的 16 个 query positions。

最开始，我也产生过一个很自然的猜测：假设模型输入是 $B\times T$，其中 $B$ 是 batch size，$T$ 是 sequence length，那么所谓“同时生成 16 个 token”，是不是把 $B$ 设置成 16？

答案是：**不是。**

真正理解这个问题以后，会发现 DFlash 的核心和 Transformer 最基础的 tensor shape 密切相关：

> 并行预测 16 个 token，增加的是 sequence / token position 这一维，而不是 batch 维。

本文不假设读者了解扩散模型。我们从最基础的 Transformer 输入输出开始，一步一步理解 DFlash 为什么能够一次预测一整个 token block。

---

## 一、先忘掉 DFlash：Transformer 的输入输出是什么？

先约定四个维度：

| 符号 | 含义 | 示例 |
|---|---|---:|
| $B$ | batch size | 2 |
| $T$ | sequence length | 128 |
| $H$ | hidden size | 4096 |
| $V$ | vocabulary size | 150000 |

输入是一批 token IDs：

```text
input_ids: [B, T] = [2, 128]
```

它表示一次处理两条序列，每条序列包含 128 个 token。经过 Embedding：

```text
[B, T]
   ↓ Embedding
[B, T, H] = [2, 128, 4096]
```

每个 token 现在都变成了一个 4096 维向量。经过若干层 Transformer 后，shape 仍然是 `[B, T, H]`。最后通过 LM Head：

```text
[B, T, H]
   ↓ Linear(H → V)
[B, T, V] = [2, 128, 150000]
```

这意味着 Transformer 可以在一次 forward 里为 128 个位置全部计算 vocabulary logits：

```text
position 0   → 150000 个 logits
position 1   → 150000 个 logits
position 2   → 150000 个 logits
...
position 127 → 150000 个 logits
```

这是理解后面所有内容的第一个关键点：**Transformer 的一次 forward，本来就会并行计算多个 token position。**

## 二、那为什么 GPT 还要一个 token 一个 token 地生成？

问题不是 Transformer 缺乏并行计算能力，而是：**未来 token 还不知道是什么，因此没有东西可以作为未来位置的输入。**

例如当前已有：

```text
我 今天 去
```

普通 GPT 先计算下一个 token“学校”，把它拼回输入，再计算“上”，然后再计算“数学”：

```text
我 今天 去
       ↓
      学校

我 今天 去 学校
            ↓
            上

我 今天 去 学校 上
               ↓
              数学
```

第 $N+1$ 个 token 必须等第 $N$ 个 token 真正生成以后才能继续。这就是 **autoregressive generation**。

## 三、DFlash 换了一个问题：先给未来准备一排空位

如果未来 token 还不知道，可以先用特殊 token 表示“这里还不知道”，例如 `<MASK>`。

假设当前 context 是“我 喜欢 吃”，我们希望预测未来一块 token。可以构造：

```text
吃  MASK  MASK  MASK  MASK  ...  MASK
↑
anchor
```

若 block size 为 16，draft block 的 shape 是：

```text
[anchor, MASK, MASK, ..., MASK]
shape = [1, 16]
```

这里特别需要注意：

```text
B = 1
S = block_size = 16
```

而不是 `B = 16`。这 16 个位置属于**同一条序列**的未来 16 个 token position。

## 四、为什么不能把 B 设置成 16？

假设输入 shape 是 `[16, 128]`，它表达的是 16 条独立序列：

```text
batch 0: sequence A
batch 1: sequence B
batch 2: sequence C
...
batch 15: sequence P
```

Batch 之间彼此隔离，attention 不会让 batch 0 去 attend batch 1。

DFlash 想表达的是：**同一个样本中，同时处理未来 16 个 token position。** 因此它需要的是 `[B, 16]`。例如 batch size 为 2：

```text
[2, 16]
   ↓ Embedding
[2, 16, 4096]
```

可以用一句话区分：

| 维度 | 表示什么 |
|---|---|
| $B$ | 有多少条彼此独立的序列 |
| $T$ / $S$ | 一条序列内部有多少个 token position |

## 五、`<MASK>` 到底是什么？

可以简单把 `<MASK>` 理解成 vocabulary 中的一个特殊 token。它有自己的 token ID，Embedding table 中也有对应的 $H$ 维向量：

```text
embedding[MASK] → 一个 H 维向量
```

所以一组 MASK 经过 Embedding 后会成为：

```text
[MASK, MASK, ..., MASK]
          ↓ Embedding
[B, 16, H]
```

例如 `[1, 16, 4096]`。

虽然这些位置最初使用同一个 MASK embedding，但它们并不完全相同，因为模型还有位置信息：

```text
MASK 0  → position 129
MASK 1  → position 130
MASK 2  → position 131
...
MASK 15 → position 144
```

借助 RoPE 等 positional encoding，模型知道“我是未来第 1 个位置”和“我是未来第 12 个位置”不是同一个位置。

## 六、Attention 本身并不“生成 token”

“Attention 一次生成了 16 个 token”严格来说并不准确。Attention 做的是：**同时计算 16 个位置的 hidden representations。**

真正把 hidden state 转换成 token 的，是最后的 LM Head 和 argmax / sampling：

```text
16 个位置
    ↓
Embedding
    ↓
Transformer / Attention
    ↓
16 个 hidden states
    ↓
LM Head
    ↓
16 × vocabulary logits
    ↓
argmax / sampling
    ↓
16 个 token
```

## 七、从 Q/K/V 的 Shape 看“并行 16 个位置”

假设：

```text
B          = 2
block_size = 16
H          = 4096
num_heads  = 32
head_dim   = 128
```

因为 $4096=32\times128$，draft block embedding 是：

```text
X: [2, 16, 4096]
```

经过 Q projection 并拆成多个 attention heads：

```text
[2, 16, 4096]
        ↓
[2, 16, 32, 128]
        ↓ transpose
[2, 32, 16, 128]
```

也就是常见的 `[B, heads, sequence, head_dim]`。这里的 16，就是 16 个并行 query positions。

## 八、为什么 Attention 喜欢 `[B, heads, T, d]`？

因为接下来需要计算 $QK^\top$。假设：

```text
Q: [B, heads, Tq, d]
K: [B, heads, Tk, d]
```

那么：

```text
Kᵀ:     [B, heads, d, Tk]
Q @ Kᵀ: [B, heads, Tq, Tk]
```

这就是 attention score matrix。若 $T_q=16$、$T_k=144$：

```text
attention_scores: [B, heads, 16, 144]
```

可以把它想象成：

```text
                 所有可见的 Key
            ┌──────────────────────┐
query 0     │ · · · · · · · · ·  │
query 1     │ · · · · · · · · ·  │
query 2     │ · · · · · · · · ·  │
...         │                      │
query 15    │ · · · · · · · · ·  │
            └──────────────────────┘
                    16 × 144
```

GPU 会把整个矩阵并行计算出来。这才是“同时处理 16 个 token position”真正发生的地方。

## 九、DFlash 不只有 `<MASK>`，还有大模型提供的 Context

如果只给模型一排 MASK，信息显然太少。DFlash 的重要设计之一，是利用 target model 已经计算出的 hidden states 帮助 draft model 猜未来。

假设当前 context 长度 $C=128$，target model 已经计算出：

```text
target_hidden: [B, 128, H] = [1, 128, 4096]
```

而 draft block 是：

```text
noise_embedding: [B, 16, H] = [1, 16, 4096]
```

于是 DFlash 手上有两份信息：

- target model 提供的 context：`[B, 128, H]`
- 待预测 block：`[B, 16, H]`

可以粗略理解成：**大模型把自己对前文的“理解”交给小模型，小模型利用这些信息快速猜接下来的 block。**

## 十、DFlash Attention 中的 Q/K/V

### Query 来自待预测 block

```text
Q = Wq(noise_embedding)
Q: [B, heads, 16, head_dim]
```

例如 `[1, 32, 16, 128]`。

### Key / Value 同时包含 context 和 draft block

可以简化为：

```text
K = concat(K_context, K_noise)
V = concat(V_context, V_noise)
```

若 context length 为 128、block size 为 16，则 KV sequence length 为 $128+16=144$：

```text
Q:   [B, heads,    16,  d]
K/V: [B, kv_heads, 144, d]
```

真实模型经常使用 GQA，因此 `num_kv_heads` 不一定等于 `num_attention_heads`，但不影响这里的核心理解。

## 十一、一次 Attention 到底发生了什么？

核心公式是：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
$$

先看 Q 和转置后的 K：

```text
Q:  [1, 32, 16, 128]
Kᵀ: [1, 32, 128, 144]
```

矩阵相乘得到：

```text
Q @ Kᵀ → [1, 32, 16, 144]
```

也就是说，16 个 query 同时计算自己应该关注哪些 context / block 信息。经过 softmax，再乘 V：

```text
[1, 32, 16, 144]
        @ V
        ↓
[1, 32, 16, 128]
```

把所有 head 拼回去：

```text
[1, 32, 16, 128]
        ↓ concat heads
[1, 16, 4096]
```

所以经过一层 Attention，shape 仍是：

```text
[B, 16, H] → [B, 16, H]
```

16 个位置一直都存在。

## 十二、最后怎么真正变成 16 个 token？

经过若干 Transformer layers 后：

```text
hidden_states: [B, 16, H] = [1, 16, 4096]
```

再经过 LM Head：

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

每个位置都有一份完整的 vocabulary logits。最后执行：

```python
pred = logits.argmax(dim=-1)
```

便得到：

```text
pred: [B, 16]
```

例如：

```text
[[苹果, ，, 因为, 它, 很, 好吃, ...]]
```

这就是所谓“一次预测一整个 block”。

## 十三、第 2 个位置知道第 1 个位置生成了“苹果”吗？

答案是：**在同一个 forward 中，它并不知道第一个位置最终被采样成了“苹果”。**

假设输入是：

```text
MASK0 MASK1 MASK2 MASK3
```

一次 forward 最后输出：

```text
苹果 ， 因为 它
```

它并不是下面这个自回归过程：

```text
MASK0 → 苹果
MASK1 看见“苹果” → ，
MASK2 看见“苹果，” → 因为
```

真正的过程是：

```text
MASK0  ─┐
MASK1  ─┤
MASK2  ─┤
MASK3  ─┤
...     ├──→ Transformer → h0 h1 h2 ... h15 → LM Head → 16 个 token
MASK15 ─┘
```

最终的离散 token，是 Transformer forward 基本结束以后才确定的。

## 十四、那 16 个位置怎么互相协调？

它们交换的不是已经确定的离散 token，而是**连续 hidden representations**。

例如第一层之后：

```text
MASK0 → h0¹：这里可能是某种食物 / 名词
MASK1 → h1¹：这里可能是标点或连接结构
MASK2 → h2¹：这里可能开始解释原因
```

下一层 Attention 又可以继续交换这些连续表示：

```text
Layer 1     → 粗略语义结构
Layer 2     → 互相协调
Layer 3     → 进一步明确
...
Final Layer → LM Head → 离散 token
```

因此，block 内的位置虽然没有看到彼此最终采样出的 token，但仍然能够通过 hidden states 建立依赖。

这也是理解扩散式语言模型很重要的直觉：**不是先确定 token 再交流，而是在连续表示空间里先协调，最后才一起落到离散 token 上。**

## 十五、为什么普通 GPT 不能直接这么做？

这里存在一个重要 trade-off。

自回归模型逐个生成：

```text
token1 → token2 → token3 → token4
```

token 2 可以明确知道 token 1 是什么，条件很强：

$$
P(x_2\mid x_1),\quad
P(x_3\mid x_1,x_2),\quad
P(x_4\mid x_1,x_2,x_3)
$$

因此生成质量好，但计算是串行的。

Block-parallel 模型则同时预测：

```text
x1  x2  x3  x4  ...  x16
↑   ↑   ↑   ↑        ↑
        同时预测
```

它能充分利用 GPU 并行，但每个位置无法直接获得前面位置最终确定的离散 token。核心挑战因此变成：**如何让同时预测的位置保持足够好的协调性？**

这正是 diffusion / block diffusion 类方法要解决的问题。

## 十六、为什么 DFlash 特别适合 Speculative Decoding？

因为 DFlash 不需要成为最终模型，它只需要**快速猜**。

假设 DFlash 一次预测：

```text
相 对 论 ， 并 对 现 代 物 理 学 产 生 了
```

Target model 再进行验证：

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

那么前面连续正确的“相对论，并对现代物理学”都可以接受。即使后面猜错，也没有关系。

DFlash 的目标不是“一次生成 16 个完美 token”，而是：

> 用一次很便宜的并行 forward，尽可能猜中一段较长的 prefix。

只要一次猜 16 个，其中前面连续猜中多个，就可能获得收益。

## 十七、为什么这比传统 Speculative Decoding 更进一步？

传统 speculative decoding 通常是：

```text
Draft Model：
token1 → token2 → token3 → ... → token16
                              ↓
Target Model：             一次验证
```

虽然 target model 的验证被并行化了，但 draft model 自己仍然是 autoregressive。

DFlash 想进一步把 draft 阶段也并行：

```text
Draft Model： [token1 token2 token3 ... token16]
                          ↑
                       一次预测
                          ↓
Target Model：          一次验证
```

因此它试图同时减少 target model 和 draft model 的串行计算。

## 十八、用完整 Shape 串一次 DFlash

假设：

| 参数 | 值 |
|---|---:|
| Batch size $B$ | 1 |
| Context length $C$ | 128 |
| Block size $S$ | 16 |
| Hidden size $H$ | 4096 |
| Attention heads | 32 |
| Head dimension $d$ | 128 |
| Vocabulary size $V$ | 150000 |

完整数据流如下：

```text
Target context IDs
[1, 128]
    ↓ Target Transformer
Target hidden
[1, 128, 4096]

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

最终得到整个 block 的预测。

## 十九、读源码时应该盯住哪些 Shape？

以后直接读 DFlash 源码，最重要的是始终盯住下面几个 tensor：

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

所谓“DFlash 一次生成 16 个 token”，从 tensor 的角度看，本质上就是：

```text
[B, 16, H]
     ↓ Transformer
[B, 16, H]
     ↓ LM Head
[B, 16, V]
     ↓ token selection
[B, 16]
```

## 二十、总结

理解 DFlash，最容易犯的错误就是把“并行生成 16 个 token”理解成 `batch_size = 16`。实际上：

- Batch 维表示多少条彼此独立的 sequence。
- Block size 表示同一条 sequence 中有多少个位置需要同时预测。
- Attention 利用 GPU 的矩阵并行能力，一次为 16 个 query positions 计算 hidden representations。
- LM Head 再把 `[B, 16, H]` 映射成 `[B, 16, V]`，为 16 个位置分别产生 vocabulary logits。
- 这些位置不会在同一次 forward 中看到彼此最终采样出的 token，但能通过连续 hidden states 协调。

所以真正值得记住的一句话是：

> DFlash 的“一次生成一批 token”，不是把 16 个 token 当成 16 个 batch，而是把它们作为同一条 sequence 中的 16 个 query positions，在一次 Transformer forward 中并行计算；这些位置通过连续 hidden states 协调，最后再由 LM Head 同时映射成离散 token。

理解到这里，再去看 DFlash 的 attention mask、block diffusion 训练方式，以及 target model 如何 verify speculative tokens，就会容易很多。
