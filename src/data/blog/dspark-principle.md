---
title: 从 DFlash 到 DSpark：并行生成之后，怎样补回 Token 依赖？
description: 用一个小型 shape 示例解释 DSpark 的 Markov Head、Confidence Head，以及它如何在保留并行生成优势的同时减少无效验证。
pubDate: 2026-08-22
updatedDate: 2026-08-22
category: ai-infra
tags:
  - DSpark
  - DFlash
  - Speculative Decoding
  - LLM Inference
draft: false
---

上一篇介绍 DFlash 时，我们把它的关键点归结为：在 sequence 维放入一组未来位置，让 draft backbone 一次算出整个 token block。

这种方法很快，但有两个实际问题：同一 block 内的 token 同时预测，后一个位置不知道前一个位置最后选中了什么，因此越靠后的 token 越容易偏离；如果仍把整块候选都送进大模型验证，又会把算力浪费在大概率无法通过的后缀上。

DSpark 针对这两个问题增加了两个很小的模块：

- **Markov Head**：在选 token 时补回相邻 token 之间的依赖；
- **Confidence Head**：估计每个位置被 target model 接受的概率，帮助系统决定验证多长。

因此可以先把 DSpark 记成：

```text
DSpark = DFlash 式并行 Backbone
       + 轻量串行 Markov Head
       + Confidence-Scheduled Verification
```

## 一、DFlash 缓解了串行计算，也削弱了块内依赖

假设要一次 draft 4 个 token。DFlash backbone 会并行得到四个 hidden state：

```text
hidden_states: [B, 4, H]
                │  │  │  │
                h0 h1 h2 h3
```

经过 LM Head 后：

```text
[B, 4, H] → [B, 4, V]
```

四个位置都有自己的 vocabulary logits，可以同时选出 `t0、t1、t2、t3`。问题也出在这里：计算 `h3` 时，`t2` 还没有被选出来。模型知道当前位置和上下文，却不知道前一个位置最终落到了哪个离散 token。

普通自回归生成有明确的依赖链：

```text
t0 → t1 → t2 → t3
```

纯并行 draft 更接近：

```text
h0 → t0
h1 → t1
h2 → t2
h3 → t3
```

这也是 block 后缀通常更难通过 verifier 的原因。

## 二、Markov Head：Transformer 跑一次，选 token 时轻量串行

DSpark 保留并行 backbone，只在最终 logits 上增加一个由前序 token 决定的修正项：

$$
z_i = \operatorname{LMHead}(h_i) + \operatorname{Markov}(t_{i-1})
$$

其中第一项提供上下文和当前位置的整体语义，第二项根据已经选出的前一个 token 调整当前词表分布。

例如 backbone 认为某个位置的“苹果”“逗号”都合理，但前一个 token 已经是“苹果”。Markov Head 可以提高逗号、句号等 token 的 logits，同时压低再次生成“苹果”的倾向。它只做局部修正，不重新执行 Attention 和 MLP。

推理过程因此变成：

```text
并行 Backbone：一次得到 h0、h1、h2、h3

轻量解码：
anchor + h0 → t0
t0     + h1 → t1
t1     + h2 → t2
t2     + h3 → t3
```

位置维上确实多了一段串行计算，所以 DSpark 被称为 **semi-autoregressive**。不过串行的只是一个很小的 head 和采样过程，昂贵的 Transformer backbone 仍然只运行一次；在 batch 维上，不同 request 也仍可并行。

## 三、为什么 Markov Head 要做成 Low-Rank？

最直接的 token 转移表是一个 `[V, V]` 矩阵：行表示前一个 token，列表示当前 token。当词表有 150000 个 token 时，这张表会有 225 亿个参数，代价过高。

DSpark 把它分解为两个较小的矩阵：

```text
previous_token_id
        │
        ▼
W1: [V, R]       token embedding
        │
        ▼
      [R]
        │
        ▼
W2: [R, V]       project to vocabulary
        │
        ▼
markov_bias: [V]
```

若 `V = 150000`、`R = 256`，参数量从 `V²` 降为约 `2VR`。对一个 batch 的某个位置，shape 是：

```text
prev_token:   [B]
embedding:    [B, R]
markov_bias:  [B, V]
base_logits:  [B, V]
final_logits: [B, V]
```

训练时，真实序列中的 previous token 已知，所有位置可以一起计算；推理时，`t_i` 要等 `t_{i-1}` 选出后才能确定。这里和普通语言模型的 teacher forcing 很相似，区别在于 DSpark 的顺序部分非常小。

## 四、Confidence Head：不是每个后缀都值得验证

Markov Head 提高 draft 质量，Confidence Head 处理另一类浪费：候选 block 已经生成了，但 target model 是否值得把它全部验证完？

Confidence Head 从每个位置的 hidden state 预测一个接受概率：

```text
hidden_states: [B, S, H]
        │ Linear(H → 1)
        ▼
confidence:    [B, S]
```

实现也可以把 previous-token 的 Markov embedding 拼到输入中，此时最后一维是 `H + R`。它预测的是“这个 draft token 会不会被 verifier 接受”，不是 draft model 自己的 softmax 概率；训练时可用实际接受/拒绝标签和 BCE loss 学习这件事。

对 speculative decoding 来说，单个位置的置信度还不够。真正有价值的是连续通过的 prefix。若每个位置的接受概率为 $p_i$，走到第 $k$ 个位置的概率可以近似写成：

$$
P(\text{prefix survives to }k) \approx \prod_{i=0}^{k}p_i
$$

例如一条请求的 confidence 很快从 `0.99` 降到 `0.2`，继续验证长后缀的收益通常很低；另一条请求直到后面仍保持较高 confidence，就可以分配更长的验证长度。

DSpark 会把这条 confidence 曲线和服务引擎的吞吐特征、当前负载结合起来，为不同请求选择不同的验证长度。这里优化的不只是“每轮接受几个 token”，还有 target model 宝贵的 batch token capacity。

## 五、完整流程

把模型侧和系统侧接起来，DSpark 的一次推理可以压成四步：

```text
1. Parallel Backbone
   draft block → hidden_states [B, S, H]

2. Base LM Head
   hidden_states → base_logits [B, S, V]

3. Semi-AR Draft
   anchor → t0 → t1 → ... → t(S-1)
   每一步只计算 Markov bias 并完成采样

4. Confidence-Scheduled Verification
   confidence [B, S] → 为每个 request 选择验证长度 K
   → Target Model 并行验证前 K 个候选
```

最终输出仍由 target model 的验证结果决定。DSpark 只负责提出候选并决定验证预算，因此它提升的是 speculative decoding 的效率，而不是用 draft model 替代 target model。

## 六、怎样理解 DSpark？

DFlash 像是让 16 个人同时完成最耗时的思考，再各自写下一个词。速度很快，但后面的人看不到前面最终写了什么。

DSpark 保留这次并行思考，只在落笔时用一张很便宜的小纸条，把前一个词传给下一个位置；同时再估计这串答案从哪里开始不可靠，避免让 verifier 检查明显无望的后缀。

Markov Head 解决的是 **draft 后缀质量**，Confidence Head 和动态调度解决的是 **verification waste**。这两部分合起来，才是 DSpark 相比 DFlash 真正多出来的东西。

## 参考资料

- [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)
- [vLLM Speculators：DSpark 算法说明](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md)
- [vLLM Ascend：Speculative Decoding Guide](https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/speculative_decoding.html)
