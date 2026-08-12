---
title: Speculative Decoding 全解析：从概率质量到 vLLM / Ascend 一次 Decode
description: 从 Leviathan 论文公式讲到 min(p,q) 与残差采样，再到 Greedy / Non-greedy、bonus、acceptance length，并对照 vLLM 与 vLLM-Ascend 关键代码与 draft 方法。
pubDate: 2026-08-12
updatedDate: 2026-08-12
category: ai-infra
tags:
  - Speculative Decoding
  - vLLM
  - vLLM-Ascend
  - LLM Inference
  - 采样算法
draft: false
---

> 大模型生成很慢，是因为每出一个 token，就要再跑一遍大模型。Speculative Decoding 想换一种做法：先让小模型猜几个 token，再让大模型一次性核对。问题是——核对之后，输出还是不是大模型原来的分布？
>
> 本文从概率账讲起，再落到 vLLM / vLLM-Ascend 的实现。

论文：Leviathan et al., ICML 2023（[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)）；Chen et al.（[arXiv:2302.01318](https://arxiv.org/abs/2302.01318)）。

代码快照：vLLM `main @ bd6536071c`；vLLM-Ascend `main @ 0391783df`。

---

## 一、标准解码为什么慢？

生成是自回归的。每出一个 token，就要把它拼进上下文，再跑一次大模型 $M_p$：

```text
prefix → target forward → 1 token → 新 prefix → target forward → …
```

瓶颈很清楚：

> 大模型每次 forward，通常只推进 1 个 token。

Speculative Decoding 的想法是：先让便宜的 draft / proposer 猜 $\gamma$ 个 token（实现里常记成 $K$），再让 target **一次**并行验证。猜得准，一次 target forward 就能往前走好几个 token。

---

## 二、「服从 target 分布」到底指什么？

不是泊松分布。

指的是词表 $V$ 上的**离散分类分布（categorical）**：

$$
p(x)=\mathrm{softmax}(\ell)_x,\quad x\in V,\quad\sum_{x\in V}p(x)=1
$$

两个容易混的东西：

- $\ell$：logits，softmax 之前的未归一化分数，可正可负
- $p(x)$：概率，accept / reject 用的是它，不是 raw logits

举个小例子。词表只有「猫 / 狗 / 鸟」，某位置输出：

$$
\ell=[2.0,\,1.0,\,0.1]
\quad\Rightarrow\quad
p\approx[0.659,\,0.242,\,0.099]
$$

后文里：

- $p$ = target 分布
- $q$ = draft 分布

目标始终是：最终吐出的 token 满足 $X\sim p$，和直接对 target 采样一样。

---

## 三、先记几个词

| 概念 | 一句话 |
|------|--------|
| Draft | proposer 猜出来的 $\tilde x_1,\ldots,\tilde x_K$ |
| Verify | target 一次 forward，算出各 draft 位（以及 bonus 位）的分布 |
| Accept / Reject | 从左到右逐位判定；**第一个 reject 截断**后面所有 draft |
| Recovered | reject 后，用残差分布或 target argmax 纠正的当前 token |
| Bonus | 全部 draft 都接受时，从同一次 forward 多出来的 $p_{K+1}$ 再采 1 个 |
| Acceptance length | 本步最终有效 token 数 |
| Greedy | temperature = 0，draft 必须等于 $\arg\max p$ |
| Non-greedy | 用 $p/q\ge u$ 做拒绝采样 |

vLLM 把最终输出拆成：

> output = accepted + recovered +（可选）bonus

论文 Algorithm 1 则把 recovered，以及「全接受时的 $t\sim p_{K+1}$」，统一写成一个 $t\sim p'$。

---

## 四、论文 Algorithm 1

输入：$M_p$（target）、$M_q$（draft）、当前 $prefix$。

**第一步：Draft。** 用 $M_q$ 串行猜 $\gamma$ 个 token：

$$
\text{for }i=1\ldots\gamma:\quad
q_i\leftarrow M_q(prefix+[x_1,\ldots,x_{i-1}]),\quad
x_i\sim q_i
$$

**第二步：Verify。** 用 $M_p$ 一次并行算出 $\gamma+1$ 个分布：

$$
p_1,\ldots,p_{\gamma+1}
\leftarrow
M_p(prefix),\;\ldots,\;M_p(prefix+[x_1,\ldots,x_\gamma])
$$

**第三步：定接受长度 $n$。**

$$
r_i\sim U(0,1),\qquad
n=\min\Big(\big\{i-1\mid r_i>\tfrac{p_i(x_i)}{q_i(x_i)}\big\}\cup\{\gamma\}\Big)
$$

**第四步：采纠正 token / bonus token $t$。**

$$
p'=
\begin{cases}
p_{n+1} & n=\gamma\\
\mathrm{norm}\big(\max(0,\,p_{n+1}-q_{n+1})\big) & n<\gamma
\end{cases}
\qquad t\sim p'
$$

返回：$prefix+[x_1,\ldots,x_n,t]$。

几个常用公式：

| 含义 | 公式 |
|------|------|
| 平均接受率 | $\alpha=\mathbb{E}[\min(p,q)]$ |
| 一步期望产出 | $\mathbb{E}[\#\mathrm{tokens}]=(1-\alpha^{\gamma+1})/(1-\alpha)$，范围 $[1,\gamma+1]$ |
| 墙钟加速比 | $(1-\alpha^{\gamma+1})/\big((1-\alpha)(\gamma c+1)\big)$，其中 $c=\mathrm{time}(M_q)/\mathrm{time}(M_p)$ |

---

## 五、拒绝采样为什么仍然等于 $p$？

这是整篇文章最容易绕进去的地方。用概率质量来想。

Draft 先抽 $x\sim q$，再按下面的概率接受：

$$
a(x)=\min\Big(1,\frac{p(x)}{q(x)}\Big)
$$

于是「抽到 $x$ 并且接受」的概率是：

$$
q(x)\,a(x)=\min\big(p(x),q(x)\big)
$$

人话版：

- $q\le p$：draft 给得不够或刚好，全部留下
- $q>p$：draft 给多了，只留 $p/q$ 这一截

Accept 路径只覆盖了 $\min(p,q)$，还缺：

$$
\max(0,\,p-q)
$$

Reject 时，不从完整的 $p$ 重采，而是从残差分布采：

$$
p'(x)=\frac{\max(0,p(x)-q(x))}{\sum_v\max(0,p(v)-q(v))}
$$

总拒绝概率恰好等于残差总质量：

$$
\mathbb{P}(\mathrm{reject})=1-\sum\min(p,q)=\sum\max(0,p-q)
$$

所以对任意 token $y$：

$$
P(\text{输出}=y)=\min(p,q)+\max(0,p-q)=p(y)
$$

为什么不能 reject 后直接从完整 $p$ 再采？因为有些 token 的额度，accept 路径已经拿满了。再从完整 $p$ 抽，它们会超额。残差采样的意义，就是**只补还缺的那一块**。

### 一个四分类验算

| | A | B | C | D |
|--|--:|--:|--:|--:|
| $q$ | 30% | 50% | 15% | 5% |
| $p$ | 40% | 30% | 20% | 10% |

第一阶段留下 $(30,30,15,5)$，合计 80%。剩下 20% 走残差 $p'=(50\%,0,25\%,25\%)$，补上 $(10,0,5,5)$。两边一加，正好是 $(40,30,20,10)=p$。

还有一句很重要：

> 每个 token 位置都是当场结清的账。Residual 只决定当前 token，不会把缺口带到下一个位置。

---

## 六、Greedy 和 Non-greedy

实现里通常分两种。

| 模式 | 怎么判定 | 拒绝时做什么 |
|------|----------|--------------|
| Greedy（temperature = 0） | $\tilde x_i=\arg\max_v p_{\mathrm{target}}(v)$ | 写 target argmax 为 recovered，截断 |
| Non-greedy | $q(\tilde x)>0$ 且 $p(\tilde x)/q(\tilde x)\ge u$，$u\sim U(0,1)$ | 从 $\propto\max(p-q,0)$ 采 recovered |

全部 draft 都接受时，两种模式都会再追加一个 bonus。

设 $K=3$，draft = `[A, B, C]`：

| 场景 | 输出 | acceptance length |
|------|------|-------------------|
| Greedy 部分命中（argmax = A,B,X） | `[A,B,X]`，无 bonus | 3 |
| Greedy 全中 | `[A,B,C,bonus]` | 4 = K+1 |
| Non-greedy 第 2 位 $p/q < u$ | `[A, recovered]`，丢掉 C | 2 |

和经典 rejection sampling 的差别：

```text
经典：  q → accept? → 是则输出 / 否则丢掉，从 q 重来
投机：  q → accept? → 是则输出 / 否则从 (p-q)+ 再采一次并输出
```

投机采样拒绝后不重抽 $q$，而是立刻用残差补齐，这样才能和「一次验证多个 draft」拼在一起。

---

## 七、一次猜多个 token 时，还多出两件事

### 1. 为什么 target 能一次验证多个位置？

Draft 猜出「苹 → 果 → 。」之后，概念上需要：

```text
P(·|我爱吃)
P(·|我爱吃苹)
P(·|我爱吃苹果)
P(·|我爱吃苹果。)
```

Transformer 有 causal attention。把整段一次送进 target，一次 forward 就能并行得到这些位置的 logits。这是能加速的工程原因。

### 2. 为什么必须从左到右？第一个 reject 后为什么全丢？

后一位成立的前提，是前面的 draft **真的被接受了**。

如果「苹」被 reject，residual 抽到「梨」，真实上下文就变成「我爱吃梨」。原来建在「我爱吃苹」上的「果」「。」全部失效，必须丢掉。

规则只有一句：

> reject 之前保留；reject 位置用 recovered 替换；之后全部作废。

Draft 是**候选未来**，不是已经确认的答案。

### 3. Bonus 和 acceptance length

槽位数始终是 $K+1$。多出来的那个 $p_{K+1}$：

- 全接受时，用来采 bonus
- 中途 reject 时，用来做 residual 采样

vLLM 日志里的约定是：

$$
\text{mean acceptance length}=1+\frac{\#\text{accepted drafts}}{\#\text{draft steps}}
$$

前面的 $1$，代表必有的那一个 sampled / recovered / bonus。最好情况：一次 target forward 推进 $K+1$ 个 token。

---

## 八、一次 Decode 在系统里怎么走

相位很重要：**本步验证的是上一步留下的 draft；验证完，再提下一步的 draft。**

```text
prev drafts
  → Scheduler 挂上 scheduled_spec_decode_tokens
  → Target forward：K draft + 1 bonus 位 logits     （VERIFY）
  → RejectionSampler：greedy 或 p/q ≥ u             （ACCEPT / REJECT）
  → 回滚 rejected 对应的 KV / num_computed_tokens
  → Proposer 再 draft 下一轮 K 个                   （下一轮 DRAFT）
```

再拆细一点：

1. Schedule：把 `request.spec_token_ids` 挂到 `scheduled_spec_decode_tokens`
2. 构造 `SpecDecodeMetadata`：$K$ 个 draft → $K+1$ 个 logits 槽
3. Target verify：一次 forward
4. Rejection sample：输出形状 `[B, K+1]`，拒绝后的空位是 `-1`
5. Scheduler 回滚：`num_computed_tokens -= num_rejected`
6. Propose next：用最终 token（EAGLE/MTP 还会用 hidden states）再 draft

---

## 九、关键代码

### SpecDecodeMetadata

```python
# vllm/v1/spec_decode/metadata.py
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor          # [num_tokens]
    num_draft_tokens: list[int]            # [batch]
    cu_num_draft_tokens: torch.Tensor
    cu_num_sampled_tokens: torch.Tensor
    target_logits_indices: torch.Tensor    # 验 draft
    bonus_logits_indices: torch.Tensor     # bonus
    logits_indices: torch.Tensor           # draft + bonus
```

构造时写得很直白：`num_sampled_tokens = num_draft_tokens + 1`。

### RejectionSampler 明确对齐论文

```python
# vllm/v1/sample/rejection_sampler.py
class RejectionSampler(nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    accepted / recovered / bonus
    output = accepted + recovered + bonus
    """
```

### Greedy kernel

```python
# rejection_greedy_sample_kernel
token_id = target_argmax_id
rejected = draft_token_id != target_argmax_id
# ...
if not rejected:
    # If all tokens are accepted, append the bonus token.
    bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
```

### Non-greedy kernel（论文里的 $p/q$）

```python
# rejection_random_sample_kernel
accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
if accepted:
    token_id = draft_token_id
else:
    rejected = True
    token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
```

Ascend 的 PyTorch 路径同式：

```python
# vllm_ascend/sample/rejection_sampler.py
acceptance_condition = (draft_token_probs > 0) & (
    target_token_probs / draft_token_probs >= uniform_token_probs
)
```

### Scheduler 回滚

```python
# vllm/v1/core/sched/scheduler.py
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
request.num_computed_tokens -= num_rejected
```

### Ascend：挂上 drafter 和 AscendRejectionSampler

```python
# vllm_ascend/worker/model_runner_v1.py
if self.speculative_config:
    self.decode_token_per_req = 1 + spec_token_num
    if get_pp_group().is_last_rank:
        self.drafter = self._get_drafter()
        self.rejection_sampler = AscendRejectionSampler(
            self.sampler, self.speculative_config, self.device
        )
```

```python
# vllm_ascend/spec_decode/__init__.py
def get_spec_decode_method(method, vllm_config, device, runner):
    if method == "ngram": ...
    elif method == "ngram_gpu": ...  # 名字带 gpu，实现是 NPU Triton
    elif method in ("eagle", "eagle3", "mtp"):
        return AscendEagleProposer(...)
    elif method == "draft_model":
        return AscendDraftModelProposer(...)
```

Ascend 和 GPU 路径的主要差别：ACLGraph；常见路径要求 $K+1\le 16$；另外还有 Block / Entropy / Synthetic 等 verify 变体。

### Acceptance length 日志

```python
# vllm/v1/spec_decode/metrics.py
# Conventionally, mean acceptance length includes the bonus token
mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)
```

---

## 十、Draft 方法对照

Verify 框架是同一套，差别在**谁来猜 draft**：

| method | 谁产生 draft | 用 target hidden states？ | 备注 |
|--------|--------------|---------------------------|------|
| `draft_model` | 独立小 LM | 否 | 最接近论文里的 $M_q$ |
| `eagle` / `eagle3` | 轻量头 / 模型 | 是 | Ascend 常用 |
| `mtp` | 目标模型 MTP 头 | 是 | 常走 Eagle 类 proposer |
| `medusa` | 多个 decoding head | 最后一层 | 不是 AR draft LM |
| `ngram` / `ngram_gpu` | prompt n-gram | 否 | 常常没有 `draft_probs` |
| `suffix` | suffix tree | 否 | Arctic Inference |
| `dflash` / `dspark` | 专用架构 | 是 | Ascend 有适配 |
| `mlp_speculator` | MLP | 意图是 | V1 目前禁用 |

---

## 十一、已经删掉或过时的概念

读代码时别踩这些：

| 状态 | 概念 | 说明 |
|------|------|------|
| REMOVED | V0 SpecDecodeWorker / `vllm/spec_decode` | 只剩 `vllm/v1/spec_decode/` |
| DISABLED | MLP Speculator（V1） | registry 临时关掉 |
| DEPRECATED | `deepseek_mtp` 等 method 名 | 统一改写成 `mtp` |
| DEPRECATED | CLI `--speculative-model` | 改用 `--speculative-config` |
| 文档过时 | `strict` / `probabilistic` | 代码是 `standard` / `synthetic` / `block` |
| TODO 删除 | Ascend `patch_rejection_sampler` | 注释写明以后要删 |
| 遗留命名 | `ngram_gpu` | Ascend 上实际是 NPU 路径 |
| 命名债务 | `use_eagle()` | 实际表示「用 target hidden states」，不单指 EAGLE |

主要文件：

- vLLM：`config/speculative.py`，`v1/sample/rejection_sampler.py`，`v1/spec_decode/`，`v1/worker/gpu_model_runner.py`，`v1/core/sched/scheduler.py`
- Ascend：`worker/model_runner_v1.py`，`spec_decode/`，`sample/rejection_sampler.py`，`ops/triton/reject_sample.py`

---

## 十二、最后记住什么？

我自己读下来，最有用的是四层看法：

1. **概率层**：$q$ 是提案，$p$ 是目标；accept + residual 保证输出仍服从 $p$
2. **单 token 层**：每个位置当场结清，残差不跨位
3. **多 token 层**：一次猜 $K$ 个，一次算出 $K+1$ 个 logits，从左到右截断
4. **路径层**：draft 是候选未来；第一个 reject 废掉旧路径

如果只留七句话：

1. Draft 是 proposal，不是已确认输出。
2. $q\le p$ 全接受；$q>p$ 只接受 $p/q$。
3. Reject 后从 $(p-q)_+$ 采，不是完整 $p$。
4. Residual 只修当前 token。
5. Target 一次 forward 能验整段，因为 causal attention 可并行算 logits。
6. 第一个 reject 之后，旧 prefix 上的后续 draft 全部作废。
7. 全接受时可多拿 bonus，来自同一次 forward 的 $p_{K+1}$。

Speculative Decoding 不是「用小模型顶替大模型」。它同时做两件事：统计上不改 target 的采样分布；工程上尽量用一次 target forward 推进多个 token。

一句话：

> 小模型负责猜未来，大模型负责更高效地验证未来。
