---
title: DSpark 在 vLLM Ascend V1 中怎么跑：从 Target Forward 到动态验证
description: 沿一次 speculative decode step 追踪 vLLM 与 vLLM Ascend 的 DSpark V1 实现，解释 context KV、并行 query block、Markov 串行采样和 confidence 动态验证如何接在一起。
pubDate: 2026-08-22
updatedDate: 2026-08-22
category: ai-infra
tags:
  - DSpark
  - vLLM
  - vLLM Ascend
  - Speculative Decoding
  - NPU
draft: false
---

上一篇从算法角度介绍了 DSpark：DFlash 式并行 backbone 负责重计算，Markov Head 用很小的串行开销补回 block 内 token 依赖，Confidence Head 再判断候选后缀是否值得验证。

这篇继续往源码里走，回答四个更具体的问题：

- target model 的 hidden states 怎样进入 draft model？
- DSpark 为什么只跑一次 draft Transformer，却能从左到右生成 $K$ 个 token？
- vLLM Ascend 为 NPU 重写了哪些数据准备和 KV Cache 逻辑？
- Confidence Head 产生的分数，最后怎样改变下一轮 target verification？

分析基于以下本地版本：

```text
vLLM:        bd6536071c
vLLM Ascend: b4b04c5eb
```

本文只讨论 `vllm_ascend/worker/model_runner_v1.py` 这条 V1 路径。当前上游 vLLM 会把 DSpark 导向 GPU V2 Model Runner；Ascend V1 则通过 `AscendDSparkProposer` 补上了自己的实现。因此，下文中的“vLLM 提供什么”主要指 V1 的通用 speculative-decoding 接口，“Ascend 实现什么”指插件里的 NPU 执行路径。

## 一、先看一次 decode step 的全貌

DSpark 没有单独接管整个推理循环。调度、target forward、验证和请求状态仍由 Model Runner 管理，drafter 只是其中一个阶段。

一次稳态 decode step 可以压成下面这条链：

```text
SchedulerOutput
      │
      ▼
NPUModelRunner.execute_model()
      │
      ├─ 准备本轮 token、position、block table
      ├─ Target Model forward
      └─ 保存 logits / hidden states / spec metadata
      │
      ▼
NPUModelRunner.sample_tokens()
      │
      ├─ AscendRejectionSampler
      │    验证上轮 draft，得到 accepted prefix + bonus token
      │
      ├─ bookkeeping
      │    修正请求长度、KV 状态和被拒绝位置
      │
      └─ propose_draft_token_ids()
           │
           ▼
      AscendDSparkProposer._propose()
           ├─ 准备 context 与 query block
           ├─ 预计算 draft context KV
           ├─ 一次 draft backbone forward
           ├─ K 次轻量 Markov 修正与 greedy sampling
           └─ 可选：Confidence Head 计算 verify length
```

这里有一个容易混淆的时间关系：**本轮 target forward 验证的是上轮生成的 draft；本轮 target 采样结束后，DSpark 才为下一轮生成新的 draft。**

`NPUModelRunner.sample_tokens()` 先调用 `_sample()`。存在 `spec_decode_metadata` 时，`_sample()` 进入 `AscendRejectionSampler`；随后内部闭包 `propose_draft_token_ids()` 才把 target 的采样结果、hidden states 和 attention metadata 交给 drafter。

## 二、初始化：vLLM 管框架，Ascend 选择自己的 Proposer

`NPUModelRunner` 继承上游 `GPUModelRunner`，但在 speculative decoding 初始化时调用 Ascend 自己的工厂：

```text
NPUModelRunner._set_up_drafter()
  └─ get_spec_decode_method(method, ...)
       └─ method == "dspark"
            └─ AscendDSparkProposer
```

`AscendDSparkProposer` 又继承 `AscendDflashProposer`，后者继承 `AscendEagleProposer`。这条继承链不是说三种算法相同，而是在复用 V1 proposer 已经具备的能力：加载 draft model、接收 target hidden states、管理 draft buffers、准备 attention metadata，并将 draft token 交回 Model Runner。

DSpark 初始化时还建立了三类关键状态：

```text
_dspark_seed_buffer:  [max_batch_size]
_dspark_draft_buffer: [max_batch_size, K + 1]
hidden_states:        [max_num_tokens, H_draft]
```

`_dspark_draft_buffer` 的第 0 列保存 seed，也就是 target 刚采样出的 next token；第 1 到 $K$ 列保存 DSpark 依次生成的候选：

```text
[seed, draft_1, draft_2, ..., draft_K]
```

V1 目前只支持 greedy DSpark。若配置 `draft_sample_method="probabilistic"`，构造函数会直接抛错。这个限制很重要，因为标准 rejection sampling 需要 draft probability，而当前路径只实现了 greedy token proposal。

## 三、Target Hidden States 没有直接当作 Query 输入

Model Runner 调用 `propose_draft_token_ids()` 时，会先找到本轮有效的 target token 和 hidden states。对于 `dspark`，通用 proposer 的 `_propose()` 还会调用：

```python
target_hidden_states = self.model.combine_hidden_states(
    target_hidden_states
)
```

如果 checkpoint 使用 target 的多个辅助层，这里会先拼接再投影到 draft hidden size。得到的 `[T, H_draft]` 不是和 MASK embedding 简单拼在一起送入 Transformer，而是走另一条路径：

```text
target hidden states [T, H_draft]
        │
        ▼
precompute_and_store_context_kv()
        │
        ▼
DSpark 自己的 draft KV Cache
```

随后，真正进入 draft backbone forward 的是每条请求的一组 query tokens。默认 `sample_from_anchor=True` 时，每条请求恰好有 $K$ 个 query 位置：第一个位置放 seed/anchor，其余位置放 parallel drafting token。

```text
request 0: [anchor, MASK, MASK, ..., MASK]  共 K 个 query
request 1: [anchor, MASK, MASK, ..., MASK]
...
```

这和 DFlash 常见的 `1 + K` 布局不同。DFlash 的 anchor 主要用于提供条件，后面的 $K$ 个 MASK 位置各预测一个 token；DSpark 默认让 anchor 位置本身预测 `draft_1`，所以 $K$ 个 query 就产生 $K$ 个候选。

若 checkpoint 关闭 `sample_from_anchor`，DSpark 也会退回 `1 + K` 布局。阅读 shape 时不能只看 `num_speculative_tokens`，还要看这个配置。

## 四、Ascend 怎样准备一个并行 Query Block？

入口是 `AscendDSparkProposer.set_inputs_first_pass()`。假设：

```text
B = 2
K = 4
T = 6                 # 本轮 target context token 总数
sample_from_anchor = True
```

那么：

```text
num_query_total  = B × K = 8
num_sample_total = B × K = 8
```

函数先把 `next_token_ids[B]` 写入 `_dspark_seed_buffer`，再调用 Ascend Triton kernel：

```text
copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid
```

这个 kernel 一次填好：

- query 的 `input_ids` 和 positions；
- context positions；
- context 与 query 的 slot mappings；
- 从 backbone 输出中抽取 logits 行的 `token_indices_to_sample`。

这些数据必须一起生成，因为 rejected token 会让逻辑序列长度回退。代码先计算：

```text
effective_seq_lens = seq_lens - num_rejected_tokens
```

再把当前 query block 加进去：

```text
new_seq_lens = effective_seq_lens + K
```

单元测试专门覆盖了这个分支。例如原长度 128、上轮拒绝 2 个、当前 `K=5`，新的 attention sequence length 是 `128 - 2 + 5 = 131`。不先回滚 rejected suffix，draft query 就会看到本不该存在的 KV。

### 为什么要按 KV Cache Group 分别准备？

`initialize_attn_backend()` 会读取 draft model 暴露的 layer names，再与 vLLM 的 `kv_cache_groups` 对齐。每个 group 单独持有：

```text
block table
query slot mapping buffer
context slot mapping buffer
attention metadata builder
```

这样做是因为 DSpark draft layers 不一定全属于同一个 KV Cache Group，也可能使用不同的 cache spec。V1 没有 V2 `BlockTables` 提供的统一封装，因此 proposer 自己维护 per-group 字典，并最终按 layer 顺序组装 `_context_slot_mapping_buffers`。

找不到 draft layer API，或者 layer names 与任何 KV group 都不重合时，初始化会直接报错，而不是静默使用错误的 target cache。

## 五、Context KV 和 Query Forward 是两段计算

准备完输入后，`AscendSpecDecodeBaseProposer._run_merged_draft()` 进入实际 draft 计算。对 DFlash/DSpark，它先调用：

```python
self.build_model_inputs_first_pass(
    num_input_tokens,
    self._context_slot_mapping_buffers,
)
```

继承自 DFlash proposer 的实现会进一步调用：

```python
self.model.precompute_and_store_context_kv(
    target_hidden_states,
    context_positions,
    context_slot_mappings,
)
```

这一步只把 target hidden states 投影为各 draft layer 所需的 K/V，并写入 **draft model 自己的 KV Cache**。它不会完整执行一次 draft Transformer。

之后才调用一次：

```python
ret_hidden_states = self.model(
    input_ids=model_input_ids,
    positions=model_positions,
    inputs_embeds=inputs_embeds,
)
```

这里的 `model_input_ids` 只有 `[B × K]` 个 query token。Attention 通过刚才构造的 metadata 读取已写好的 context KV，因此可以把计算理解为：

```text
Target hidden states ──projection──> Draft context K/V
                                          ▲
                                          │ attend
[anchor/MASK query block] ──backbone──> [B × K, H_draft]
```

DSpark 的并行骨架就落在这一次 model call 中。它没有为每个候选 token 重跑 Transformer。

## 六、真正的 Semi-Autoregressive 循环在哪里？

backbone 返回 hidden states 后，代码按 `token_indices_to_sample` 取出 $B\times K$ 行，计算 base logits：

```text
sample_hidden_states: [B × K, H_draft]
raw_logits:           [B × K, V]
logits:               [B, K, V]
```

随后 `AscendSpecDecodeBaseProposer._run_merged_draft()` 进入 DSpark 分支：

```python
draft_token_ids[:, 0] = seed

for idx in range(K):
    markov_emb = model.markov_embed(draft_token_ids[:, idx])
    logits_bias = model.markov_bias(markov_emb)
    logits[:, idx] += logits_bias
    draft_token_ids[:, idx + 1] = logits[:, idx].argmax(-1)
```

若 Markov rank 是 $R$，每一步的 shape 是：

```text
previous token: [B]
markov_embed:   [B, R]
markov_bias:    [B, V]
base_logits:    [B, V]
sampled token:  [B]
```

position 维上的循环确实执行 $K$ 次，但循环里只有 embedding lookup、低秩投影、logit add 和 argmax。batch 中的 $B$ 条请求仍然同时计算，Transformer 也没有再次运行。

最后返回的是：

```python
draft_token_ids[:, 1:]
```

shape 为 `[B, K]`。第 0 列 seed 只是 Markov 链的起点，不是 speculative candidate。

## 七、Confidence Head 怎样接入？

Ascend 为 Qwen3 DSpark 增加了 `DSparkConfidenceHead`：

```text
hidden state [H]
      +
Markov embedding [R]
      │ concat
      ▼
[H + R] → ReplicatedLinear → [1]
```

`AscendQwen3DSparkForCausalLM` 在配置启用时创建这个 head，并单独加载 checkpoint 中的 `confidence_head.*` 权重。如果配置要求启用但 checkpoint 没有对应权重，代码会关闭该 head；若只有部分参数缺失，则直接报错，避免用半套随机参数参与调度。

当 `additional_config.dynamic_spec_config.method="dspark"` 时，Markov 循环结束后会调用 `DynamicSpecScheduler.update()`：

```text
last_hidden_states: [B, K, H]
Markov inputs:      [seed, draft_1, ..., draft_(K-1)]
                    [B, K]
        │
        ▼
confidence logits: [B, K]
        │ sigmoid
        ▼
token_probs:       [B, K]
```

这里使用的是 `draft_token_ids[:, :K]`，而不是 `[:, 1:]`。原因是第 $i$ 个 confidence 需要与生成该位置时实际使用的 previous token 对齐：第一个位置用 seed，第二个位置用 `draft_1`，依此类推。

## 八、动态验证调度器实际做了什么？

`DynamicSpecScheduler` 先把逐 token 的条件接受概率转成 prefix survival：

```python
survival = torch.cumprod(token_probs, dim=1)
```

假设两条请求的 confidence 是：

```text
request A: [0.9, 0.8, 0.7, 0.6]
request B: [0.9, 0.9, 0.9, 0.9]
```

那么 survival 是：

```text
A: [0.90, 0.72, 0.504, 0.3024]
B: [0.90, 0.81, 0.729, 0.6561]
```

后面的 token 只有在前缀全部通过时才有价值，因此调度器比较 survival，而不是把每个位置的 confidence 当成互不相关的分数。

### 1. 先估计共享预算

每隔 `budget_update_interval` 步，`compute_verify_budget()` 统计 survival 高于 `budget_threshold` 的位置总数，除以 request 数并向上取整，得到共享的平均预算 `budget_k`。

这一步有一次 `.item()`，会产生 NPU 到 CPU 的同步，但只发生在预算更新步。代码还包含一个针对既有测量的奇偶宽度修正：若新预算是偶数且尚未达到最大 $K$，会向上调成下一个奇数。

### 2. 再把预算分给不同请求

`allocate_verify_budget()` 先保证每条请求至少得到 `min_verify_tokens`，再把剩余的全局 token budget 分给 batch 中最大的 survival 值：

```text
每条请求先保底 min_k
        │
        ▼
剩余 survival 展平
        │ global top-k
        ▼
scatter_add 到各 request
        │
        ▼
num_verify_tokens [B]
```

由于一条请求内部的 cumulative survival 单调不增，全局 top-k 自然倾向于先选前面的 token，结果仍然是每条请求的 prefix length，而不是中间挖出几个不连续位置。

### 3. Model Runner 在哪里真正裁剪？

调度结果暂存在：

```text
dynamic_spec.num_verify_tokens: [B]
```

`NPUModelRunner.take_draft_token_ids()` 在把 draft 交回 scheduler 前按请求裁剪：

```python
draft_token_ids = [
    tokens[:k]
    for tokens, k in zip(out.draft_token_ids, per_req_k)
]
```

下一轮 scheduler 只会把这些保留下来的 token 放进 target verification workload。

这个顺序说明了一个重要边界：**当前动态策略不会减少本轮 DSpark backbone 和 Markov Head 生成 $K$ 个 draft 的成本；它减少的是下一轮 target model 要验证的 suffix。**

## 九、Target Verification 怎样保持输出正确？

下一轮 `execute_model()` 会把不同请求上轮保留的 draft token 一起打包。`_prepare_inputs()` 记录每条请求的 `num_draft_tokens`，并构造 target logits indices、draft token IDs 和累计长度。Target Model 一次 forward 为这些候选和额外的 bonus 位置计算 logits。

随后 `sample_tokens()` 中的 `AscendRejectionSampler` 比较 target distribution 与 draft candidates，输出连续 accepted prefix，并在首次拒绝处使用 target 结果继续生成。被拒绝的 suffix 会在下一次输入准备时从逻辑长度和 slot mapping 中回滚。

所以 DSpark 的职责是：

```text
提出更可能通过的候选
        +
决定候选前缀保留多长
```

最终 token 仍由 target verification 决定。Greedy 路径下，draft 猜错只会缩短本轮推进距离，不会直接把错误 token 写进最终输出。

## 十、vLLM 与 vLLM Ascend 的职责边界

把整个实现拆开，会更容易看清插件为什么不只是“换一个设备名”。

| 层面 | vLLM V1 提供 | vLLM Ascend V1 实现 |
| --- | --- | --- |
| 推理循环 | SchedulerOutput、Model Runner 生命周期、spec metadata、request bookkeeping | `NPUModelRunner` 适配 NPU 执行与异步拷贝 |
| Speculative 框架 | proposer 接口、target verification 数据结构、通用 rejection sampling 语义 | Ascend proposer 工厂、`AscendRejectionSampler` |
| DFlash 基础 | target hidden-state 输入与 block proposer 的基本抽象 | NPU input packing、context KV、Ascend attention metadata |
| DSpark 采样 | 模型暴露的 `markov_embed()` / `markov_bias()` 接口 | V1 中的顺序 Markov greedy loop |
| KV 管理 | KV Cache Group 与 block table 抽象 | per-group slot mapping 和 draft attention backend 对接 |
| 动态验证 | scheduler 能接收不同长度的 draft list | Confidence Head、survival 预算与按请求裁剪 |

Ascend 改动最多的地方不是 DSpark 的数学公式，而是公式周围的数据布局：int32 positions、NPU slot mapping、不同 KV groups、rejected-token 回滚、device/CPU 同步和 attention metadata。

## 十一、当前 V1 实现的边界

从代码和文档能确认以下限制：

- DSpark V1 proposal 只支持 greedy sampling；
- `AscendDSparkProposer` 明确关闭自身的 graph path，使用 eager 执行；
- confidence-based dynamic verify length 仍标记为 exploratory；
- 当前动态 DSpark 文档只承诺 Qwen 系列，不能据此推断 DeepSeek-V4 等模型也接入了相同 Confidence Head 路径；
- 当前 `DynamicSpecScheduler` 使用阈值、历史周期与 global top-k，是工程启发式，不是论文中结合完整 engine throughput profile 求解的调度器；
- 单元和 E2E 测试覆盖可执行路径、shape、回滚与 acceptance baseline，但不等于复现论文吞吐数字。

这些限制不影响理解主链路，却决定了部署时能打开哪些开关，也决定了性能对比应该怎样描述。

## 总结

在 vLLM Ascend V1 中，DSpark 不是一套绕开 Model Runner 的独立引擎。它嵌在现有 speculative decoding 循环里：target model 先验证上轮 draft 并产生新的 seed；proposer 把 target hidden states 投影进自己的 context KV Cache，再用 anchor/MASK query block 跑一次并行 backbone；最后只让 Markov Head 沿 position 维顺序执行，生成下一轮候选。

Confidence Head 也没有改变 draft 生成本身。它把每个位置的接受概率变成 prefix survival，由 `DynamicSpecScheduler` 分配 verify budget，Model Runner 再裁剪每条请求送往下一轮 target verification 的候选前缀。

整条因果链可以压成一句话：

> vLLM 管理 speculative decoding 的生命周期，vLLM Ascend 把 DFlash/DSpark 的 context KV、query block、Markov 采样和动态验证落到 NPU 的数据布局与执行路径上。

## 源码索引

- vLLM `vllm/v1/worker/gpu_model_runner.py`：V1 Model Runner 与 speculative-decoding 通用流程
- vLLM `vllm/v1/spec_decode/dflash.py`：V1 DFlash proposer 的 context/query 设计
- vLLM Ascend `vllm_ascend/worker/model_runner_v1.py`：NPU V1 target、sampling 与 draft 串联
- vLLM Ascend `vllm_ascend/spec_decode/dspark_proposer.py`：DSpark 输入布局、KV groups 和 seed buffers
- vLLM Ascend `vllm_ascend/spec_decode/dflash_proposer.py`：context KV 与 query-only forward
- vLLM Ascend `vllm_ascend/spec_decode/llm_base_proposer.py`：Markov 顺序采样主循环
- vLLM Ascend `vllm_ascend/spec_decode/utils.py`：`DynamicSpecScheduler`
- vLLM Ascend `vllm_ascend/models/qwen3_dspark.py`：Confidence Head 与权重加载
- vLLM Ascend `vllm_ascend/sample/rejection_sampler.py`：NPU rejection sampling
