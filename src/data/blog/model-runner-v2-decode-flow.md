---
title: Model Runner V2 到底管什么：从一次 Decode 串起 Scheduler、DeepSeek V4 和 DSpark
description: 沿一次 MRV2 decode iteration 追踪 SchedulerOutput 如何变成 input_ids、positions、block table 和 attention metadata，再进入 target model、采样验证与 DSpark proposal。
pubDate: 2026-08-24
updatedDate: 2026-08-24
category: ai-infra
tags:
  - Model Runner V2
  - vLLM
  - vLLM Ascend
  - DSpark
  - Speculative Decoding
draft: false
---

理解 Model Runner V2 的 decode 流程，需要先明确三个相关机制：Speculative Decoding 为什么能一次验证多个 token，DeepSeek V4 的 target model 怎样完成一次 forward，以及 DSpark 怎样用并行 backbone 和 Markov Head 生成一组 draft token。

这些内容还缺一个连接点：谁把 Scheduler 给出的请求状态整理成模型能吃的 Tensor？谁准备 position、KV Cache 地址和 attention metadata？target forward 结束后，又是谁把 hidden states 交给 DSpark？

答案是 Model Runner。

上一篇追的是 vLLM Ascend V1 中的 `AscendDSparkProposer`。这篇先不比较 V1 和 V2，只沿 Model Runner V2 的一轮 decode 往下走。目标是把 Scheduler、DeepSeek V4 和 DSpark 放回同一条执行链里。

分析基于：

```text
vLLM Ascend: main @ b4b04c5eb
对应 main2main 版本: vLLM v0.27.1
```

MRV2 仍在快速变化，上游 `main` 的部分函数签名可能已经不同。本文只抓稳定的职责边界和执行顺序，不展开 block/slot 的内部计算。

## 一、Runner 不是 Model

先把三个角色分开。

| 组件 | 它决定什么 | 它不做什么 |
|---|---|---|
| Scheduler | 本轮跑哪些请求、每个请求算几个 token、使用哪些新 KV block、带上哪些 draft token | 不构造设备 Tensor，也不执行 Transformer |
| Model Runner | 维护执行侧请求状态，把调度结果整理成 batch、Tensor 和 metadata，调用模型并处理采样结果 | 不定义 Attention、MoE 和 mHC 怎么算 |
| Model | 给定输入后执行 embedding、Attention、MoE、norm 等模型计算 | 不认识完整的 `SchedulerOutput`，也不决定请求怎样拼 batch |

可以把 SchedulerOutput 看成一张施工单。它会说：

```text
req-0 本轮算 1 个 token
req-1 本轮验证 4 个 draft token
req-1 新增这些 KV block
req-2 已结束
```

但 `DeepseekV4Model.forward()` 不能直接接这张施工单。它需要的是：

```text
input_ids
positions
inputs_embeds（如果有）
以及 Attention 在 forward context 中读取的 metadata
```

Model Runner 做的事，就是把前一种描述翻译成后一种描述。

vLLM Ascend 的 V2 Runner 也明确保留了这个分工：`NPUModelRunner` 继承上游 `GPUModelRunner`。Ascend 文件补 NPU 特有的输入、attention 和 graph 行为，完整执行骨架仍由上游 Runner 提供。

```python
class NPUModelRunner(GPUModelRunner):
    """Model runner for Ascend NPUs."""
```

所以阅读 `vllm_ascend/worker/v2/model_runner.py` 时，不要只找一个从头写到尾的 `execute_model()`。Ascend 的 `execute_model()` 主要包了一层 FlashComm 处理，然后调用 `super().execute_model(...)`。真正的 request update、target forward、sampling 和 proposal 分散在上游 MRV2 主干及 Ascend override 中。

## 二、一轮 decode 的完整顺序

MRV2 把一次生成拆成两个阶段：

```text
execute_model()
  更新请求状态、准备输入、执行 target forward
        │
        ▼
  暂存 ExecuteModelState
        │
        ▼
sample_tokens()
  计算 logits、验证/采样、更新状态、生成下一轮 draft
```

放进 speculative decoding 后，稳态执行是：

```text
上一轮 DSpark 生成的 draft tokens
              │
              ▼
SchedulerOutput
              │
              ▼
Model Runner V2
  ├─ 更新 request state 和 block table
  ├─ prepare_inputs
  ├─ prepare_attn
  └─ model_state.prepare_attn
              │
              ▼
Target Model Forward
  └─ hidden states / aux hidden states
              │
              ▼
Target logits → Sampling / Rejection Sampling
              │
              ├─ 产生本轮有效输出
              └─ 回写 request state
                         │
                         ▼
              DSpark Speculator.propose()
                         │
                         ▼
                 下一轮 draft tokens
```

这里的先后顺序很重要：本轮 target forward 验证的是上一轮 draft；本轮采样和状态更新结束后，DSpark 才生成下一轮 draft。

如果只画成 `Target → DSpark → Verification`，就把两个 iteration 混在一起了。

## 三、SchedulerOutput 先进入 Runner 的状态表

SchedulerOutput 不是完整请求快照，而是本轮执行所需的增量信息。Runner 自己长期维护：

```text
req_id_to_index
all_token_ids
last_sampled_tokens
draft_tokens
num_computed_tokens
prefill_len
block_tables
```

进入 `execute_model()` 后，上游 Runner 先处理结束、释放、新增和继续运行的请求，再应用 block table 更新：

```text
finish_requests
free_states
add_requests
update_requests
block_tables.apply_staged_writes
```

这里能看出 Scheduler 和 Runner 对状态的不同视角：Scheduler 用 request ID 表达调度决定；Runner 用固定的 request slot 和预分配 buffer 组织设备执行。

例如 `req-17` 在 Runner 中可能映射到第 3 个 slot：

```text
req-17
   │ req_id_to_index
   ▼
slot 3
   ├─ all_token_ids[3]
   ├─ num_computed_tokens[3]
   ├─ draft_tokens[3]
   └─ block_tables[3]
```

后面的 input preparation 都围绕这些 slot 做批量 gather，而不是让 Python 每轮重新拼一套零散对象。

## 四、`input_ids`、`positions` 和 `seq_lens` 从哪里来？

Ascend 的关键覆盖函数是 `NPUModelRunner.prepare_inputs()`。它接收 SchedulerOutput 和当前 batch descriptor，最后返回 `AscendInputBatch`。

### `num_scheduled_tokens`：Scheduler 决定，Runner 重排

SchedulerOutput 里的 `num_scheduled_tokens` 是一个 `req_id → token count` 映射。Runner 会按当前 batch 顺序取出这些值，形成 NumPy 数组：

```text
{"req-A": 1, "req-B": 4}
          │ sort / gather
          ▼
num_scheduled_tokens = [4, 1]
```

顺序可能变化，所以后续还有 `idx_mapping` 把 batch 中的位置映射回 Runner 的 request slot。

### `input_ids`：Runner 从三类 token 中拼出来

`input_ids` 不是 Scheduler 直接创建的。Runner 从自己的请求状态和 Scheduler 带回的信息里收集：

- prefill 时，从 `all_token_ids` 取尚未计算的 prompt token；
- 普通 decode 时，取上一轮的 `last_sampled_tokens`；
- speculative verification 时，再拼上 `draft_tokens`。

对应的两个关键 helper 是：

```text
prepare_prefill_inputs()
combine_sampled_and_draft_tokens()
```

后一个函数还会生成 `logits_indices`。target forward 会为本轮所有输入位置产生 hidden states，但不一定每一行都需要过 LM Head。`logits_indices` 标出需要计算 logits、执行采样或验证的那些位置。

### `positions`：由已计算长度推出来

Runner 已经知道每个请求此前计算了多少 token，也知道本轮安排了多少 token。`prepare_pos_seq_lens()` 根据 `num_computed_tokens` 和 `query_start_loc` 填充 position 与 sequence length。

假设 batch 里只有一个请求，进入本轮前已经计算了 8 个 token，本轮是普通 decode：

```text
num_computed_tokens = 8
num_scheduled_tokens = [1]
query_start_loc = [0, 1]

input_ids = [上一轮采样出的 token]
positions = [8]
seq_lens = [9]
```

如果本轮要验证 3 个旧 draft，target 通常需要处理 draft 验证位置和一个可继续采样的位置。此时本轮 query 不再只有一行，position 会连续展开：

```text
positions = [8, 9, 10, 11]
```

具体 token 数还会受到 speculative method 和 bonus-token 规则影响，但生产关系不变：Scheduler 决定本轮安排多少 token，Runner 根据持久状态生成 position 和 seq_lens。

Ascend 这里还有一个额外动作。NPU attention backend 仍需要 CPU 侧的 `seq_lens`，因此 `NPUModelRunner` 维护 `seq_lens_cpu`，并在 speculative decoding 拒绝 token 后同步修正过的 `num_computed_tokens`。这是 Ascend 覆盖上游输入准备逻辑的原因之一。

## 五、block table、slot mapping 和 attention metadata 不是一回事

这三个名字经常一起出现，但职责不同。

### block table：一个请求占了哪些物理 KV block

Scheduler 负责分配 block。Runner 收到新增 block ID 后，把它们写入持久的 `BlockTables`。准备当前 batch 时，再按 `idx_mapping` 收集每个请求的 block table。

```text
Scheduler 分配 block ID
          │
          ▼
Runner 持久维护 BlockTables
          │ gather current requests
          ▼
当前 batch 的 block_tables
```

### slot mapping：本轮新 KV 写到哪里

Runner 根据当前请求、position、query start location 和 block table，计算本轮每个 token 对应的物理 KV slot：

```text
block table + positions + query_start_loc
                    │
                    ▼
               slot mapping
```

Scheduler 不需要知道某个 query token 最终落到 KV tensor 的哪个元素；这是 Runner 和 KV Cache backend 之间的工作。

### attention metadata：backend 执行 Attention 需要的完整说明

Runner 先执行 `prepare_attn()` 得到 block tables 和 slot mappings，再调用：

```text
model_state.prepare_attn(...)
```

这里会结合：

```text
query_start_loc
seq_lens
block_tables
slot_mappings
prefill / decode 状态
KV cache group
attention backend
graph mode
```

生成各 attention backend 消费的 metadata。

因此生产关系可以压成一句话：Scheduler 分配 block，Runner 维护 block table 并计算 slot mapping，`model_state` 按 backend 组织 attention metadata，Attention layer 在 forward 时消费它们。

## 六、这些数据怎样进入 DeepSeek V4 forward？

Runner 会把显式模型参数整理成 `model_inputs`：

```python
model_inputs = {
    "input_ids": input_batch.input_ids,
    "positions": input_batch.positions,
    "inputs_embeds": inputs_embeds,
    "intermediate_tensors": None,
    **self.model_state.prepare_inputs(input_batch, self.req_states),
}
```

attention metadata 没有作为一个普通的 `forward(attn_metadata=...)` 参数一路传下去。Runner 在调用模型前进入 `set_forward_context(...)`，把 metadata 和按 layer 整理的 slot mapping 放进当前 forward context：

```text
set_forward_context(attn_metadata, slot_mapping, ...)
                       │
                       ▼
              self.model(**model_inputs)
                       │
                       ▼
        Attention layer 从 context 取 metadata
```

这就接上了上一篇 DeepSeek V4 文章中的 `dsa_forward`：模型层只发起 DSA 自定义 op；Ascend Attention 实现再从 `ForwardContext` 里取出本轮 metadata 和 KV Cache。

Target model 完成 mHC、DSA、MoE 等计算后返回 hidden states。LM Head 仍不在模型主干 forward 里直接执行，Runner 会在 sampling 阶段按 `logits_indices` 选出需要的 hidden-state 行，再调用：

```text
self.model.compute_logits(sample_hidden_states)
```

所以 Runner 和 Model 的边界很清楚：Model 负责把输入算成 hidden states 和 logits；取哪些行算 logits、这些 logits 用来普通采样还是验证 draft，由 Runner 决定。

## 七、Speculator 接在 sampling 后面

`execute_model()` 完成 target forward 后，不急着生成 draft，而是把本轮数据存进 `ExecuteModelState`：

```text
input_batch
attention metadata
slot mappings by layer
hidden states
aux hidden states
finished request IDs
```

接着 `sample_tokens()` 取出这份临时状态。

如果本轮没有 draft，Runner 使用普通 Sampler。如果 Scheduler 带来了上一轮 draft，并且配置了 RejectionSampler，就用 target logits 验证这些 draft：

```text
hidden states
    │ logits_indices
    ▼
target logits
    │
    ├─ 普通 sampling
    └─ rejection sampling
          ├─ accepted tokens
          ├─ recovered / bonus token
          └─ num_rejected
```

之后 `postprocess_sampled()` 更新：

```text
num_computed_tokens
last_sampled_tokens
all_token_ids
模型相关状态
```

这一步必须发生在下一次 proposal 前。DSpark 需要知道本轮到底接受了多少 token、拒绝了多少 token，以及新的 anchor token 是什么。

## 八、MRV2 怎样调起 AscendDSparkSpeculator？

`NPUModelRunner.__init__()` 发现开启 speculative decoding 后，会调用 Ascend 自己的 `init_speculator()`。当配置命中 `use_dspark()` 时，返回：

```text
AscendDSparkSpeculator
    └─ 继承上游 DSparkSpeculator
```

Ascend 类没有重写整套 DSpark 算法。它补的是 NPU 运行所需的部分：

- 使用 Ascend attention metadata builder；
- 收集 Ascend attention backend；
- 调整 slot mapping dtype；
- 把 Ascend graph manager 和 update stream 接到 Speculator；
- 为 draft forward 构造 NPU 所需的 attention metadata。

KV Cache 初始化阶段，Runner 还会调用 Speculator 的 `set_attn()`，把这些基础设施交给它：

```text
model_state
kv_cache_config
block_tables
target input buffers
target attention groups
```

到 `sample_tokens()` 的末尾，Runner 调用 `speculator.propose(...)`，传入：

```text
input_batch
target attention metadata
target slot mappings
target hidden states / aux hidden states
num_sampled / num_rejected
last_sampled tokens
next prefill tokens
temperature / seeds
```

对于 DeepSeek V4，还有一个细节。普通 target 输出已经过 `hc_head` 收回单流；某些 MTP 路径需要 pre-`hc_head` 状态，Runner 会通过模型暴露的 `get_mtp_target_hidden_states()` 取得它。DSpark 则主要消费 target 的辅助层 hidden states，再由自己的 projection 合并到 draft hidden size。Runner 不解释这些 hidden states 的数学含义，只负责按模型和 Speculator 约定把正确的 Tensor 交过去。

接下来的计算就是上一篇 DSpark 文章里的内容：

```text
target aux hidden states
        │
        ▼
DSpark context KV / input preparation
        │
        ▼
parallel draft backbone
        │
        ▼
base logits [B, K, V]
        │
        ▼
sequential Markov sampling
        │
        ▼
draft tokens [B, K]
```

产生的 draft tokens 被写回 Runner 的 `req_states.draft_tokens`，再通过 `DraftTokensHandler` 交还调度侧，供下一轮 target verification 使用。

## 九、把关键状态的生产者和消费者列清楚

| 状态 | 谁准备或决定 | 谁维护 | 谁消费 |
|---|---|---|---|
| `num_scheduled_tokens` | Scheduler | SchedulerOutput；Runner 转成 batch 数组 | input preparation、attention、sampling |
| `input_ids` | Runner 从 prompt、last sampled 和 draft tokens 组合 | Runner input buffers | Target model embedding |
| `positions` | Runner 根据已计算长度生成 | Runner input buffers | RoPE、Attention、Model |
| `seq_lens` | Runner 根据 computed + scheduled 生成 | Runner；Ascend 另有 CPU 副本 | Attention backend |
| block allocation | Scheduler | Runner `BlockTables` 接收增量 | Runner、KV Cache backend |
| 当前 block table | Runner gather | 当前 execution batch | Attention metadata/backend |
| slot mapping | Runner 计算 | 当前 execution batch | KV 写入、Attention |
| target attention metadata | `model_state.prepare_attn()` | 当前 forward context | Target Attention backend |
| target hidden states | Target model | `ExecuteModelState` 暂存 | LM Head、Sampler、Speculator |
| draft attention metadata | Speculator | Speculator | DSpark Attention backend |
| draft KV | DSpark model/backend 写入 | Speculator draft state | 后续 DSpark proposal |
| draft tokens | DSpark Speculator | Runner request state | Scheduler、下一轮 target verification |

这张表比记 helper 名更有用。helper 会改，生产者和消费者的边界相对稳定。

## 十、batch=1 再走一遍

最后用一个最小例子把流程串起来。假设：

```text
batch = 1
prompt 已完成
进入本轮前已计算 8 个 token
greedy sampling
当前没有需要验证的 draft
```

进入 target forward 前：

```text
num_scheduled_tokens = [1]
input_ids = [上一轮 sampled token]
positions = [8]
seq_lens = [9]
query_start_loc = [0, 1]
block table = [该请求当前使用的物理 KV blocks]
slot mapping = [position 8 对应的物理 KV slot]
attention metadata = 当前 decode backend 所需的描述
```

然后依次发生：

```text
Target forward
  → hidden_states [1, H]

logits_indices
  → 取出需要采样的 hidden state

compute_logits
  → logits [1, V]

greedy sampling
  → sampled token

postprocess_sampled
  → 更新 last_sampled_tokens
  → 更新 num_computed_tokens

AscendDSparkSpeculator.propose
  → 接收 target hidden / aux hidden
  → 准备 draft-side positions、KV 和 attention metadata
  → 运行一次 DSpark backbone
  → Markov Head 依次生成 K 个 draft token

DraftTokensHandler
  → 把 draft 交给 Scheduler
  → 下一轮 target 一次验证多个位置
```

这个例子暂时不需要算 block index 和 slot offset。先回答两件事就够了：谁创建输入，谁消费输入。

## 十一、两个容易看错的地方

第一，Runner 不只是调用 `model.forward()`。从 Scheduler 的逻辑请求到设备侧固定 buffer，再到 KV Cache 地址、Attention backend、Sampler 和 Speculator，这些状态都在 Runner 这里接起来。Model 看到的只是当前 forward 所需的 Tensor。

第二，DSpark 生成和 target 验证不在同一个相位。稳态下的顺序是：

```text
验证上一轮 draft
  → 更新状态
  → 生成下一轮 draft
```

比较 V1 Proposer 与 MRV2 Speculator 时，也需要沿这个顺序检查两边的状态更新。

## 十二、分析边界与后续验证方向

本文聚焦 execution flow，以下实现细节不在本次分析范围内：

- `BlockTables.compute_slot_mappings()` 怎样把 position 映射到物理 KV slot；
- target KV 和 draft KV 各自怎样分组、分配和回滚；
- full graph 下 input buffer 和 metadata 为什么要保持固定地址；
- adaptive verification 怎样改变每个请求下一轮的 scheduled token 数。

这些问题都指向同一个关键点：一个 token 最终写入哪块 KV Cache。进一步分析时需要重点验证：

> 对同一个 DSpark decode step，V1 `AscendDSparkProposer` 和 MRV2 `AscendDSparkSpeculator` 分别由谁维护 request state、block table、slot mapping、draft KV 和 attention metadata？

从整体关系看，Speculative Decoding 定义“先猜再验证”的机制；DeepSeek V4 target model 负责算出可信的 hidden states 和 logits；DSpark 负责低成本地产生下一组候选；Model Runner V2 则按正确的状态和时序把这些组件连接起来。

## 参考源码

- [vLLM GPU Model Runner V2](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py)
- [vLLM DSparkSpeculator](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py)
- [vLLM Ascend NPUModelRunner V2](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/model_runner.py)
- [vLLM Ascend Speculator 工厂](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/spec_decode/__init__.py)
- [vLLM Ascend AscendDSparkSpeculator](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py)
