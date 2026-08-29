---
title: Model Runner V2 到底管什么：从一次 Decode 串起 Scheduler、DeepSeek V4 和 DSpark
description: 沿一次 MRV2 decode iteration，分析 SchedulerOutput 如何转换为 input_ids、positions、block table 和 attention metadata，并进入 target forward、采样验证与 DSpark proposal。
pubDate: 2026-08-24
updatedDate: 2026-08-29
category: ai-infra
tags:
  - Model Runner V2
  - vLLM
  - vLLM Ascend
  - DSpark
  - Speculative Decoding
draft: false
---

在 vLLM 的推理流程中，Scheduler、Model Runner 和 Model 分别处理调度、执行编排和模型计算。三者的代码边界并不完全等同于运行时边界，特别是在引入 Speculative Decoding、DeepSeek V4 和 DSpark 之后，一轮 decode 会同时涉及请求状态、KV Cache、Attention metadata、target forward、draft 验证和下一轮 proposal。

Model Runner 位于这些模块的交界处。它接收 Scheduler 的调度结果，维护执行侧状态，准备模型输入和 KV Cache 寻址信息；target forward 结束后，它还要组织采样、更新请求状态，并把 hidden states 交给 DSpark。

本文沿 Model Runner V2（下文简称 MRV2）的一轮 decode 展开，重点回答三个问题：

1. `SchedulerOutput` 如何转换为模型可以执行的 Tensor 和 metadata；
2. DeepSeek V4 的 target forward 如何接入这套执行流程；
3. target 采样完成后，MRV2 如何调用 DSpark 生成下一轮 draft token。

本文分析基于以下版本：

```text
vLLM Ascend: main @ b4b04c5eb
对应 main2main 版本: vLLM v0.27.1
```

MRV2 仍在快速演进，上游 `main` 的函数签名可能已经变化。本文关注相对稳定的职责边界和执行顺序，不讨论 block index 与 slot offset 的具体计算。

## 1. Scheduler、Model Runner 与 Model 的职责边界

先看三类组件分别负责什么。

| 组件 | 主要职责 | 不负责的内容 |
|---|---|---|
| Scheduler | 选择本轮请求，决定每个请求计算多少 token，分配新的 KV block，并携带待验证的 draft token | 不构造设备 Tensor，不执行 Transformer |
| Model Runner | 维护执行侧请求状态，将调度结果整理为 batch、Tensor 和 metadata，调用模型并处理采样结果 | 不定义 Attention、MoE 和 mHC 的计算公式 |
| Model | 根据输入执行 embedding、Attention、MoE、norm 等计算 | 不处理完整的 `SchedulerOutput`，不决定请求如何组成 batch |

SchedulerOutput 描述的是调度决定。例如：

```text
req-0 本轮计算 1 个 token
req-1 本轮验证 4 个 draft token
req-1 新增一组 KV block
req-2 已结束
```

`DeepseekV4Model.forward()` 接收的则是模型输入：

```text
input_ids
positions
inputs_embeds（如果有）
以及 Attention 从 forward context 读取的 metadata
```

Model Runner 负责完成这两种表示之间的转换。

vLLM Ascend V2 延续了这一分工。`NPUModelRunner` 继承上游 `GPUModelRunner`，在通用执行框架上增加 NPU 相关的输入、Attention 和图执行适配。

```python
class NPUModelRunner(GPUModelRunner):
    """Model runner for Ascend NPUs."""
```

因此，在 `vllm_ascend/worker/v2/model_runner.py` 中看不到一份独立实现的完整 `execute_model()`。Ascend 的 `execute_model()` 主要处理 FlashComm，再调用 `super().execute_model(...)`。请求更新、target forward、sampling 和 proposal 的主体仍在上游 MRV2，Ascend 只覆盖平台相关部分。

## 2. 一轮 Decode 的总体流程

MRV2 将执行拆为两个阶段：

```text
execute_model()
  更新请求状态
  准备模型输入与 Attention metadata
  执行 target forward
        │
        ▼
  暂存 ExecuteModelState
        │
        ▼
sample_tokens()
  计算 logits
  验证 draft 或执行普通采样
  更新请求状态
  生成下一轮 draft
```

加入 Speculative Decoding 后，稳态流程如下：

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

这里需要区分两个 iteration：本轮 target forward 验证上一轮生成的 draft；本轮采样和状态更新完成后，DSpark 才生成下一轮 draft。将流程写成 `Target → DSpark → Verification`，会把 proposal 和 verification 的时序混在一起。

## 3. SchedulerOutput 如何进入 Runner

`SchedulerOutput` 是本轮执行的增量信息，不是完整的请求快照。Runner 持续维护以下状态：

```text
req_id_to_index
all_token_ids
last_sampled_tokens
draft_tokens
num_computed_tokens
prefill_len
block_tables
```

进入 `execute_model()` 后，Runner 依次处理已结束、已释放、新增和继续运行的请求，然后提交 block table 的增量更新：

```text
finish_requests
free_states
add_requests
update_requests
block_tables.apply_staged_writes
```

Scheduler 以 request ID 表达调度结果，Runner 则使用固定 request slot 和预分配 buffer 组织设备执行。假设 `req-17` 映射到 Runner 的第 3 个 slot，它的状态可以表示为：

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

输入准备阶段会根据这些 slot 批量 gather 数据，避免每轮都由 Python 重新组装零散对象。

## 4. 模型输入的构造过程

Ascend 的主要覆盖入口是 `NPUModelRunner.prepare_inputs()`。它接收 `SchedulerOutput` 和当前 batch descriptor，返回 `AscendInputBatch`。这一阶段需要准备 `num_scheduled_tokens`、`input_ids`、`positions`、`seq_lens` 和 `logits_indices`。

### 4.1 `num_scheduled_tokens`：调度决定，Runner 重排

`SchedulerOutput.num_scheduled_tokens` 是 `req_id → token count` 的映射。Runner 根据当前 batch 顺序取值，转换为 NumPy 数组：

```text
{"req-A": 1, "req-B": 4}
          │ sort / gather
          ▼
num_scheduled_tokens = [4, 1]
```

batch 顺序可能与 request slot 顺序不同，`idx_mapping` 用于建立两者之间的映射。

### 4.2 `input_ids`：由三类 Token 组合

`input_ids` 由 Runner 从请求状态中生成，来源分为三类：

- prefill：从 `all_token_ids` 读取尚未计算的 prompt token；
- 普通 decode：使用上一轮的 `last_sampled_tokens`；
- speculative verification：在 sampled token 后拼接 `draft_tokens`。

对应的两个关键 helper 是：

```text
prepare_prefill_inputs()
combine_sampled_and_draft_tokens()
```

`combine_sampled_and_draft_tokens()` 同时生成 `logits_indices`。target forward 会为本轮输入位置计算 hidden states，但只有部分位置需要经过 LM Head。`logits_indices` 标记需要计算 logits、采样或验证的行。

### 4.3 `positions` 与 `seq_lens`：由请求进度生成

Runner 已知每个请求的 `num_computed_tokens` 和本轮的 `num_scheduled_tokens`。`prepare_pos_seq_lens()` 根据这两类数据与 `query_start_loc` 填充 position 和 sequence length。

以单请求普通 decode 为例。进入本轮前已经计算 8 个 token：

```text
num_computed_tokens = 8
num_scheduled_tokens = [1]
query_start_loc = [0, 1]

input_ids = [上一轮采样出的 token]
positions = [8]
seq_lens = [9]
```

如果本轮需要验证 3 个旧 draft，target 通常要处理 draft 验证位置和一个可继续采样的位置，position 会连续展开：

```text
positions = [8, 9, 10, 11]
```

实际 token 数还受 speculative method 和 bonus-token 规则影响，但数据来源不变：Scheduler 决定本轮计算量，Runner 根据持久状态生成 `positions` 和 `seq_lens`。

Ascend Attention backend 还需要 CPU 侧的 `seq_lens`。因此，`NPUModelRunner` 维护 `seq_lens_cpu`，并在 speculative decoding 拒绝 token 后同步修正 `num_computed_tokens`。这也是 Ascend 覆盖上游输入准备逻辑的原因之一。

## 5. 从 Block Table 到 Attention Metadata

block table、slot mapping 和 attention metadata 都与 KV Cache 有关，但处在不同层次。

### 5.1 Block Table：记录请求占用的物理 KV Block

Scheduler 分配 block，Runner 接收新增 block ID，并写入持久的 `BlockTables`。准备本轮 batch 时，Runner 按 `idx_mapping` 收集各请求的 block table。

```text
Scheduler 分配 block ID
          │
          ▼
Runner 持久维护 BlockTables
          │ gather current requests
          ▼
当前 batch 的 block_tables
```

### 5.2 Slot Mapping：确定本轮 KV 的写入位置

Runner 根据请求状态、position、query start location 和 block table，计算每个 query token 对应的物理 KV slot：

```text
block table + positions + query_start_loc
                    │
                    ▼
               slot mapping
```

Scheduler 只负责 block 分配，不需要知道 query token 最终对应 KV Tensor 中的哪个元素。物理 slot 的计算由 Runner 和 KV Cache backend 完成。

### 5.3 Attention Metadata：描述 Backend 的执行条件

Runner 先通过 `prepare_attn()` 得到 block tables 和 slot mappings，再调用：

```text
model_state.prepare_attn(...)
```

`model_state.prepare_attn()` 综合以下信息，为不同 Attention backend 生成 metadata：

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

这三类数据的生产关系可以概括为：Scheduler 分配 block；Runner 维护 block table 并计算 slot mapping；`model_state` 按 backend 生成 attention metadata；Attention layer 在 forward 中消费这些 metadata。

## 6. DeepSeek V4 Target Forward 如何接入

Runner 将显式模型参数整理为 `model_inputs`：

```python
model_inputs = {
    "input_ids": input_batch.input_ids,
    "positions": input_batch.positions,
    "inputs_embeds": inputs_embeds,
    "intermediate_tensors": None,
    **self.model_state.prepare_inputs(input_batch, self.req_states),
}
```

Attention metadata 不会通过 `forward(attn_metadata=...)` 逐层传递。调用模型前，Runner 进入 `set_forward_context(...)`，将 metadata 和按 layer 组织的 slot mapping 放入当前 forward context：

```text
set_forward_context(attn_metadata, slot_mapping, ...)
                       │
                       ▼
              self.model(**model_inputs)
                       │
                       ▼
        Attention layer 从 context 读取 metadata
```

在 DeepSeek V4 中，模型层通过 `dsa_forward` 发起 DSA 自定义 op，Ascend Attention 实现从 `ForwardContext` 读取本轮 metadata 和 KV Cache。mHC、DSA、MoE 等模型计算完成后，target model 返回 hidden states。

LM Head 不在模型主干 forward 中直接执行。sampling 阶段，Runner 按 `logits_indices` 选出需要的 hidden-state 行，再调用：

```text
self.model.compute_logits(sample_hidden_states)
```

Model 负责从输入计算 hidden states 和 logits；Runner 决定哪些 hidden states 需要计算 logits，以及这些 logits 用于普通采样还是 draft 验证。

## 7. Sampling、状态更新与 Draft 验证

`execute_model()` 完成 target forward 后，将本轮执行数据暂存在 `ExecuteModelState` 中：

```text
input_batch
attention metadata
slot mappings by layer
hidden states
aux hidden states
finished request IDs
```

`sample_tokens()` 随后读取这份状态。没有 draft 时，Runner 使用普通 Sampler；当 Scheduler 携带上一轮 draft 且配置了 RejectionSampler 时，target logits 用于验证这些 draft：

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

采样完成后，`postprocess_sampled()` 更新：

```text
num_computed_tokens
last_sampled_tokens
all_token_ids
模型相关状态
```

状态更新必须先于下一次 proposal。DSpark 需要读取本轮的接受数量、拒绝数量和新的 anchor token，才能准备下一轮 draft。

## 8. MRV2 如何调用 AscendDSparkSpeculator

`NPUModelRunner.__init__()` 检测到 speculative decoding 配置后，会调用 Ascend 的 `init_speculator()`。当配置满足 `use_dspark()` 时，工厂返回：

```text
AscendDSparkSpeculator
    └─ 继承上游 DSparkSpeculator
```

DSpark 的通用算法仍由上游 `DSparkSpeculator` 实现。Ascend 子类补充 NPU 执行所需的能力：

- 使用 Ascend attention metadata builder；
- 收集 Ascend attention backend；
- 调整 slot mapping dtype；
- 将 Ascend graph manager 和 update stream 接入 Speculator；
- 为 draft forward 构造 NPU attention metadata。

KV Cache 初始化时，Runner 通过 Speculator 的 `set_attn()` 传入以下基础设施：

```text
model_state
kv_cache_config
block_tables
target input buffers
target attention groups
```

到 `sample_tokens()` 末尾，Runner 调用 `speculator.propose(...)`，主要输入包括：

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

DeepSeek V4 还涉及不同 hidden states 的选择。普通 target 输出已经经过 `hc_head`，恢复为单流；部分 MTP 路径需要 pre-`hc_head` 状态，Runner 通过模型提供的 `get_mtp_target_hidden_states()` 获取。DSpark 主要消费 target 的辅助层 hidden states，再通过自身 projection 合并到 draft hidden size。

Runner 不解释这些 hidden states 的数学含义，只按照 Model 与 Speculator 的接口约定传递正确的 Tensor。DSpark 后续执行如下：

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

生成的 draft tokens 写回 `req_states.draft_tokens`，并由 `DraftTokensHandler` 交给调度侧，在下一轮 target forward 中接受验证。

## 9. 关键状态的生产与消费关系

MRV2 的函数和类仍可能调整，但关键状态的生产者、维护者和消费者相对稳定。

| 状态 | 谁准备或决定 | 谁维护 | 谁消费 |
|---|---|---|---|
| `num_scheduled_tokens` | Scheduler | `SchedulerOutput`；Runner 转为 batch 数组 | input preparation、Attention、sampling |
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

排查 MRV2 的执行问题时，可以先沿这张表确认某个状态由谁生成、在哪里保存、由谁读取，再进入具体 helper。

## 10. 单请求 Decode 示例

下面用 `batch = 1` 的普通 decode 串联上述过程。假设：

```text
batch = 1
prompt 已完成
进入本轮前已计算 8 个 token
greedy sampling
当前没有需要验证的 draft
```

target forward 前，Runner 准备的数据如下：

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

随后依次执行：

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
  → 将 draft 交给 Scheduler
  → 下一轮 target 一次验证多个位置
```

这个例子省略了 block index 和 slot offset 的计算，但已经覆盖 Model Runner 的主要职责：将逻辑请求转换为模型输入，组织 target forward 与采样，再为下一轮 proposal 准备状态。

## 11. 实现中容易混淆的两个问题

### 11.1 Runner 的职责不止调用 `model.forward()`

Runner 连接了 Scheduler 的逻辑请求、设备侧固定 buffer、KV Cache 地址、Attention backend、Sampler 和 Speculator。Model 只接收当前 forward 所需的 Tensor，不感知完整调度状态。

这一区别也决定了排查顺序：模型输入错误时，应先检查 Runner 的请求状态和输入准备；Attention 结果错误时，再沿 block table、slot mapping 和 metadata 继续定位；只有输入与 metadata 一致后，才进入模型算子内部。

### 11.2 Proposal 与 Verification 分属相邻两轮

Speculative decoding 的稳态顺序是：

```text
验证上一轮 draft
  → 更新请求状态
  → 生成下一轮 draft
```

对比 V1 Proposer 和 MRV2 Speculator 时，也需要按照这个顺序检查状态更新，避免将本轮 proposal 与本轮 verification 错误对应。

## 12. 总结与后续分析

MRV2 负责推理执行阶段的状态编排。Scheduler 决定本轮运行哪些请求、计算多少 token、分配哪些 KV block；Runner 将这些决定转换为 `input_ids`、`positions`、block table、slot mapping 和 attention metadata；DeepSeek V4 target model 根据这些输入计算 hidden states 和 logits；采样完成并更新请求状态后，DSpark 再生成下一轮候选。

本文没有展开以下实现细节：

- `BlockTables.compute_slot_mappings()` 如何将 position 映射到物理 KV slot；
- target KV 与 draft KV 如何分组、分配和回滚；
- full graph 模式下为何要求 input buffer 和 metadata 保持固定地址；
- adaptive verification 如何影响每个请求下一轮的 scheduled token 数。

这些问题最终都落到 KV Cache 的物理寻址和生命周期管理上。继续比较 V1 与 MRV2 时，可以围绕下面的问题展开：

> 对同一个 DSpark decode step，V1 `AscendDSparkProposer` 和 MRV2 `AscendDSparkSpeculator` 分别由谁维护 request state、block table、slot mapping、draft KV 和 attention metadata？

## 参考源码

- [vLLM GPU Model Runner V2](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py)
- [vLLM DSparkSpeculator](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py)
- [vLLM Ascend NPUModelRunner V2](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/model_runner.py)
- [vLLM Ascend Speculator 工厂](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/spec_decode/__init__.py)
- [vLLM Ascend AscendDSparkSpeculator](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py)
