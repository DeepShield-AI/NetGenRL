# AGENTS.md

## Project Overview

本项目是一个基于 Transformer 的网络流量序列生成模型，用于根据流量类别（label）和 flow length 生成 packet-level 特征序列。

当前正式模型实现位于：

* `stateFormer/transformer.py`

`stateFormer/newTransformer.py` 是历史测试/实验版本，**不是当前正式模型实现**。除非任务明确要求，否则不要以 `newTransformer.py` 作为修改目标。

---

## Git / Workspace Rules

必须严格遵守：

1. 当前工作分支为 `state-for-codex`。
2. **禁止切换 Git branch。**
3. **禁止修改其他 branch。**
4. **禁止 checkout、reset、rebase、merge 或其他可能影响其他 branch 的操作。**
5. 不要修改用户已有的未提交修改。
6. 除非用户明确要求，**不要执行 `git commit`**。
7. **禁止向任何远程仓库 push、提交或修改远程分支。**
8. 不要执行可能破坏工作区的 Git 操作。
9. 修改前后应检查 `git status`，确保只产生任务要求的修改。
10. 如果发现当前 branch 不是 `state-for-codex`，**停止并报告，不要自行切换。**
11. 工作的conda环境为`exp`环境

---

## Repository Structure

主要代码：

* `driver.py`：项目整体流程入口。
* `pre_process/pcap_process.py`：PCAP → flow/JSON 的预处理。
* `pre_process/ChiMerge.py`：ChiMerge 分箱。
* `pre_process/baseline.py`：baseline 分箱。
* `stateFormer/dataset.py`：JSON 数据 → 模型输入。
* `stateFormer/transformer.py`：**当前正式模型实现，以及生成过程。**
* `stateFormer/newTransformer.py`：测试/实验版本，不作为默认修改目标。
* `stateFormer/model_train.py`：模型训练及 loss。
* `post_process/check_bestmodel.py`：checkpoint 评估/选择。
* `post_process/data_save.py`：生成结果及反分箱保存。

---

## Data Representation

一个样本对应一个双向 flow。

Flow-level attributes：

* `src_ip`
* `src_port`
* `dst_ip`
* `dst_port`
* `protocol`

Packet-level attributes：

* `time`
* `pkt_len`
* `flags`
* `ttl`

当前模型输入字段顺序通常为：

```text
[time, pkt_len, flags, ttl, src_port, dst_port, src_ip, dst_ip]
```

其中 meta attributes 会在每个 packet timestep 重复。

模型输入经过 bins 转换为离散 ID。

---

## Model Architecture

正式模型：`stateFormer/transformer.py`

核心流程：

```text
Discrete packet features
↓
Per-field Embeddings
↓
Length Embedding
↓
Label-specific Input Head
↓
Position Embedding
↓
Causal Transformer Encoder
↓
512-dim Hidden
↓
64-dim State (μ, logσ²)
↓
Reparameterized State
+
32-dim Noise
+
Current-packet Attribute Condition
↓
Per-field Output Layers
↓
Categorical Packet Features
```

主要维度：

* `hidden_dim = 512`
* `embedding_dim = 128`
* `attribute_dim = 64`
* `state_dim = 64`
* `noise_dim = 32`
* `length_emb_dim = 64`
* `transformer_layers = 4`

Transformer 使用 causal mask，因此 timestep `t` 的 hidden 只能依赖当前位置及之前的输入。

---

## State Representation

State 位于：

```text
stateFormer/transformer.py
```

通过：

```python
mu = self.state_mu(hidden)
logvar = self.state_logvar(hidden)
```

得到，再通过 reparameterization：

```python
std = torch.exp(0.5 * logvar)
s_t = mu + eps_state * std
```

得到 state。

State dimension 为 64。

需要注意：

* state 不是显式 recurrent state。
* 不存在 `state[t+1] = f(state[t], x[t])`。
* 每个 timestep 的 state 由 causal Transformer hidden 产生。
* 因此 state 可以依赖历史，但其历史信息最终需要通过 Transformer hidden → 64-dim state 传递。
* 生成时 `get_hidden()` 对当前完整历史重新计算 Transformer，并使用最后一个 timestep 的 `mu` 作为当前 state。

任何涉及 state 的修改，都必须同时检查：

* `forward()`
* `get_hidden()`
* `sample()`
* training
* generation

避免训练和生成使用不同的 state 定义。

---

## Attribute Autoregression

当前模型同时具有：

### Temporal autoregression

由 causal Transformer 实现。

### Intra-packet attribute autoregression

当前 packet 中，后面的字段使用已经生成的前序字段作为条件：

```text
field 0
↓
field 1
↓
field 2
↓
...
```

相关实现：

```text
condition_fix
step_w_hidden
```

训练时使用真实的当前字段作为条件；生成时使用模型刚刚生成的字段。

因此修改 attribute conditioning 时必须同时考虑 teacher forcing 与 free-running generation 的差异。

---

## Noise

正式模型 `transformer.py` 中的 noise 当前设计为：

```text
random noise + length
```

即 noise 本身不直接依赖历史 packet。

不要把 `newTransformer.py` 中的 position-aware noise 实现误认为正式模型设计。

---

## Training

训练入口主要位于：

```text
stateFormer/model_train.py
```

训练时：

* 使用右移后的序列作为 temporal input。
* 当前目标 timestep 使用真实数据作为 prediction target。
* 当前 timestep 的 attribute conditioning 使用真实字段。
* 因此存在 teacher forcing。

主要 loss：

1. 每个字段的 categorical NLL。
2. State 的 KL loss。
3. padding 通过 mask 排除。
4. label weight 用于加权。
5. 当前 KL 系数为 `beta = 0.1`。

修改 loss 时必须明确检查：

* loss normalization
* padding mask
* label weight
* KL 对 state 的梯度
* NLL → state → Transformer 的梯度路径

---

## Generation

生成逻辑主要位于：

```text
stateFormer/transformer.py
```

核心流程：

```text
Generated prefix
↓
get_hidden()
↓
Transformer
↓
last hidden → state μ
↓
get_noise()
↓
step_w_hidden()
↓
逐字段采样
↓
append packet
↓
repeat
```

生成阶段没有真实未来 packet，也没有真实当前字段条件，因此与训练阶段存在 teacher-forcing / exposure-bias 差异。

任何模型修改必须同时检查 training 和 generation。

---

## Known Important Issues

当前代码分析表明：

1. State 是 64-dimensional bottleneck，而 Transformer hidden 为 512-dimensional。
2. 没有显式 recurrent state transition。
3. 当前 loss 主要约束逐 timestep 的 token prediction，没有显式长期一致性目标。
4. KL loss 会限制 state 携带的信息量。
5. Training 使用真实历史，generation 使用模型生成的历史，因此存在 exposure bias。
6. 当前 `max_seq_len` 通常为 16。
7. 正式模型应以 `stateFormer/transformer.py` 为准。

这些是**分析结论，不代表任务要求立即修改，部分issue为故意设计**。除非用户明确要求，否则不要主动改变这些设计。

---

## Modification Guidelines

进行模型修改时：

1. 优先阅读相关代码的完整调用链，而不是只修改单个函数。
2. 明确区分 training 和 generation。
3. 修改 state 时同时检查 `forward()`、`get_hidden()` 和 `sample()`。
4. 修改 attribute conditioning 时同时检查 teacher forcing 和 generation。
5. 修改 loss 时检查梯度实际经过哪些模块。
6. 不要把 `newTransformer.py` 的测试设计引入正式模型，除非用户明确要求。
7. 不要修改原始数据文件。
8. 不要为了测试而改变实验配置，除非任务明确要求。
9. 尽可能进行最小化修改，不改变与当前任务无关的模型行为。
10. 修改后至少进行语法检查和适当的 smoke test。

---

## Working Style

在开始修改前：

1. 检查当前 branch 和工作区状态。
2. 阅读相关代码及其调用关系。
3. 说明准备修改的位置和原因。
4. 再进行修改。

修改后：

1. 检查 diff。
2. 确认没有产生无关修改。
3. 运行适当的最小测试。
4. 报告修改内容和测试结果。

除非用户明确要求，否则：

* 不创建 commit。
* 不 push。
* 不操作远程仓库。
* 不切换 branch。
* 不修改与当前任务无关的文件。

