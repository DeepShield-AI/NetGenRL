## 2026-10-09

### 可运行性验证：初始化检查

- 已阅读 `AGENT.md`、`README.md` 和 `config.json`。
- 已确认当前分支为 `state-for-codex`；未切换分支、未提交、未回滚已有修改。
- 当前工作区原始状态：`config.json` 已有未提交修改，`AGENT.md` 和本实验记录文件为未跟踪文件；均予以保留。
- 当前配置数据集为 `iscx_small`，对应 `data-json/iscx_small`、`bins/bins_iscx_small.json` 和 `model/iscx_small` 均存在。
- 后续验证将优先使用隔离的临时输出目录，避免覆盖已有模型、结果和 bins；原始数据不修改。
- 待执行：`exp` 环境版本检查、语法检查、模块导入、数据/配置校验和小规模端到端训练。

### 环境与静态检查结果

- `exp` 环境：Python `3.9.23`，PyTorch `2.7.1`，CUDA 构建版本 `12.9`，但 `torch.cuda.is_available()` 为 `False`，本次只能使用 CPU。
- 依赖：NumPy `2.0.2`、SciPy `1.13.1`、POT `0.9.5`、Scapy `2.6.1`；`gensim` 未安装，但当前实际导入链未使用该包。
- `python -m compileall -q driver.py pre_process post_process stateFormer` 已通过。
- `stateFormer`、分箱、训练和后处理模块可导入。
- `driver`/`pre_process.pcap_process` 在受限沙箱中导入 Scapy 时因网卡探测权限得到 `PermissionError: [Errno 1] Operation not permitted`；这是执行环境限制，待用允许网卡探测的方式复核。
- `config.json` 可解析；`iscx_small` 有 4 个 JSON 类别文件、共 3,089 个 flow，`bins/bins_iscx_small.json` 包含 8 个配置字段。

### 模型 smoke test

- 使用 `exp` 环境在 CPU 上读取真实 `iscx_small` 数据，完成 2 个样本 batch 的 forward、分类 NLL 反向传播和 `sample()` 生成。
- 结果：输入形状 `[2, 16, 8]`，输出形状 `[2, 16, 1370]`，state 形状 `[2, 16, 64]`，反向传播成功；生成形状 `[2, 16, 8]` 且所有字段 ID 均在 bins 范围内。
- 测试脚本首次因命令字符串中的 f-string 引号错误退出，未涉及项目代码；修正命令后通过。
- 待执行：隔离目录中的 1 epoch 端到端分箱、训练、checkpoint 选择和生成。

### 隔离端到端验证结果

- 测试目录：`runtime_test_20261009/`；复制了 `data-json/iscx_small`，未使用原有 bins、model 或 result 输出目录。
- 使用配置中的 `max_seq_len=16`、`batch_size=64`、`epoch=1`、`checkpoint=1`、`expand_times=1`；数据集、属性和模型维度保持不变。
- 流程完整通过：ChiMerge 分箱、baseline 分箱、1 epoch 训练、checkpoint 选择、四个类别生成。
- 训练日志：设备为 `cuda`，epoch loss `125.64429473876953`，耗时约 `0.98 s`；选择的最佳模型为 `generator_1.pth`，OT `8.906076421762307`，HM `212.25202468669056`。
- 生成结果：`skype` 3640 flows、`spotify` 274 flows、`voipbuster` 1838 flows、`youtube` 426 flows；checkpoint 文件可用 CPU 加载。
- ChiMerge 输出 `RuntimeWarning: invalid value encountered in divide`，但流程继续完成；这是现有分箱数值处理的遗留风险，本次按限制未修改。
- Scapy 在受限沙箱中导入时会因系统网卡探测权限失败；使用允许该探测的执行方式运行完整 driver 后通过。因此 PCAP 预处理在本受限沙箱内未独立验证，但 driver 的实际导入和后续流程已完成。

### 结论

- 已通过：语法编译、正式模型模块导入、配置/数据/bins 校验、forward/backward、采样、隔离目录端到端训练和生成。
- 未通过：无阻断可运行性错误。
- 未验证/遗留：GPU 在普通受限命令中显示不可用，完整端到端授权执行显示使用 CUDA；`gensim` 未安装，但当前代码导入链不依赖它；PCAP 读取未在受限沙箱内独立执行；ChiMerge 数值 warning 未处理。
- 本次代码修复：无。未修改模型架构、训练策略、loss、属性生成机制或原始数据；只追加本实验记录并创建隔离测试产物。

## 2026-10-10

### 起始占位符与分箱 ID 分离：实施前记录

- 动机：真实样本的分箱编号从 0 开始，与训练移位输入和生成前缀的全零起始占位符冲突。
- 采用真实分箱 ID `original_id + 1`；0 保留为特殊输入和右侧 padding。右侧 padding 由真实序列长度 mask 从训练目标中排除。
- 计划修改正式调用链：`stateFormer/dataset.py`、`stateFormer/transformer.py`、`stateFormer/model_train.py`、`post_process/check_bestmodel.py`、`post_process/data_save.py`。历史实验文件 `newTransformer.py` 不在当前正式训练调用链内，不修改。
- 运行前已确认 `config.json` 指向 `iscx_small`，参数为 `max_seq_len=16`、`batch_size=64`、`epoch=100`、`checkpoint=50`、`expand_times=0`。
- 为避免覆盖原始 bins 定义，训练产生的 bins 将写入独立实验目录；模型 checkpoint 与生成结果使用配置中的既有目录（用户明确允许覆盖），日志写入新的独立日志文件。PCAP 和原始 JSON 不作为写入目标。
- 当前工作区已有用户修改：`config.json`、`visual.ipynb`；未跟踪的 `AGENT.md`、`EXPERIMENT.md` 也保留，不回滚。
- 下一步：实施最小代码修改，运行 ID 范围、padding/mask、forward/backward、采样和后处理解码测试，再执行完整 100 epoch 流程。

### 编码修改与训练前验证

- `stateFormer/dataset.py`：真实 `find_interval()` 结果统一加一，超出区间也返回 `1..N`；原 bins 区间不改。
- `stateFormer/transformer.py`：每个 embedding 和输出层类别数改为 `N+1`；0 号 embedding 保留给占位/padding，输出 logits 和自回归采样均屏蔽 0。
- `stateFormer/model_train.py`：NLL 按扩展后的每字段输出宽度计算，训练目标使用数据集产生的 `1..N` ID，原有长度 mask 保持不变。
- `post_process/check_bestmodel.py`、`post_process/data_save.py`：新增统一解码函数，仅在后处理边界执行 `generated_id - 1`，并拒绝 0、负数和越界 ID；真实数据读取和 bins 区间定义未改。
- `compileall` 通过。
- 训练前检查通过：原 bins 数量 `[100, 500, 10, 60, 250, 250, 100, 100]`，模型 embedding/output 数量为 `[101, 501, 11, 61, 251, 251, 101, 101]`；有效真实 ID 最小为 1；padding 保持 0；forward、反向传播、采样和解码边界测试通过。
- `git diff --check` 仍被已有 `config.json` 第 14 行尾随空格触发；该行不是本次修改内容，未改动。
- 待执行：使用当前 `config.json` 完整运行 100 epoch，并验证 checkpoint、评估、生成结果和编号范围。

### 首次 100 epoch 运行

- 已实际运行当前 `config.json` 的完整流程，达到 `Epoch [100/100]`。
- 结果：checkpoint 50 和 100 均完成评估，最佳模型为 100；生成阶段完成四个类别。
- 运行中保留了既有 `ChiMerge.py:49` 的 `RuntimeWarning: invalid value encountered in divide`。
- 训练后复核发现 padding 目标在输出层屏蔽 0 后仍会先进入 NLL，再乘长度 mask；当前实现使用有限最小 logits 尚未中断，但为避免 padding 的无效类别损失参与数值计算，追加将 masked target 替换为合法类别 1 的最小修复。该修改不改变有效序列目标或训练策略。
- 因此需在该修复后重新完整运行 100 epoch，最终结果以修复后的运行记录为准。

### 最终 100 epoch 运行与验证

- 修复 padding target 输入后，重新使用 `conda run --no-capture-output -n exp python driver.py` 完整运行 `config.json`。配置数据集为 `iscx_small`；参数未调整：`max_seq_len=16`、`batch_size=64`、`epoch=100`、`checkpoint=50`、`expand_times=0`。
- 流程实际达到 `Epoch [100/100]`，训练日志显示设备为 `cuda`；epoch 100 loss 为 `3.4690608978271484`。
- checkpoint 50 评估：OT `2.6894773702678014`，HM `206.10917592503603`。
- checkpoint 100 评估：OT `2.0715092242963595`，HM `205.60367586179564`；最佳模型为 100。
- checkpoint `model/iscx_small/generator_50.pth`、`generator_100.pth` 和最终 `generator.pth` 均存在，并可由当前模型定义加载。
- 本次生成结果位于 `result/iscx_small/`：`skype.json` 1820 flows、`spotify.json` 137 flows、`voipbuster.json` 919 flows、`youtube.json` 213 flows。四个文件均可解析，flows 和 series 非空，packet 字段为数值；生成阶段未遇到反分箱索引异常。
- 运行日志：`log/iscx_small_offset_20261010_v2.log`。首次未包含 padding target 安全修复的 100 epoch 运行日志保留于 `log/iscx_small_offset_20261010.log`，最终验收以 v2 为准。
- 本次允许并已覆盖 `bins/bins_iscx_small.json`、`model/iscx_small` 对应写入文件及 `result/iscx_small/*.json`；原始 PCAP 和 `data-json/iscx_small/*.json` 未作为写入目标。
- 旧的 `model/iscx_small/generator_10.pth`、`generator_20.pth`、`generator_30.pth`、`generator_40.pth` 未被本次 checkpoint 间隔覆盖，仍是旧输出维度格式，不兼容新增的 0 特殊位模型；本次评估未加载它们，也未删除这些既有文件。
- 唯一观察到的运行 warning 为 `pre_process/ChiMerge.py:49` 的 `RuntimeWarning: invalid value encountered in divide`；分箱流程继续完成，ChiMerge 算法未修改。
- 最终检查：代码语法编译通过；偏移 ID 范围、padding/起始占位、模型前反向、采样和解码边界测试通过；评估与结果 JSON 检查通过。
- 工作区未清理或回滚用户已有的 `config.json`、`visual.ipynb` 修改；`AGENT.md`、`EXPERIMENT.md` 仍为未跟踪文件。全量 `git diff --check` 仍会报告已有 `config.json` 第 14 行尾随空格，本次源代码修改未引入该问题。
