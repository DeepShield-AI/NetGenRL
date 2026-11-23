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

## 2026-10-16 MIRAGE-AppAct-2024 适配前调查

### 操作与口径

- 开始时确认分支为 `state-for-codex`，工作区已有 `config.json`、`pre_process/ChiMerge.py`、`visual.ipynb` 修改；均未触碰。阅读 `AGENT.md`、本记录、`pre_process/pcap_process.py`、`stateFormer/dataset.py`、`config.json`、`driver.py`、`README.md`，以当前正式数据调用链为准。保留此前“真实 bin ID 从 1 开始、0 为占位/padding”的约定。
- 外部数据 `/data/xiaozj/MIRAGE-AppAct-2024` 只读；逐文件按行切分顶层 flow，并逐 flow 解析与计数，未一次加载全量数据；对全部 2,245 个文件完成扫描，未发现文件读取或记录解析错误。重复检查使用 flow 值的字节摘要，未将重复摘要直接当作完整重复记录。
- 生成 `mirage_dataset_report.md`，含文件/目录规模、字段粒度与缺失、完整应用/活动/组合分布、质量问题和下一步适配方案；本阶段未转换数据、改模型或训练。

### 统计结果

- 文件均为 JSON：2,245 个、144,240,074,368 字节（144.240 GB），分布在 20 个一级应用目录；Zoom 中有 2 个仅含 `{}` 的空文件。
- 原始流级记录 106,157 条，逐包时间戳元素合计 305,825,577 个。每条记录为 5 元组键映射至 `packet_data`（逐包平行数组）、`flow_features`（流级汇总）和 `flow_metadata`（标签及流级元数据）。核心逐包数组同长、无 null、5 元组有效且非空的 flow 为 106,157 条。
- 当前模型所需的逐包 TTL 在全部记录中不存在，因此可直接满足现有八字段输入的 flow 为 **0**。逐包 `iat` 与相邻时间戳差一致；`IP_packet_bytes` 为 IP 层长度，尚不能等同于现有 Scapy `len(packet)`；`TCP_flags` 为字符串，UDP 流也带该数组，但不代表观测到 TCP flags。
- 应用类别以 `flow_metadata.BF_label` 统计：46 个非空取值加 1 个 null；目录名与 `BF_label` 不一致的 flow 有 28,835 条。活动 `BF_activity` 缺失 24,828 条，字面 `Unknown` 4,008 条；共有 142 种应用—活动组合。缺失活动不能从 `BF_sublabel` 推断，后者为应用包名/进程子标签。详细分布见报告。
- 流长度中位数 26 包、P90 612、P99 81,838、最大 1,665,842；按 `max_seq_len=16`，81,134 条 flow 会截断。发现相同 flow 值字节摘要 12 条；`BF_num_packets` 与时间戳数组长度不一致为 0；全量检查未发现非单调时间戳、非法 IAT、非法包长或非法方向值。`L4_raw_payload` 在 14,336 条 flow 中与时间戳数组不等长，不作为现有输入字段。

### 结论与后续方案

- 先确认逐包 TTL 是否能从对应原始 PCAP 经 5 元组、方向与时间戳严格匹配获取；若没有真实来源，后续需明确设计无 TTL 输入，不能填零或伪造。此项阻断当前模型的直接适配。
- 标签任务需明确是应用、活动还是应用—活动组合；以记录内标签为准，单独处理缺失、`Unknown`、背景和系统进程。若开展适配，应逐文件/流式处理，核验方向、端口、包长度定义和 TCP flags 映射，保留原始数据只读，并评估现有训练读取端一次性加载类别 JSON 的内存问题。

## 2026-10-16 MIRAGE-AppAct-2024 预处理实施

### 实施与命令

- 本次用户明确要求缺失数值属性以 `0` 填充，因此将源数据缺失的逐包 TTL 写为数值 `0`；这是一项有损占位，不能视作观测到的 TTL。此前调查关于“不能推断真实 TTL”的事实仍成立。
- 新增 `pre_process/mirage_process.py`；递归读取原始 JSON，先逐文件、逐行扫描 `BF_label` 与逐包数组偏移/长度，再以种子 `20241016` 对每个记录内应用标签做全局 reservoir 抽样，每类最多 2,000 条。第二遍仅按偏移读取被选 flow 的前 300 包，避免加载完整 144 GB 数据或百万包长的整条 flow。终端与 `/data/xiaozj/mirage-json/processing.log` 同步记录日志。
- 小规模测试：`python pre_process/mirage_process.py --input-dir /data/xiaozj/MIRAGE-AppAct-2024/Line --output-dir /tmp/mirage-test-v6 --max-flows-per-class 2 --max-packets-per-flow 300 --seed 17`；处理 1,592 条原始 flow，6 个非空类别共采样并成功转换 11 条，另 1 条缺失 `BF_label`。输出 JSON 与真实 `CustomDataset` 加载器检查通过。测试过程中依次修复脚本运行时项目模块路径、限定 `packet_data` 字段读取范围、区分 flow/包上限；最终小测试为通过版本。
- 全量命令：`python pre_process/mirage_process.py --input-dir /data/xiaozj/MIRAGE-AppAct-2024 --output-dir /data/xiaozj/mirage-json --max-flows-per-class 2000 --max-packets-per-flow 300 --seed 20241016`。原始数据保持只读；未运行训练、未改模型或训练代码。

### 全量统计与验证

- 扫描 2,245 个源 JSON，原始 flow 数 **106,157**，与前次调查一致；46 个非空 `BF_label` 类别。总采样 **40,510** 条、成功 **40,510** 条、选中后转换失败 **0** 条；另有 **1** 条缺失应用标签，计为失败且未输出。写出 **2,932,009** 个包、46 个类别 JSON 文件，共 **141,246,980 字节**（约 134.7 MiB；不含日志和报告）。
- 每类原始数、采样数、成功数、失败数、包总数、文件字节数和 MiB 见 `/data/xiaozj/mirage-json/processing_report.json` 的 `classes`；其中 `Discord` 最大，17,177,091 字节（16.381 MiB），所有类别输出数均不超过 2,000 条。
- 自动逐类重新解析输出 JSON，检查 flow/packet 字段与数值类型、每条最多 300 包、类别上限，并对每类首末样本使用当前 `stateFormer.dataset.CustomDataset` 与既有 bins 作 8 字段加载测试；46 类均通过。报告总数与实际文件大小已复核一致，源文件读取错误为 0；`python -m py_compile pre_process/mirage_process.py` 与目标文件的 `git diff --check` 通过。

### 映射与已知限制

- `time` 用相邻时间戳差并按现有预处理四舍五入到 2 位小数，首包为 0；原始 `iat` 仅用于一致性检查。首包方向作为 flow 正向，逆向包长取负；逐包端口与方向逐条核验。TCP flags 字符串映射为位掩码，UDP flags 依现有预处理规则设为 0。
- `IP_packet_bytes` 为 IP 层包长；源 JSON 没有可验证的链路层头长度，无法精确复现现有 Scapy `len(packet)`，输出 `pkt_len` 使用 IP 层包长及方向符号。该特征语义差异在后续使用时需要保留。
- 含标点的原始应用标签为兼容加载器 `filename.split('.')[0]` 的命名逻辑，输出文件名使用可逆百分号编码，如 `com.android.networkstack` → `com%2Eandroid%2Enetworkstack.json`；原标签与文件名映射保存在处理报告中。若未来以这些类别训练，类别映射需使用编码后的文件名 stem。
- 原始标签同时包含 `Background`、系统进程和其他应用进程，本次按用户要求保留所有非空 `BF_label`，未替用户决定排除哪些类。输出仅保存前 300 包；现有加载器在当前配置下还会截到 16 包。验证证明加载器可读取所选样本，不代表已有 ISCX bins 适合 MIRAGE 的最终训练分布；本次未生成 MIRAGE bins，也未训练。

## 2026-10-16 UTMobileNet2021 Wild Test Data 适配前调查

### 操作与口径

- 确认当前分支为 `state-for-codex`；保留原有工作区修改。阅读 `AGENT.md`、`pre_process/pcap_process.py`、`stateFormer/dataset.py` 与既有实验记录。仅调查 `/data/xiaozj/UTMobileNet2021/Wild Test Data`，原始 CSV 只读；逐文件、逐行扫描，避免一次加载全量数据。
- 统计原始 CSV 行作为**包记录**；只对可解析的 IPv4/TCP 或 UDP 包，在单文件内按双向五元组、无超时切分作**推定分组**。这不是源数据定义的真实 flow。行级重复检查分别使用完整行、以及忽略索引与 `frame.number` 的字段值。生成 `utmn_dataset_report.md`，未转换数据、修改模型或启动训练。

### 全量统计

- 目标目录无一级子目录，只有 14 个 CSV，共 **200,000,099 字节（190.735 MiB）**、**686,065 条原始包行**。13 个 CSV 有额外空索引列与 `location`，`google-drive` 文件没有。
- **685,728** 条包具备有效 IPv4/TCP 或 UDP 核心字段（TCP 634,958、UDP 50,770）；无效 337 条，其中缺 IPv4 字段 281、协议 2 不支持 43、复合/其他核心字段异常 13。有效包推定为 **1,319 组**（TCP 801、UDP 518），组长中位数 15、P90 317、P99 9,623、最大 71,161 包。1,258 组含双向包。
- 文件名前缀得到 14 个**候选应用**。各候选应用的有效包/推定组：facebook 21,131/46；gmail 4,374/41；google-drive 218,144/159；google-maps 56,561/64；hangout 12,924/131；hulu 85,263/213；instagram 46,281/56；messenger 1,385/42；netflix 58,846/61；pinterest 59,297/63；reddit 39,580/147；spotify 33,961/127；twitter 23,716/50；youtube 24,265/119。行内无独立应用或活动标签，故真实应用、活动及应用—活动组合的 flow 分布不可统计。
- `frame.time`、`frame.len`、`frame.cap_len` 均无空值；`ip.src/dst/proto/ttl/len` 各缺 281 行（0.041%）。TCP 端口与 `tcp.flags.ns/fin` 各缺 51,107 行（7.449%，多为非 TCP）；UDP 端口各缺 635,129 行（92.576%，多为非 UDP）。`frame.cap_len < frame.len` 为 459,466 行（66.971%）；有效核心包的 `frame.len-ip.len` 为 16/22/18 字节三种值，对应 685,429/286/13 包。
- 同一推定组按 CSV 行序有 750 次时间倒退（TCP 742、UDP 8）；完整行重复与忽略索引及帧号后的重复均为 0。逐包 IAT 无现成列，必须在可靠分流和排序后推导。

### 结论与后续方案

- 现有模型要双向 flow JSON、IPv4 整数端点、端口和协议，以及逐包 IAT、有向包长、完整 TCP flags 位掩码、TTL。源 CSV 是包级；TTL 和原始 `frame.len` 可用，但 TCP 只给 NS、FIN，不能恢复其他 flags 位。**可直接满足完整模型输入的真实 flow 为 0**；UDP 518 个推定组只是需核验的候选，不能当成已验证样本。
- 后续先查可信原始 PCAP/字段说明以恢复完整 TCP flags、核对帧长封装语义，并制定超时与五元组复用、时间乱序处理规则；确认文件名应用标签可信度及活动标签来源后，才考虑流式适配。此次未把缺失 IP、TTL 或 TCP flags 填零或伪造；MIRAGE 预处理阶段由用户要求的 TTL=0 占位约定不用于本调查。

## 2026-10-16 UTMobileNet2021 Wild Test Data 预处理实施

### 脚本、口径与运行命令

- 新增 `pre_process/utmn_process.py`。只读扫描目标目录内的 CSV，文件名前缀作为候选应用标签，同类文件合并输出一个 `[应用].json`。逐文件将可解析的包按双向 IPv4 五元组分组，稳定按绝对时间戳排序，同组超过 60 秒空闲间隔时切分；每个重建段保留排序后的前 300 包。输出是**按明确规则重建的 flow**，源 CSV 并无真实 flow ID 或边界。
- 小规模测试：从 Gmail CSV 取表头与前 200 包行写入 `/tmp/utmn-smoke-input`，运行 `python pre_process/utmn_process.py --input-dir /tmp/utmn-smoke-input --output-dir /tmp/utmn-smoke-output --max-packets-per-flow 300`；200 条记录生成 13 条重建 flow，JSON 校验及 `CustomDataset` 加载通过。
- 全量最终命令：`python pre_process/utmn_process.py --input-dir '/data/xiaozj/UTMobileNet2021/Wild Test Data' --output-dir /data/xiaozj/utmn --max-packets-per-flow 300 --idle-timeout-seconds 60`。首次在沙箱中创建指定外部输出目录时遇到只读文件系统，随后获授权重跑；最终版本为修正截断后的 `last_time` 并加入加载器验证后的复跑结果。终端日志同步写到 `/data/xiaozj/utmn/processing.log`，详细统计在 `/data/xiaozj/utmn/processing_report.json`。原始 CSV 未修改，未启动训练。

### 全量结果与验证

- 14 个 CSV、686,065 条原始包行；685,728 条可解析并参与分组；337 条排除，原因是无效/缺失/复合 IPv4 地址 294 条、非 TCP/UDP 的协议 2 共 43 条。1,319 个单文件双向五元组组经 187 次 60 秒空闲切分后生成 **1,506 条重建 flow**；它们不是源定义的真实 flow。按行序有 750 次组内时间倒退，输出前按时间稳定排序。保留 77,146 包；每条最多 300 包，截去 608,582 个超限包。
- 各候选应用的“重建 flow 数 / JSON 字节数”：facebook 68/164,234；gmail 46/73,346；google-drive 174/370,050；google-maps 80/214,500；hangout 134/316,651；hulu 239/449,987；instagram 63/205,691；messenger 57/66,729；netflix 84/266,724；pinterest 73/97,501；reddit 163/804,132；spotify 141/483,593；twitter 55/72,523；youtube 129/219,025。14 个 JSON 合计 3,804,686 字节。每类原始包数、异常数及 MiB 见处理报告。
- 最终复跑自动重新解析 14 个输出 JSON，逐条检查必需元字段、逐包字段、数值类型、非空与 300 包上限；全部通过。使用现有 `CustomDataset`、`bins/bins_iscx_small.json` 加载并逐一读取 1,506 条样本，得到 `[16,8]` 输入，加载机械兼容性通过。`python -m py_compile pre_process/utmn_process.py` 通过。

### 特征映射与限制

- `frame.time` 的 CDT/CST 使用 `America/Chicago` 解析；排序后相邻时间差按既有代码四舍五入到 0.01 秒，首包为 0。`frame.len` 作为有方向符号的包长；正向由重建段首包端点定义，逆向为负；真实 `ip.ttl` 保留。源有效包没有缺失 TTL，未发生 TTL=0 填补。端口和 IP 来自实际包头。
- TCP 仅有 `tcp.flags.ns`、`tcp.flags.fin`；逐包只把实际可见位编码到 `flags`，其余未提供的位按 0 占位，**不能解释为 SYN/ACK/PSH/RST/URG 实际为 0**。本次 634,958 个 TCP 包均受此限制；UDP 的 `flags=0` 是既有预处理的“不适用”规则。现有配置 `flags` 上限为 255，若将来出现 NS=1，脚本会显式排除并计数，当前数据没有。CSV 行内也没有可核验的应用/活动标签；文件名前缀只是候选应用。`frame.len` 与现有 Scapy `len(packet)` 的链路封装等价性尚未验证。使用旧 ISCX bins 通过加载检查不表示适合 UTMN 的分箱分布，更不表示 TCP flags 语义已完整兼容。
- 复核现有配置的 `pkt_len` 最大值为 1,500：最终保留包中 **6,932** 个 `frame.len` 超过该值，全部来自 reddit，最大 **49,220** 字节。JSON 保留原始长度及方向，不作截断或伪造；当前旧 bins 的 `CustomDataset` 对超范围值会压到边界，因此加载通过不代表包长信息被完整保留。脚本已将超限计数和最大包长写入 `processing_report.json`，并以同一全量命令最终复跑；最终 14 个 JSON 的 flow 数、包数和文件大小未变。

## 2026-10-16 UTMobileNet2021 双来源合并预处理

### 修改与执行

- 修改 `pre_process/utmn_process.py`：`--input-dir` 可重复指定多个目录；同时识别旧目录的 `[应用]_manual_...csv` 和新增目录的 `[应用]_man-[活动]_...csv` 文件名，仍按**应用**合并成一个 JSON。各 CSV 独立执行双向五元组聚合、稳定时间排序、60 秒空闲切分及每条最多 300 包；来源分布在 `processing_report.json` 的 `sources` 和每类 `sources` 中单独保留，避免跨 CSV/来源误合并会话。新增必需 CSV 表头检查，并将 IPv6、其他非 IPv4 链路类型与真正无效 IPv4 分开统计。
- 新增目录 `/data/xiaozj/UTMobileNet2021/Action-Specific Wild Test Data` 有 43 个 CSV、563,096,482 字节；42 个为 43 列布局，1 个缺空索引列和 `location`，均与旧来源的既有布局兼容。抽样确认 `frame.time`、`frame.len`、IPv4、端口、TTL 与 TCP NS/FIN 字段名称不变。文件名含活动文本，但 CSV 行内无可核验的活动标签，输出仍只按候选应用类别分组。
- 双目录小样本测试各取 Gmail 200 包，共生成 32 条重建 flow（旧 13、新 19），单一 `gmail.json`、JSON 校验及 `CustomDataset` 加载通过。最终运行命令：`python pre_process/utmn_process.py --input-dir '/data/xiaozj/UTMobileNet2021/Wild Test Data' --input-dir '/data/xiaozj/UTMobileNet2021/Action-Specific Wild Test Data' --output-dir /data/xiaozj/utmn --max-packets-per-flow 300 --idle-timeout-seconds 60`。原始 CSV 只读；更新指定输出目录、`processing_report.json` 和 `processing.log`；未训练。

### 来源与合并统计

| 来源 | CSV | 原始包行 | 可参与重建包 | 排除行 | 重建 flow | 输出保留包 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wild Test Data | 14 | 686,065 | 685,728 | 337 | 1,506 | 77,146 |
| Action-Specific Wild Test Data | 43 | 2,050,305 | 1,953,067 | 97,238 | 3,551 | 166,055 |
| 合并 | 57 | 2,736,370 | 2,638,795 | 97,575 | 5,057 | 243,201 |

- 新增来源使输出增加 **3,551 条重建 flow、166,055 个保留包**，并新增 `dropbox`、`pandora`、`skype` 三个应用；合并后 17 个应用 JSON 共 12,018,070 字节。双来源单文件双向五元组组数 4,679，经 378 次空闲切分得 5,057 条重建 flow；这些是规则生成的样本，不是源数据标注的真实 flow。逐包 300 上限截去 2,395,594 包；组内原行序时间倒退 2,348 次，输出前已稳定排序。
- 各应用的“旧来源 / 新来源 / 合并重建 flow”：dropbox 0/58/58；facebook 68/33/101；gmail 46/81/127；google-drive 174/57/231；google-maps 80/115/195；hangout 134/96/230；hulu 239/731/970；instagram 63/144/207；messenger 57/53/110；netflix 84/230/314；pandora 0/535/535；pinterest 73/139/212；reddit 163/417/580；skype 0/177/177；spotify 141/312/453；twitter 55/186/241；youtube 129/187/316。各应用原始/保留包和排除行的来源分布见 `/data/xiaozj/utmn/processing_report.json`。
- 新增来源排除原因：IPv6 **96,316**、其他非 IPv4 链路类型 **753**、不支持的 L4 **162**、无效 IPv4 **7**，合计 97,238。旧来源对应 170/106/43/18，合计 337。新增来源部分文件以 IPv6 为主，因现有模型元字段为 IPv4 整数而不能可靠纳入；例如新增 Pinterest 的 48,459 行中有 44,656 个 IPv6 帧。

### 验证与保留限制

- 最终复跑自动解析 17 个 JSON，逐 flow 和逐包检查 schema、数值类型、非空 `series`、300 包上限；全部通过。现有 `CustomDataset` 使用 `bins/bins_iscx_small.json` 逐一读取全部 **5,057** 条，输入形状为 `[16,8]`。报告中的来源合计、类别合计、输出文件字节数与实际文件已核对一致；脚本语法编译及目标文件 `git diff --check` 通过。
- TCP 仍仅有 NS、FIN，可见位被编码，其余缺失位写 0 占位；输出不能视为完整 TCP flags。新增来源有 1,512,844 个受此限制的 TCP 包，合并为 2,147,802 个。保留包中 15,775 个帧长超过当前配置 `pkt_len` 上限 1,500；旧 ISCX bins 加载通过只证明接口可读，不证明特征无损或适合 UTMN。IPv6 未被伪造为 IPv4，原始文件未修改。

## 2026-10-16 UTMN 从 CSV 重新生成与 CPU 兼容性复核

- 按本次要求检查 `ChiMerge.py`、`CustomDataset` 和既有 `data-json/iscx_small/youtube.json`：训练目录必须只含类别 JSON；每个文件顶层为 flow 列表，flow 含 `last_time`、`series` 与五个整数元字段，逐包含 `time`、`pkt_len`、`flags`、`ttl`。`ChiMerge.py:305-307` 会对训练目录中**每个文件**直接执行 `json.load`。当前 `config.json` 实际读取 `data-json/utmn`，而指定重建输出是 `/data/xiaozj/utmn`；项目目录中旧 `processing.log` 会触发 `Extra data: line 1 column 5`，旧 `processing_report.json` 虽是合法 JSON 但不是 flow 列表。
- 检查两个原始 CSV 来源的全部 57 个表头：均只有 `ip.src`、`ip.dst`，没有 IPv6 地址列。抽样 IPv6 帧的 `ip.src/dst` 为空；因此无法获取 128 位地址，更无法可靠截取后 32 位。继续将这类记录按 `unsupported_ipv6` 排除并明确计数，不伪造 IP。两个来源共排除 IPv6 96,486 行（旧 170、新 96,316）。本次未修改任何原始 CSV。
- 修改 `pre_process/utmn_process.py`：增加可指定的 `--artifacts-dir`，日志和统计报告只能放在训练 JSON 目录外；在报告中说明 IPv6 转换无原始地址可用。使用 `--input-dir` 两次从两个原始目录全量重建，临时输出 `/tmp/utmn-rebuild-stage-20261016`，临时报告和日志 `/tmp/utmn-rebuild-artifacts-20261016`；保留原双向五元组聚合、稳定时间排序、60 秒空闲切分及每 flow 最多 300 包规则。
- 临时目录含且仅含 17 个类别 JSON，合计 5,057 条重建 flow、243,201 个保留包；来源分别为 Wild Test Data 1,506 flow / 77,146 包、Action-Specific Wild Test Data 3,551 flow / 166,055 包。原始包 2,736,370，参与重建 2,638,795，排除 97,575。各应用数量与前次全量统计一致，未发现重建导致额外丢样；详细分布见处理报告。
- CPU 验证在隔离目录调用**原有** `ChiMerge.chimerge()` 成功生成 `bins_utmn.json`，随后原有 `CustomDataset` 以这些临时分箱逐一读取全部 5,057 个样本，形状 `[16,8]`。首次误用系统 Python 遇 NumPy/SciPy 二进制版本冲突；改用项目 `exp` 环境且设置 `CUDA_VISIBLE_DEVICES=-1` 后通过。ChiMerge 仍有既有 `invalid value encountered in divide` warning，未修改算法。未运行 `driver.py` 或任何 CUDA/GPU 计算。
- 验证后将旧 `/data/xiaozj/utmn` 完整备份到 `/data/xiaozj/utmn-backup-before-rebuild-20261016`，校验临时与待替换 JSON 的 SHA-256 一致后替换正式目录；日志和报告复制到 `/data/xiaozj/utmn-processing-artifacts-20261016`。正式目录仅有 17 个类别 JSON，合计 12,018,070 字节，再次由 `CustomDataset` 在 CPU 上逐一读取全部 5,057 条。
- **当前配置尚不能直接训练这份新输出**：`config.json` 仍指向项目内 `data-json/utmn`，其中仍有旧日志和报告。为遵守物理 GPU 0 现有训练任务不可受干扰的约束，本次没有改配置、移动项目内训练目录或启动训练；后续应在现有任务结束后，备份该目录并使配置指向经过验证的纯类别 JSON。TCP flags 仅 NS/FIN、超过旧配置包长分箱上限等既有语义限制仍保留。

## 2026-10-18 按类别随机划分 JSON 数据集

- 新增 `pre_process/split_dataset.py`，输入限定为 `data-json` 下的一个直接子目录。项目类别文件顶层是 flow 列表；抽查 `data-json/mirage` 的类别文件与现有模型 `series` 结构一致。脚本逐类别检查顶层列表及模型所需元字段、逐包字段，按源 flow **索引**随机打乱，每类按约 4:1 划分；固定种子和类别文件名共同决定随机顺序，重跑结果可复现。内容相同的 flow 仍作为不同源记录独立随机抽样，不按内容去重或强制放在同一侧。
- 类别仅有 1 条时放入训练、测试为空；2 条及以上时测试至少 1 条、训练至少 1 条。输出到同级 `<数据集名>-train`、`<数据集名>-test`，保持类别文件名和 flow JSON 结构。先在临时目录写入、重新解析并按源索引验证互斥和完整覆盖，再公布输出；若目标目录已存在，拒绝覆盖。原始数据不修改。
- 用 1 条、2 条以及含相同内容 flow 的 5 条合成类别验证：划分分别为 1/0、1/1、4/1；计数守恒、写出重读和固定种子可复现检查通过。`python -m py_compile pre_process/split_dataset.py` 与目标文件 `git diff --check` 通过。本次仅实现脚本及临时验证，未对正式数据集执行划分，未启动训练或修改模型代码。
- 用法：`python pre_process/split_dataset.py --input-dir data-json/mirage --seed 42`。脚本输出各类及总计的原始/训练/测试 flow 数；正式目录 `data-json/mirage-train`、`data-json/mirage-test` 仅在用户实际执行该命令时创建。

## 2026-10-19 CESNET-2025-05 适配前调查

- 已阅读 `AGENT.md`、`EXPERIMENT.md`、`pre_process/pcap_process.py`、`stateFormer/dataset.py`、`ChiMerge.py` 和当前 `config.json`（保留其已有用户修改；当前配置数据集为 `ssrc`）。模型契约仍是类别 JSON 顶层 flow 列表，五个整数元字段 `src_ip/src_port/dst_ip/dst_port/protocol`，逐包 `series` 字段 `time/pkt_len/flags/ttl`。
- 原始目录 `/data/xiaozj/CESNET-2025-05` 只读；使用系统已有的匹配 NumPy 1.26.4 + PyArrow 16.1 包缓存，按 Parquet metadata 和 `iter_batches` 流式扫描，未安装依赖、未写原始目录、未转换数据或训练。目录无一级子目录，31 个 `.parquet` 日文件、31 个 row group，schema 一致；总字节 **2,178,214,834（约 2,077.31 MiB）**，总行 **15,263,858**。
- 实际字段审计：全量均 `PROTOCOL=17/UDP`；`DST_IP_VERSION` 为 IPv4 13,592,483、IPv6 1,671,375。`DST_IP`/`DST_IP_SUBNET` 都是非空 10 位十六进制标识（不能用 IP 解析，也没有源 IP/源端口）；目的端口 147 种、范围 22--64031，其中 443 有 15,260,077 行。没有应用或活动标签列，故应用/活动/组合流分布及有效标签记录数均不可统计。
- `PPI` 是三条等长嵌套数组（间隔、方向、长度），全量 362,031,329 个元素且无 null；间隔单位由和与 `PPI_DURATION` 核验为毫秒，方向仅 ±1，长度范围 4--1472，最大序列 30。`TIME_FIRST/TIME_LAST` 是流级时间，端点差与 `DURATION` 全量一致；`PACKETS/PACKETS_REV` 是流级包数，总和 **5,359,198,989**，不是逐包字段。`PPI_LEN < PACKETS+PACKETS_REV` 有 **7,619,676** 条，只有 7,644,182 条 PPI 完整覆盖流级包数；不能将有限 PPI 冒充完整流轨迹。
- 质量检查：21 条 `QUIC_PACKETS` 长度与 `PPI_LEN` 不一致；5 条 `PPI_DURATION` 比 `DURATION` 多超过 30 ms；27 条完整 PPI 的方向长度与流级字节/头长交叉关系不一致。所有 Parquet 字段 null 数均为 0；SNI 为空 449,166 行、非空值 99,192 种，用户代理为空 15,249,192 行。前 31,744 行及按文件顺序前 100,000 行完整摘要均未发现重复，但未做全量精确去重。
- 结论：按现有八字段输入契约，原始记录 15,263,858 条中可直接作为模型输入的记录为 **0**。源数据缺真实双端 IP、源端口、逐包时间戳/TTL/TCP flags 和标签，禁止通过填零、截取标识或把流级统计伪装成逐包字段适配。调查报告见 `cesnet_dataset_report.md`；建议先获取官方字段字典/外部标识映射、标签及逐包采集字段，再另行设计可审计转换方案。

## 2026-10-19 CESNET 官方标签关联调查

- 只读核验官方 `CESNET/cesnet-datazoo` 仓库提交 `85afe5613a9a1ea840d174d779042ee13f87b4e2`。`CESNET_QUICEXT_25` 使用 `CESNET-QUICEXT-25.h5`，HDF5 按 `/flows/DYYYYMMDD` 日期表组织；每行有 `ID`、`APP`、`CATEGORY` 等字段，`APP` enum 是逐条应用标签。`servicemap.csv` 仅按 `Tag` 映射 Service Provider/Service Category，不携带样本 ID；CATEGORY 是服务类别，不是活动标签。
- 官方常量明确 `M-2025-5` 包含 20250501--20250531；QUICEXT-25 覆盖 2024-06 至 2025-05，原始规格约 194,296,462 条、50 个应用及 3 个背景类。官方没有单独按月 HDF5，DataZoo 会下载整年数据库后按日期表筛选；XS/S/M/L 是抽样规格。未下载整库。
- 检查本机 `/data/xiaozj`：只有已调查的 `/data/xiaozj/CESNET-2025-05` Parquet 和同名 ZIP，没有 `CESNET-QUICEXT-25.h5`、servicemap.csv 或其他标签数据库。Parquet schema/metadata 没有 `ID`、`APP`、`CATEGORY`；RangeIndex 只是文件行索引。虽然两边可能共享目的标识、端口、协议、时间、字节/包数和 QUIC CID 等候选字段，官方未声明这些是公共唯一键，不能凭行号、时间顺序或同名日期匹配。
- 因缺少 HDF5，尚未能验证 ID/复合键唯一性、冲突率或标签覆盖率；当前可安全关联的 Parquet 行数为 **0**，不是断言官方标签不存在。servicemap 和 enum 本身也不能给本地行赋标签。
- 结论：**官方有 2025 年 5 月带标签数据的获取途径，但目前无法证明与已下载 Parquet 逐条对应。** 后续如取得官方 HDF5，应按同月日期表先抽取 ID/APP/CATEGORY，再用 ID 或两端唯一、字段完全相同的复合键逐条匹配并报告命中/歧义；若无可证唯一键，应直接使用官方同月 HDF5 样本，不向本地 Parquet 猜配标签。详细证据和获取限制见 `cesnet_dataset_report.md` 新增章节。

## 2026-10-19 CESNET HDF5 下载成本与存储调查

- 复用 DataZoo 官方仓库提交 `85afe5613a9a1ea840d174d779042ee13f87b4e2`，确认 ORIG 下载 URL 为 `https://liberouter.org/datazoo/download?bucket=cesnet-quicext-25&file=CESNET-QUICEXT-25.h5`；XS/S/M/L 文件由同一 URL 参数分别请求 `CESNET-QUICEXT-25-XS.h5`、`...-S.h5`、`...-M.h5`、`...-L.h5`。下载器按 HTTP Content-Length 分块直写，数据库是 `.h5`；`servicemap.csv` 独立下载。官方样本量 XS/S/M/L 为 10M/25M/50M/100M，ORIG 为 194,296,462。
- 官方 DataZoo metadata API `https://zenodo.org/api/records/17249078` 返回 12 个按月 ZIP（Parquet，而非 HDF5），总压缩大小 28,186,538,748 字节。`2025-05.zip` 为 2,167,459,774 字节、MD5 `7528eab9e82ca832b6f086be7f80d6d2`；本机同名 ZIP 大小/MD5 一致，现有解压 Parquet 为 2,178,214,834 字节。该 Parquet/ZIP 无标签字段，不能与带标签 HDF5混为一份发布。
- 官方没有公布 HDF5 或 XS/S/M/L 文件大小；liberouter 的 HEAD/流式响应分别超时或返回 503，未取得 Content-Length，也未下载正文。容量计划以五月 Parquet 的每行存储代理线性外推，并按 HDF5 为代理值 1--5 倍给范围，不将估算伪称官方大小。ORIG 规划 HDF5 约 27.7--138.7 GB，下载和筛选副本同时存在时规划约 60--278 GB；XS 约 1.4--7.2 GB，临时峰值约 3--15 GB。详见报告表格及限制。
- `df -B1`：`/data` 可用 388,524,875,776 字节（约 362.0 GiB）；`/tmp`/根盘可用 103,654,199,296 字节（约 96.5 GiB）。本机未找到 HDF5/servicemap。官方 HDF5 内为 `/flows/DYYYYMMDD` 每日表，`M-2025-5` 是 31 个日期表的筛选配置，不能单独通过当前 DataZoo 类下载；官方 Zenodo 月度 ZIP 是 Parquet。
- 推荐先联系维护者确认带标签的五月 HDF5/标签导出，并优先用 XS 验证官方 ID/APP/CATEGORY 与日表读取；如只有整年文件，不建议直接下载 ORIG，需先取得准确 Content-Length/校验和、规划 `/data` 下载临时文件及筛选副本余量。XS/S/M/L 的随机抽样不能保证覆盖当前 Parquet，不能拿来逐条猜配标签。全程未下载大库、未修改原始文件、未转换或训练；详细估算、三方案比较和推荐见 `cesnet_dataset_report.md` 新增章节。

## 2026-10-19 CESNET-QUICEXT-25 XS HDF5 单独估算

- 阅读已有 `EXPERIMENT.md` 与 `cesnet_dataset_report.md`，复用官方仓库/Zenodo 证据。仅对 `https://liberouter.org/datazoo/download?bucket=cesnet-quicext-25&file=CESNET-QUICEXT-25-XS.h5` 做**一次**流式响应头请求，读取超时 25 秒，无 Content-Length/Content-Type；未读取正文或保存文件，不再重试。官方 XS 是 10,000,000 流，文件名和 HDF5 结构可由 DataZoo 代码确认，实际文件大小/内部压缩方式仍未公开或查明。
- 估算依据：已下载官方五月 ZIP 2,167,459,774 字节，解压 Parquet 2,178,214,834 字节/15,263,858 流，约 142.7 字节/流。线性换算 10M 流约 1.43 GB Parquet 等价体量；按 HDF5 相对密度 1--5 倍的**规划假设**，XS 下载/最终 `.h5` 占用约 **1.4--7.2 GB**，若未压缩可能超出。`.h5` 直接读取，未见另行解压要求；实际传输字节数应由未来 Content-Length 确定。
- `df -B1 /data` 当前可用 **388,524,875,776 字节（约 361.8 GiB）**。官方断点下载器直接写目标文件，通常无需完整第二份临时文件；若另存备份需再加 1.4--7.2 GB。按日期比例 31/365 估算的五月筛选副本约 0.12--0.61 GB，另预留索引/日志/处理余量，建议为 XS 峰值留 **10 GB，稳妥留 15 GB**。这是估算，最终以 HDF5 文件大小和真实五月表大小复核。
- 详细计算和限制已追加到 `cesnet_dataset_report.md`。本阶段未下载 HDF5、未修改原始数据或模型、未转换或训练。

## 2026-10-19 CESNET XS 下载尝试

- 本次用户授权下载并审计 XS。先复核 `AGENT.md`、既有报告/实验约定和官方 DataZoo URL；确认分支 `state-for-codex`，工作区已有 `config.json` 修改保持不动。`df -B1 /data` 显示可用 **388,524,875,776 字节**；指定 `/data/xiaozj/cesnet-datazoo/` 及 XS 文件在尝试前不存在。
- 对 `https://liberouter.org/datazoo/download?bucket=cesnet-quicext-25&file=CESNET-QUICEXT-25-XS.h5` 做一次流式连接尝试，设置连接超时 8 秒、读取超时 25 秒；得到 `ReadTimeout`，未收到 Content-Length 或正文，未创建文件。鉴于前次调查已有持续超时/503，本次依用户“若持续超时或失败，记录错误并停止”约束立即停止，未重试。
- 因未取得 HDF5，实际文件大小、校验和、完整性、日期范围/日流数、APP/CATEGORY 分布、缺失和重复 ID、属性审计及真实模型适配验证均**未执行、不能推断**。详细失败及保留的容量估算见 `cesnet_dataset_report.md` 新章节。未修改原始 Parquet、模型或配置，未转换或启动训练。

## 2026-10-19 CESNET-TLS22 适配前调查

- 已阅读 `AGENT.md`、`EXPERIMENT.md`、`pre_process/pcap_process.py`、`stateFormer/dataset.py` 和现有 CESNET 报告；保留用户已有 `config.json` 修改，未修改模型或训练代码。模型契约仍为双向 flow 的五个整数元字段和逐包 `time/pkt_len/flags/ttl`。
- 原始目录 `/data/xiaozj/cesnet-tls22` 只读检查：一级周目录 `W-2021-40`、`W-2021-41`，14 个 `flows-YYYYMMDD.csv.gz`、17 个 JSON 统计文件；flow 压缩文件合计 **9,482,149,835 字节**，目录约 8.9G。日期覆盖 2021-10-04--2021-10-17，流 CSV 总行 **141,720,670**，与 `stats-dataset.json` 及每日统计 JSON 完全一致。
- 通过 gzip CSV 分块读取审计全部 141,720,670 行。表头为 30 个字段，记录是流级统计 + PPI 有限包序列混合格式。所有 CSV 字段无空值，数值解析失败 0；`APP` 198 类、`CATEGORY` 21 类、应用—类别组合 198 种，标签缺失 0。应用标签完整计数在源目录 `stats-dataset.json`，报告列出高频/低频类别与服务类别分布。
- 字段实测：PPI 三数组为 IPT/DIR/SIZE，累计 **2,199,133,576** 个元素，`PPI_LEN` 3--30（中位数 13、P90 30）；DIR 为 ±1、首 IPT 抽样为 0、SIZE 抽样范围 1--1460。流级包数合计约 **29,307,501,175**，中位数 25、P90 83、P99 2,244、最大 18,907,538。PPI 长度小于流级包数的记录 **141,720,226**，不能当完整包轨迹。
- 质量：`PPI_DURATION<0` 有 2 条，`PPI_DURATION>DURATION` 有 8,768 条；PPI 时长与全流时长不同是定义差异信号，不可静默当解析错误。ID 从 0 连续递增至 141,720,669，按日文件边界连续，无非连续跳变/相邻重复；未做全量任意字段哈希去重，不能宣称无重复记录。
- 当前模型需要的源/目的 IP、端口、L4 协议、逐包 TTL、逐包 TCP flags 均未提供；`TCP_FLAGS`/`FLAG_*` 是流级汇总或出现性字段，不能伪装成逐包 flags；PPI SIZE 是传输 payload，不等同现有 Scapy 帧长。按现有八字段契约，原始记录中直接有效记录数 **0**。详细字段表、标签分布、质量问题及适配建议见 `cesnet_tls22_dataset_report.md`。本阶段未转换、未修改原始数据、未训练。

## 2026-10-19 CESNET-TLS22 分类 JSON 预处理

- 新增 `pre_process/cesnet_process.py`。源目录实际是 14 个 `.csv.gz`，统计 JSON 不含待转换流；脚本流式读取 CSV，按行内 `CATEGORY` 在全部文件上做种子 42 的 reservoir 抽样，每类至多 2,000 条。输入原始文件保持只读。类别名直接用于输出文件名；空格及 `&` 被保留，文件名不含点号以适配 `CustomDataset`。
- 小规模检查命令：`python pre_process/cesnet_process.py --input-dir /data/xiaozj/cesnet-tls22 --output-dir /tmp/cesnet-smoke-json --artifacts-dir /tmp/cesnet-smoke-artifacts --max-records 10000 --max-flows-per-class 20`。扫描 10,000 行，抽取并成功写入 389 flow；原 `CustomDataset` 在 `exp` 环境、`CUDA_VISIBLE_DEVICES=-1` 下读取 389 条、张量 `[16,8]`。
- 全量命令：`python pre_process/cesnet_process.py --input-dir /data/xiaozj/cesnet-tls22 --output-dir /tmp/cesnet-full-stage --artifacts-dir /tmp/cesnet-full-artifacts --max-flows-per-class 2000 --max-packets-per-flow 30 --seed 42`。扫描 141,720,670 行；21 类每类抽取 2,000，选中 42,000、成功 42,000、失败 0，保留 646,807 个 PPI 包；源行格式/类别排除 0，输出合计 33,895,237 字节。逐类原始流、成功流、包数、文件字节及失败原因见 `/data/xiaozj/cesnet-processing-artifacts/processing_report.json`，终端同内容日志见同目录 `processing.log`。
- 已将临时目录验证后的 21 个 JSON 发布至 `/data/xiaozj/cesnet-json`，日志和报告留在数据目录外。全量临时 JSON 解析、字段与长度检查通过；原 `CustomDataset` 读取 42,000 条、张量 `[16,8]`。正式目录再次复核见后续验证记录。
- 映射限制：`PPI` 的 IPT 毫秒转换为秒并按旧代码保留两位小数；DIR 确定包长符号，SIZE 是传输层 payload 字节，**不是** `pcap_process.py` 的 `len(packet)`。源数据没有双端 IP、端口、逐包 TTL 或逐包 TCP flags，按本次要求这些 schema 字段用数值 0；流级 `TCP_FLAGS` 未冒充逐包 flags。`protocol=6` 是 TLS22 数据集级 TCP 约定，源 CSV 无逐行协议列；`last_time` 是累计 PPI IPT，非绝对时间。未抽中的 PPI 未作完整解析，异常数只对被选记录有效；源数据此前调查的负 `PPI_DURATION` 等质量问题不能据此声明消失。旧 ISCX bins 的加载成功只证明接口兼容，不能证明特征语义或分箱适合该数据。未改配置、模型或训练代码，未训练。
- 发布后发现 URL 编码会使 `Analytics & Telemetry` 的文件 stem 与 `CATEGORY` 不一致；已调整脚本保留原类别名，并更正正式文件及报告。正式目录再次解析 21 个 JSON：42,000 flow、646,807 包；`CustomDataset` 在 CPU 上逐一读取全部 42,000 项且每项形状 `[16,8]`。`git diff --check` 与脚本语法检查通过。


## 2026-11-05 恢复训练中的 state KL loss

- 用户确认在 `state-for-codex` 分支实施；修改前工作区干净，本次未切换分支。
- 参照 commit `a9f413972d782e69ea81de24c6a23e64e24c2024`，在 `stateFormer/model_train.py` 恢复读取 `mu/logvar` 并计算标准正态先验的 KL，系数使用该 commit 实际值 `beta=0.001`。`AGENT.md` 中的 `beta=0.1` 与该历史代码不一致，本次未修改该约定文件。
- 损失为 `NLL字段平均 + 0.001 * KL`；KL 对 state 维求和，使用与 NLL 相同的长度 mask 和类别权重，再对 batch/timestep 求和。在字段循环外计算一次，与历史实现循环内重复加入 KL 后再除以字段数的有效系数一致。
- 保留当前真实 ID `1..N`、特殊 ID `0`、`N+1` 输出宽度、padding target 安全替换、反分箱及 GPU 后处理和 checkpoint 续训逻辑。未修改 `transformer.py`、历史实验文件 `newTransformer.py`、数据、配置及既有实验产物。
- 日志增加 NLL、KL 和 beta*KL，沿用最后一个 batch 的损失口径。
- 按用户要求仅做简单语法检查：使用 `/data/xiaozj/anaconda3/envs/exp/bin/python` 对 `driver.py`、`stateFormer/*.py`、`post_process/*.py` 共 7 个文件执行内存中的 `compile(..., "exec")`，全部通过，不导入或执行模块。`git diff --check` 通过。
- 未运行训练、采样、梯度或数值验证，未启动正式实验，未提交或 push。


## 2026-11-05 iscx_small：恢复 KL 后的 1000 epoch 后台实验 v1

- 用户授权在 `state-for-codex` 从头运行完整分箱、训练、checkpoint 选择与生成，覆盖同名输出而不续训。保留此前 KL 修复及其日志记录，未修改原始数据。
- `config.json` 数据集改为 `iscx_small`；`max_seq_len=16`、`batch_size=64`、`epoch=1000`、`checkpoint=100`、`expand_times=1`，保持八字段输入和 ChiMerge 分箱，KL `beta=0.001`。原配置除数据集名外未调整。
- `exp` 环境：PyTorch 2.7.1，沙箱外 CUDA 可用；物理 GPU 0 为 NVIDIA GeForce RTX 3080（10 GiB）。启动前两张 GPU 均无计算负载，选择 GPU 0。沙箱内无法访问驱动，因此 GPU 检查、导入和后台启动均通过授权的沙箱外执行完成。
- 输入 `data-json/iscx_small/` 四类：skype 1820、spotify 137、voipbuster 919、youtube 213，共 3089 flow。输出路径：`bins/bins_iscx_small.json`、`model/iscx_small/`、`result/iscx_small/`。本次启动前这些目标均不存在；启动脚本仍包含针对这些精确路径的覆盖清理，以保证无旧编号 checkpoint 进入续训。未清理其他数据集或历史隔离实验目录。
- 日志：`log/iscx_small_20261105_v1.log`（stdout/stderr 合并，无缓冲，启动时覆盖）；配置快照：`log/iscx_small_20261105_v1.config.json`；状态：`log/iscx_small_20261105_v1.status.json`；后台包装脚本：`log/iscx_small_20261105_v1_runner.py`。
- 启动时间：2026-11-05T14:43:05+08:00；PID=23735，PGID=23735。使用独立 session 后台运行，终端返回后仍继续执行。日志已确认 `No numbered checkpoint found; starting from epoch 0.`、设备 `cuda`、从 Epoch 1 开始，NLL、KL 和 beta*KL 为有限值。ChiMerge 保留已有 `invalid value encountered in divide` warning，但分箱已完成。
- 初始耗时参考：旧同数据集日志每轮约 0.76–0.83 秒；本次前两轮约 1.02/0.85 秒。预计训练约 14–17 分钟，含 checkpoint 写入、10 次选模评估与生成总计约 15–25 分钟，预计北京时间 14:58–15:08 完成；此为估计，最终以状态和日志为准。
- 完成时包装脚本将检查 10 个编号 checkpoint（100 至 1000）、四类 JSON 的可解析性、生成计数、序列长度、包字段有限值和 1000 条有限损失日志，成功后写 `status=completed`；异常写 `status=failed`。`expand_times=1` 表示生成两轮，预期 skype 3640、spotify 274、voipbuster 1838、youtube 426，共 6178 flow。停止信号可能使状态保留在 running，需结合进程和日志判断。

启动命令（已执行，重复运行会覆盖本次产物，监测时不要重跑）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/iscx_small_20261105_v1_runner.py > log/iscx_small_20261105_v1.log 2>&1'
```

监测命令：

```bash
cd /home/xiaozj/cgan
tail -f log/iscx_small_20261105_v1.log
cat log/iscx_small_20261105_v1.status.json
ps -p 23735 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
ls -lh model/iscx_small/generator_*.pth
```

停止命令（先核对 PID/PGID 和命令仍属于本次任务；向本次进程组发送 TERM）：

```bash
ps -p 23735 -o pid,pgid,etime,args
kill -TERM -- -23735
ps -p 23735 -o pid,pgid,stat,args
```

- 本记录描述启动与早期运行验证，尚不代表 1000 epoch、评估或生成已完成；未创建 commit 或 push。


## 2026-11-05 双 GPU 正式实验：1000 epoch、无 expand

- 用户授权覆盖同数据集既有 bins、model、result，从头训练，后台运行两条独立队列；每个数据集完整完成分箱 → 训练 → checkpoint 选择 → 生成 → 结果校验后，才进入同 GPU 的下一数据集。未切换分支，当前为 `state-for-codex`。
- GPU 0 顺序：`utmn` → `ssrc` → `iscx` → `mirage` → `cesnet3`。
- GPU 1 顺序：`utmn-train` → `ssrc_train` → `iscx_train` → `mirage-train` → `cesnet3-train`。用户所称 `ssrc-train`、`iscx-train` 在磁盘上的实际目录名为 `ssrc_train`、`iscx_train`，输入、输出与日志统一使用实际名称，未创建或更改原始数据目录。
- 参数：`epoch=1000`、`checkpoint=100`、`batch_size=64`、`max_seq_len=16`、`expand_times=0`。使用当前正式八字段 Transformer、ChiMerge 和 KL `beta=0.001`；未改模型代码或 `newTransformer.py`。保留原始 JSON、用户现有 `model_train.py` 与 `visual.ipynb` 未提交修改。
- 各任务配置独立写入 `log/formal_20261105_v1/[数据集].config.json`，不依赖随后修改的全局 `config.json`。本次没有修改全局配置。运行代码快照保存在 `log/formal_20261105_v1/source/`，覆盖 driver、训练、正式模型、预处理与后处理共 14 个 Python 文件；输入与代码 SHA-256、顺序和计数见 `manifest.json`。准备阶段解析全部类别 JSON、检查模型使用字段和非空 series，并对快照代码进行语法编译，已通过。
- 每个任务开始前先复核输入 SHA-256 和 CUDA，再清理精确对应的 `model/[数据集]/`、`result/[数据集]/` 及 `bins/bins_[数据集].json`。因此不存在加载旧编号 checkpoint 的机会；尚未轮到的任务暂时保留旧输出，到任务开始才覆盖。其他实验目录保持原样。
- 每个任务 stdout/stderr 写入并覆盖 `log/[数据集]_20261105_v1.log`；两条队列管理日志为 `log/formal_20261105_v1/gpu0.queue.log`、`gpu1.queue.log`。队列状态为 `gpu0.status.json`、`gpu1.status.json`；每个数据集另有 `[数据集].status.json`。数据集任务失败即停止该队列，另一队列继续，不把失败当作完成。
- 输出：`bins/bins_[数据集].json`、`model/[数据集]/generator.pth` 及 `generator_100.pth` 至 `generator_1000.pth` 共 10 个编号 checkpoint、`result/[数据集]/[类别].json`。无 expand，生成量应等于输入量。完成阶段会核对 1000 条有限损失日志、checkpoint 编号、结果类别、JSON 可解析性、生成计数、序列长度和包字段有限值，全部通过才写 `status=completed`。
- 启动前两张 RTX 3080 空闲，上一轮 `iscx_small` 日志已于 14:57:29 标记 COMPLETED；本次不影响该数据集产物。启动时磁盘剩余约 58.8 GiB；旧输出将按任务覆盖，不累计备份。

| 数据集 | GPU | 类别数 | 输入／预期生成 flow |
|---|---:|---:|---:|
| utmn | 0 | 17 | 5,057 |
| ssrc | 0 | 9 | 7,426 |
| iscx | 0 | 10 | 6,781 |
| mirage | 0 | 20 | 33,975 |
| cesnet3 | 0 | 21 | 209,999 |
| utmn-train | 1 | 17 | 4,047 |
| ssrc_train | 1 | 9 | 5,942 |
| iscx_train | 1 | 10 | 5,475 |
| mirage-train | 1 | 20 | 27,180 |
| cesnet3-train | 1 | 21 | 167,999 |

启动与早期验证：

- 启动时间：2026-11-05T15:38:50+08:00。
- GPU 0：队列 PID/PGID/SID=26245，初始 worker PID=26248；GPU 1：队列 PID/PGID/SID=26246，初始 worker PID=26247。worker 随数据集更换，队列 PID/PGID 不变。两条队列均已脱离终端，PPID=1，各 worker 与对应队列共享进程组。
- GPU 0 `utmn`、GPU 1 `utmn-train` 已完成分箱，日志均明确 `No numbered checkpoint found; starting from epoch 0.`、设备 `cuda`，NLL、KL、beta*KL 为有限值。启动后十余秒已分别记录 6/8 个 epoch；物理 GPU 0 worker=26248、GPU 1 worker=26247，两卡 GPU 利用率约 73%/68%，显存约 1070/1015 MiB。此为启动证据，不能代表整轮实验完成。
- ChiMerge 沿用既有数值 warning，若流程未报错则继续；完整错误将记录到对应数据集日志和状态。

耗时估计（不是完成承诺）：

- 采用同八字段、ChiMerge 的历史日志：全量 UTMN/SSRC/ISCX 每轮约 2.30/2.00/2.03 秒，MIRAGE 33.91 秒，CESNET3 204.42 秒；相应训练集约 1.74/1.59/1.75/26.54/160.04 秒。本次开头 UTMN/UTMN-train 实测约 2.31/1.76 秒，与历史口径接近。
- 单纯训练的历史线性估计：GPU 0 合计约 68 小时，GPU 1 约 53 小时；计入评估、生成、checkpoint 保存和负载波动，GPU 0 预计 60–80 小时，GPU 1 预计 48–65 小时，整体预计 60–80 小时（约 2.5–3.3 天）。按本次 11 月 5 日 15:38 启动，整体大致在北京时间 11 月 8 日 03:38 至 23:38 完成。
- 六字段等宽/等频消融的耗时不用于此次八字段 ChiMerge 估计。主要耗时来自 MIRAGE/CESNET3，实际进度以当前日志和状态为准。

启动命令（已执行，不要重复运行覆盖当前任务）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'exec /data/xiaozj/anaconda3/envs/exp/bin/python -u log/formal_20261105_v1/run.py lane 0 > log/formal_20261105_v1/gpu0.queue.log 2>&1'
setsid -f bash -c 'exec /data/xiaozj/anaconda3/envs/exp/bin/python -u log/formal_20261105_v1/run.py lane 1 > log/formal_20261105_v1/gpu1.queue.log 2>&1'
```

监测命令：

```bash
cd /home/xiaozj/cgan
cat log/formal_20261105_v1/gpu0.status.json
cat log/formal_20261105_v1/gpu1.status.json
tail -f log/formal_20261105_v1/gpu0.queue.log log/formal_20261105_v1/gpu1.queue.log
# 按队列状态中的 active 数据集选择日志；下列为启动时的两个任务。
tail -f log/utmn_20261105_v1.log log/utmn-train_20261105_v1.log
ps -p 26245,26246 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
# 查看各数据集阶段、错误和完成记录。
cat log/formal_20261105_v1/utmn.status.json
```

停止命令（先核对命令仍为本次队列及 PGID，避免 PID 重用；停止进程组会同时停止当前 worker 和后续排队任务）：

```bash
ps -p 26245,26246 -o pid,pgid,etime,args
# 仅停止 GPU 0 队列：
kill -TERM -- -26245
# 仅停止 GPU 1 队列：
kill -TERM -- -26246
# 或同时停止两条队列：
kill -TERM -- -26245 -26246
ps -p 26245,26246 -o pid,pgid,stat,args
```

- TERM 处理会尽力将队列和 worker 状态记为 stopped；强制终止可能留下 running 状态，必须结合进程与日志判断。未提交或 push。


## 2026-11-05 停止正式队列，启动 iscx_small_new 的 beta=0.1 实验

- 用户要求停止当前任务。已核实旧队列 PGID 26245/26246，先发送 TERM；训练 worker 26248/26247 已退出且 GPU 计算进程清空，但队列父进程卡在 futex 等待，随后对两个已核实进程组发送 KILL。最终 `ps` 确认四个进程均不再存在，并将两条队列状态补记为 stopped，保留所有已产生的 bins、checkpoint、日志和其他数据集输出。
- 停止时最后记录：UTMN epoch 673、UTMN-train epoch 874；其后八个排队数据集未启动，没有继续执行新任务。旧队列不代表完成的正式实验。
- 新实验输入仍为原始 `data-json/iscx_small/`，四类共 3089 flow。为使用统一 driver 的数据集命名机制，仅在 `log/iscx_small_new_20261105_v1/input/iscx_small_new` 创建指向该输入的符号链接；没有复制、修改或重命名原始数据。
- 新实验输出全部隔离：`bins/iscx_small_new/bins_iscx_small_new.json`、`model/iscx_small_new/`、`result/iscx_small_new/`。启动前这些目录均不存在；脚本使用 `mkdir(exist_ok=False)`，若发现同名目录就退出，不删除或覆盖任何既有结果。日志创建也启用 shell noclobber，防止重复启动覆盖。
- 参数沿用最近正式任务：`epoch=1000`、`batch_size=64`、`checkpoint=100`、`max_seq_len=16`、`expand_times=0`，八字段、ChiMerge、GPU 0；beta 改为 0.1。该系数仅写入独立代码快照 `log/iscx_small_new_20261105_v1/source/stateFormer/model_train.py`，主工作区训练代码仍为 beta=0.001。真实 ID/padding 修复、GPU 后处理和损失归一化方式保留。
- 配置、代码快照及输入 SHA-256 见 `log/iscx_small_new_20261105_v1/config.json`、`source/`、`manifest.json`；启动脚本 `run.py`。快照和启动脚本语法检查通过。本次未修改全局 config、用户已有模型代码修改或 visual.ipynb。
- stdout/stderr：`log/iscx_small_new_20261105_v1.log`；状态：`log/iscx_small_new_20261105_v1/status.json`。完成时检查 1000 条有限损失日志、10 个编号 checkpoint、结果 JSON 可解析性、包字段有限值和生成计数，成功才写 completed。
- 后台启动：2026-11-05T16:07:33+08:00；PID/PGID=29627。预期生成 skype 1820、spotify 137、voipbuster 919、youtube 213，共 3089 flow，无 expand。
- 根据今天先前 iscx_small 实验约 14 分钟的实际耗时，预计本次完整分箱、训练、选模和生成约 15–25 分钟，大致北京时间 16:22–16:32 完成；最终以日志和完成状态为准，启动不代表完成。

启动命令（已执行，不要重复启动）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/iscx_small_new_20261105_v1/run.py > log/iscx_small_new_20261105_v1.log 2>&1'
```

监测命令：

```bash
cd /home/xiaozj/cgan
tail -f log/iscx_small_new_20261105_v1.log
cat log/iscx_small_new_20261105_v1/status.json
ps -p 29627 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
```

停止指令（先核对 PID/PGID 和命令属于本次实验）：

```bash
ps -p 29627 -o pid,pgid,etime,args
kill -TERM -- -29627
ps -p 29627 -o pid,pgid,stat,args
```

- 原 `iscx_small` 和旧正式实验产物保留，未提交或 push。


## 2026-11-05 beta 对比输出统一命名，启动 beta=0.01

- 用户授权将已完成 beta=0.001 的 iscx_small 输出统一为 `iscx_small_0_001`，beta=0.1 的 iscx_small_new 输出统一为 `iscx_small_0_1`；另启动 beta=0.01 的 iscx_small 实验，输出 `iscx_small_0_01`。
- 已确认 beta=0.1 实验于 16:21:56 完成：1000 epoch，最佳 checkpoint=700，四类共生成 3089 flow；本次启动前两张 GPU 均空闲，无须停止旧任务。
- 已重命名 `model/iscx_small` → `model/iscx_small_0_001`、`result/iscx_small` → `result/iscx_small_0_001`；将根级 `bins/bins_iscx_small.json` 移入新建 `bins/iscx_small_0_001/`，并命名为 `bins_iscx_small_0_001.json`。
- 已重命名 `model/iscx_small_new` → `model/iscx_small_0_1`、`result/iscx_small_new` → `result/iscx_small_0_1`、`bins/iscx_small_new` → `bins/iscx_small_0_1`；其中 bins 文件也同步改为 `bins_iscx_small_0_1.json`。
- 所有目标命名在操作前均检查不存在；移动前后逐文件 SHA-256 校验一致，未覆盖文件。原始输入 `data-json/iscx_small`、历史日志/配置/代码快照以及 `baseline-transformer` 下独立实验保持原样；此前记录中的旧输出路径已由本节新路径替代。
- 移动后重新解析结果：beta=0.001 为 skype 3640、spotify 274、voipbuster 1838、youtube 426，共 6178 flow（旧实验 expand_times=1）；beta=0.1 为 skype 1820、spotify 137、voipbuster 919、youtube 213，共 3089 flow（expand_times=0）。此命名变更不重跑旧实验，也不修改其数据量。
- 新 beta=0.01 实验沿用最近一轮参数：`epoch=1000`、`batch_size=64`、`checkpoint=100`、`max_seq_len=16`、`expand_times=0`，八字段、ChiMerge、GPU 0，预期共生成 3089 flow。使用新代码快照 `log/iscx_small_0_01_20261105_v1/source/`，仅将快照内 beta 改为 0.01，主工作区训练代码仍为 0.001；全局 config 和用户其他修改保持原样。
- 新输出：`bins/iscx_small_0_01/bins_iscx_small_0_01.json`、`model/iscx_small_0_01/`、`result/iscx_small_0_01/`；脚本拒绝已有输出目录、已有运行状态和已有日志文件，保证不覆盖历史结果。
- 日志 `log/iscx_small_0_01_20261105_v1.log`；状态 `log/iscx_small_0_01_20261105_v1/status.json`；配置、输入/代码 SHA-256 与重命名校验记录见同目录 `config.json`、`manifest.json`。输入通过该目录 `input/iscx_small_0_01` 的符号链接只读访问原始 iscx_small，不复制或修改原始数据。
- 新快照和启动脚本语法编译通过。已于 2026-11-05T16:31:26+08:00 后台启动，PID/PGID=30835；预计总耗时 15–25 分钟，大致北京时间 16:46–16:56 完成，最终以完成校验和日志为准。

启动命令（已执行）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/iscx_small_0_01_20261105_v1/run.py > log/iscx_small_0_01_20261105_v1.log 2>&1'
```

监测命令：

```bash
cd /home/xiaozj/cgan
tail -f log/iscx_small_0_01_20261105_v1.log
cat log/iscx_small_0_01_20261105_v1/status.json
ps -p 30835 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
```

停止命令（先确认 PID/PGID 与命令仍属于本次实验）：

```bash
ps -p 30835 -o pid,pgid,etime,args
kill -TERM -- -30835
ps -p 30835 -o pid,pgid,stat,args
```

- 本节记录后台启动，不代表 beta=0.01 已完成；完成后脚本会校验 1000 条有限损失日志、10 个编号 checkpoint、结果 JSON 结构和生成计数，成功才写 completed。未提交或 push。


## 2026-11-05 iscx_small beta=0.025 / 0.05 双 GPU 并行实验

- 按用户授权同时启动 beta=0.025 与 beta=0.05 两个 iscx_small 实验；输出名称分别为 `iscx_small_0_025`、`iscx_small_0_05`。不覆盖 beta=0.001/0.01/0.1 或其他既有产物，不修改原始输入。
- 先确认 beta=0.01 实验已于 16:45:49 完成：1000 epoch、最佳 checkpoint=800、生成 3089 flow，状态为 completed；启动前两张 GPU 空闲，没有干扰其他训练任务。
- 两个实验均保持 `epoch=1000`、`batch_size=64`、`checkpoint=100`、`max_seq_len=16`、`expand_times=0`、八字段输入、ChiMerge 及 GPU 后处理。使用与已完成 beta=0.01 实验相同的代码快照，唯一训练代码差异是 beta；主工作区 `model_train.py`、全局 config 和用户其他修改保留原状。
- 输入均为 `data-json/iscx_small/`，skype 1820、spotify 137、voipbuster 919、youtube 213，共 3089 flow。各实验通过自己的 `input/[实验名]` 符号链接只读访问原始数据，保留输入 SHA-256、代码 SHA-256 和配置快照。两个代码快照及启动脚本语法检查已通过。
- 每个实验均拒绝已有输出目录、状态或同名日志。完成后自动校验 1000 条有限损失日志、100至1000共10个编号 checkpoint、生成类别/计数、JSON 可解析性、序列长度与包字段有限值，全部通过才写 completed。

| beta | 物理 GPU | bins 文件 | model 目录 | result 目录 |
|---|---:|---|---|---|
| 0.025 | 0 | `bins/iscx_small_0_025/bins_iscx_small_0_025.json` | `model/iscx_small_0_025/` | `result/iscx_small_0_025/` |
| 0.05 | 1 | `bins/iscx_small_0_05/bins_iscx_small_0_05.json` | `model/iscx_small_0_05/` | `result/iscx_small_0_05/` |

- 日志分别为 `log/iscx_small_0_025_20261105_v1.log`、`log/iscx_small_0_05_20261105_v1.log`，合并 stdout/stderr 且无缓冲。对应同名目录内保存 `run.py`、`config.json`、`manifest.json`、`status.json` 和 `source/`。
- 启动时间：2026-11-05T16:54:35+08:00；beta=0.025 PID/PGID=32479，beta=0.05 PID/PGID=32480，均使用独立 session 后台运行。
- 参考今天两次已完成同参数实验实际总耗时约 14分23秒，两项并行预估总墙钟耗时约15–25分钟，预计北京时间 17:09–17:19 完成。并行负载可能影响实际耗时，以日志和 completed 状态为准。

启动命令（已执行，不要重复运行）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/iscx_small_0_025_20261105_v1/run.py > log/iscx_small_0_025_20261105_v1.log 2>&1'
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/iscx_small_0_05_20261105_v1/run.py > log/iscx_small_0_05_20261105_v1.log 2>&1'
```

监测命令：

```bash
cd /home/xiaozj/cgan
tail -f log/iscx_small_0_025_20261105_v1.log log/iscx_small_0_05_20261105_v1.log
cat log/iscx_small_0_025_20261105_v1/status.json
cat log/iscx_small_0_05_20261105_v1/status.json
ps -p 32479,32480 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
```

停止命令（先核对 PID/PGID 和命令仍属于本次实验）：

```bash
ps -p 32479,32480 -o pid,pgid,etime,args
# 仅停止 beta=0.025：
kill -TERM -- -32479
# 仅停止 beta=0.05：
kill -TERM -- -32480
# 或同时停止：
kill -TERM -- -32479 -32480
```

- 本节记录启动，并非完成结果；未创建 commit 或 push。


## 2026-11-05 限定修复：分箱单位、频数累计与前缀续写对齐

- 用户明确授权三项修改，仍在 `state-for-codex`；未切换分支。用户确认元字段“每 flow 创建一行、仅首行累计直方图”和生成流级属性与最终保存值的差异均为故意设计，不再作为本次修复目标；保留这些逻辑及其他既有设计。
- `pre_process/ChiMerge.py`：在 `cal_bins_one_category()` 中，将物理数值区间映射为 `round(value/min_unit)` 的整数网格，再以步长1执行低频合并与 chi merge；例如0秒和0.01秒映射为0和1。合并完成后乘回 min_unit 并按原精度舍入，因此对外返回和 bins JSON 仍使用原字段单位。`chi_cal()` 补充整数网格输入约定；检验公式、自由度和阈值策略均未改。
- 同文件 `construct_intervals_data()` 将 `tmp_dict[k][id] = v` 改为 `+= v`，对同一统计行内裁剪或舍入到相同数值的频数进行累计，保留各行独立性和计数守恒。
- `stateFormer/transformer.py`：仅在 `sample(x=prefix)` 分支内部历史前拼接零 BOS，使用 `[BOS,prefix]` 计算后续 state；返回值仍是 prefix 加新生成包，不包含 BOS。默认从零生成、forward、get_hidden、noise 和字段生成逻辑保持原样。
- 新增 `tests/test_binning_and_prefix.py`，8项 CPU 回归测试覆盖：双侧边界裁剪累计及顺序无关性、舍入冲突频数守恒、时间网格统计量和返回秒单位、物理单位缩放下合并结果一致、元字段布局保留、续写历史/BOS/有效 ID 和 state 与训练对应 mu 的一致性、完整前缀直接返回、默认从零生成。
- 首次测试中一个 spy 在合并后读取了被原地修改的区间引用，造成测试断言失败；改为在调用时复制区间和计数后，8项全部通过。产品实现未为该测试问题额外修改。
- 验证命令：`CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /data/xiaozj/anaconda3/envs/exp/bin/python -m unittest discover -s tests -p 'test_binning_and_prefix.py' -v`，8项通过。对两个修改模块和测试文件执行内存语法编译通过；本次修改文件的 `git diff --check` 通过。
- 额外用 AST 对比 HEAD 确认 `construct_data_dic/threshold_merge/chi_merge` 及 `forward/get_hidden/get_noise/step_w_hidden` 未改。原有 `model_train.py`、`visual.ipynb` 和历史 EXPERIMENT 内容的用户修改保留。
- 本次未启动正式训练、未重建或覆盖任何既有 bins/model/result、未修改 `newTransformer.py`、GPU 后处理、beta 或全局配置。后续若采用新生成的 bins，应与对应新训练权重配套使用，历史产物继续保留原分箱定义。未 commit 或 push。


## 2026-11-05 ChiMerge 修改前后分箱数比较：iscx_small / iscx

- 按用户要求仅运行预处理分箱，不训练、不选模、不生成流量。使用 `exp` 环境 CPU，`CUDA_VISIBLE_DEVICES=-1`。
- 修改前代码取自当前 HEAD `97b8d08a64cacf66fc268b2ab47a770ef7654224` 的 `pre_process/ChiMerge.py`，修改后取当前工作区文件；分别保存为 `log/chimerge_compare_20261105_v1/before.py`、`after.py`，未执行 checkout 或切换分支。配置快照及两版本源码 SHA-256 见同目录 `config.json`、`summary.json`。
- 两版本使用完全相同的原始 JSON、当前八字段配置和 `max_seq_len=16`；每个数据集在两次计算前后校验输入 SHA-256 一致。元字段统计布局、检验自由度和阈值设计保留。
- 输出隔离至 `bins/chimerge_compare_20261105_v1/before/bins_[数据集].json` 与 `after/bins_[数据集].json`，未覆盖任何已有 bins、模型或生成结果。

| 属性 | iscx_small 修改前 | iscx_small 修改后 | iscx 修改前 | iscx 修改后 |
|---|---:|---:|---:|---:|
| time | 53 | 160 | 82 | 245 |
| pkt_len | 340 | 340 | 554 | 554 |
| flags | 9 | 9 | 13 | 13 |
| ttl | 52 | 52 | 66 | 66 |
| src_port | 95 | 95 | 235 | 235 |
| dst_port | 46 | 46 | 77 | 77 |
| src_ip | 16 | 16 | 29 | 29 |
| dst_ip | 266 | 266 | 408 | 408 |

- 仅 time 改变：iscx_small 53 → 160（+107），iscx 82 → 245（+163）。其余七个字段不仅分箱数相同，完整区间列表也相同。比较为网格单位修复和累计频数修复共同生效后的结果，未单独消融两项改动；分箱数增加本身不代表生成质量改善。
- 完整结果、每版耗时、输入及代码哈希见 `log/chimerge_compare_20261105_v1/summary.json`，表格见 `summary.csv`，执行输出见 `compare.log`；旧算法的既有数值 warning 已保留于日志。
- 执行命令：`CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/chimerge_compare_20261105_v1/compare.py > log/chimerge_compare_20261105_v1/compare.log 2>&1`。进程退出码0，日志以 COMPLETED 结束。
- 未修改模型和训练代码，未提交或 push。


## 2026-11-05 清理分箱对比临时产物，启动 ISCX beta=0.01 / 0.025

- 按用户要求清除刚才分箱数对比新增的代码和输出：删除 `log/chimerge_compare_20261105_v1/`（8个脚本、快照、日志及汇总文件）和 `bins/chimerge_compare_20261105_v1/`（4份隔离 bins）。先确认对比日志 COMPLETED，再仅删除上述两个精确目录；保留此前算法修复、8项回归测试和所有其他历史数据/实验输出。前一节的比较数值和运行历史保留，其中临时文件链接现已失效。
- 新实验使用完整 `data-json/iscx/`，不是 iscx_small，也不是 iscx_train；10类共6781 flow。物理 GPU 0 负责 beta=0.01，物理 GPU 1 负责 beta=0.025，同时独立后台运行。
- 参数沿用此前测试：`epoch=1000`、`batch_size=64`、`checkpoint=100`、`max_seq_len=16`、`expand_times=0`，八字段输入、ChiMerge、checkpoint OT选模及生成。这里“测试”沿用原有训练目录的 checkpoint 评估流程，没有新增独立测试集评估。
- 两项均使用当前工作区代码快照，包含时间整数网格、裁剪/舍入累计频数、前缀 BOS 对齐修复；不改变用户明确保留的元字段统计和生成/保存设计。仅分别修改各快照 beta，主代码和全局配置保持原样。
- 输出分别为 `bins/iscx_0_01/bins_iscx_0_01.json`、`model/iscx_0_01/`、`result/iscx_0_01/`，以及对应 `iscx_0_025` 路径。新输出均在启动前确认不存在，拒绝已有目录、日志和状态，不加载旧 checkpoint，不覆盖已有产物。
- 日志以 beta 作后缀：`log/iscx_20261105_v1_0_01.log`、`log/iscx_20261105_v1_0_025.log`；分别合并 stdout/stderr、无缓冲。状态、配置、输入/代码 SHA-256 和代码快照分别保存在同名日志目录中的 `status.json`、`config.json`、`manifest.json`、`source/`，启动脚本为 `run.py`。
- 输入通过各自 `input/iscx_0_01` / `input/iscx_0_025` 符号链接只读访问原始 ISCX；配置仅位于独立目录，不修改或重命名原始数据。源码快照和启动脚本语法编译通过。
- 启动时间：2026-11-05T18:10:01+08:00；beta=0.01 PID/PGID=35381，beta=0.025 PID/PGID=35382。已确认两进程 PPID=1、各自独立 session，分别在物理 GPU 0/1 占用约864 MiB显存，GPU利用率约88%/86%。
- 两份日志均确认 `No numbered checkpoint found; starting from epoch 0.`、设备cuda，从Epoch 1开始。最初4轮记录的每轮耗时稳定在约2.12/2.10秒，NLL、KL、beta*KL均为有限值。两者分箱数均为 time245、pkt_len554、flags13、ttl66、src_port235、dst_port77、src_ip29、dst_ip408，与修复后的比较一致。
- 预计单项训练约35–38分钟，含10个 checkpoint 评估和最终生成，总墙钟耗时预估40–50分钟；两项并行，整体预计北京时间 18:50–19:00 完成，以日志与完成校验为准。
- 每项预期生成与输入等量的6781 flow，完成后检查10个编号 checkpoint、1000条有限损失日志、类别集合、JSON可解析性、生成计数、序列长度及包字段有限值，全部通过才写completed。启动不代表实验已完成。

启动命令（已执行，不要重复启动）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/iscx_20261105_v1_0_01/run.py > log/iscx_20261105_v1_0_01.log 2>&1'
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/iscx_20261105_v1_0_025/run.py > log/iscx_20261105_v1_0_025.log 2>&1'
```

监测命令：

```bash
cd /home/xiaozj/cgan
tail -f log/iscx_20261105_v1_0_01.log log/iscx_20261105_v1_0_025.log
cat log/iscx_20261105_v1_0_01/status.json
cat log/iscx_20261105_v1_0_025/status.json
ps -p 35381,35382 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
```

停止命令（先核实 PID/PGID 和命令仍为本次任务）：

```bash
ps -p 35381,35382 -o pid,pgid,etime,args
# 仅停止 beta=0.01：
kill -TERM -- -35381
# 仅停止 beta=0.025：
kill -TERM -- -35382
# 或同时停止：
kill -TERM -- -35381 -35382
```

- 当前仍在 `state-for-codex`；既有用户修改保留，未commit或push。


## 2026-11-05 ISCX beta=0.05 / 0.1 双 GPU 并行测试

- 用户授权在 ISCX 上增加 beta=0.05、0.1 两项测试，沿用上一轮 ISCX 测试方式：完整分箱 → 1000 epoch训练 → 每100 epoch的10个checkpoint评估选模 → 生成与校验。`batch_size=64`、`max_seq_len=16`、`expand_times=0`，八字段及修复后的 ChiMerge、前缀对齐保持一致。
- 启动前确认 beta=0.01/0.025 两项状态均为completed，分别于18:49:36/18:48:28完成，各生成6781 flow，最佳checkpoint分别700/1000；两张GPU空闲，无需停止其他任务。
- 新任务使用与已完成 ISCX beta=0.01 相同的代码快照，仅改变各自快照 beta。输入为 `data-json/iscx/`，10类、6781 flow，全部文件SHA-256与上一轮一致。主工作区代码、全局config和历史实验产物不修改。
- GPU 0：beta=0.05，实验名 `iscx_0_05`，输出 `bins/iscx_0_05/bins_iscx_0_05.json`、`model/iscx_0_05/`、`result/iscx_0_05/`；日志 `log/iscx_20261105_v1_0_05.log`。
- GPU 1：beta=0.1，实验名 `iscx_0_1`，对应输出 `bins/iscx_0_1/bins_iscx_0_1.json`、`model/iscx_0_1/`、`result/iscx_0_1/`；日志 `log/iscx_20261105_v1_0_1.log`。
- 所有目标在准备时均不存在；脚本拒绝已有输出目录、状态和日志，保证不覆盖、不加载已有checkpoint。每个任务的同名日志目录中保存run.py、config.json、manifest.json、source/与status.json，包含输入/源码哈希；通过input符号链接只读访问原始ISCX。快照和启动脚本语法编译通过。
- 后台启动时间：2026-11-05T18:56:10+08:00。beta=0.05 PID/PGID=39250，beta=0.1 PID/PGID=39251，各自独立session。任务结束前会校验10个编号checkpoint、1000条有限损失日志、生成类别、计数、JSON可解析性及包字段有限值，成功才标记completed。
- 沿用此前checkpoint评估口径：与训练目录真实数据比较OT，不包含新增独立测试集。两项各预期生成6781 flow，无expand。
- 上一轮相同规模实测总耗时约38.4–39.6分钟，预计本轮并行总墙钟耗时约40–50分钟，即北京时间 19:36–19:46 完成；启动不代表完成，实际以日志与状态为准。

启动命令（已执行，不要重复启动）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/iscx_20261105_v1_0_05/run.py > log/iscx_20261105_v1_0_05.log 2>&1'
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/iscx_20261105_v1_0_1/run.py > log/iscx_20261105_v1_0_1.log 2>&1'
```

监测命令：

```bash
cd /home/xiaozj/cgan
tail -f log/iscx_20261105_v1_0_05.log log/iscx_20261105_v1_0_1.log
cat log/iscx_20261105_v1_0_05/status.json
cat log/iscx_20261105_v1_0_1/status.json
ps -p 39250,39251 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
```

停止命令（先确认PID/PGID与命令仍属于本轮实验）：

```bash
ps -p 39250,39251 -o pid,pgid,etime,args
# 仅停止 beta=0.05：
kill -TERM -- -39250
# 仅停止 beta=0.1：
kill -TERM -- -39251
# 或同时停止：
kill -TERM -- -39250 -39251
```

- 仍在state-for-codex，保留用户现有修改，未commit或push。


## 2026-11-05 beta=0.05 正式双 GPU 实验 v2

- 用户已将主工作区 `stateFormer/model_train.py` 的 beta设为0.05，本次准备时通过AST核实该赋值，并原样保存当前代码快照，未覆盖用户修改。仍在 `state-for-codex`，未切换分支。
- GPU 0严格依次运行：`utmn` → `ssrc` → `iscx` → `mirage` → `cesnet3`。
- GPU 1严格依次运行对应训练集：`utmn-train` → `ssrc_train` → `iscx_train` → `mirage-train` → `cesnet3-train`。SSRC/ISCX训练集使用磁盘实际下划线命名，未创建替代数据集或更改原始目录。
- 每个数据集完整完成分箱 → 训练 → 10个checkpoint评估选模 → 生成 → 输出校验后，才进入同GPU下一个数据集；两GPU队列独立并行，一条失败即停止该队列，另一条继续。
- 参数：`beta=0.05`、`epoch=1000`、`batch_size=64`、`checkpoint=100`、`max_seq_len=16`、`expand_times=0`；沿用八字段、GPU后处理，包含时间整数网格、裁剪/舍入频数累加和前缀BOS修复，保留用户明确指定的元字段统计及生成/保存设计。
- 配置分别保存到 `log/formal_20261105_v2/[数据集].config.json`，源码快照位于同目录 `source/`（14个Python文件）；顺序、输入计数/哈希、源码哈希见manifest.json。启动后每个worker只导入此快照，后续工作区编辑不会改变排队任务的训练代码。全局config保持原样，各任务配置明确覆盖expand_times为0。
- 准备阶段完整解析十个数据集的类别JSON，核对非空series及模型使用字段的有限值，保存SHA-256；所有源码快照及运行脚本语法编译通过。每个数据集开始前再次检查输入哈希和CUDA。
- 用户授权覆盖的精确输出是 `bins/bins_[数据集].json`、`model/[数据集]/`、`result/[数据集]/`；每个任务开始前清理这三个对应目标，确保不读取旧编号checkpoint或旧结果。尚未开始的数据集保留旧产物，轮到该任务时才清理。beta对比实验目录、iscx_small系列、baseline及原始数据均不作为清理目标。
- 日志位于 `log/[数据集]_20261105_v2.log`，合并stdout/stderr、无缓冲；v1日志和所有既有实验记录保留。所有v2日志在准备时确认不存在，创建日志时使用exclusive模式，防止意外重启覆盖。
- 输出checkpoint为generator.pth及generator_100.pth至generator_1000.pth共10个编号文件；结果为每类JSON，无expand，生成flow数应与输入一致。完成阶段校验1000条有限损失日志、编号集合、输出类别、JSON可解析性、生成计数、长度及包字段有限值，通过才写completed。checkpoint选模沿用训练目录的OT评估口径，未增加独立测试集。

| 数据集 | GPU | 类别数 | 输入／预期生成 flow |
|---|---:|---:|---:|
| utmn | 0 | 17 | 5,057 |
| ssrc | 0 | 9 | 7,426 |
| iscx | 0 | 10 | 6,781 |
| mirage | 0 | 20 | 33,975 |
| cesnet3 | 0 | 21 | 209,999 |
| utmn-train | 1 | 17 | 4,047 |
| ssrc_train | 1 | 9 | 5,942 |
| iscx_train | 1 | 10 | 5,475 |
| mirage-train | 1 | 20 | 27,180 |
| cesnet3-train | 1 | 21 | 167,999 |

启动与早期验证：

- 启动前两张RTX3080均空闲；上一轮ISCX beta=0.05/0.1实验已分别于19:35:46/19:34:44完成，各生成6781 flow，不受本次同名正式目录覆盖影响。磁盘剩余约57 GiB，同名旧输出按任务替换，不累计备份。
- 启动时间：2026-11-05T19:51:23+08:00；GPU 0队列PID/PGID/SID=43650，初始worker PID=43654；GPU 1队列PID/PGID/SID=43651，初始worker PID=43653。父队列PPID=1，worker随数据集切换而变化，PGID保持为各自队列。
- 已检查两张物理GPU分别运行对应worker，显存约982/992 MiB；两份首任务日志均确认从epoch 0开始、设备cuda、beta=0.05，NLL/KL/beta*KL为有限值。本轮UTMN初始稳定每轮约2.35秒、UTMN-train约1.78秒。
- v2包装脚本避免在信号处理函数中调用Popen.wait/poll，改为标记停止并在主循环处理，以避免v1的信号等待死锁。已使用/tmp内不涉及GPU或训练的模拟worker验证两种TERM方式：向整进程组发送TERM、仅向父队列发送TERM，均退出码143、状态stopped且worker被回收；测试通过。

运行时间预估：

- 历史八字段ChiMerge实验单纯训练线性估计：GPU 0约68小时、GPU 1约53小时；本轮小数据初速接近历史值。修复时间网格后部分分箱数增加，大数据集实际速度尚未测得，需为编码和评估留出余量。
- GPU 0全队列约65–90小时，GPU 1约50–75小时，整体约65–90小时（约2.7–3.8天）；以本次启动时间计算，整体预计北京时间 2026-11-08 12:51 至 2026-11-09 13:51 完成。主要耗时为MIRAGE/CESNET3，此为估计，最终以日志、队列和任务完成校验为准。

启动命令（已执行，不要重复启动）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'set -C; exec /data/xiaozj/anaconda3/envs/exp/bin/python -u log/formal_20261105_v2/run.py lane 0 > log/formal_20261105_v2/gpu0.queue.log 2>&1'
setsid -f bash -c 'set -C; exec /data/xiaozj/anaconda3/envs/exp/bin/python -u log/formal_20261105_v2/run.py lane 1 > log/formal_20261105_v2/gpu1.queue.log 2>&1'
```

监测命令：

```bash
cd /home/xiaozj/cgan
cat log/formal_20261105_v2/gpu0.status.json
cat log/formal_20261105_v2/gpu1.status.json
tail -f log/formal_20261105_v2/gpu0.queue.log log/formal_20261105_v2/gpu1.queue.log
# 根据队列active字段切换到当前数据集日志；下面是初始两个任务。
tail -f log/utmn_20261105_v2.log log/utmn-train_20261105_v2.log
cat log/formal_20261105_v2/utmn.status.json
ps -p 43650,43651 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
```

停止指令（先核实PID/PGID与命令仍属于本轮队列；停止队列会终止当前worker及后续任务）：

```bash
ps -p 43650,43651 -o pid,pgid,etime,args
# 仅停止GPU 0队列：
kill -TERM -- -43650
# 仅停止GPU 1队列：
kill -TERM -- -43651
# 或同时停止两条队列：
kill -TERM -- -43650 -43651
ps -p 43650,43651 -o pid,pgid,stat,args
```

- TERM会将队列与worker状态记为stopped；若被强制KILL或主机异常，状态可能停留running，需结合实际进程与日志判断。本节只证明启动成功，尚不代表十组正式实验完成。未commit或push。


## 2026-11-09 beta=0.01 新正式训练：仅GPU 1，先*-train再完整数据

- 用户已将主代码beta改为0.01，本次准备阶段通过AST核实 `transformer_train()` 中beta=0.01，并保存当前源码快照；仍在state-for-codex，未切换分支或修改用户代码。
- 仅使用物理GPU 1。严格顺序为 `utmn-train` → `ssrc_train` → `iscx_train` → `mirage-train` → `cesnet3-train` → `utmn` → `ssrc` → `iscx` → `mirage` → `cesnet3`。每项完整完成分箱、训练、checkpoint选模、生成及校验后才进入下一项。
- 用户称“测试集（*-train）”，执行以括号中指定的*-train目录为准，不改用*-test。SSRC/ISCX对应目录的实际命名为ssrc_train/iscx_train，输入与日志保留此命名，未重命名或修改任何源数据。
- 配置：beta=0.01、epoch=1000、batch_size=64、checkpoint=100、max_seq_len=16、expand_times=0。保留八字段、修复后的ChiMerge和前缀BOS对齐，以及用户明确保留的元字段统计、生成/保存设计。全局config保持原样，任务独立配置明确设置expand_times=0。
- 当前已存在旧 `iscx_0_01` 对比输出，为保证不覆盖，新建日期隔离层 `formal_20261109_v1`，各数据集下一级目录均使用 `[实际数据集名]_0_01`。路径为：
  - `bins/formal_20261109_v1/[数据集]_0_01/bins_[数据集]_0_01.json`
  - `model/formal_20261109_v1/[数据集]_0_01/generator.pth` 及100至1000共10个编号checkpoint
  - `result/formal_20261109_v1/[数据集]_0_01/[类别].json`
- 例如完整ISCX输出是 `result/formal_20261109_v1/iscx_0_01/`，既有根级 `result/iscx_0_01/` 保留。本轮没有删除或覆盖任何旧bins/model/result；包装脚本已移除清理逻辑，输出目录使用mkdir(exist_ok=False)，若存在则直接失败，不续训。
- 中间stdout/stderr无缓冲合并写入 `log/[实际数据集名]_20261109_v1.log`，例如utmn-train_20261109_v1.log、ssrc_train_20261109_v1.log。新日志全部确认不存在，创建采用exclusive模式，拒绝重复启动覆盖。
- `log/formal_20261109_v1/` 保存run.py、manifest.json、各任务config、source/代码快照及状态；input/中通过数据集别名符号链接只读访问原始data-json目录。准备时完整解析十份数据的全部类别JSON，核对非空series、使用字段及有限值，保存输入和14个源码文件的SHA-256；快照和包装脚本语法编译通过，各worker启动时再次校验输入哈希。
- 快照beta保持用户设定0.01，后续工作区修改不会影响排队任务的代码。每个任务创建新的空模型目录，日志必须从epoch 0开始。模型不使用已有权重或结果。

| 输入数据集／执行顺序 | 输出目录名 | 输入／预期生成flow | 上轮同规模实际总耗时 |
|---|---|---:|---:|
| utmn-train | utmn-train_0_01 | 4,047 | 0.54小时 |
| ssrc_train | ssrc_train_0_01 | 5,942 | 0.49小时 |
| iscx_train | iscx_train_0_01 | 5,475 | 0.53小时 |
| mirage-train | mirage-train_0_01 | 27,180 | 7.72小时 |
| cesnet3-train | cesnet3-train_0_01 | 167,999 | 47.17小时 |
| utmn | utmn_0_01 | 5,057 | 0.70小时 |
| ssrc | ssrc_0_01 | 7,426 | 0.68小时 |
| iscx | iscx_0_01 | 6,781 | 0.66小时 |
| mirage | mirage_0_01 | 33,975 | 10.00小时 |
| cesnet3 | cesnet3_0_01 | 209,999 | 59.07小时 |

- 本轮无expand，预期总生成473881 flow。完成阶段核对1000条有限损失日志、10个编号checkpoint、结果类别、JSON解析、计数、长度和包字段有限值，通过后才写completed。一个任务失败就停止整条队列，避免跳过失败而继续。
- checkpoint选模仍沿用同任务数据目录的OT评估，未新增独立测试集评估。本轮不使用之前生成结果作为输入。

启动与资源检查：

- 启动前确认上一轮beta=0.05两队列均completed，分别结束于11月8日04:18:43（GPU1）和18:58:12（GPU0）。GPU1空闲；GPU0另有PID76138在运行，本轮未停止、修改或占用GPU0。
- 新队列启动于 2026-11-09T20:16:26+08:00；队列PID/PGID/SID=80372，初始worker PID=80373。队列PPID=1，已确认后台独立session持续运行；worker随任务变化，队列PGID保持不变。
- 初始UTMN-train已完成分箱并从epoch 0开始，设备cuda、beta=0.01；NLL/KL/beta*KL为有限值，稳定每轮约1.76秒。物理GPU1 worker=80373，显存约992MiB；GPU0原进程仍在运行。启动前磁盘剩余约61GiB。

运行时间预估：

- 上轮相同十项实际总耗时相加约127.6小时（前五项约56.5小时、后五项约71.1小时）；上轮两卡并行，本轮仅一张GPU顺序执行，因此需要累计两部分耗时。
- 本轮前五个*-train任务预计约55–65小时，后五个完整数据约65–80小时，总墙钟耗时预计120–145小时，约5–6天。按启动时间估计，整体大致于北京时间 2026-11-14 20:16 至 2026-11-15 21:16 完成。beta变化、CPU竞争和大数据集评估都会影响耗时，以实际日志及状态为准。

启动命令（已执行，不要重复启动）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/formal_20261109_v1/run.py lane 1 > log/formal_20261109_v1/gpu1.queue.log 2>&1'
```

监测命令：

```bash
cd /home/xiaozj/cgan
cat log/formal_20261109_v1/gpu1.status.json
tail -f log/formal_20261109_v1/gpu1.queue.log
# 按状态中的active字段选择当前日志；启动时是UTMN-train。
tail -f log/utmn-train_20261109_v1.log
cat log/formal_20261109_v1/utmn-train.status.json
ps -p 80372 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
```

停止命令（先核对PID/PGID与命令仍属于本轮队列；该操作同时停止当前worker和所有后续排队任务）：

```bash
ps -p 80372 -o pid,pgid,etime,args
kill -TERM -- -80372
ps -p 80372 -o pid,pgid,stat,args
```

- TERM处理沿用已验证的v2主循环停止机制，避免在信号处理函数内阻塞wait；强制KILL/主机异常可能留下running状态，须结合实际进程判断。本节仅记录启动成功，十项任务尚未完成。未commit或push。


## 2026-11-09 顺序更正并清理重启：*-train与完整数据交替运行

- 用户更正顺序并明确要求停止原任务、删除其产生文件、重新启动。上一节“先五个*-train、后五个完整数据”的顺序已作废，旧PID80372/worker80373及旧停止指令已失效；其余参数和存储要求保持不变。
- 已核实原队列PGID80372，发送TERM后队列状态stopped，ps确认父队列和worker均不存在，nvidia-smi确认GPU1计算进程退出。原任务最后记录UTMN-train epoch 104/1000，没有任何数据集完成；旧worker未留存运行。
- 确认停止后，删除原任务新增的 `bins/formal_20261109_v1/`、`model/formal_20261109_v1/`、`result/formal_20261109_v1/`、`log/formal_20261109_v1/` 及清单中已创建的 `log/utmn-train_20261109_v1.log`。其中包括旧分箱、generator.pth、generator_100.pth、空结果目录、旧状态、代码快照、配置与队列日志；未删除或覆盖其他历史实验和原始数据。
- 随后在相同路径创建全新的正确顺序运行，重新生成代码/配置/输入哈希清单，未加载已删除的checkpoint。旧任务元信息仅以清理说明写入新的restart_info.json和本节历史记录，不保留旧实验产物。
- 仅物理GPU1，新顺序严格为：`utmn-train` → `utmn` → `ssrc_train` → `ssrc` → `iscx_train` → `iscx` → `mirage-train` → `mirage` → `cesnet3-train` → `cesnet3`。SSRC/ISCX训练目录沿用磁盘实际下划线命名。每个任务完整分箱、训练、选模、生成和校验后才进入下一项。
- beta=0.01、1000 epoch、batch_size=64、checkpoint=100、max_seq_len=16、expand_times=0；输出仍为bins/model/result各自 `formal_20261109_v1/[数据集]_0_01/`，bins文件为 `bins_[数据集]_0_01.json`，日志 `log/[实际数据集名]_20261109_v1.log`。新运行拒绝已有目标目录和日志，不覆盖其他既有beta结果。
- 重新准备时解析并核验十个数据集，保存14个源文件快照及SHA-256；脚本和源码语法编译通过。已检查manifest、队列脚本的顺序完全一致，全部gpu=1、beta=0.01，目标输出路径在启动前均不存在。
- 新启动时间：2026-11-09T20:20:40+08:00。新队列PID/PGID/SID=81124，初始worker PID=81125；已脱离终端，父队列PPID=1。GPU1仅新worker在运行，显存约992MiB；GPU0原进程76138未受影响。
- 新UTMN-train日志明确从epoch 0开始，已记录Epoch 1及后续训练，NLL/KL/beta*KL有限，每轮约1.76秒；这证明全新启动，不是从旧epoch104或checkpoint100续训。
- 总任务量不变，预计总耗时仍约120–145小时（5–6天），基于本次重新启动时间，整体预计北京时间 2026-11-14 20:20 至 2026-11-15 21:20 完成。顺序改变只改变各数据集的开始/完成先后，最终耗时仍以实际日志为准。

重新启动命令（已执行，不要重复启动）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/formal_20261109_v1/run.py lane 1 > log/formal_20261109_v1/gpu1.queue.log 2>&1'
```

当前有效监测命令：

```bash
cd /home/xiaozj/cgan
cat log/formal_20261109_v1/gpu1.status.json
tail -f log/formal_20261109_v1/gpu1.queue.log
# 根据active字段切换到当前任务日志；初始为UTMN-train。
tail -f log/utmn-train_20261109_v1.log
ps -p 81124 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
```

当前有效停止指令（先核对PID/PGID与命令；停止当前任务及后续队列）：

```bash
ps -p 81124 -o pid,pgid,etime,args
kill -TERM -- -81124
ps -p 81124 -o pid,pgid,stat,args
```

- 保留EXPERIMENT历史并以本节纠正原顺序与PID；未修改用户模型代码或全局配置，未commit或push。


## 2026-11-10 停止并保留beta=0.01队列，启动beta=0.001正式串行实验

- 用户已将主代码beta改为0.001，并要求停止当前任务、保留旧文件，在GPU1开启全新串行训练和生成。已通过AST核实用户赋值，未修改主代码或全局config，仍在state-for-codex。
- 已核实旧队列PID/PGID81124、当前MIRAGE worker82716，于13:57:06发送TERM并正常退出；旧队列状态stopped，ps和GPU检查确认两进程均不存在。原 `log/bins/model/result` 下formal_20261109_v1及所有独立日志和历史实验文件均保留，未清理。
- 停止前旧队列完成7项：utmn-train、utmn、ssrc_train、ssrc、iscx_train、iscx、mirage-train；完整mirage最后记录661/1000，cesnet3-train和cesnet3尚未开始。旧任务保持原状态和产物，不用于新队列的续训。
- 新运行隔离名formal_20261110_v1，beta=0.001，仅物理GPU1。顺序不变：utmn-train → utmn → ssrc_train → ssrc → iscx_train → iscx → mirage-train → mirage → cesnet3-train → cesnet3；每项完整分箱、训练、checkpoint评估选模、生成和校验后才运行下一项。
- 参数不变：1000 epoch、batch_size=64、checkpoint=100、max_seq_len=16、expand_times=0，八字段、已修复ChiMerge及前缀BOS对齐，保留用户指定的元字段统计和生成/保存设计。
- 新结果为 `bins/formal_20261110_v1/[实际数据集名]_0_001/bins_[实际数据集名]_0_001.json`、`model/formal_20261110_v1/[实际数据集名]_0_001/`、`result/formal_20261110_v1/[实际数据集名]_0_001/`。对应旧正式实验及历史beta=0.001结果不受影响。新任务拒绝已有输出目录，不删除旧文件，不使用旧模型或生成结果。
- 中间日志为 `log/[实际数据集名]_20261110_v1.log`；队列管理、状态和配置/代码快照位于 `log/formal_20261110_v1/`。日志创建使用exclusive模式防止覆盖。源码快照14个Python文件、beta=0.001、输入/源码SHA-256及10份配置均已校验，语法编译通过。
- 全部源数据保持只读，通过新运行目录下的input符号链接访问；本轮预期无expand生成总计473881 flow。各任务完成时检查1000条有限损失日志、10个编号checkpoint、生成类别/计数、JSON可解析性及包字段有限值，成功才写completed。checkpoint评估沿用同任务数据目录OT口径，未新增独立测试集。
- 新队列启动于 2026-11-10T13:57:58+08:00；PID/PGID/SID=89189，初始UTMN-train worker=89190。已确认旧进程退出，新worker在物理GPU1运行，显存约992MiB；初始日志明确从epoch 0开始，已到Epoch 11，NLL/KL/beta*KL均有限，每轮约1.73–1.74秒。启动前磁盘剩余约54GiB。

每个数据集预期耗时（含分箱、1000轮训练、10个checkpoint选模及生成，不含等待队列前序任务）：

| 顺序 | 数据集 | 预计耗时 |
|---|---|---:|
| 1 | utmn-train | 30–40分钟 |
| 2 | utmn | 40–50分钟 |
| 3 | ssrc_train | 25–35分钟 |
| 4 | ssrc | 30–45分钟 |
| 5 | iscx_train | 28–38分钟 |
| 6 | iscx | 35–45分钟 |
| 7 | mirage-train | 7–9小时 |
| 8 | mirage | 9–12小时 |
| 9 | cesnet3-train | 44–53小时 |
| 10 | cesnet3 | 55–65小时 |

- 估计依据：刚停止的GPU1队列实际总耗时约为utmn-train33.1分钟、utmn43.1分钟、ssrc_train29.6分钟、ssrc36.4分钟、iscx_train30.9分钟、iscx38.8分钟、mirage-train7.65小时；完整mirage近期每轮约35.49秒，1000轮纯训练约9.86小时。CESNET3两项参考11月5日正式v2已完成的约47.17/59.07小时。beta变化、系统负载和OT评估有波动，保留估计范围。
- 总墙钟时间预计约120–145小时（约5–6天），大致北京时间 2026-11-15 13:57 至 2026-11-16 14:57 完成。当前仅证明启动，不能视作任务已完成。

启动命令（已执行）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/formal_20261110_v1/run.py lane 1 > log/formal_20261110_v1/gpu1.queue.log 2>&1'
```

当前监测命令：

```bash
cd /home/xiaozj/cgan
cat log/formal_20261110_v1/gpu1.status.json
tail -f log/formal_20261110_v1/gpu1.queue.log
# 按active字段切换到当前数据集日志；启动时为UTMN-train。
tail -f log/utmn-train_20261110_v1.log
ps -p 89189 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
```

当前停止指令（先确认PID/PGID和命令仍属于本轮队列；停止当前worker及后续排队任务）：

```bash
ps -p 89189 -o pid,pgid,etime,args
kill -TERM -- -89189
ps -p 89189 -o pid,pgid,stat,args
```

- 原队列停止指令不再用于本轮任务，旧产物均保留。未commit或push。


## 2026-11-13 保存检查点并暂停beta=0.001队列，释放GPU1与进程内存

- 用户要求检查进度、保存当前检查点以便下次继续，并暂停释放显存及内存。本次未修改主模型、训练代码、配置或原始数据。
- 暂停前队列PID/PGID89189，当前worker150808，物理GPU1上运行完整cesnet3；前9项均已完成。已重新解析前9项全部结果JSON，逐类生成数与输入一致，且每项10个编号checkpoint均存在，不仅依赖状态文件。

| 数据集 | 进度 | 已生成flow |
|---|---|---:|
| utmn-train | 1000/1000，生成已校验 | 4,047 |
| utmn | 1000/1000，生成已校验 | 5,057 |
| ssrc_train | 1000/1000，生成已校验 | 5,942 |
| ssrc | 1000/1000，生成已校验 | 7,426 |
| iscx_train | 1000/1000，生成已校验 | 5,475 |
| iscx | 1000/1000，生成已校验 | 6,781 |
| mirage-train | 1000/1000，生成已校验 | 27,180 |
| mirage | 1000/1000，生成已校验 | 33,975 |
| cesnet3-train | 1000/1000，生成已校验 | 167,999 |
| cesnet3 | 已完成49/1000，暂停 | 尚未选模／生成 |

- 保存步骤：先核对PID/PGID后向本轮进程组发送SIGSTOP，冻结权重写入；冻结时日志已打印第49轮，generator.pth写入时间晚于该轮日志。将其复制为独立暂停文件，通过CPU `torch.load(weights_only=True)`、178个tensor的有限值检查及当前快照模型的strict load_state_dict形状检查；文件大小102,695,179字节，SHA-256与冻结的generator.pth一致。原generator.pth保留。
- 独立暂停检查点：`model/formal_20261110_v1/cesnet3_0_001/paused_epoch_49_20261113.pth`；最近完整轮次49，下次从第50轮继续到1000，未保存的第50轮部分batch工作不计入完成进度。
- 暂停元信息：`log/formal_20261110_v1/pause_20261113.json`，保存checkpoint/bins/源码哈希、配置、类别映射、完成队列及恢复轮次。输入文件哈希也已与本轮manifest核对。
- 核验检查点成功后，对冻结的原进程组发送TERM再CONT，使已挂起信号正常处理并退出；ps确认队列89189与worker150808均不存在。GPU1显存由约1123MiB降至15MiB，无GPU1计算进程，相关训练进程RAM已随退出释放。GPU0另有PID150640占用约8340MiB，未干预该任务。
- 队列gpu1.status.json和cesnet3.status.json已明确记录paused、checkpoint、第49轮及next_epoch=50，日志追加PAUSED记录；暂停时刻为2026-11-13T12:25:37+08:00。不再将旧进程PID视为活跃任务，所有旧产物保留。
- 重要恢复限制：原训练代码只保存模型state_dict，没有保存Adam优化器状态或随机数状态，无法事后补回；本检查点支持从第49轮权重继续训练，但Adam/RNG将重新初始化，不是逐步数值完全相同的恢复。
- 已准备 `log/formal_20261110_v1/resume_cesnet3_from_pause.py`，仅对剩余cesnet3运行；恢复前校验权重、bins、输入和代码快照哈希，固定已记录类别映射，跳过ChiMerge，并强制从该暂停文件/epoch49加载。不会重训前9项、重建bins、加载其他dataset权重或从epoch0开始。完成剩余951轮后照原规则评估100至1000共10个checkpoint并生成结果、校验后更新整队列completed。
- 续训stdout/stderr由脚本exclusive创建 `log/cesnet3_20261113_resume_v1.log`，不覆盖原日志；最终检查原日志1–49轮与续训日志50–1000轮共同完整覆盖1000轮。脚本语法检查通过；当前仅准备，未执行任何续训，任务仍paused。

当前暂停状态与检查点检查命令：

```bash
cd /home/xiaozj/cgan
cat log/formal_20261110_v1/gpu1.status.json
cat log/formal_20261110_v1/pause_20261113.json
ls -lh model/formal_20261110_v1/cesnet3_0_001/paused_epoch_49_20261113.pth
nvidia-smi
```

下次明确要求恢复时使用以下专用命令（本次未执行；不要重跑原lane/case命令，因为原入口拒绝已有状态/输出）：

```bash
cd /home/xiaozj/cgan
setsid -f env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/formal_20261110_v1/resume_cesnet3_from_pause.py
```

恢复后的监测：

```bash
cat log/formal_20261110_v1/gpu1.status.json
tail -f log/cesnet3_20261113_resume_v1.log
```

- 当前不保留占用GPU/RAM的STOP进程，也不进行后续训练；保持state-for-codex，未commit或push。


## 2026-11-13 CESNET3训练速度优化：实施、等价验证与候选淘汰

- 用户授权在GPU1实施并验证上轮方案，不进行完整实验，达不到预期的方法不保留。当前仍在state-for-codex，GPU0有其他任务，本次仅GPU1运行短程验证。
- 修改前已读取AGENT及相关调用链，将当前dataset.py、model_train.py、transformer.py（含用户现有beta=0.001修改）保存到 `log/training_optimization_20261113_v1/baseline/`，记录SHA-256。未回滚用户已有beta或visual.ipynb修改。

保留的实现：

- `stateFormer/dataset.py` 新增 `preencode=False` 兼容开关。启用时一次性编码序列/one-hot标签/长度/类别权重，后续getitem仅索引CPU缓存；编码保留原文件/样本顺序、前16包截断、真实ID1..N、特殊0、padding和float32/long类型。
- 分箱查询每个值只round一次，且预编码期间按属性/舍入值复用首次查找结果。仍逐区间first-match，不替换为假设区间无重叠的二分查找；边界共享、上下界裁剪和类别逆频权重均保留。
- 预编码后释放原始JSON列表、标签列表和临时查找字典；CESNET3缓存为234,358,884字节（约223.5MiB），构建耗时10.44秒。
- `stateFormer/model_train.py` 默认通过 `model_paras.get('preencode', True)` 启用缓存，输出编码准备耗时；可设preencode=false关闭。Dataset本身默认仍为lazy，避免改变其他预处理调用的行为。
- 训练函数transformer_train经AST确认与修改前完全相同：NLL/KL定义、beta、优化器、batch_size、精度、checkpoint保存频率均不变；transformer.py经逐字节确认与本次开始时相同。

验证（限定规模，无正式实验）：

1. `tests/test_dataset_cache.py` 3项CPU回归通过：原/新序列等价、共享边界first-match、舍入/越界、padding/截断、one-hot和类别权重、负索引、shuffle、RNG不变及缓存释放。
2. 全量CESNET3共209,999样本逐项对照修改前原实现，seq/label/length/weight的值和dtype全部完全一致，预编码不改变CPU随机状态，固定shuffle种子下首batch完全一致。原实现全量访问加比较耗时174.88秒；缓存纯访问0.508秒，两者工作量不同，不直接作为训练加速比。
3. GPU1两步严格对照中，预测、mu/logvar/state/noise、NLL/KL/总损失、各参数梯度、Adam状态及更新后权重逐位一致。首次测试脚本仅复位CUDA RNG，漏掉模型在CPU生成的noise随机状态而断言失败；补齐CPU+CUDA RNG复位后通过，并非产品数值差异。
4. 固定CPU/CUDA及Python随机状态，实际get_gen_data生成3条不同长度flow，反分箱后的JSON完全相同；生成逻辑没有修改。
5. 直接调用修改前/后的真实transformer_train，各对同1024条真实样本运行16个batch、仅1个限定子集轮次（start_epoch=49、epochs=50），临时保存的全部权重逐位相同；日志两者NLL/KL/总损失一致，临时输出自动清除，不写正式模型目录。

速度对比（同一1024条真实样本、batch64，4batch预热，3次轮换测试顺序）：

| 方法 | 每16个batch耗时中位数 | 决策 |
|---|---:|---|
| 原始逐样本编码 | 0.999949秒 | 基线 |
| 预编码缓存 | 0.370691秒 | 保留，约2.70倍加速 |
| 缓存+pin_memory+异步传输 | 0.368117秒 | 仅比缓存约0.69%改善，不保留 |

- 按类别分组计算输入头仅在验证脚本的临时模型中尝试，虽保留空head梯度连接，mu最大差异仍约6.4969063e-06；不符合本轮严格逐位一致条件，因此弃用候选，正式transformer.py无此修改。这不等于证明微小数值差异一定降低质量，但本轮选择更保守的等价方案。
- checkpoint写盘实测约0.060–0.061秒/次，远小于训练时间，因此不引入异步写盘，不改变每轮generator.pth及每100轮编号保存策略。
- 所有未保留候选只存在于已结束的验证进程/脚本，正式训练路径没有pin_memory、异步传输、类别分组或异步保存改动。训练精度、模型结构、batch_size、训练轮数和原bins均不减少。
- 16batch速度外推完整CESNET3约76.0秒/轮，较既有约207秒/轮显著下降；这不是完整epoch实测，不作为正式完成时间承诺。若速度可持续，剩余951轮纯训练约20.1小时，另计选模和生成。

检查点与后续恢复：

- 原 `log/formal_20261110_v1/source/` 冻结快照、原bins、原始CESNET3输入及第49轮暂停检查点SHA-256均重新校验一致。pause_20261113.json仍为paused，未执行正式续训。
- 新优化续训快照位于 `log/training_optimization_20261113_v1/resume_source/`，仅覆盖dataset.py和model_train.py，其他模块从暂停任务原快照复制。源码哈希见optimized_resume_manifest.json；不修改旧快照以绕过检查。
- 新入口 `log/formal_20261110_v1/resume_cesnet3_optimized.py` 已语法检查但未执行，使用已验证缓存版本；仍从原第49轮权重开始第50轮、保留bins和类别映射，完成后才选模生成。原暂停任务未保存Adam/RNG，此已有恢复限制仍存在。
- 将来明确要求恢复时，推荐使用下列新入口；本次未执行：

```bash
cd /home/xiaozj/cgan
setsid -f env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/formal_20261110_v1/resume_cesnet3_optimized.py
# 恢复后新日志，不覆盖原日志：
tail -f log/cesnet3_20261113_optimized_resume_v1.log
```

- 测试进程全部结束后，GPU1显存15MiB、无GPU1计算进程；GPU0原任务未干预。正式队列仍paused，没有完整epoch/CESNET3完整实验或新正式生成产物。
- 验证结果和日志集中在 `log/training_optimization_20261113_v1/`：summary.json、dataset_summary.json、gpu_summary.json、training_entry_summary.json及各validation日志。3项单元测试、源文件/入口语法检查通过。dataset/test文件diff检查通过；model_train.py的全局diff检查仍提示用户原有beta=0.001行尾空格，该行本次未修改。
- 保留用户其他修改，未commit或push。


## 2026-11-13 beta=0.01 预编码加速新实验：CESNET3-train → CESNET3

- 用户已将主代码beta修改为0.01，本次通过AST核实并保存当前代码快照；要求基于刚验证的加速修改在GPU1依次对cesnet3-train、cesnet3从头训练和生成，不覆盖旧模型或中间文件。
- 顺序固定：`cesnet3-train` → `cesnet3`；每项分箱、训练、10个checkpoint选模、生成和最终校验全部完成后才进入下一项。仅物理GPU1；GPU0既有PID162927未停止或干预。
- 超参数：beta=0.01、epoch=1000、batch_size=64、checkpoint=100、max_seq_len=16、expand_times=0、preencode=True。八字段、ChiMerge、KL定义、精度、Adam和checkpoint写盘策略均沿用原正式训练；不使用已弃用的pinned异步传输、类别头分组或异步保存。
- 新隔离名 `cesnet_beta0_01_fast_20261113_v1`。输出分别为：
  - `bins/cesnet_beta0_01_fast_20261113_v1/cesnet3-train_0_01/bins_cesnet3-train_0_01.json`
  - `model/cesnet_beta0_01_fast_20261113_v1/cesnet3-train_0_01/`
  - `result/cesnet_beta0_01_fast_20261113_v1/cesnet3-train_0_01/`
  - 第二项使用同层级的 `cesnet3_0_01` 目录和 `bins_cesnet3_0_01.json`。
- 中间stdout/stderr日志为 `log/cesnet3-train_20261113_v1.log`、`log/cesnet3_20261113_v1.log`，无缓冲合并输出，exclusive创建防覆盖。队列日志为 `log/cesnet_beta0_01_fast_20261113_v1/gpu1.queue.log`。
- manifest/config/source/status保存于 `log/cesnet_beta0_01_fast_20261113_v1/`。准备阶段完整解析两份数据，核对字段、有限值和非空series，保存输入和14个源码文件SHA-256，语法编译及快照哈希检查通过。明确配置preencode=True，worker只导入本轮快照，不受随后工作区编辑影响。
- 输入是原始 `data-json/cesnet3-train`（21类167999 flow）、`data-json/cesnet3`（21类209999 flow），通过本轮input符号链接只读访问。无expand，预期各生成相同数量。没有复用旧bins、权重或生成结果，也没有恢复第49轮旧任务。
- 新目标路径在准备时均不存在，worker使用mkdir(exist_ok=False)，存在则失败；启动从epoch0，无旧checkpoint加载。旧formal_20261105_v2、formal_20261109_v1、formal_20261110_v1及beta对比产物、原日志和暂停检查点全部保留。旧beta=0.001任务仍paused，不修改原恢复入口。
- 启动前GPU1空闲，磁盘剩余约58GiB。启动于2026-11-13T16:19:14+08:00；队列PID/PGID/SID=166721，初始worker PID=166723。父队列PPID=1，独立session后台运行；GPU1 worker占用约1080MiB，GPU0原任务仍在运行。
- 首项日志已确认 `No numbered checkpoint found; starting from epoch 0.`、cuda、`Dataset ready: preencode=True, samples=167999, seconds=7.955`，缓存加速已生效。已记录Epoch 1/1000，最近每轮61.94秒。
- 完成时检查1000条有限损失日志、10个编号checkpoint、类别、JSON解析、生成数、长度和包字段有限值，通过才写completed。checkpoint评估沿用同任务数据目录OT口径，没有额外划分测试集。

运行时间预估（包含分箱、1000轮训练、checkpoint选模及生成；单项耗时不包括等待前序任务）：

| 数据集 | 预计耗时 | 依据 |
|---|---:|---|
| cesnet3-train | 17–20小时 | 加速小批次折算约61秒/轮，约16.9小时纯训练，另计约1小时选模及波动 |
| cesnet3 | 22–26小时 | 小批次折算约76秒/轮，约21.1小时纯训练，另计约1–2小时选模生成及波动 |

- 两项串行，整体约39–46小时。首项预计北京时间2026-11-14 09:19至2026-11-14 12:19完成；整体预计2026-11-15 07:19至2026-11-15 14:19完成。这是基于加速验证的外推，后续以实际epoch耗时和状态更新为准。

启动命令（已执行，不要重复启动）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/cesnet_beta0_01_fast_20261113_v1/run.py lane 1 > log/cesnet_beta0_01_fast_20261113_v1/gpu1.queue.log 2>&1'
```

监测命令：

```bash
cd /home/xiaozj/cgan
cat log/cesnet_beta0_01_fast_20261113_v1/gpu1.status.json
tail -f log/cesnet_beta0_01_fast_20261113_v1/gpu1.queue.log
tail -f log/cesnet3-train_20261113_v1.log
# 第二项开始后：
tail -f log/cesnet3_20261113_v1.log
ps -p 166721 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
```

停止指令（先核实PID/PGID和命令仍属于本轮任务；停止当前worker与后续队列）：

```bash
ps -p 166721 -o pid,pgid,etime,args
kill -TERM -- -166721
ps -p 166721 -o pid,pgid,stat,args
```

- 仍在state-for-codex，用户已有修改保留；未commit或push。本节记录启动成功，不代表完整实验已结束。

- 启动后补核：CESNET3-train已完成Epoch 1/1000，实测61.94秒，NLL/KL/总损失有限，与加速外推约61秒/轮接近；旧beta=0.001暂停状态及检查点哈希再次核实保持不变。


## 2026-11-15 beta=0.001 预编码加速 CESNET3 单数据集全新实验

- 用户已将主代码beta设为0.001，要求基于预编码加速修改，仅对完整cesnet3在GPU1从头训练1000 epoch及生成，不覆盖旧输出、不使用旧模型。准备阶段通过AST核实用户赋值并保存当前代码快照，未改用户代码或全局配置。
- 仅物理GPU1，队列只有cesnet3。参数：beta=0.001、epoch=1000、batch_size=64、checkpoint=100、max_seq_len=16、expand_times=0、preencode=True；八字段、修复后的ChiMerge、KL、Adam及每轮generator.pth/每100轮编号checkpoint策略均保持原先条件。
- 输入原始data-json/cesnet3，21类209999 flow。完整预检查类别JSON、字段有限值和非空series，保存输入哈希；通过新运行input符号链接只读访问源数据，不读取生成结果作为训练输入。
- 隔离名cesnet_beta0_001_fast_20261115_v1。全新输出：
  - `bins/cesnet_beta0_001_fast_20261115_v1/cesnet3_0_001/bins_cesnet3_0_001.json`
  - `model/cesnet_beta0_001_fast_20261115_v1/cesnet3_0_001/`
  - `result/cesnet_beta0_001_fast_20261115_v1/cesnet3_0_001/`
- 中间stdout/stderr无缓冲合并写 `log/cesnet3_20261115_v1.log`；队列管理日志 `log/cesnet_beta0_001_fast_20261115_v1/gpu1.queue.log`。同目录保存run.py、manifest.json、cesnet3.config.json、source/、gpu1.status.json及cesnet3.status.json。
- 新日志和输出目录在准备时确认不存在，日志exclusive创建、输出mkdir(exist_ok=False)，因此不会覆盖已有文件或续训。14个源码文件语法和哈希检查通过，source中明确包含preencode实现，配置强制preencode=True。
- 上一轮加速beta=0.01两项已全部completed：cesnet3-train约18.48小时，cesnet3约23.20小时（11月15日10:00结束），原结果/模型均保留；此前formal_20261110_v1 beta=0.001任务仍paused，第49轮检查点哈希核实未变。本次既不使用该暂停权重，也不运行其恢复脚本。
- 启动前GPU1空闲，GPU0 PID217598有其他计算任务，未干预。磁盘剩余约51GiB。启动于2026-11-15T11:53:49+08:00；队列PID/PGID/SID=220370，worker PID=220372，父队列PPID=1，独立session后台运行。已核实新worker只在GPU1，显存约1098MiB、利用率约96%。
- 首项日志确认 `No numbered checkpoint found; starting from epoch 0.`、cuda、`Dataset ready: preencode=True, samples=209999, seconds=9.972`；已完成前3轮，最近每轮约77.92秒，NLL/KL/总损失均为有限值。预编码加速已实际生效。
- 完成时检查1000条有限损失日志、100至1000的10个编号checkpoint、输出类别、JSON解析、无expand生成计数209999、长度及包字段有限值，成功才标记completed。checkpoint选模仍沿用同数据目录的OT评估。

运行时间估计：

- 上轮同CESNET3加速实测：训练约21.92小时，选模约72.9分钟，生成约4.1分钟，总计约23小时12分钟；本轮初始每轮约77–78秒，与该规模接近。
- 预计本次包含分箱、1000轮训练、10个checkpoint评估及生成共22–26小时。按启动时间估计，约北京时间2026-11-16 09:53至2026-11-16 13:53完成；实际受负载影响，以日志及校验状态为准。

启动命令（已执行，不要重复启动）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/cesnet_beta0_001_fast_20261115_v1/run.py lane 1 > log/cesnet_beta0_001_fast_20261115_v1/gpu1.queue.log 2>&1'
```

监测命令：

```bash
cd /home/xiaozj/cgan
tail -f log/cesnet3_20261115_v1.log
cat log/cesnet_beta0_001_fast_20261115_v1/gpu1.status.json
cat log/cesnet_beta0_001_fast_20261115_v1/cesnet3.status.json
ps -p 220370 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
```

停止指令（先核对PID/PGID与命令仍属于本轮任务）：

```bash
ps -p 220370 -o pid,pgid,etime,args
kill -TERM -- -220370
ps -p 220370 -o pid,pgid,stat,args
```

- 本节记录从头启动，尚未完成1000 epoch、选模或生成；仍在state-for-codex，未commit或push。


## 2026-11-16 beta=0.01 测试子集训练与25份合成数据集

- 用户已将beta改为0.01；本次通过AST核实，使用当前预编码加速代码。输入明确为iscx_test、ssrc_test、mirage-test、utmn-test、cesnet3-test，不使用完整集、train子集或已有生成结果替代。
- GPU0顺序：iscx_test → ssrc_test → mirage-test → utmn-test；GPU1单独运行cesnet3-test，两条队列独立后台并行。每个输入仅训练一次1000 epoch并选出同一最佳checkpoint，再依次独立调用生成函数expand_times=0、1、2、3、4。总计5次训练、25份合成数据集，不是25次训练。
- 保持batch_size=64、checkpoint=100、max_seq_len=16、preencode=True，八字段、修复后的ChiMerge、KL/Adam/精度等不变。每个输入从epoch0开始，使用全新模型目录，不加载旧权重。
- expand_times的语义沿用原代码：0至4分别生成输入数量的1至5倍；每次重新采样生成，五份结果互相独立，不将小集合复制或拼接成大集合。每个输入五份总量为15倍；全部预期总生成790590 flow、25个目录集、385个类别JSON文件。
- 新隔离名test_beta0_01_20261116_v1，目标在准备时均不存在。分箱 `bins/test_beta0_01_20261116_v1/[数据集]_0_01/bins_[数据集]_0_01.json`，模型 `model/test_beta0_01_20261116_v1/[数据集]_0_01/`。
- 合成结果为 `result/test_beta0_01_20261116_v1/expand_0/[数据集]_0_01/` 至 `expand_4/[数据集]_0_01/`；各目录含该输入的全部类别JSON。模型和bins只存一份，五个expand共用选出的checkpoint。
- 日志 `log/[数据集]_20261116_v1.log`，两条队列管理日志位于 `log/test_beta0_01_20261116_v1/gpu0.queue.log` 与gpu1.queue.log。各任务配置、输入/源码哈希、冻结source/及status位于同运行目录。生成期间状态记录active_expand_times及synthetic_sets，每份结果有逐类计数、路径、耗时及validated标记；另写[数据集].synthetic_summary.json。
- 准备阶段解析全部原始JSON、检查字段与非空series，保存输入及14个代码快照的SHA-256，源文件/包装脚本语法检查通过。五输出包装逻辑用/tmp小规模fixture验证输出目录隔离及1至5倍计数，全部通过。该独立smoke首次遗漏绑定time模块，补齐测试namespace后通过；实际包装脚本本身已有time导入，运行任务未受影响。
- 每种expand完成后立即重新解析全部输出，校验类别、每类数量=输入*(expand+1)、长度1..16和包字段有限值。每个输入必须5份全部验证通过、1000条有限损失日志及10个编号checkpoint通过，才标记completed。出错即停止对应GPU队列，不跳过失败。
- 此任务按用户指定在test子集上训练；checkpoint选模仍按原流程使用同输入目录OT比较，未增加独立验证集或重新划分。所有旧模型、结果、日志、原始数据和暂停检查点保留，不覆盖。

| 输入数据集 | GPU | 输入flow | expand 0/1/2/3/4预期各自生成flow | 单项预计耗时 |
|---|---:|---:|---|---:|
| iscx_test | 0 | 1,417 | 1417 / 2834 / 4251 / 5668 / 7085 | 7–10分钟 |
| ssrc_test | 0 | 1,484 | 1484 / 2968 / 4452 / 5936 / 7420 | 7–10分钟 |
| mirage-test | 0 | 6,795 | 6795 / 13590 / 20385 / 27180 / 33975 | 40–60分钟 |
| utmn-test | 0 | 1,010 | 1010 / 2020 / 3030 / 4040 / 5050 | 7–10分钟 |
| cesnet3-test | 1 | 42,000 | 42000 / 84000 / 126000 / 168000 / 210000 | 4.5–5.5小时 |

启动验证与时间估计：

- 启动前两张GPU均空闲，前一轮CESNET3 beta=0.001已于11月16日11:08:34完成。磁盘约49GiB可用，预计本轮模型与25份结果可容纳，未清理其他文件。
- 启动时间2026-11-16T14:52:38+08:00；GPU0队列PID/PGID/SID=232246，初始worker=232248；GPU1队列PID/PGID/SID=232247，初始worker=232249。父队列PPID=1、独立session，已确认对应worker运行在物理GPU0/1。
- iscx_test初始实测每轮约0.400秒，cesnet3-test约15.32秒，两者均明确epoch0起步、preencode=True、NLL/KL/总loss有限。显存分别约840/1118MiB，GPU利用率约94%/95%。
- 上表单项耗时包含1000轮训练、每轮/间隔checkpoint写盘、10个checkpoint选模及五次生成和输出校验，不包含等待同GPU前序任务。小数据集估计需考虑每轮写盘固定成本；MIRAGE/UTMN依据batch数、类别头规模及已验证GPU吞吐推算。
- GPU0四项预计约61–90分钟，GPU1约4.5–5.5小时；两卡并行，总墙钟预计4.5–5.5小时，约北京时间2026-11-16 19:22至2026-11-16 20:22完成。后续以各数据集实际日志、生成进度与状态为准；本节只证明启动，不代表25份结果已完成。

启动命令（已执行，不要重复启动）：

```bash
cd /home/xiaozj/cgan
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=0 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/test_beta0_01_20261116_v1/run.py lane 0 > log/test_beta0_01_20261116_v1/gpu0.queue.log 2>&1'
setsid -f bash -c 'set -C; exec env CUDA_VISIBLE_DEVICES=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u log/test_beta0_01_20261116_v1/run.py lane 1 > log/test_beta0_01_20261116_v1/gpu1.queue.log 2>&1'
```

监测命令：

```bash
cd /home/xiaozj/cgan
cat log/test_beta0_01_20261116_v1/gpu0.status.json
cat log/test_beta0_01_20261116_v1/gpu1.status.json
tail -f log/test_beta0_01_20261116_v1/gpu0.queue.log log/test_beta0_01_20261116_v1/gpu1.queue.log
# 根据active切换当前任务日志：
tail -f log/iscx_test_20261116_v1.log log/cesnet3-test_20261116_v1.log
cat log/test_beta0_01_20261116_v1/iscx_test.status.json
ps -p 232246,232247 -o pid,ppid,pgid,sid,etime,stat,args
nvidia-smi
```

停止指令（先核实PID/PGID与命令仍属于本轮队列）：

```bash
ps -p 232246,232247 -o pid,pgid,etime,args
# 仅GPU0：
kill -TERM -- -232246
# 仅GPU1：
kill -TERM -- -232247
# 或两条均停止：
kill -TERM -- -232246 -232247
```

- 仍在state-for-codex，用户已有代码/笔记本修改保留，未commit或push。

## 2026-11-17 等宽/等频分箱接入与正式实验准备（未启动）

- 用户授权修改接入代码，beta保留0.01；指定iscx、ssrc、mirage、utmn、cesnet3，不使用对应训练集。正式实验不启动；测试仅使用GPU0。
- 当前分支bin-exp；保留用户已有AGENT.md修改。本轮未修改模型结构、model_train.py、loss、全局config.json或原始数据。
- driver.py新增--config和--binning-method，按binning.method选择chimerge/equal_width/equal_freq；缺省仍为ChiMerge。移除未使用的PCAP导入。新配置固定随机种子及排序后的类别映射，既有无seed配置沿用原类别枚举方式。
- baseline.py统一两方法的字段箱数配置，等频按累计计数计算线性分位点，无需展开重复样本；处理常量和舍入后的重复边界，空输入明确报错。保留原编码的共享边界首次匹配、1..N真实ID及现有反分箱语义。
- 正式配置仅准备在configs/binning_20261117_v1/，五输入×两方法共10份。八字段、max_seq_len=16、batch_size=64、epoch=1000、checkpoint=100、expand_times=1（2倍生成量）、preencode=true、seed=20261117；beta=0.01仍由未改动的训练代码提供。
- 两种方法共用配置范围和箱数上限：time100、pkt_len500、flags10、ttl60、src/dst_port各250、src/dst_ip各100。等宽固定配置范围；等频数据裁剪后计算分位点，重复值可能使实际箱数减少。逐包字段取前16包、meta每flow统计一次。
- bins/model/result分别隔离在各自binning_20261117_v1/方法/目录。fresh_start=true在任何分箱/训练前拒绝已有目标文件或目录，不自动续训。10项正式输出路径检查均未存在；未创建正式bins/model/result、后台队列或GPU分配。
- 语法解析、git diff --check、配置数量/输入目录/参数校验通过。100组等频计数结果与NumPy展开样本quantile对照通过，常量字段、等宽边界、空输入检查通过。
- 沙箱GPU不可见后，经宿主执行在物理GPU0（NVIDIA GeForce RTX 3080）运行validation/smoke_binning.py：五数据集×两方法共10项，每项2类×2条真实flow、batch4、仅1轮。训练、选模、生成、JSON有限值、生成计数/长度、真实ID/padding以及fresh-start重复输出拒绝均通过，进程正常退出。限定测试使用配置副本，正式epoch不变；临时输入、bins、checkpoint及生成结果自动清理。
- 本次没有正式训练结果，未启动任何正式实验。使用方式、分箱语义及限制见configs/binning_20261117_v1/README.md。

## 2026-11-17 双GPU等频/等宽正式基线（后台运行）

- 用户明确授权后台正式运行。物理GPU0=等频(equal_freq)，物理GPU1=等宽(equal_width)；两卡各自固定顺序 `iscx → ssrc → mirage → utmn → cesnet3`，只使用完整命名输入，不使用对应train子集。每项完整完成分箱→训练→10个checkpoint选模→生成→结果校验后才进入下一项；单项失败停止本GPU队列，另一队列独立继续。
- 新运行名 `binning_formal_20261117_v1`。独立配置、冻结源码、输入/源码/配置SHA-256和逐类flow数保存于 `log/binning_formal_20261117_v1/manifest.json`、`source/`、`[方法]_[数据集].config.json`。本轮配置从准备版复制，显式改为epoch=1000、expand_times=0（按输入逐类生成1倍flow）；batch_size=64、checkpoint=100、max_seq_len=16、preencode=true、seed=20261117，八字段，beta=0.01，模型结构及loss未变。全局config与先前准备版配置未修改。
- 逐文件解析、非空flow、字段有限值和输入哈希检查已通过：iscx 10类6781 flow、ssrc 9类7426、mirage 20类33975、utmn 17类5057、cesnet3 21类209999。worker在开始前再次核对输入、源码及配置哈希；不加载旧模型、不覆盖旧输出。
- 全新输出：`bins/binning_formal_20261117_v1/[方法]/[数据集]/bins_[数据集].json`；`model/binning_formal_20261117_v1/[方法]/[数据集]/`；`result/binning_formal_20261117_v1/[方法]/[数据集]/`。分箱方法/字段箱数及配置范围沿用本轮已验证接入。目标存在即报错；未复用11月3日实验配置或产物。
- 各任务stdout/stderr合并且无缓冲写入新日志 `log/binning_formal_20261117_v1/[方法]_[数据集].log`，exclusive创建防止覆盖；每项阶段、耗时、最佳epoch、校验计数及失败原因写入同名`.status.json`。两条队列日志 `gpu0.queue.log`、`gpu1.queue.log`，队列状态 `gpu0.status.json`、`gpu1.status.json`。
- 启动：北京时间2026-11-17 16:09:52。GPU0 lane PID/PGID/SID=262392，worker=262394；GPU1 lane PID/PGID/SID=262393，worker=262395。lane PPID=1、独立会话；worker与所属lane同PGID。nvidia-smi核实worker262394在GPU0（UUID GPU-1f72e708-dcfc-5400-3a85-e61d6272edc1，858MiB），worker262395在GPU1（UUID GPU-dd306551-104a-e6e1-34ec-1c79160bbe17，864MiB）。启动前两卡无计算进程，剩余磁盘约52GiB。
- 启动核实两项ISCX均从epoch0开始、preencode=true、设备cuda；已超过第5轮且loss/NLL/KL有限，初始稳定每轮约1.88–1.90秒（日志内计时不含随后checkpoint保存）。这是启动证据，不代表任何正式任务已完成。

### 耗时预估（两方法分别适用）

| 数据集 | GPU0等频单项耗时 | GPU1等宽单项耗时 | expand=0各自产生flow |
|---|---:|---:|---:|
| iscx | 30–45分钟 | 30–45分钟 | 6781 |
| ssrc | 35–50分钟 | 35–50分钟 | 7426 |
| mirage | 3–5小时 | 3–5小时 | 33975 |
| utmn | 30–45分钟 | 30–45分钟 | 5057 |
| cesnet3 | 20–27小时 | 20–27小时 | 209999 |

- 单项包含分箱、1000轮训练、每轮/间隔保存、10个checkpoint选模、生成及校验，不包含在本GPU队列中等待前序项的时间。每条队列合计约24.6–34.4小时；两卡并行总墙钟约25–35小时，预计北京时间2026-11-18 17:10至2026-11-19 03:10全部完成，非时间保证。
- 依据本轮ISCX初始实测、当前batch数/类别头规模外推，以及近期同八字段/预编码CESNET3实测总耗时23.20小时（训练21.92小时、选模72.9分钟、生成4.1分钟，见log/cesnet_beta0_01_fast_20261113_v1/cesnet3.status.json）。两种分箱的实际词表不同，两卡并行的CPU/磁盘竞争也会影响耗时；目前没有本轮完整数据集的实测时间，因此给两方法相同的宽区间，后续按日志重算。初始依据保存在initial_eta.json。未使用11月3日基线耗时。

### 监控指令

在 `/home/xiaozj/cgan` 下执行：

```bash
/data/xiaozj/anaconda3/envs/exp/bin/python log/binning_formal_20261117_v1/run.py status
cat log/binning_formal_20261117_v1/gpu0.status.json
cat log/binning_formal_20261117_v1/gpu1.status.json
tail -f log/binning_formal_20261117_v1/gpu0.queue.log log/binning_formal_20261117_v1/gpu1.queue.log
# 初始两任务；后续按队列status中的active替换日志名
tail -f log/binning_formal_20261117_v1/equal_freq_iscx.log log/binning_formal_20261117_v1/equal_width_iscx.log
nvidia-smi
```

全部完成以两份lane状态为completed、各自completed包含5项，并且10份case状态为completed为准。worker仅在完整1000条有限epoch日志、10个编号checkpoint、全部结果JSON可解析、逐类数量/每flow长度与输入匹配、字段有限值校验成功后写completed；不能只看GPU占用或checkpoint推断完成。

### 停止指令

分别停止对应整条队列（立即终止当前worker，不启动后续项，保留已写入的模型/分箱/结果/日志）：

```bash
# 停止GPU0等频队列
/data/xiaozj/anaconda3/envs/exp/bin/python log/binning_formal_20261117_v1/run.py stop 0
# 停止GPU1等宽队列
/data/xiaozj/anaconda3/envs/exp/bin/python log/binning_formal_20261117_v1/run.py stop 1
# 然后确认状态和GPU释放
/data/xiaozj/anaconda3/envs/exp/bin/python log/binning_formal_20261117_v1/run.py status
nvidia-smi
```

停止入口会核对PID的启动标识、完整脚本路径、lane参数及PGID，防止PID复用误停其他任务；向lane发送SIGTERM，由lane终止其worker，15秒未退出则强制结束worker。停止不会额外保存正在计算的半轮，现有checkpoint仅包含模型权重，不包含Adam/RNG。不要重复launch同一运行目录，也不要用泛化pkill停止其他实验。

## 2026-11-18 ChiMerge去除threshold_merge的CPU分箱对照（已完成，未训练）

- 用户要求检查去除threshold_merge的影响，只运行预处理，不训练、不使用GPU；现有GPU任务未停止或改动。
- pre_process/ChiMerge.py的cal_bins_one_category、cal_bins、chimerge新增use_threshold_merge=True参数，默认保持原算法；False直接将初始区间交给chi_merge。仅绕过低频预合并，alpha=0.01、df计算、整数网格、类别间整合及最终边界后处理保持原实现。
- 新增独立CPU脚本pre_process/compare_chimerge_threshold.py，无torch/driver/model导入，强制CUDA_VISIBLE_DEVICES=-1、BLAS/OMP/MKL单线程。输入为原始data-json/{iscx,ssrc,mirage,utmn,cesnet3}；八字段、前16包、当前config.json属性范围与min_unit，两方案在同输入重新拟合。未调用训练入口或生成模型。
- 新运行名chimerge_threshold_ablation_20261118_v1。10份完整bins位于bins/chimerge_threshold_ablation_20261118_v1/{original,no_threshold}/bins_[数据集].json；未覆盖旧bins。逐次日志、配置、脚本/源码快照、输入哈希、耗时、警告、summary位于log/chimerge_threshold_ablation_20261118_v1/。
- 结果：ISCX总箱数1627→3027（1.860倍）、SSRC687→976（1.421倍）、MIRAGE4801→6787（1.414倍）、UTMN1792→3647（2.035倍）、CESNET3 3344→4433（1.326倍）。时间字段依次245→751、480→754、565→2190、124→541、657→1681。并非所有字段都增加，例如MIRAGE目的IP1833→1571、UTMN TTL75→62；预合并会影响后续贪心合并路径，不能将新结果当成对旧箱的严格细分。
- 原版/去预合并分箱实际CPU耗时（秒）：ISCX2.27/5.57、SSRC0.96/2.69、MIRAGE5.90/22.99、UTMN0.35/0.72、CESNET3 5.57/5.87；仅为分箱函数时间，不含报告统计及其输入扫描。
- 验证：默认路径与修改前冻结实现的类别级及整合级fixture输出一致；禁用路径通过替换threshold_merge为报错函数验证未调用它。语法/diff检查通过。10份输出每份8字段，端点均有限；40项对照无倒置区间、无观测重叠、无边界兜底后不可编码内部空隙。原始范围外观测一致：ISCX time11次、SSRC time29次、UTMN pkt_len841次，沿用首末箱兜底。chi_cal既有零期望除法warning在两种方案均出现，本次未改变统计量或修复该警告。
- 统计包括逐字段箱数、单点箱、空箱、宽度中位数/P90/最大值、最大箱占比、范围外/内部空隙/重叠观测。meta占用每flow一次，序列字段按保留packet统计。报告log/chimerge_threshold_ablation_20261118_v1/REPORT.md，机器可读comparison.csv/comparison.json。status.json=completed，5数据集、40项字段对照；没有本轮model/result目录。
- 本次只比较离散化结果，尚不能据此判断模型生成质量或下游性能。

复现需选择新运行名，避免覆盖已保存结果（CPU-only）：

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u pre_process/compare_chimerge_threshold.py --run-id chimerge_threshold_ablation_20261118_v2
```

## 2026-11-18 GPU1等待当前队列后训练去threshold_merge基线（已后台排队）

- 用户授权：基于已完成的no_threshold分箱，对iscx、ssrc、mirage、utmn、cesnet3进行训练与生成，使用GPU1，等待当前任务结束，不干预当前任务，不覆盖旧产物。
- 新运行名chimerge_no_threshold_train_20261118_v1。顺序`iscx → ssrc → mirage → utmn → cesnet3`；每项1000 epoch、batch_size=64、checkpoint=100、max_seq_len=16、expand_times=0（逐类生成输入1倍flow）、preencode=true、seed=20261117、beta=0.01。只使用完整命名数据集，不使用train子集。未改模型或loss代码。
- 使用bins/chimerge_threshold_ablation_20261118_v1/no_threshold/bins_[数据集].json的精确副本，独立保存到bins/chimerge_no_threshold_train_20261118_v1/bins_[数据集].json；5份SHA-256逐一一致，总箱数依次3027、976、6787、3647、4433。worker直接调用训练、选模、生成函数，完全跳过重新分箱，避免driver重写bins。
- 训练模型写入model/chimerge_no_threshold_train_20261118_v1/[数据集]/，生成JSON写入result/chimerge_no_threshold_train_20261118_v1/[数据集]/；目标目录存在即失败，日志exclusive创建，不自动复用旧checkpoint。旧分箱、旧模型、旧结果、旧日志均保留。
- 新日志/状态/配置/清单/冻结源码统一在log/chimerge_no_threshold_train_20261118_v1/：gpu1.queue.log、gpu1.status.json、[数据集].log、[数据集].status.json、[数据集].config.json、manifest.json、source/。清单含输入/分箱/配置/源码哈希、类别计数及截断长度；输入与CPU分箱比较时哈希一致。新队列开始每项前再次验证这些哈希。
- 依赖是当前binning_formal_20261117_v1的GPU1整条等宽队列，原lane PID262393/start_ticks120945199。新队列每30秒检查：原队列身份未变、状态completed且5项completed齐全、5份case状态均completed、原lane对应进程已退出、nvidia-smi只读查询GPU1无计算进程。全部满足才开始训练；原队列failed/stopped或GPU1另有任务时继续等待。不会以epoch1000、checkpoint存在或GPU瞬时空闲替代完成判定。
- 等待进程在依赖放行前不导入torch/模型，不做CUDA可用性探测，不创建CUDA上下文。GPU0不属于本队列；原GPU0/1进程均不发信号、不改文件。单项训练/选模/生成/校验失败会停止新队列后续项。
- CPU测试覆盖：前序running/failed/stopped不放行且不调用GPU查询；前序completed但进程仍存活不放行；GPU1有计算进程不放行；仅GPU0活动时允许GPU1开始。配置、源码及分箱哈希、全新模型/结果/任务日志路径和语法检查均通过。本次没有为验证而提前在GPU上训练。
- 后台等待进程于北京时间2026-11-18 15:27:14启动，PID/PGID/SID=271940、PPID=1，独立会话。启动核实status=waiting、active=null、wait_reason=predecessor status=running；原GPU1仍为equal_width_cesnet3，worker266121。nvidia-smi仅显示原GPU0 worker266216和GPU1 worker266121，新等待进程没有GPU显存。本节只表示已排队，尚未训练新基线。
- 新队列只有在完整1000轮有限日志、10份编号checkpoint、全部类别结果JSON数量/长度/字段有限值校验通过后才将单项标为completed；现有checkpoint仍只保存模型权重。

监控新队列（在/home/xiaozj/cgan执行）：

```bash
/data/xiaozj/anaconda3/envs/exp/bin/python log/chimerge_no_threshold_train_20261118_v1/run.py status
cat log/chimerge_no_threshold_train_20261118_v1/gpu1.status.json
tail -f log/chimerge_no_threshold_train_20261118_v1/gpu1.queue.log
# 依赖放行且iscx开始后才会创建此任务日志
tail -f log/chimerge_no_threshold_train_20261118_v1/iscx.log
```

取消等待或停止本新队列（只核对并停止本运行的lane；不会停止前序等宽队列）：

```bash
/data/xiaozj/anaconda3/envs/exp/bin/python log/chimerge_no_threshold_train_20261118_v1/run.py stop 1
```

stop核对新lane脚本路径、参数、PID启动标识及PGID；等待期间只停止等待器，训练开始后还会停止其当前worker并禁止后续项，保留所有已写文件。不要重复launch此运行目录。

## 2026-11-18 恢复主代码原始ChiMerge

- 用户要求恢复原先ChiMerge方法。pre_process/ChiMerge.py已逐字节恢复到HEAD及此前原版冻结实现，移除use_threshold_merge开关，重新固定执行threshold_merge→chi_merge。
- 分箱对比脚本改为显式加载已保存的消融源码快照log/chimerge_threshold_ablation_20261118_v1/ChiMerge.py，保留历史对照可复现性，主ChiMerge不再承担消融开关。
- CPU检查确认主路径实际调用threshold_merge，独立冻结消融路径仍支持开关；语法和diff检查通过。没有重新分箱或训练。
- 未改动历史bins/模型/结果/日志或后台进程。已排队去threshold_merge训练仍按此前授权使用其独立冻结代码和分箱；本次恢复主文件不取消该队列。

## 2026-11-18 主模型移除KL损失贡献和独立noise分支（仅代码修改，未运行）

- 用户明确授权直接修改为NLL-only、无noise_head输入，输出只拼接state与属性条件；要求不运行实验。
- stateFormer/model_train.py删除beta、KL计算及KL日志字段，总损失直接为g_loss=nll_loss。原字段平均NLL、padding mask、安全target、类别权重、Adam及保存策略保持原实现。
- stateFormer/transformer.py删除noise_heads、noise维度、独立噪声采样及get_noise；forward和step_w_hidden仅拼接64维state和64维属性条件，输出层Linear输入由160维变为128维；sample的无前缀/有前缀两条路径均同步移除noise调用和参数。forward返回preds/mu/logvar/state，不再返回noise。
- 保留state_mu/state_logvar及训练时s_t=mu+eps_state*std，生成get_hidden仍使用mu；Transformer、长度/类别条件、字段自回归和真实ID/padding机制不变。
- 检查训练、checkpoint选择和最终生成均导入同一stateFormer.transformer.TransGenerator，其构造接口及sample接口未变，后处理无需修改。新结构的输出层宽度和参数集不同，原带noise checkpoint不能直接加载到此主实现；后续新实验需独立模型目录并从头训练。
- 仅执行AST语法/结构检查、调用参数检查和git diff --check：确认无noise分支残留、两条sample路径一致、唯一g_loss赋值为nll_loss、state重参数化保留。没有导入或执行模型，没有forward/backward、训练、采样、GPU操作或新实验。
- 用户已有AGENT.md和其他未提交修改保留。本轮不修改AGENT.md；其中原架构/noise/KL说明与本次主代码消融不同，以本条及当前代码为准。newTransformer.py、配置、分箱、历史结果均未修改。
- 两条既有运行binning_formal_20261117_v1及chimerge_no_threshold_train_20261118_v1的冻结源码SHA-256重新核对未变；本次未向后台进程发信号、未修改其快照/配置/分箱/日志，因此原实验仍使用原KL/noise实现。

## 2026-11-18 GPU0完整ChiMerge＋无KL/无独立noise模型（已后台排队）

- 用户授权在GPU0等待当前任务结束后，对iscx、ssrc、mirage、utmn、cesnet3运行完整ChiMerge及当前NLL-only/no-noise-head模型，不影响当前任务，不覆盖旧产物。
- 新运行名chimerge_no_kl_no_noise_20261118_v1；GPU0依次`iscx → ssrc → mirage → utmn → cesnet3`。沿用1000 epoch、batch_size=64、checkpoint=100、max_seq_len=16、expand_times=0、preencode=true、seed=20261117；使用完整命名数据集，不使用对应训练集子目录。总损失仅NLL，独立noise_heads/get_noise已移除，输出仅拼接64维state＋64维属性条件；state重参数化训练、生成使用mu保持当前实现。
- 当前完整ChiMerge和模型/训练/选模/生成源码均冻结在log/chimerge_no_kl_no_noise_20261118_v1/source/，按哈希检查；5份独立config、原始输入哈希/类别计数/截断长度、依赖身份及代码哈希见manifest.json。未使用GPU1去threshold_merge队列的模型或分箱。
- GPU0依赖为binning_formal_20261117_v1的原等频整条队列（lane PID262392、start_ticks120945199）：要求同一身份、队列及全部5份case均completed、5项completed列表齐全、lane进程已退出，以及nvidia-smi只读确认GPU0无计算进程。每30秒检查；running/failed/stopped、进程未退出或GPU0有其他任务均继续等待。GPU1任务或其等待器不属于本队列，不修改、不发信号。
- 后台等待器不导入torch/模型、不探测CUDA或分配显存。放行后，每项先使用冻结完整ChiMerge重新拟合（包含threshold_merge→chi_merge），再训练→10个checkpoint选模→生成→结果校验。完整bins还与此前CPU对照的original bins逐字段比对；reference_bins/为独立核对副本，并非直接用它替代本轮分箱。新bins在训练/生成前后核对哈希。
- 新输出：bins/chimerge_no_kl_no_noise_20261118_v1/bins_[数据集].json；model/chimerge_no_kl_no_noise_20261118_v1/[数据集]/；result/chimerge_no_kl_no_noise_20261118_v1/[数据集]/。模型从epoch0起步，不加载旧checkpoint；已有目标文件/目录即报错。新日志log/chimerge_no_kl_no_noise_20261118_v1/[数据集].log以exclusive模式创建，任务状态[数据集].status.json，队列日志/状态gpu0.queue.log和gpu0.status.json。
- CPU依赖逻辑测试通过：前序running/failed/stopped或进程未退出均不放行且不调用GPU查询；GPU0忙继续等待，GPU1单独忙不阻止GPU0。5份配置、全部冻结源码/参考bins哈希、全新输出路径、模型静态结构和语法检查通过。没有在原任务完成前执行模型或GPU smoke。
- 新等待器于北京时间2026-11-18 15:42:21启动，PID/PGID/SID=272892、PPID=1。核实status=waiting、active=null、beta=0、use_noise=false、wait_reason=predecessor status=running。原GPU0等频CESNET3仍运行（worker266216，核实时epoch787/1000）；GPU1原worker266121及其独立等待器271940保持。nvidia-smi只显示原GPU0/1 worker，新等待器没有显存分配。
- 单项成功须完整1000轮有限loss日志、10个编号checkpoint、逐类JSON生成数量为输入1倍、长度与输入前16包匹配、字段有限值检查全部通过。失败停止本队列后续项并保留文件；当前已排队不代表新模型训练已开始或完成。

监控GPU0新队列（在/home/xiaozj/cgan下）：

```bash
/data/xiaozj/anaconda3/envs/exp/bin/python log/chimerge_no_kl_no_noise_20261118_v1/run.py status
cat log/chimerge_no_kl_no_noise_20261118_v1/gpu0.status.json
tail -f log/chimerge_no_kl_no_noise_20261118_v1/gpu0.queue.log
# 放行、ISCX任务开始后才创建单项日志
tail -f log/chimerge_no_kl_no_noise_20261118_v1/iscx.log
```

取消等待或停止本新队列：

```bash
/data/xiaozj/anaconda3/envs/exp/bin/python log/chimerge_no_kl_no_noise_20261118_v1/run.py stop 0
```

该停止入口验证本运行的脚本路径、lane参数、PID启动标识及PGID；等待期只停止本等待器，训练期终止其worker并禁止后续项，保留已写文件；不会停止前序等频队列或GPU1实验。不要重复launch本运行目录。

## 2026-11-20 原版模型32/64包实验预检查（未启动正式实验）

- 用户仅授权检查ISCX、SSRC、MIRAGE、UTMN数据长度与训练资源，拟GPU0最大32包、GPU1最大64包并行。未开始训练、选模或生成；主代码仍保留现有无KL/no-noise修改，没有恢复覆盖主代码或更改任何旧实验。
- 原版参照log/binning_formal_20261117_v1/source/：完整ChiMerge（包含threshold_merge）、noise_heads存在、beta=0.01、state64/noise32、hidden512、4层8头。AST核对beta与noise分支。GPU仅通过nvidia-smi只读查询，没有CUDA上下文、模型前向或反向。
- CPU统计原始JSON完整流长度（非16包截断长度）：ISCX6781 flow，>=32有821（12.11%）、>=64有647（9.54%）、最长858740；SSRC7426 flow，>=32有1264（17.02%）、>=64有338（4.55%）、最长27083；MIRAGE33975 flow，>=32有15075（44.37%）、>=64有9401（27.67%）、最长300；UTMN5057 flow，>=32有1317（26.04%）、>=64有884（17.48%）、最长300。
- ISCX/MIRAGE/UTMN的每个类别均有>=32和>=64的流。SSRC没有>=32流的类别为bruteforce_dns、ddos_ack、ddos_dns；没有>=64流的类别还包括ddos_syn。最大长度实验应保留所有短流并padding/mask，不能为凑32/64包而删除类别/样本。
- 为评估真实词表规模，仅CPU重新拟合32/64包原版ChiMerge，结果隔离于log/length_precheck_20261120_v2/length_{32,64}/bins/，不是正式训练产物。八字段总箱数（32/64）：ISCX1637/1590、SSRC637/608、MIRAGE5110/5037、UTMN1791/1683。拟合均完成，没有词表膨胀；记录现有chi_cal零期望除法warning，不修改统计量。
- 原版模型只在PyTorch meta设备构造以统计参数，无权重存储分配或数据计算，并确认torch.cuda.is_initialized()==False。MIRAGE最大约25.62M参数；FP32参数+梯度+Adam两组状态合计约390.9MiB。batch64、长度64下，MIRAGE单份log-prob张量约78.8MiB，单份全类别hidden输出160MiB，4层单份FFN隐藏张量合计128MiB，4层注意力矩阵合计32MiB；这些是主要组成的静态大小，并非完整反向传播/分配器/工作区峰值。
- Dataset CPU预编码缓存最大为MIRAGE长度64约135.6MiB。主机约62GiB RAM，可用约57GiB。磁盘剩余约31GiB；按每项1000epoch、checkpoint100、最终权重+10个编号权重估算，8项权重裸张量合计约7.03GiB，另需结果JSON/临时写盘/日志，建议预留至少10–12GiB。本次仅沿用这些条件估算，没有创建正式配置或输出队列。
- GPU只读采样：两张RTX3080各10240MiB；GPU0 used83/free9786MiB，GPU1 used15/free9861MiB，均无计算进程，温度50/48℃。此前两条消融队列状态均completed，未对进程发信号。
- 判断：数据允许两种长度实验，当前静态规模和CPU/磁盘资源没有明显硬性阻断；显卡容量具备可行性，但尚未实测32/64包训练峰值，不能保证正式batch64安全。64包相对16包，线性激活约4倍、显式注意力矩阵约16倍；正式启动前需另行授权有界、逐级GPU显存验证，尤其不能把历史16包显存采样当作本轮峰值。不能将原始最大流长度用于张量分配，应固定截断32/64；后续从头训练，不加载16包checkpoint。
- 完整逐类长度、直方图、8组箱数/参数/张量大小、原版源码哈希及机器资源见log/length_precheck_20261120_v2/report.json；CPU预检查脚本副本precheck_cpu.py。最初v1增量JSON读取在超长流上耗时异常，仅停止该CPU读取器，改逐类别文件解析完成v2；v1标为aborted_cpu_reader。没有启动GPU实验或影响其他任务。

## 2026-11-20 原版模型32/64包有界GPU验证（8项通过，未启动完整实验）

- 用户授权仅小规模可控GPU验证，不运行完整实验。GPU0验证max_seq_len=32，GPU1验证64，分别覆盖ISCX、SSRC、MIRAGE、UTMN。使用独立log/length_gpu_smoke_20261120_v1/source/冻结原版：完整ChiMerge预检查bins、noise_heads存在、KL beta=0.01、FP32、Adam(lr1e-4,betas0.5/0.999)。主工作区无KL/no-noise模型未改动。
- 数据来自原始data-json，每类最多2条>=64包真实flow及1条<32包短flow，CPU截断最多64包，保留完整类别映射和原始32/64分箱词表。实际样本数ISCX30、SSRC19、MIRAGE60、UTMN51，不读取生成结果作为训练输入。每个batch档使用重复采样的真实满长度flow，避免用padding导致显存低估。
- batch按1→2→4→8→16→32→64逐级增加，每档仅执行原版transformer_train的2个batch、1个限定轮次；另做1个真实混合长度batch检查mask/padding，每项共15个优化步骤，8项共120步。该入口期间仅在新临时目录写generator.pth并自动清理；没有1000轮任务、正式模型目录或后台队列。选模OT流程未在本次smoke中执行。
- 每项另使用4条样本验证无前缀生成和带2包前缀续生成，检查输出shape、前缀保持及有效ID1..N。每个训练档检查参数和梯度有限值，全部日志Loss/NLL/KL有限。训练采样均为原实现，不调整模型维度、loss或精度。
- 保护措施：启动前确认两卡无计算进程；每worker只暴露一张卡，PyTorch分配器限额为单卡50%；外部每约0.5秒只读采样整卡显存/温度，显存used>6144MiB、free<4096MiB、温度>=85℃或单worker超过180秒即停止本验证子进程；下一batch若上档reserved两倍超过4.5GiB则停止升级。分配器限额不是整卡硬限制，外部监控采样不是连续峰值保证。未触发任何阈值或OOM。

| 数据集 | GPU0/32包 PyTorch峰值allocated/reserved（MiB） | GPU0整卡采样最大used（MiB） | GPU1/64包 PyTorch峰值allocated/reserved（MiB） | GPU1整卡采样最大used（MiB） |
|---|---:|---:|---:|---:|
| ISCX | 794.9 / 858 | 1221 | 1265.1 / 1476 | 1773 |
| SSRC | 740.6 / 832 | 1195 | 1178.2 / 1386 | 1683 |
| MIRAGE | 1072.7 / 1334 | 1699 | 1722.3 / 2050 | 2347 |
| UTMN | 913.9 / 1054 | 1419 | 1448.9 / 1684 | 1981 |

- 8项均到达batch64且passed。最重为MIRAGE64包：PyTorch峰值reserved2050MiB，整卡采样used2347MiB（约2.29GiB），采样最低free7529MiB；全程采样最高温56℃。结果说明该环境下原版32/64包、batch64的限定训练及少量采样具备可行性；不等于完整1000轮长期稳定性、全量CPU读取峰值、OT选模峰值或生成质量验证。
- 所有worker正常退出，临时checkpoint目录全部清理。结束只读nvidia-smi确认无计算进程，GPU0恢复used83/free9786MiB，GPU1 used15/free9861MiB。没有遗留GPU任务或启动正式训练。
- 汇总与逐项记录在log/length_gpu_smoke_20261120_v1/summary.json、gpu[0/1]_[数据集].json及同名.log；源代码/小样本/bins/脚本哈希和限额见manifest.json。验证脚本validation/smoke_length_gpu.py及其运行目录副本保留，重复运行需使用新目录，不能覆盖本轮。

## 2026-11-20 16包混合表示有界GPU测试：整数时间＋IP等宽＋其余原值（5项通过）

- 用户要求评估16包“无分箱（IP等宽）”小规模测试是否需减少bin，并明确时间按整数分箱。实际口径为混合表示：time在原范围0..10000上按1秒桶，按已有两位小数精度用[0,0.99]、[1,1.99]等不重叠闭区间实现，另含10000上限单点桶，共10001箱，超范围首末箱兜底；src_ip/dst_ip各100箱，区间与先前等宽实验逐项一致；pkt_len/flags/ttl/src_port/dst_port按前16包/flow的完整不同观测值建立单点区间与连续ID，不按配置范围裁剪包长或端口。其余属性保留当前JSON/编码使用的数值精度，不合并观测值。
- 模型沿用刚才原版验证：独立noise_heads＋KL beta=0.01、FP32、Adam1e-4；源代码冻结在log/hybrid16_gpu_smoke_20261120_v1/source/，未修改主工作区无KL/no-noise实现或任何正式配置。
- CPU扫描5个完整输入拟合完整词表，再每类取最多2条>=16包flow及1条短flow，最多16包，实际小样本ISCX30、SSRC20、MIRAGE60、UTMN51、CESNET3 63。CPU验证IP区间、整数时间边界、其他字段单点ID、padding/有效ID通过。每项训练使用完整词表和全部类别头，不以缩词表替代资源验证。
- 只在GPU0依次测试5项，GPU1不运行测试。batch1→2→4→8→16→32→64各2个满16包真实样本训练batch（不足时重复选样），再1个混合长度batch；每项15步、总计75步。另各4条样本做无前缀及带2包前缀采样，检查shape、有效ID、前缀一致。未执行全量训练、1000轮实验或OT选模，临时generator.pth自动清理。
- 保护措施与长度验证一致：启动前确认GPU0无计算进程；PyTorch分配器50%限额；整卡used>6144MiB、free<4096MiB、温度>=85℃、worker>180秒或batch升级预估超4.5GiB即停止；不触发OOM后继续扩大batch。整卡监控约0.5秒采样，不是连续硬限制。

| 数据集 | 类别数合计 | 实际占用的1秒时间桶 | batch64训练峰值allocated/reserved（MiB） | 整卡采样最大used（MiB） |
|---|---:|---:|---:|---:|
| ISCX | 16789 | 322 | 825.3 / 1042 | 1407 |
| SSRC | 19187 | 178 | 873.4 / 1128 | 1491 |
| MIRAGE | 32974 | 405 | 1309.8 / 1820 | 2183 |
| UTMN | 16252 | 58 | 894.6 / 1128 | 1493 |
| CESNET3 | 13125 | 87 | 883.3 / 1174 | 1539 |

- 5项均passed并到达batch64，参数/梯度和Loss/NLL/KL均有限，无OOM/断线/阈值停止。最大整卡used约2.13GiB（MIRAGE），采样最低free7686MiB；最高温55℃。
- 资源判断：在本次16包、batch64、原版模型的限定条件下，没有因显存而减少bin数量的必要。IP200类仅占MIRAGE32974类的约0.61%；各100箱改各50箱仅节约100类，对FP32参数/梯度/Adam约0.441MiB、单份batch64/16包logits约0.391MiB，非总峰值节省，不值得优先牺牲IP分辨率。
- 可选后续优化：时间10001桶中实际仅占58..405桶；可研究仅编码实际出现的1秒桶，以减少空类别且保留分辨率，但必须定义未见桶处理及保存映射。本次未作该优化，也没有比较生成质量；MIRAGE最大非IP词表为src_port17105类。不能从短测试推断完整训练长期稳定性或生成质量。
- 原范围外时间观测ISCX11次、SSRC29次，其余0，维持同一范围兜底；UTMN包长保留原值，包括超出旧ChiMerge配置范围的值。本方案不是严格全字段无分箱。
- 新词表、小样本、日志、逐项报告和summary.json/manifest.json全部位于log/hybrid16_gpu_smoke_20261120_v1/。验证脚本validation/smoke_hybrid_16_gpu.py及运行目录副本已保留；既有实验日志/模型/结果未覆盖。
- 测试进程正常结束、临时权重目录全部清理。结束nvidia-smi无计算进程，GPU0恢复used83/free9786MiB，GPU1 used15/free9861MiB。未启动正式实验或遗留后台任务。

## 2026-11-20 GPU0完整数据正式混合表示：时间>=5000合并尾桶（已后台启动）

- 用户授权GPU0正式运行ISCX、SSRC、MIRAGE、UTMN、CESNET3完整数据，时间5000以上合并一桶，其余保持刚验证的混合表示，不覆盖已有产物。新运行名hybrid16_time5000_formal_20261120_v1。
- 边界按“5000以上”含5000处理：现有两位小数精度下，时间桶为[0,0.99]...[4999,4999.99]共5000个1秒桶，加[5000,10000]尾桶，总5001箱。原时间范围0..10000不变，>10000继续归尾桶，负值仍按首箱兜底。实际CustomDataset编码验证4999.99→ID5000、5000/5000.01/10000/10000.01/20000→ID5001。
- 其余字段完整区间列表与hybrid16_gpu_smoke_20261120_v1逐项一致：源/目的IP原全范围等宽各100箱，pkt_len/flags/ttl/src_port/dst_port按不同观测值一一编码，不合并数值、不按旧ChiMerge范围裁剪包长。时间、IP仍按原后处理箱内取值方式反分箱，非时间/IP单点箱还原原值。本轮不是ChiMerge或严格全字段无分箱实验。
- 参数：max_seq_len16、batch_size64、epoch1000、checkpoint100、expand_times0（逐类生成原输入1倍flow）、preencode=true、seed20261120。模型为已验证原版独立noise_heads＋KL beta0.01、FP32、Adam1e-4；不使用当前主工作区无KL/no-noise消融代码，不加载旧权重。冻结源码在本运行source/，模型相关文件与小规模验证快照哈希一致；driver单独冻结当前版本，仅用其类别映射/模块入口。
- 顺序固定`iscx → ssrc → mirage → utmn → cesnet3`，同一GPU一次一个worker。每项训练→10个checkpoint选模→生成→逐类数量/截断长度/有限值/完整epoch日志与checkpoint校验，全部成功才进入下一项，失败停止队列。输入data-json原完整数据，不用small/train/test或smoke子集替代。输入/分箱/配置/源码哈希、逐类计数、截断长度保存在manifest.json；启动每项前重新核对。
- 新词表总数：ISCX11789、SSRC14187、MIRAGE27974、UTMN11252、CESNET3 8125（均比此前通过GPU验证的对应词表减少5000类）。输入flow依次6781、7426、33975、5057、209999，合计263238。5项配置为log/hybrid16_time5000_formal_20261120_v1/[数据集].config.json。
- 新输出路径：bins/hybrid16_time5000_formal_20261120_v1/bins_[数据集].json；model/hybrid16_time5000_formal_20261120_v1/[数据集]/；result/hybrid16_time5000_formal_20261120_v1/[数据集]/。模型/结果目录存在即失败，日志exclusive创建，原始数据及任何旧实验输出均未修改。
- 新任务stdout/stderr写入log/hybrid16_time5000_formal_20261120_v1/[数据集].log，状态[数据集].status.json记录阶段/耗时/最佳epoch/生成计数；队列日志gpu0.queue.log，队列状态gpu0.status.json。全部使用新文件，不覆盖既有日志。原有generator.pth的每轮更新仅发生在本轮新建模型目录内部。
- GPU安全：仅暴露物理GPU0，PyTorch分配器限50%；lane约每5秒读nvidia-smi，整卡used>6144MiB、free<4096MiB、温度>=85℃、监控不可用或日志出现非有限Loss/NLL/KL时停止当前worker并终止后续项，不自动减batch或缩短epoch。该限额不是整卡硬限制，采样也不能保证捕获瞬时峰值。GPU1未启动任务。
- 启动前GPU0/1均无计算进程，GPU0 used83/free9786MiB，磁盘约86GiB可用。预检查通过：原模型beta/noise、所有源代码/配置/输入/bin哈希、5项非时间字段不变、实际编码时间边界和全新模型/结果/日志路径；未修改全局config或主模型代码。
- 后台启动时间北京时间2026-11-20 15:07:44，lane PID/PGID/SID331800、PPID1，worker331806，同PGID。核实worker位于GPU0（UUID GPU-1f72e708-dcfc-5400-3a85-e61d6272edc1）、显存约1188MiB。ISCX从epoch0开始、完整6781样本preencode=true，初始已超过22/1000，loss/NLL/KL有限，每轮约2.56秒。这是启动证据，不代表正式实验已完成。

监控（在/home/xiaozj/cgan下执行）：

```bash
/data/xiaozj/anaconda3/envs/exp/bin/python log/hybrid16_time5000_formal_20261120_v1/run.py status
cat log/hybrid16_time5000_formal_20261120_v1/gpu0.status.json
tail -f log/hybrid16_time5000_formal_20261120_v1/gpu0.queue.log
# 当前ISCX；后续按status中的active替换文件名
tail -f log/hybrid16_time5000_formal_20261120_v1/iscx.log
nvidia-smi
```

停止本新队列（验证PID启动标识、脚本路径、lane参数和PGID，仅停止本lane及其worker，保留已写产物）：

```bash
/data/xiaozj/anaconda3/envs/exp/bin/python log/hybrid16_time5000_formal_20261120_v1/run.py stop 0
```

停止不额外保存半轮进度，现有checkpoint仅含模型权重。队列completed且completed列表有5项、5份case状态均completed才视为整轮完成；不要重复launch同一目录。

## 2026-11-20 新增扁平属性Transformer基线（代码接入＋GPU1有界验证，未正式运行）

- 用户同意扁平Transformer方案，要求保留原代码，以新增文件接入，可做简单耗时/显存测试，GPU只用物理GPU1。本轮新增stateFormer/flat_transformer.py、stateFormer/flat_train.py、post_process/flat_pipeline.py、flat_driver.py及validation测试脚本，没有修改原transformer.py/model_train.py/dataset.py、原后处理、ChiMerge或driver.py，也未覆盖全局config.json；既有未提交修改保留。
- FlatTransGenerator按包优先展开八字段为属性序列。16包对应128token；输入为BOS、y0、y1...，目标为y0、y1、y2...。输入值embedding使用前一token所属字段，目标字段embedding使用当前待预测字段；另加入packet位置、类别、长度条件和逐位置32维独立高斯噪声投影。512维、4层8头因果Transformer直接经字段专属输出头得到N_f+1分类logits，0号输出屏蔽。
- 不使用packet-state瓶颈、mu/logvar、state重参数化、condition_fix或输出端独立noise_heads。长度/类别仍为外部条件。训练仅按原mask、类别权重和字段平均方式求NLL，无KL/beta。各字段只计算自己的目标位置，避免分配[batch,全部属性token,全部字段词表之和]大张量。
- sample固定整条流的位置噪声bank，在逐属性重算前缀时重复使用同一噪声前缀；每8个token还原为一个packet，返回[batch,max_packets,8]。流级meta仍按原设计每包重复，反分箱/JSON保存保留首包meta处理。第一版无KV-cache，生成会反复计算变长历史。
- 独立flat_driver.py显式要求model_type=flat_transformer_v1且仅允许完整ChiMerge。复用现有CustomDataset编码；在新训练入口属性级右移。选模复用原OT/HM计算与最小OT选择规则；反分箱复用现有函数。没有全局猴子补丁或替换原模型导入。
- 新checkpoint包含model_type、构造参数、字段顺序/类别映射/bins和输入SHA-256/seed元数据及权重。加载时校验类型和元数据，拒绝其他架构及不匹配数据。仅保存权重与配置，不保存Adam/RNG，不提供自动续训；新入口拒绝既有bins文件、模型和结果目录。
- 5份独立配置仅准备于configs/flat_transformer_20261120_v1/{iscx,ssrc,mirage,utmn,cesnet3}.json：完整ChiMerge、16包、batch64（有界验证已通过）、1000epoch、checkpoint100、expand_times0、seed20261120。目标bins/model/result均为各自flat_transformer_20261120_v1命名空间；目前没有创建这些正式输出目录或启动正式任务。原始模型作为对照应使用原版noise＋KL(beta0.01)冻结快照，而非当前主文件无KL/no-noise消融版。
- CPU正确性检查validation/test_flat_transformer.py共7项通过（约0.94秒）：属性移位及前字段embedding、未来目标/未来噪声无泄漏、固定噪声下teacher forcing与逐前缀预测一致、带权mask NLL与手算一致/忽略padding/噪声梯度存在、固定噪声bank采样/合法ID/前缀保持、checkpoint精确往返及身份拒绝、极小真实数据完整ChiMerge→1轮训练→checkpoint选模→JSON生成及重复输出拒绝。CPU集成模型仅32维1层、2类各2条，临时目录已清理，未初始化CUDA。
- GPU验证validation/benchmark_flat_gpu.py仅GPU1，使用5个数据集16包完整原版ChiMerge词表与已有真实小样本（不是混合分箱），模型真实512维4层。batch1→2→4→8→16→32→64，每档1步预热＋3步计时；全部用满16包真实样本，必要时重复采样。每项另采样单流3次计时。5项均到达batch64且passed，loss/参数/梯度有限，输出shape与ID范围正确。另对MIRAGE原版packet模型测batch8/64作为参考，来源明确为noise＋KL beta0.01冻结实现。
- 保护：GPU1启动前无计算进程；PyTorch分配器50%限额，整卡used>6144MiB、free<4096MiB、温度>=85℃、单worker>240秒或batch升级预算超过4.5GiB则停止本测试。只监控/停止本GPU1验证子进程，不发信号给GPU0。相邻GPU0若达到85℃也会停止新增GPU1验证以避免继续增加热负荷。本次无任何阈值停止、OOM或非有限值。

| 扁平模型数据集 | batch64训练步中位耗时 | PyTorch峰值allocated/reserved（MiB） | GPU1整卡采样最大used（MiB） | 单流16包生成中位耗时 |
|---|---:|---:|---:|---:|
| ISCX | 60.40毫秒 | 1629.4 / 1838 | 2135 | 0.142秒 |
| SSRC | 60.16毫秒 | 1618.2 / 1824 | 2121 | 0.142秒 |
| MIRAGE | 61.84毫秒 | 1660.9 / 1856 | 2153 | 0.143秒 |
| UTMN | 60.77毫秒 | 1629.9 / 1838 | 2135 | 0.142秒 |
| CESNET3 | 61.14毫秒 | 1645.4 / 1858 | 2155 | 0.143秒 |

- 时间口径：GPU驻留minibatch的forward＋loss＋backward＋Adam，CUDA synchronize计时，取3步中位；不含全量数据加载、预编码、checkpoint写盘和OT选模。生成是单流bins ID自回归，不含JSON反分箱保存。整卡最大值来自约0.5秒采样，不等于连续瞬时峰值。
- MIRAGE参考原packet模型同batch64：25.67毫秒/训练步、峰值allocated667.8/reserved758MiB、整卡采样1053MiB、单流生成0.035秒。新扁平模型对应训练约2.41倍耗时、生成约4.1倍；原packet参数约25.52M，新flat约15.83M，非等参数架构比较，不把该比值视为纯序列展平的独立因果效应。
- 汇总log/flat_transformer_smoke_20261120_v1/summary.json为passed，6项（5 flat＋1原版参考）全部通过；逐项.json/.log、源代码快照/哈希、bins/小样本哈希、CPU测试摘要在同目录。GPU测试不保存模型权重或正式生成数据。
- 结束核实GPU1无计算进程、恢复used15/free9861MiB。GPU0仍为原hybrid16_time5000_formal_20261120_v1的worker331806，占用约1190MiB；未停止、重启、修改其快照或输出。全部旧文件保持，未启动flat正式实验。

仅验证的复现入口（本次已执行）：

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 /data/xiaozj/anaconda3/envs/exp/bin/python validation/test_flat_transformer.py
```

后续正式flat入口（本次没有执行；需另行安排GPU、输出目录与日志）：

```bash
CUDA_VISIBLE_DEVICES=1 /data/xiaozj/anaconda3/envs/exp/bin/python -u flat_driver.py --config configs/flat_transformer_20261120_v1/iscx.json
```

GPU性能脚本使用exclusive运行目录，不能覆盖同名结果重复执行；如需复测需使用新运行目录。

## 2026-11-20 GPU1完整数据扁平Transformer正式队列（已后台启动）

- 用户授权在GPU1对ISCX、SSRC、MIRAGE、UTMN、CESNET3完整数据执行扁平模型训练、选模、生成，并估算单项与总耗时。新运行名flat_formal_20261120_v1，顺序`iscx → ssrc → mirage → utmn → cesnet3`，一项全部成功后再进入下一项。
- 参数沿用已验证配置：完整原版ChiMerge（含threshold_merge）、八字段、最大16包/128属性token、batch_size64、1000epoch、checkpoint100、expand_times0（逐类生成输入1倍flow）、seed20261120；512维4层8头、128维字段值embedding、逐位置32维噪声投影，NLL-only，无packet-state/KL。不是混合整数时间/IP分箱实验，不加载原packet模型权重。
- 完整输入data-json/{iscx,ssrc,mirage,utmn,cesnet3}，不是smoke/train/test子集。输入flow数6781、7426、33975、5057、209999，类别数10、9、20、17、21，总计263238 flow。逐文件JSON、非空series、字段有限值/哈希均已核验，逐类计数和前16包长度保存于manifest.json；每worker开始前重新核对文件集合和哈希。
- 当前扁平模型/训练/后处理/入口源码与已通过GPU1测试的源文件哈希一致；完整相关代码冻结到log/flat_formal_20261120_v1/source/，5份独立配置[数据集].config.json、哈希和队列保护阈值存manifest.json。快照flat_driver的工作区根路径仅在worker内部显式指回/home/xiaozj/cgan，防止将source快照目录当成data-json根；不改动工作区代码或旧实验快照。
- 新分箱：bins/flat_formal_20261120_v1/bins_[数据集].json；模型：model/flat_formal_20261120_v1/[数据集]/；结果：result/flat_formal_20261120_v1/[数据集]/。包含模型类型/参数和字段/类别/bins/input身份的checkpoint与model metadata一起保存。新路径存在即报错，不自动续训或覆盖旧模型/结果。
- 新日志：log/flat_formal_20261120_v1/[数据集].log（exclusive创建、无缓冲stdout/stderr），阶段/耗时/最佳epoch/选模指标/生成校验写入[数据集].status.json；队列gpu1.queue.log/gpu1.status.json。1000轮有限NLL日志、100..1000共10个checkpoint、10组有限OT/HM、每类JSON数量和长度/字段有限值均通过才标记completed。单项失败停止后续项。
- GPU1独占可见设备，分配器上限50%，约每5秒监控整卡used>6144MiB、free<4096MiB、温度>=85℃及非有限loss，异常即终止本worker、停止本队列。不自动减少batch或epoch。不操作GPU0进程。分配器不是整卡硬上限，外部采样不是连续峰值保证。
- 启动前GPU1无计算进程，used15/free9861MiB；磁盘约85GiB可用、主机可用RAM约55GiB。正式启动北京时间2026-11-20 15:41:35，lane PID/PGID/SID336187、PPID1，初始worker336192。nvidia-smi核实worker在物理GPU1（UUID GPU-dd306551-104a-e6e1-34ec-1c79160bbe17），初始显存约2012MiB。ISCX完整ChiMerge箱数245/554/13/66/235/77/29/408，16包128token，已进入训练，初始每轮约6.3–6.5秒、NLL有限。本节是启动证据，非完成结果。

### 单项及全队列耗时预估

- 训练依据：此前同模型/完整词表GPU1 batch64三步中位数，ISCX60.40、SSRC60.16、MIRAGE61.84、UTMN60.77、CESNET3 61.14毫秒/步；按ceil(flow/64)×1000计算，不减少epoch。初始完整ISCX实测与此接近。
- 为避免将batch1生成时间错误外推到batch64，本轮另外做了有界GPU1采样校准：每数据集1次预热+3次batch64、16包/128token采样，无训练或JSON生成。中位数依次1.2467、1.2506、1.2465、1.2452、1.2484秒/批。明细[数据集].generation_calibration.json/.log，脚本calibrate_generation.py；均有50%分配器及显存/温度/120秒超时保护。
- 选模对100..1000共10个checkpoint分别按完整输入逐类生成再算OT/HM，最终最佳checkpoint再生成1次。因此按逐类ceil(n_class/64)计算11遍采样开销，并额外考虑精确OT求解、CPU编码/反分箱、数据传输及checkpoint写盘。单流0.143秒的小测试耗时未直接按总flow数相乘。

| 数据集 | 纯训练外推 | 10次选模采样＋最终生成外推 | 单项总耗时预计（含其他开销/波动） |
|---|---:|---:|---:|
| ISCX | 1.78小时 | 0.42小时 | 2.2–2.8小时 |
| SSRC | 1.96小时 | 0.45小时 | 2.3–3.0小时 |
| MIRAGE | 9.12小时 | 2.07小时 | 11–14小时 |
| UTMN | 1.35小时 | 0.33小时 | 1.6–2.2小时 |
| CESNET3 | 55.74小时 | 12.58小时 | 68–82小时 |

- 全部5项GPU1串行合计约85.1–104小时（约3.5–4.3天）。按启动时间预计北京时间2026-11-24 04:48至23:42全部完成。单项不含等待前序项；总计已包含全部顺序等待。范围考虑持续负载/温度导致的时钟变化、CPU数据/反分箱、保存和精确OT求解波动，非保证。原始计算的batch数、实测秒数及区间见initial_eta.json；后续应按实际完整epoch/选模阶段更新。

### 监控和停止

在/home/xiaozj/cgan下执行：

```bash
/data/xiaozj/anaconda3/envs/exp/bin/python log/flat_formal_20261120_v1/run.py status
cat log/flat_formal_20261120_v1/gpu1.status.json
tail -f log/flat_formal_20261120_v1/gpu1.queue.log
# 当前ISCX，后续按active字段替换日志名
tail -f log/flat_formal_20261120_v1/iscx.log
nvidia-smi
```

只停止本扁平模型GPU1队列（核实脚本/PID启动标识/PGID后终止本lane和其worker，保留已写文件）：

```bash
/data/xiaozj/anaconda3/envs/exp/bin/python log/flat_formal_20261120_v1/run.py stop 1
```

checkpoint保存权重和模型/数据元信息，不含Adam/RNG；停止不保存半轮。不要重复launch同一目录，也不要用泛化pkill。

### 同时发现的GPU0旧队列状态

- 只读核实发现hybrid16_time5000_formal_20261120_v1于2026-11-20 15:41:09因既有监控读到85℃自动停止：gpu0.status.json=failed，error为GPU safety stop，exit143；ISCX最后日志epoch726/1000。此发生于GPU1新正式队列15:41:35启动前。可确认温度保护触发，不能仅凭该记录判定散热耦合的因果。
- 本次未向该GPU0进程发送信号、修改其温度阈值或自动重启；旧日志/权重保留，恢复前需要另行核验checkpoint完整性和温度条件。本GPU1队列独立运行。

## 2026-11-20 GPU1调整CESNET3重跑、GPU0恢复，温度停止阈值改90℃（两条已启动）

- 用户先要求停止GPU1当前实验，仅将CESNET3改为500轮/每50轮检查点，其余数据集保持，并重启GPU0；随后明确温度停止阈值90℃而非85℃。本次继续完成两项重启，所有新队列运行时判断均为temp>=90；显存限制不变（分配器50%、used>6144MiB或free<4096MiB停止），约5秒监控。旧v1记录保留原85℃历史，不重写旧日志或模型。

### GPU1：扁平模型从头重跑v2

- 已使用身份核验停止入口终止flat_formal_20261120_v1的lane336187/worker336192，状态stopped，旧ISCX日志最后到67/1000。旧源码、checkpoint、日志和结果目录全部保留，不复用其权重。
- 新运行flat_formal_20261120_v2：顺序iscx→ssrc→mirage→utmn→cesnet3，5项均从epoch0开始，完整ChiMerge、16包/128属性token、batch64、expand_times0及已验证扁平模型/数据/seed均保持。ISCX/SSRC/MIRAGE/UTMN为epoch1000/checkpoint100；仅CESNET3为epoch500/checkpoint50，仍有50..500共10个编号checkpoint和10次选模。
- v2源码从v1冻结副本逐文件校验复制；输入SHA-256、配置及新目标路径核对通过。worker完整性验证改为按各自epoch/checkpoint计算，正确接受500轮/50保存和原1000轮/100保存，且校验每一轮日志序号完整，不只计数。
- 新bins/model/result均使用各自flat_formal_20261120_v2命名空间；日志log/flat_formal_20261120_v2/[数据集].log，队列gpu1.queue.log，状态gpu1.status.json/各数据集.status.json，清单manifest.json。全部新文件/目录，未覆盖v1。
- 根据原GPU1训练及batch64采样计时更新：CESNET3纯训练外推约27.87小时，10次选模采样＋最终生成仍约12.58小时，因此单项预计40–50小时；其余4项估计不变，全部约57.1–72小时。initial_eta.json已更新，不能把整轮时间简单减半。

### GPU0：混合表示实验从700轮恢复v2

- 原hybrid16_time5000_formal_20261120_v1在85℃停止后，latest generator.pth仅24836103字节，CPU torch.load报PytorchStreamReader缺data/40，确认不完整；未删除或覆盖。generator_700.pth为87772027字节，CPU加载、全部参数有限和原版模型strict load均通过，SHA-256=b31ef06f87f09c3ce6296c531853065aeeeccde6a53718fa43a4e51c06c542c4。
- 新运行hybrid16_time5000_resume_20261120_v2。5份原bins精确复制且哈希匹配，模型为原noise＋KL beta0.01，混合表示仍为time>=5000尾桶/IP各100箱/其余原值编码，16包、batch64、1000轮、checkpoint100、expand_times0。
- ISCX新模型目录复制原100..700编号checkpoint并核对哈希，native find_latest_checkpoint确认从700恢复，实际继续701..1000；旧日志701..726因无完整最新检查点需重算。其余SSRC/MIRAGE/UTMN/CESNET3从epoch0开始。原checkpoint不含Adam/RNG，恢复初始化新Adam和随机状态，不与不间断训练逐步等价，此限制已向用户说明。
- 模型、bins、结果均写入各自hybrid16_time5000_resume_20261120_v2命名空间；新日志log/hybrid16_time5000_resume_20261120_v2/[数据集].log，与旧日志完全分开。旧权重、旧坏latest及旧日志保留。新结果从未存在，不覆盖旧结果。
- 仅在新worker中包装torch.save为临时文件写完再rename，训练和loss数学未变；完整latest保存后写model/.../[数据集]/latest_complete_epoch.json。若温度停止发生在写盘过程中，上一个完整latest仍保留，避免再次破坏已完成权重。恢复日志仅要求701..1000齐全，但选模仍用复制的100..700和新800..1000共10个checkpoint。

### 启动核实及监控

- 两条新队列北京时间2026-11-20 15:55:22启动：GPU0 lane PID/PGID/SID337663、worker337674；GPU1 lane337666、worker337677，独立会话，父进程脱离终端。
- 核实GPU0明确记录从epoch700恢复，已到704/1000，NLL/KL有限；latest_complete_epoch.json=704，运行温度约63℃。GPU1从1/1000重新开始，第一轮NLL与v1相同，约6.52秒，温度约58℃。这是重新启动证据，尚未完成。两脚本源码和manifest阈值均90℃，未升高显存限额。

```bash
# GPU1新扁平队列：监控 / 停止
/data/xiaozj/anaconda3/envs/exp/bin/python log/flat_formal_20261120_v2/run.py status
tail -f log/flat_formal_20261120_v2/gpu1.queue.log log/flat_formal_20261120_v2/iscx.log
/data/xiaozj/anaconda3/envs/exp/bin/python log/flat_formal_20261120_v2/run.py stop 1

# GPU0恢复队列：监控 / 停止
/data/xiaozj/anaconda3/envs/exp/bin/python log/hybrid16_time5000_resume_20261120_v2/run.py status
tail -f log/hybrid16_time5000_resume_20261120_v2/gpu0.queue.log log/hybrid16_time5000_resume_20261120_v2/iscx.log
/data/xiaozj/anaconda3/envs/exp/bin/python log/hybrid16_time5000_resume_20261120_v2/run.py stop 0
```

停止命令仍核验各自脚本、lane参数、PID启动标识及PGID，只影响本队列；不要重复launch同一运行目录。

## 2026-11-20 GPU0改为时间3000尾桶＋端口等宽250箱，从头训练

- 用户要求先停止过热的GPU0任务，再将时间3000以上并为一箱、端口使用等宽分箱并重新训练。调用旧队列stop 0时，hybrid16_time5000_resume_20261120_v2已经于15:59:54因监控读到90℃自动失败退出（exit143），并非本次停止命令再次发信号。随后nvidia-smi和进程身份核实GPU0无计算进程，旧lane337663/worker337674均已退出；GPU1的flat_formal_20261120_v2 lane337666/worker337677持续运行，没有修改或停止。
- 新实验命名空间：hybrid16_time3000_ports250_20261120_v1。顺序ISCX→SSRC→MIRAGE→UTMN→CESNET3，全部完整数据、每流最多16包、batch64、1000轮、每100轮checkpoint、expand_times=0、seed20261120；原版packet-state模型、独立noise、KL beta=0.01。分箱维度已变，五项均从epoch0初始化，不能加载旧权重继续；本次不改变主工作区模型源码，使用旧GPU0经过验证的冻结原版source副本并逐文件校验SHA-256。
- 时间[0,3000)保持1秒整数分箱，即[k,k+0.99]，k=0..2999；尾箱为[3000,10000]，共3001箱。含3000边界；>10000仍按原编码器映射到最后一箱，反分箱保持原最大值10000。src_port/dst_port均使用pre_process.baseline.cal_bins_equal_width：范围0..65535，各250箱，整数边界与原equal_width基线一致，共享边界保持原编码器first-match。IP各100箱不变；pkt_len/flags/ttl原始单值箱不变。

| 数据集 | 原5000尾桶且端口原值的总箱数 | 新3000尾桶且端口250箱的总箱数 |
|---|---:|---:|
| ISCX | 11789 | 4785 |
| SSRC | 14187 | 3913 |
| MIRAGE | 27974 | 6488 |
| UTMN | 11252 | 5311 |
| CESNET3 | 8125 | 6623 |

- 输出独立：bins/hybrid16_time3000_ports250_20261120_v1/bins_[数据集].json；model/hybrid16_time3000_ports250_20261120_v1/[数据集]/；result/hybrid16_time3000_ports250_20261120_v1/[数据集]/；log/hybrid16_time3000_ports250_20261120_v1/[数据集].log。配置、清单、输入/源码/分箱哈希、cpu_preflight.py/json/log及队列状态也位于新log目录。旧日志、权重和结果不覆盖。
- CPU-only小规模预检5/5通过：使用已有小规模真实样本，编码样本30/20/60/51/63；检查时间2999.99/3000/10000/10001和端口0/65535边界、全部ID范围，每项仅一个batch2训练更新、有限参数及梯度、临时权重保存/严格加载、两条生成序列。临时模型自动清理，无GPU测试负载；正式运行前另核对完整输入、冻结源码和配置哈希。语法检查通过。
- GPU0启动前增加空闲且温度≤75℃的冷却等待，温度未达标只等待，不占用GPU计算。运行中仍约5秒检查，temp>=90℃、used>6144MiB、free<4096MiB、非有限loss或监控失败时停止本队列；分配器50%上限不变。保留原子checkpoint保存及latest_complete_epoch.json。减少箱数不保证持续温度降低；没有改变GPU功耗、风扇或GPU1配置。
- 已于北京时间16:01:54后台启动，独立lane PID/PGID/SID339561，worker339568。ISCX明确从0开始、完整6781样本预编码成功并完成第1/1000轮，NLL/KL有限，第1轮约2.27秒；16:02附近外部快照GPU0约1053MiB/79℃，GPU1约2035MiB/79℃，两卡各自对应worker。此为启动验证，尚未完成正式实验，短时温度不能代表长期温度。

在/home/xiaozj/cgan中监控或停止本次GPU0队列：

```bash
/data/xiaozj/anaconda3/envs/exp/bin/python log/hybrid16_time3000_ports250_20261120_v1/run.py status
cat log/hybrid16_time3000_ports250_20261120_v1/gpu0.status.json
tail -f log/hybrid16_time3000_ports250_20261120_v1/gpu0.queue.log log/hybrid16_time3000_ports250_20261120_v1/iscx.log
# 后续数据集按status的active字段更换日志名
nvidia-smi
# 核验脚本/PID启动标识/PGID后，仅终止本队列及其worker
/data/xiaozj/anaconda3/envs/exp/bin/python log/hybrid16_time3000_ports250_20261120_v1/run.py stop 0
```

不要重复launch相同目录；停止不保存半轮，权重仍不含Adam/RNG。GPU1继续使用原v2队列的监控/停止入口。

## 2026-11-20 GPU0恢复完整ChiMerge＋原版模型，并记录过热诊断

- 用户要求GPU0过热终止后恢复ChiMerge＋原有模型进行训练生成，并判断实验设置与散热原因。已确认hybrid16_time3000_ports250_20261120_v1于16:03:30达到90℃后自动停止（exit143）；lane339561/worker339568均已退出，无残留GPU0计算进程。GPU1 flat_formal_20261120_v2的lane337666/worker337677持续运行，未停止、未改配置。
- 新运行名chimerge_original_thermal_20261120_v1，已于北京时间16:08:47后台启动。GPU0 lane PID/PGID/SID343227、worker343234。五个完整数据集按ISCX→SSRC→MIRAGE→UTMN→CESNET3串行，全部从epoch0训练：max_seq_len16、batch64、epoch1000、checkpoint100、expand_times0、seed20261120、preencode=True；随后原选模与生成流程。旧文件均保留。
- 恢复的是完整八字段ChiMerge（包含threshold_merge和chi_merge），不是混合编码。复用bins/chimerge_threshold_ablation_20261118_v1/original已完成的原方法分箱，五项均与已完成chimerge_no_kl_no_noise实验中的分箱逐字段相同，并核对original分支的统计、完整输入SHA-256及流数；新bins是原文件逐字节复制。此次只复用分箱，不复用无KL实验的模型源码或权重。
- 原版模型来自旧GPU0冻结source并逐文件核对哈希：stateFormer/transformer.py含独立noise_heads，model_train.py含beta=0.01和实际KL贡献；保持原packet-state/属性自回归与原生成方式。冻结ChiMerge.py与当前恢复后的主工作区ChiMerge.py逐字节相同。未修改主工作区的模型消融代码或GPU1代码。

| 数据集 | 完整ChiMerge总分箱数 | 训练流数 |
|---|---:|---:|
| ISCX | 1627 | 6781 |
| SSRC | 687 | 7426 |
| MIRAGE | 4801 | 33975 |
| UTMN | 1792 | 5057 |
| CESNET3 | 3344 | 209999 |

- 新输出均使用chimerge_original_thermal_20261120_v1命名空间：bins/[运行名]/bins_[数据集].json、model/[运行名]/[数据集]/、result/[运行名]/[数据集]/；日志log/[运行名]/[数据集].log，配置/清单/状态也在新log目录。CPU预检5/5通过：真实小样本编码及ID范围、每项一个batch2训练更新、有限参数和梯度、临时权重严格加载、两条生成序列；无额外GPU压力测试。语法和冻结文件哈希检查通过。
- 继续执行启动前GPU0空闲且温度≤75℃、运行时temp>=90℃停止、分配器50%、used>6144MiB/free<4096MiB停止、非有限loss/监控失败停止。原子权重保存及latest_complete_epoch.json保留。没有改变硬件功率上限、风扇控制、GPU时钟或GPU1训练设置。

### 当前诊断证据与边界

- 旧GPU0 ISCX的5000尾桶/端口原值共11789箱：v1用时2005秒到85℃停，重启v2用时272秒到90℃停；3000尾桶/端口250等宽共4785箱，用时96秒到90℃停，当时仅1053MiB显存。这些起始热状态、运行阶段和GPU1状态不同，不能用停止用时直接量化分箱对温度的因果效应；明确不是OOM或显存限制触发。
- 本次恢复完整ChiMerge的ISCX仅1627箱，启动GPU0约59℃/31W/空闲；约31秒已到80℃，约101秒到89℃。16:11:14（约147秒）仍在训练第67轮附近，GPU0约88℃、风扇100%、显存955MiB、瞬时功耗250W；同刻GPU1约78℃、风扇91%、显存2035MiB、功耗344W。20秒后的已采样GPU0功耗大致264–314W，运行期间GPU利用率通常较高；瞬时点不能代替热稳态平均。
- gpu_before.txt与gpu_after_observation.txt记录驱动详细温度/功耗/时钟/降频原因。本次GPU0从SW Thermal Slowdown Not Active变为Active，累计计数从833577485974us增加到833676240831us（约98.75秒）；GPU1该项在这两个快照中均Not Active，计数132819439252us不变。此处使用同次运行前后差值，不把跨历史启动的累计计数当作本次证据。
- 初步判断：过热并非混合分箱或大词表独有；在完整ChiMerge＋原版模型、显存不到1GiB的情况下，GPU0仍快速升至88–89℃并发生驱动温度降频，风扇读数已100%。现有证据更支持GPU0在当前持续负载/双卡环境中的散热余量不足，训练负载是触发因素；不支持仅靠减少bins就能解决。功耗低于GPU1仍更热是辅助线索，非严格同负载对照。
- 尚不能仅凭软件传感器区分显卡散热接触/积灰/风扇实际转速问题、机箱风道/卡间距、进风温度或GPU1热耦合；风扇百分比不证明所有物理风扇运转正常。也不能将不同工作负载、不同卡位置的比较当作硬件故障定论。需要后续有条件进行相同负载/相同冷却起点的单卡对照及现场散热检查；本次不为诊断停止GPU1，不擅自调整硬件参数。
- 当前只是启动及短期温度观察，不代表热稳态或五项实验完成；若再次90℃停止，队列会保留权重/日志并退出，不自动反复重启。正式训练及生成是否完成必须看每项status=completed和生成数量验证。

### 监控和停止

约5秒一条thermal_telemetry.jsonl，包含两张卡的时间、UUID、显存、温度、风扇、利用率、功耗/上限、SM/显存时钟、P-state。仅只读查询GPU1，不向GPU1发送信号。thermal_report.py只读这些日志汇总，无CUDA调用。

```bash
/data/xiaozj/anaconda3/envs/exp/bin/python log/chimerge_original_thermal_20261120_v1/run.py status
cat log/chimerge_original_thermal_20261120_v1/gpu0.status.json
tail -f log/chimerge_original_thermal_20261120_v1/iscx.log
tail -f log/chimerge_original_thermal_20261120_v1/thermal_telemetry.jsonl
# 更新汇总；后续按active更换数据集日志名
/data/xiaozj/anaconda3/envs/exp/bin/python log/chimerge_original_thermal_20261120_v1/thermal_report.py
# 身份核验后只停止此GPU0队列，保留所有文件
/data/xiaozj/anaconda3/envs/exp/bin/python log/chimerge_original_thermal_20261120_v1/run.py stop 0
```

### 本次原版对照最终观察：再次触发90℃保护，已停止

- 上述启动观察之后，GPU0在16:11:19达到90℃自动终止：距16:08:47启动152秒；gpu0.status.json=failed，exit143，error=GPU safety stop: used=955 free=8914 temperature=90。ISCX最后完成69/1000轮，后续4个数据集未启动，选模及生成均未进行。因此本次五数据集训练生成尚未完成，不能把“已恢复并启动”理解为“已生成”。没有自动再次重启。
- 最后现场核实lane343227/worker343234均退出，GPU0计算进程为空、显存83MiB、利用率0%、温度回落到66℃；GPU1仍是原lane337666/worker337677，利用率99%、约78℃、2035MiB，未受停止影响。
- model/chimerge_original_thermal_20261120_v1/iscx/generator.pth及latest_complete_epoch.json保存第69轮完整权重。CPU torch.load成功，134个参数张量均有限，76024075字节，SHA-256=8a0b58ed1cb1aa16ef81653b878df27c203ef281cfc8d6bcbaec65afe84d4d05；详情stopped_checkpoint_check.json。未到第100轮，尚无编号checkpoint；权重没有Adam/RNG，现有原生恢复入口只找编号权重，后续若决定从69恢复需显式处理，不能直接重新launch旧目录或声称自动无缝续训。
- 最终判断更明确：完整ChiMerge＋原版模型也从59℃在152秒内到90℃，并有SW Thermal Slowdown Active、风扇100%的记录，排除了“只有新混合分箱才会过热”的解释。显存955MiB远低于保护限制。更倾向GPU0当前散热能力/进风环境在持续负载下不足，双卡热耦合仍是可能因素；不能只据软件读数断言显卡某个硬件部件损坏。实验负载会产热，不代表分箱或模型逻辑错误。
- 在散热问题处理前不建议反复满负载重启。可先现场检查进风、两卡间距、积灰、物理风扇运转与散热器接触；后续必要时在GPU1自然空闲后进行同负载单卡对照。降低GPU0功率上限是可选的后续缓解措施，本次没有修改硬件设置。

## 2026-11-20 GPU队列对调：GPU0扁平模型无主动温度停止，GPU1混合分箱90℃保护

- 用户明确要求将GPU1现有工作迁往GPU0，取消该任务主动温度保护、依赖GPU自带高温降频；GPU1改跑时间3000尾桶、IP/端口等宽基线，保留90℃停止。本次只修改新队列的脚本判断，未修改GPU驱动/固件的温度机制、硬件关断保护、风扇、功率上限或时钟。所有旧运行文件均保留，主工作区模型消融代码不改。

### GPU0：扁平Transformer迁移继续

- 原GPU1 flat_formal_20261120_v2通过身份核验stop 1于16:15:28停止，原lane337666/worker337677均退出。ISCX最后记录第180/1000轮，原generator.pth修改时间1795162522715111367ns晚于日志1795162522682110946ns，结合旧训练器“每轮先日志、后原子保存”及退出后文件不再变动，确认最新完整权重属于180轮；CPU typed checkpoint加载、metadata一致性、模型strict load、参数有限检查通过。旧权重和日志不覆盖。
- 新目录名flat_migrate_gpu0_20261120_v3，ISCX复制旧第100轮编号检查点以及最新权重为resume_input.pth，逐文件记SHA-256；bins逐字节复制并核对metadata输入/分箱哈希。从181训练到1000，余下SSRC/MIRAGE/UTMN各1000轮/100保存，CESNET3为500轮/50保存，均从0开始；5项顺序不变，完整ChiMerge、flat模型、16包/128 token、batch64、expand_times0、seed20261120、原选模及最终生成保持。
- 迁移用新冻结flat_train.py仅添加start_epoch偏移与新检查点顶层training_epoch标记，保留原损失、模型和Adam数学；run.py对迁移项复用原train/select/generate，复制的第100轮检查点与后续200..1000共同供10次选模。新generator.pth及编号检查点继续原子保存。旧检查点不含Adam/RNG，因此新建Adam并重置随机状态；本次迁移不与连续不间断训练逐步等价，此限制已告知用户。
- 新队列温度策略：manifest.safety.temp_limit_c=null、active_temperature_stop=false；运行判断仅used>6144MiB或free<4096MiB，不含temp比较；无启动温度门槛。仍每约5秒记录温度、风扇、功耗、利用率等，继续50%分配器上限、非有限loss/异常/监控失败保护。温度本身不再触发脚本停止；没有禁用GPU自身降频或固件保护。
- 16:18:42已后台启动，lane PID/PGID/SID348429，worker348439。确认日志从181/1000开始，第一轮迁移训练约6.51秒，NLL有限；初始外部快照约2071MiB/70℃。此为启动证据，非训练完成证据。
- 输出路径：bins/flat_migrate_gpu0_20261120_v3/bins_[数据集].json、model/flat_migrate_gpu0_20261120_v3/[数据集]/、result/flat_migrate_gpu0_20261120_v3/[数据集]/；日志log/flat_migrate_gpu0_20261120_v3/[数据集].log及gpu0.queue.log/gpu0.status.json。CPU迁移检查脚本cpu_resume_check.py及报告记录轮次偏移、typed checkpoint往返、有限参数验证；只做一个小模型CPU batch，不增加GPU测试负载。

### GPU1：3000尾桶＋IP100等宽＋端口250等宽的原版模型基线

- 新目录名hybrid16_time3000_ports250_gpu1_20261120_v2。五项ISCX→SSRC→MIRAGE→UTMN→CESNET3均完整数据、从epoch0训练1000轮/每100轮checkpoint、16包、batch64、expand_times0、seed20261120、preencode=True。原版packet-state＋独立noise＋KL beta0.01；该基线的CESNET3沿用原1000/100，500/50仅属于扁平模型队列。
- 时间[0,3000)按1秒整数区间[k,k+0.99]，尾箱[3000,10000]，共3001箱；>10000按既有编码器进入最后箱。IP在0..4294967295范围各100等宽箱；src_port/dst_port在0..65535范围各250等宽箱，沿用原equal_width边界与first-match；pkt_len/flags/ttl保持原单值编码。五份bins与GPU0旧hybrid16_time3000_ports250_20261120_v1逐字节一致并核对哈希，总箱数ISCX4785、SSRC3913、MIRAGE6488、UTMN5311、CESNET3 6623。复制的是分箱及冻结原版源码，未复用旧训练权重。
- 源码/输入/配置哈希检查通过，复用相同源码/分箱此前5/5通过的CPU编码、边界、训练梯度、权重保存和生成预检记录；GPU映射及新目录逐项核验。启动前GPU1空闲且温度≤75℃，运行时temp>=90℃停止；50%分配器、used>6144MiB/free<4096MiB、非有限loss/监控失败保护均保留，权重原子写入并记录latest_complete_epoch.json。
- 16:18:44已后台启动，lane PID/PGID/SID348470，worker348477。ISCX完整6781样本预编码成功，明确从0开始；已确认前3轮NLL/KL有限，首轮NLL/KL与旧GPU0同配置首轮相同（不据此声称后续跨GPU逐步等价），约2.29秒；初始外部快照约985MiB/59℃。
- 输出路径：bins/hybrid16_time3000_ports250_gpu1_20261120_v2/bins_[数据集].json、model/hybrid16_time3000_ports250_gpu1_20261120_v2/[数据集]/、result/hybrid16_time3000_ports250_gpu1_20261120_v2/[数据集]/；日志log/hybrid16_time3000_ports250_gpu1_20261120_v2/[数据集].log及gpu1.queue.log/gpu1.status.json。

### 两条新队列的监控与停止

两条均独立session，脱离终端后台运行。各自thermal_telemetry.jsonl每约5秒只读记录两卡温度、风扇、利用率、功耗/上限、时钟、显存及UUID；GPU0记录但不依温度停止，GPU1仍有90℃判断。不能从当前epoch或checkpoint存在推断选模/生成完成，需检查各项status=completed及生成数量验证。

```bash
# GPU0：扁平模型迁移队列
/data/xiaozj/anaconda3/envs/exp/bin/python log/flat_migrate_gpu0_20261120_v3/run.py status
tail -f log/flat_migrate_gpu0_20261120_v3/gpu0.queue.log log/flat_migrate_gpu0_20261120_v3/iscx.log
tail -f log/flat_migrate_gpu0_20261120_v3/thermal_telemetry.jsonl
# 身份核验后仅停止此GPU0队列
/data/xiaozj/anaconda3/envs/exp/bin/python log/flat_migrate_gpu0_20261120_v3/run.py stop 0

# GPU1：混合分箱基线
/data/xiaozj/anaconda3/envs/exp/bin/python log/hybrid16_time3000_ports250_gpu1_20261120_v2/run.py status
tail -f log/hybrid16_time3000_ports250_gpu1_20261120_v2/gpu1.queue.log log/hybrid16_time3000_ports250_gpu1_20261120_v2/iscx.log
tail -f log/hybrid16_time3000_ports250_gpu1_20261120_v2/thermal_telemetry.jsonl
# 身份核验后仅停止此GPU1队列
/data/xiaozj/anaconda3/envs/exp/bin/python log/hybrid16_time3000_ports250_gpu1_20261120_v2/run.py stop 1
nvidia-smi
```

后续按active字段替换当前数据集日志名；不要重复launch同一目录，不使用泛化pkill。新队列manifest记录输入、源码、配置、迁移权重和runner哈希，旧历史85/90℃记录不改写。

## 2026-11-20 GPU1追加完整ChiMerge＋原版模型32/64包队列（仅等待，未开始训练）

- 用户要求在GPU1追加32/64包训练且不影响当前任务。结合先前明确指定的长度实验范围，本次使用ISCX、SSRC、MIRAGE、UTMN四个完整数据集，各32与64包，共8项，不加入CESNET3。将“当前任务开始后进行”按不争用GPU的约束接为当前GPU1五数据集hybrid16_time3000_ports250_gpu1_20261120_v2全部成功完成后执行；不是立即并行，也不是只等当前ISCX结束。
- 新运行chimerge_length32_64_gpu1_20261120_v1。固定顺序：iscx_l32→ssrc_l32→mirage_l32→utmn_l32→iscx_l64→ssrc_l64→mirage_l64→utmn_l64，均在物理GPU1串行。每项完整原始data-json/[数据集]，保留所有短流并padding/mask，不筛除类别、不使用对应训练集或生成数据。max_seq_len分别32/64，batch64、1000轮、每100轮保存、expand_times0、seed20261120、preencode=True；每项从epoch0重新训练并进行原检查点选模和最终生成。
- 使用完整原版ChiMerge（threshold_merge＋chi_merge）及原版packet-state模型，noise_heads存在、KL beta0.01、512 hidden/4层8头、state64/noise32。source从当前GPU1冻结原版复制并逐文件核对，与length_gpu_smoke_20261120_v1相应源码哈希一致。主工作区源码和两条当前任务的脚本均未修改。
- 分别复用log/length_precheck_20261120_v2/length_32或length_64/bins已拟合的完整ChiMerge；逐字节匹配此前有界GPU测试的对应长度分箱，不复用16包箱。32/64总箱数：ISCX1637/1590，SSRC637/608，MIRAGE5110/5037，UTMN1791/1683；统计与precheck/report及8项GPU smoke均一致。完整原始输入SHA-256、类流数、截断后逐流长度、配置/分箱/源码哈希已写manifest。准备只做CPU读取和校验，不创建CUDA上下文，不额外运行GPU测试。
- 沿用GPU1温度保护：启动前空闲且≤75℃，运行时温度≥90℃停止；分配器50%、used>6144MiB/free<4096MiB、非有限loss/监控不可用时停止。原子权重保存及latest_complete_epoch.json保留。既有8项batch64有界验证均passed，最重MIRAGE64包reserved2050MiB、整卡采样used2347MiB；并不保证长期温度或完整选模阶段峰值。

### 依赖等待与核验

- 独立后台等待器已于16:28:33启动，PID/PGID/SID354043，gpu1.status.json=waiting，active=null，completed=[]，原因Predecessor status=running; waiting for successful completion。目前没有case worker、正式模型或结果目录，不占用GPU。
- 等待时只每30秒读取前序JSON状态，不调用nvidia-smi、不import torch、不创建CUDA上下文。仅当前序gpu1.status=completed、completed列表完整且顺序为5项、active=null，逐项status=completed并核对generated_counts与前序manifest全部输入数量一致，且旧lane/worker均退出后才继续核验GPU1实际空闲与资源/温度，随后启动首项。前序failed/stopped或结果未验证会保持等待，不因GPU短暂空闲抢先启动。
- CPU依赖门测试6项通过：前序running/failed时等待且不查询GPU，部分完成阻塞，旧进程仍活着阻塞，全部成功验证且进程退出才放行，任一结果无效仍阻塞。测试报告queue_gate_test.json；无torch导入。语法、8份配置/分箱/源码哈希、独立新路径检查通过，git diff --check通过。
- 启动后只读核实当前GPU0 lane348429/worker348439及GPU1 lane348470/worker348477未变，两项仍training；ISCX分别已到264/1000和273/1000。nvidia-smi只有原两条worker，没有新等待器的CUDA进程；当前两份run.py哈希仍与各自manifest一致。

### 独立输出及监控停止

- bins/chimerge_length32_64_gpu1_20261120_v1/length_[32或64]/bins_[数据集].json
- model/chimerge_length32_64_gpu1_20261120_v1/length_[32或64]/[数据集]/
- result/chimerge_length32_64_gpu1_20261120_v1/length_[32或64]/[数据集]/
- log/chimerge_length32_64_gpu1_20261120_v1/[数据集]_l[32或64].log及对应config/status；队列gpu1.queue.log/gpu1.status.json。不同长度、不同数据集和既有全部运行互相隔离，不覆盖旧文件。正式训练日志在对应case开始后才创建。

```bash
# 查看等待原因/当前case及整体进度
/data/xiaozj/anaconda3/envs/exp/bin/python log/chimerge_length32_64_gpu1_20261120_v1/run.py status
cat log/chimerge_length32_64_gpu1_20261120_v1/gpu1.status.json
tail -f log/chimerge_length32_64_gpu1_20261120_v1/gpu1.queue.log
# 等对应训练开始后，按active替换日志名
tail -f log/chimerge_length32_64_gpu1_20261120_v1/iscx_l32.log
# 仅取消本追加队列；若仍等待则只停等待器，不影响当前GPU1任务
/data/xiaozj/anaconda3/envs/exp/bin/python log/chimerge_length32_64_gpu1_20261120_v1/run.py stop 1
```

停止入口核验本队列脚本、PID启动标识及PGID，只向本lane发信号并由其停止自己的worker，不停止前序依赖或GPU0任务。不要重复launch同一目录。


## 2026-11-22 GPU1 test扩展 beta=0.05：旧队列停止，新任务准备后按用户要求停止等待

- 用户要求中断GPU1旧实验，以ISCX、SSRC、UTMN、MIRAGE、CESNET的test数据训练，expand_times分别0、1、2、3、4；补充明确beta=0.05。随后用户明确要求“停止等待进程，直至要求开始”，本节最终状态以该指令为准：已停止，禁止自动继续。
- 旧队列chimerge_length32_64_gpu1_20261120_v1通过其身份校验stop入口停止，lane354043和worker494218均已退出。前6项完成保留，mirage_l64训练中断，latest_complete_epoch.json为649/1000；utmn_l64未开始。已有数据、日志、分箱及权重不删除不覆盖。
- 新命名空间：/home/xiaozj/cgan/log/test_beta0_05_gpu1_20261122_v1/。输入依次为data-json/iscx_test、ssrc_test、utmn-test、mirage-test、cesnet3-test；逐类SHA256均与/home/xiaozj/experiment/real_data_test对应数据集一致。输入流数依次1417、1484、1010、6795、42000，类别10、9、17、20、21。cesnet-test是另一份8400流版本，本次未选择。
- 五项拟采用正式packet-state/noise模型、完整ChiMerge、16包、batch64、1000轮、每100轮checkpoint、preencode=True、seed20261122。每个数据集只训练并选模一次，随后分别独立采样5份输出。expand_times=0..4在现有代码中对应1..5倍输入流数，共25份合成集，合计790590条流，均不混入原始记录。
- 当前主工作区model_train.py为无KL消融版本；本次冻结旧GPU1原版源码到新source目录，核验旧manifest哈希，仅在新副本将beta = 0.01改为beta = 0.05，保持g_loss = nll_loss + beta * kl_loss。未修改cgan主工作区任何Python源码。新配置、输入软链接和源码哈希均已准备，静态语法、5份配置、训练器仅beta差异检查通过。
- 新增辅助代码位于/home/xiaozj/experiment/code/prepare_test_beta005.py、smoke_test_beta005.py、cgan_beta005_wait.py；运行控制器及检查脚本另冻结到新运行目录的run.py、smoke.py、background.py。
- 首次GPU小批量检查在创建CUDA上下文时即失败：CUDA unknown error，未执行任何训练batch。显式GPU1 UUID探测也失败。nvidia-smi -i 1仍可查询GPU1，但GPU0报0000:01:00.0 Unknown Error。未重置GPU、重启服务器或修改驱动、风扇、功率设置，未断言故障因果关系。
- 曾于18:51:20启动独立后台等待器PID/PGID/SID523845，状态waiting_cuda，每60秒只做可用性探测，计划CUDA恢复后先做五项小批量检查再串行正式任务；保留原GPU1 90℃、显存及监控异常停止约束。用户最新要求后，通过background.py stop核验PID启动标识、命令及PGID并终止。最终background.status.json=stopped，signal15；无gpu1.status.json、smoke.json或本次model目录，表明正式训练与生成均未开始。
- 预估：历史相同test输入beta0.01五项训练、选模和5次生成共约6.1小时，其中ISCX约10分钟、SSRC约11分钟、UTMN约10分钟、MIRAGE约54分钟、CESNET约4.7小时。本次beta0.05尚无有效GPU计时，保守估计可用GPU上的执行时间6–8小时；当前已停止，不提供完成时刻，需用户明确要求后才重新开始。
- 将来预留的新输出目录为/home/xiaozj/cgan/{bins,model,result}/test_beta0_05_gpu1_20261122_v1/，result下为expand_0..expand_4/[dataset]_beta0_05。当前仅准备日志、配置及源码，未产生正式合成数据。旧结果全部保留。

只读状态检查（不启动任务）：
```bash
/data/xiaozj/anaconda3/envs/exp/bin/python /home/xiaozj/cgan/log/test_beta0_05_gpu1_20261122_v1/background.py status
cat /home/xiaozj/cgan/log/test_beta0_05_gpu1_20261122_v1/background.status.json
```

本次停止入口（已执行）：
```bash
/data/xiaozj/anaconda3/envs/exp/bin/python /home/xiaozj/cgan/log/test_beta0_05_gpu1_20261122_v1/background.py stop
```

此运行目录已有stopped状态，禁止直接重复launch覆盖；后续用户授权启动时须重新核验GPU/CUDA及配置，并采用保留旧状态的新控制记录或新目录。
