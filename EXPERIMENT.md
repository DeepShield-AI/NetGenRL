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
