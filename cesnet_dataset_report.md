# CESNET-2025-05 数据集调查报告

## 调查范围与方法

- 原始目录：`/data/xiaozj/CESNET-2025-05`，只读调查；使用 PyArrow Parquet metadata 和 `iter_batches(batch_size=8,192/16,384)` 流式读取，未将全量数据加载到内存。
- 文件为目录根下的 31 个 `.parquet` 日文件（`20250501.parquet`--`20250531.parquet`），无一级子目录。文件总字节数 **2,178,214,834**（约 **2,077.31 MiB / 2.03 GiB**），单文件大小范围 **32,062,539--110,116,550 字节**，总行数 **15,263,858**；每文件一个 row group，schema 在 31 个文件中一致。
- 每行是一个带流级统计和有限长度包特征的流记录，不是原始逐包表。不存在应用类别或活动标签列，因此无法从该数据本身统计应用/活动/应用-活动流分布。
- 命令要点：`find /data/xiaozj/CESNET-2025-05 -maxdepth 1 -type f` 检查目录；PyArrow `ParquetFile(...).metadata` 读取每日行数，`iter_batches` 做全量字段审计。系统 Python 原有 NumPy 2.3.3 与 PyArrow 16.1 ABI 不匹配，调查进程只在运行时通过 `PYTHONPATH` 指向本机缓存的 NumPy 1.26.4；没有改动项目环境。

## 真实 schema 与粒度

|字段|Parquet 类型|粒度/实测含义|缺失率|可用性与备注|
|---|---|---|---:|---|
|`DST_IP`|string|流级目的地址标识|0/15,263,858|值均为 10 位小写十六进制标识，抽样不能用 `ipaddress` 解析为 IPv4/IPv6；不是可恢复 IP|
|`DST_IP_SUBNET`|string|流级目的子网标识|0|同为 10 位标识，不能还原子网地址|
|`DST_IP_VERSION`|int64|流级 IP 版本|0|IPv4 **13,592,483**，IPv6 **1,671,375**|
|源 IP/源端口|不存在|无|100%|无法组成模型要求的双端五元组|
|`DST_PORT`|int64|流级目的端口|0|147 个不同值；443 占 15,260,077 行，范围 22--64031|
|`PROTOCOL`|int64|流级 L4 协议|0|全量均为 **17/UDP**；无 TCP 样本|
|`TIME_FIRST`,`TIME_LAST`|timestamp[ns,tz]|流级首末时间|0|端点差与 `DURATION` 全量一致（误差 <1 µs），不是逐包时间戳|
|`DURATION`|double|流级持续时间（秒）|0|范围 0.000159--299.998542|
|`DST_ASN`,`DST_COUNTRY`|int64,string|流级目的 ASN/国家|0|网络归属元数据，不是应用或活动类别|
|`FLOW_END_REASON`|int64|流级结束原因编码|0|不是逐包事件或标签|
|`PPI`|list<list<int64>>，实际三数组|有限包序列：间隔、方向、长度|0（数组元素也无 null）|三数组全量同长；共 362,031,329 个元素，长度最大 30；第一数组单位为毫秒，和/1000 与 `PPI_DURATION` 全量一致；第二数组仅 ±1，首个方向全为 +1；第三数组范围 4--1472|
|`PPI_LEN`|int64|`PPI` 序列长度|0|中位数/ P90/P99/最大均为 30；不是完整流长|
|`PPI_DURATION`|double|有限 PPI 序列累计时间（秒）|0|与 PPI 第一数组之和/1000 全量一致；不能等同于全流 `DURATION`|
|`PACKETS`,`PACKETS_REV`|int64|流级正向/反向包数统计|0|合计包数 **5,359,198,989**，中位数 30、P90 193、P99 5,029、最大 18,715,521|
|`BYTES`,`BYTES_REV`|int64|流级正向/反向字节统计|0|不是单包长度；范围分别 1,228--14,980,135,037 和 39--22,350,187,406|
|`QUIC_PACKETS`|list<int64>|有限包序列的 QUIC 包元数据/标记|0|21 行长度与 `PPI_LEN` 不一致；不能解释为 TCP flags|
|`QUIC_VERSION`,`QUIC_CLIENT_VERSION`,`QUIC_TOKEN_LENGTH`,`QUIC_MULTIPLEXED`,`QUIC_ZERO_RTT`|int64/int64/uint64/int64/int64|流级 QUIC 数值特征|0|不是模型输入字段|
|`QUIC_OCCID`,`QUIC_OSCID`,`QUIC_SCID`,`QUIC_RETRY_SCID`,`QUIC_SNI`,`QUIC_USER_AGENT`|string|流级 QUIC 标识/域名/客户端元数据|0（字符串可为空）|SNI 为空 449,166 行，非空 99,192 个不同值；用户代理为空 15,249,192 行|
|`QUIC_TLS_EXT_LEN`,`QUIC_TLS_EXT_TYPE`|list<int64>|流级 TLS 扩展信息|0|数组长度全量一致；不是逐包 flags|
|`PPI_ROUNDTRIPS`|int64|流级往返次数|0|统计特征|
|`PHIST_SRC_SIZES`,`PHIST_DST_SIZES`|list<int64>|流级正/反向包长直方图|0|数组长度全为 8；不能恢复逐包长度|
|`PHIST_SRC_IPT`,`PHIST_DST_IPT`|list<int64>|流级正/反向 IAT 直方图|0|数组长度全为 8；不能恢复逐包 IAT|

## 质量与一致性

- `PPI` 三个数组全量无 null，方向元素仅 `1/-1`（计数分别 **177,527,577 / 184,503,752**）；但 `PPI_LEN < PACKETS+PACKETS_REV` 的记录有 **7,619,676** 条，表示 PPI 是最多 30 个包的有限观测序列。仅 **7,644,182** 条（约 50.08%）PPI 长度与流级包数合计相等。
- 21 条记录的 `QUIC_PACKETS` 长度与 `PPI_LEN` 不一致；另有 5 条 `PPI_DURATION-DURATION > 30 ms`，均应作为适配时的异常记录复核。`TIME_*`、`DURATION`、PPI 间隔关系全量一致。
- 对按文件名排序后的前 100,000 行记录做完整字段摘要比较，未发现完全重复项。其余约 15.16M 行没有做全量精确去重，不能据此宣称全量无重复。
- 对 PPI 长度等于流级总包数的 7,644,182 条记录，按方向统计的 PPI 事件数量与 `PACKETS/PACKETS_REV` 全量一致；27 条的方向 PPI 长度和与流级 `BYTES/BYTES_REV` 扣除 IPv4/IPv6 + UDP 头长的关系不一致，需保留为交叉字段异常而非推断具体原因。
- 按文件各取前 1,024 行（共 31,744 行）抽查，`DST_IP` 均不能按标准 IP 地址解析；完整 PPI 记录中，第三数组按方向求和与 `BYTES/BYTES_REV` 的差通常为每包 IPv4+UDP 28 字节或 IPv6+UDP 48 字节。结合全量 27 条例外，只能说它**近似符合 UDP 负载长度**，不能认定为 Scapy `len(packet)` 的帧长。逐包绝对时间并未存储；可结合 `TIME_FIRST` 与 PPI 间隔重建近似时间，但精度、完整性需另验。

### 标签分布

|口径|类别数|流记录数|说明|
|---|---:|---:|---|
|应用类别|0 个源标签字段|无法统计|全部 15,263,858 行均无可核验应用标签|
|活动类别|0 个源标签字段|无法统计|全部 15,263,858 行均无可核验活动标签|
|应用—活动组合|0 个源标签字段|无法统计|不能从域名或端口自动制造组合|

`QUIC_SNI` 是握手域名候选，不是标签。最高频的非空域名依次为 `graph.facebook.com` 793,002、`www.youtube.com` 531,413、`calendar.google.com` 470,733；同一应用可能对应多个域名，一个域名也未必唯一对应应用。不能把这个分布当作应用类别分布。
- 15,263,858 行均有非空 Parquet 字段（缺失率 0）；这不等于模型字段齐全。TCP flags、TTL、逐包绝对时间、源端点、应用/活动标签均不可用。

## 与现有模型的差异

现有 `pre_process/pcap_process.py`/`CustomDataset` 要求每个类别 JSON 中的 flow 含整数 `src_ip,dst_ip,src_port,dst_port,protocol`，并在 `series` 中逐包提供 `time,pkt_len,flags,ttl`；`time` 是相邻包时间差，`pkt_len` 是带方向符号的包长度，`flags` 是 TCP 位掩码，`ttl` 是逐包 TTL。CESNET 记录只有流级目的端口、目的标识、UDP、流级首末时间和最多 30 个 `(IAT, direction, length)` 特征，且无标签。

因此，按“原始记录”计数为 15,263,858，按现有八字段输入契约的**直接有效记录数为 0**。不能通过填 0、截取标识、把流级统计冒充逐包字段或伪造标签来适配；即使使用 PPI，最多只能设计一个明确标注为“有限包特征”的新适配方案，仍缺 IP/TTL/flags/标签，且不应直接喂给当前训练代码。

## 适配建议

1. 先确认 CESNET 官方字段字典：`DST_IP` 的十六进制标识是否有外部映射，以及是否存在未提供的源端点、标签和 TTL/flags 原始数据。
2. 若目标是应用分类，应取得带应用/活动标注的版本或建立可审计的标签映射；当前 SNI 只能作为候选域名，不等于应用标签。
3. 若保留现有模型，必须补充真实逐包采集或导出字段，并定义 `PPI` 截短记录、UDP flags（模型现有惯例为 0）和包长层级的语义；不得把 `PPI` 的第三数组直接宣称为 Scapy `len(packet)`。
4. 适配前应单独处理上述 21+5 条异常并报告排除影响，完成标签、字段和重复记录方案后再写转换脚本；本阶段未转换、未改模型、未训练。

## 2026-10-19 官方应用标签来源及逐条关联调查

### 官方实现核验

本次只读检查 [CESNET DataZoo 官方仓库](https://github.com/CESNET/cesnet-datazoo) 的提交 `85afe5613a9a1ea840d174d779042ee13f87b4e2`（2026-03-23）。核心证据：

|证据位置|核验结果|
|---|---|
|`cesnet_datazoo/datasets/datasets.py` 的 `CESNET_QUICEXT_25`|数据库名 `CESNET-QUICEXT-25.h5`，官方 bucket 为 `cesnet-quicext-25`；可选 `size=ORIG` 或 XS/S/M/L 采样规格|
|`cesnet_datazoo/datasets/datasets_constants.py`|`M-2025-5` 明确包含 `20250501` 至 `20250531`；APP enum 有 50 个应用 + 3 个背景类；CATEGORY enum 为服务类别编码|
|`cesnet_datazoo/constants.py`|HDF5 记录字段 `ID`、`APP`、`CATEGORY`；`servicemap.csv` 含 `Tag`、`Service Provider`、`Service Category` 映射|
|`cesnet_datazoo/datasets/cesnet_dataset.py`|HDF5 位于 `data_root/<size>/CESNET-QUICEXT-25[-size].h5`，与独立 `servicemap.csv` 一同下载；数据库按 `/flows/DYYYYMMDD` 日期表组织，表内 APP/CATEGORY enum 经官方枚举解码|
|`cesnet_datazoo/pytables_data/pytables_dataset.py`|数据加载器从 HDF5 行的 `APP` 得到逐条应用标签；内部的 `(TABLE, INDEX)` 是数据库定位坐标，不是公开 Parquet 的连接键|
|官方 `docs/features.md`|`ID` 是 per-dataset unique flow identifier；`DST_IP`/`DST_IP_SUBNET` 为 SHA 匿名标识，无地址逆转换；PPI 的 SIZE 官方定义为 transport payload size|
|官方 `metadata.csv`|整个 QUICEXT-25 原始规格标注 194,296,462 条样本，提供 XS/S/M/L，时间覆盖 2024-06 至 2025-05；并非仅 2025-05 的单月 HDF5 发布|

官方应用标签确实存在于**HDF5 每行 APP 枚举字段**，而不是 servicemap 单独按流映射。`servicemap.csv` 仅把已知应用 Tag 关联到 provider 和 category；仅拿到 servicemap 或代码中的 enum，都无法给本地 Parquet 的某一行标注 APP。`CATEGORY` 是服务类别，并非用户行为/活动标签。

### 本地 Parquet 与 HDF5 的关联条件

复查本地 31 个 Parquet 的统一 schema 与 Pandas metadata：均没有 `ID`、`APP`、`CATEGORY`，且 metadata 仅含 `ARROW:schema` 和 `pandas`，其中的 RangeIndex 只是 Parquet 文件行索引。`DST_IP`、目的端口、协议、首末时间、包数字节数及 QUIC CID 等字段可构成**待验证候选复合键**，但官方代码没有声明其为全局唯一连接键；目的标识、端口、时间或 SNI 本身都不能证明一对一。不能凭同名日期、行号或时间排序把 HDF5 `APP` 填到 Parquet。

本机 `/data/xiaozj` 的数据目录中只找到当前 31 个 Parquet 和同名 ZIP，没有 QUICEXT-25 的 HDF5 数据库或 `servicemap.csv`。因此无法做两边 ID 对照、复合键唯一性/冲突率验证或计算真实标签覆盖率。**当前可确认关联并标注的本地 Parquet 行数为 0；真实潜在覆盖率未知，而非判定其真实标签全缺失。** 官方 HDF5 是否恰好由这批 Parquet 以相同采样和版本构建，也尚未验证。

### 获取可行性与后续核验

官方 DataZoo 源码明确支持 `M-2025-5`，故**同月的官方带标签样本有获取途径**：取得 `CESNET-QUICEXT-25.h5` 原始规格后按 31 个日期表读取 `APP`/`CATEGORY` 和 `ID`。但官方类按数据库文件下载整年 HDF5，再在本地选择月份；仓库没有单独按月数据库下载接口。本次未下载大库；官方文件服务的轻量响应头请求超时，在线可达性和实际文件大小未获验证。XS/S/M/L 是抽样数据库，即使可下载，也不能据此推断覆盖本地 15,263,858 条的比例。

若今后已有官方 HDF5 文件，应先只读抽取同月日期表的 `ID`、`APP`、`CATEGORY` 和候选字段，并确认版本/日级数量。若能从 Parquet 找回官方 `ID`，优先用该 ID；否则逐日建立多字段**精确**复合键，仅接受两端都唯一且字段完全相同的匹配，并报告唯一命中、无匹配和歧义条数，再用额外独立字段交叉核验。匹配覆盖率及类别计数只能在这一步通过后给出。若无可证唯一连接键，应直接使用官方 HDF5 的 2025-05 样本开展分类研究，不向本地 Parquet 猜配标签。

本节结论：**官方确有 2025-05 应用标签来源，但当前无法证明这些标签与已下载 Parquet 逐条对应，不能安全附加标签。**

## 2026-10-19 官方 HDF5 下载成本与存储需求调查

### 官方地址、格式和已确认大小

DataZoo 官方代码 `cesnet_datazoo/datasets/datasets.py` 定义：

```text
https://liberouter.org/datazoo/download?bucket=cesnet-quicext-25&file=CESNET-QUICEXT-25.h5
```

`size="ORIG"` 使用上述文件名；`XS/S/M/L` 会由代码将文件名改为 `CESNET-QUICEXT-25-XS.h5`、`...-S.h5`、`...-M.h5`、`...-L.h5`，并通过同一 bucket 下载。官方 Python 下载器使用 HTTP `Content-Length`、分块 Range 请求和直接写文件；代码没有压缩扩展名，HDF5 是下载后的数据库文件，另有独立 `servicemap.csv`。官方常量规定 XS/S/M/L 分别为 10M/25M/50M/100M 样本，ORIG 为 194,296,462 样本。

本次查询 Zenodo 官方记录 [17249078](https://zenodo.org/records/17249078) 得到 12 个**按月 Parquet ZIP** 文件，而非 HDF5。`2025-05.zip` 元数据为 **2,167,459,774 字节**，MD5 `7528eab9e82ca832b6f086be7f80d6d2`；本机 `/data/xiaozj/CESNET-2025-05.zip` 大小和 MD5 完全一致，已解压 Parquet 总大小为 **2,178,214,834 字节**。Zenodo 记录总 ZIP 大小为 **28,186,538,748 字节**（12 个月，含各月压缩包）。该月 ZIP 的官方字段说明没有 `ID/APP/CATEGORY`，因此它不能替代带标签 HDF5。

截至本调查，官方仓库、文档和 Zenodo metadata 没有公布 HDF5/XS/S/M/L 的 Content-Length、压缩包大小或解压后大小表。对 liberouter HDF5/servicemap 端点的 HEAD/流式响应尝试分别超时或返回 503，未下载正文；所以不能把未经服务端确认的数值写成“实际 HDF5 大小”。

### HDF5 内部结构和五月筛选

官方代码会把一个 HDF5 数据库放在 `data_root/<size>/`，每日表路径为 `/flows/DYYYYMMDD`。因此 `M-2025-5` 是 DataZoo 的**日期选择配置**，不是单独 HDF5 文件；它包含 `D20250501`--`D20250531`。每行的 `APP`/`CATEGORY` 是表内枚举字段，`ID` 是数据集内 flow 标识。官方没有提供只下载单月 HDF5 的 liberouter 参数；可行做法是下载一个规格后只读扫描五月表，或取得官方另行提供的月度数据库。

不会为调查下载完整数据库。已有月度 Parquet ZIP 只能说明同月公开流记录的压缩/解压成本，不能证明 HDF5 行能与其逐条匹配。

### 磁盘空间和估算

本机空间（`df -B1`）：`/data` 总 983,350,071,296、可用 **388,524,875,776 字节（约 362.0 GiB）**；`/tmp` 与 `/` 同盘，可用 **103,654,199,296 字节（约 96.5 GiB）**。当前月度 Parquet ZIP 已占约 2.17 GB，包含解压 Parquet 约 2.18 GB。

下表是**规划代理值，不是官方 HDF5 文件大小**：按已知五月 Parquet 的 15,263,858 行线性折算，并列出 1--5 倍系数范围以覆盖 HDF5 压缩/结构差异。实际下载前必须取得响应 `Content-Length`；下载器还需要同目录临时未完成文件空间，筛选副本需要额外空间。

|规格|官方样本数|Parquet 线性代理|HDF5 规划范围（代理×1--5）|建议临时空间（下载+筛选副本）|
|---|---:|---:|---:|---:|
|XS|10M|约 1.43 GB|约 1.4--7.2 GB|约 3--15 GB|
|S|25M|约 3.57 GB|约 3.6--17.9 GB|约 8--36 GB|
|M|50M|约 7.14 GB|约 7.1--35.7 GB|约 15--72 GB|
|L|100M|约 14.3 GB|约 14.3--71.4 GB|约 30--143 GB|
|ORIG|194.3M|约 27.7 GB|约 27.7--138.7 GB|约 60--278 GB|

这些范围只用于风险规划。`/data` 的 362 GiB 理论上可能容纳 ORIG 的某些估计情形，但若 HDF5 接近范围上限并同时保留下载临时文件、五月筛选副本和现有数据，余量不足；`/tmp` 的 96.5 GiB 不适合作为 ORIG 下载和解压位置。若仅下载官方 `2025-05.zip`，其已知 2.17 GB 下载量和约 2.18 GB 解压空间已经验证可承受。

### 三种获取方案比较与推荐

|方案|成本/结果|评价|
|---|---|---|
|下载 ORIG HDF5 后筛选五月|需先取得未公开的 HDF5 Content-Length；下载整年 194.3M 样本，再读取 `/flows/D202505xx`|标签最完整、可用官方 `ID/APP/CATEGORY`；成本和磁盘风险最高|
|下载 XS/S/M/L|官方样本数固定为 10M/25M/50M/100M，具体 HDF5 大小未公开；五月只是其中日期子集|成本较低，但抽样覆盖和本地 Parquet 不一定一致，不适合声称完整五月标签覆盖|
|寻找官方单月带标签文件|Zenodo 已有 `2025-05.zip`，但它是无标签 Parquet；DataZoo/liberouter 未发现单月 HDF5 接口|若 CESNET 后续提供带标签五月 HDF5，这是成本最低、关联最清晰的方案；当前尚不可用|

**推荐：先向数据集维护者索取或确认带标签的 2025-05 HDF5/标签导出，再下载 XS 或单月版本做小规模验证；不要直接下载 ORIG。** 若必须使用官方 HDF5 且没有单月版本，优先选择 XS，确认表结构、`APP` 标签和 `ID` 关联后再评估 S/M；只有在获得确切 Content-Length、校验和及足够 `/data` 余量后才考虑 ORIG。无论选择哪种规格，都不能将五月 HDF5 标签未经公共 ID/唯一键验证附加到现有 Parquet。

## 2026-10-19 XS 版本大小与存储需求估算

### XS 端点结果

按要求仅对官方 XS URL 做了一次带流式响应的元数据请求：

```text
https://liberouter.org/datazoo/download?bucket=cesnet-quicext-25&file=CESNET-QUICEXT-25-XS.h5
```

请求在 25 秒连接/读取超时，未取得 HTTP `Content-Length`、`Content-Type` 或重定向地址，也没有写入文件；本机未发现 `CESNET-QUICEXT-25-XS.h5`。此前对 liberouter 端点已有超时/503 记录，本次不再重复请求。官方仓库只确认这是 HDF5 数据库命名和 1,000 万样本规格，没有公布 XS 的实际大小、内部 HDF5 压缩过滤器或压缩比。因此 XS 的实际下载字节数目前**未查明**。

### 估算依据与范围

- 已知同月官方 `2025-05.zip` 压缩下载大小为 **2,167,459,774 字节**，解压 Parquet 为 **2,178,214,834 字节**，对应 **15,263,858** 行，约 **142.7 字节/行**（这是 Parquet 代理，不是 HDF5 大小）。
- DataZoo 官方 XS 规格为 **10,000,000** 条流，约为本地五月 Parquet 行数的 0.655倍。按 Parquet 行密度线性换算，XS 的 Parquet 等价规模约 **1.43 GB（1.33 GiB）**。
- HDF5 行还包含 `ID`、`APP`、`CATEGORY` 枚举及 PyTables 表/索引/块元数据；实际压缩过滤器未从文档或端点获得。为覆盖“接近 Parquet 密度、标签/表开销、HDF5 物理布局差异”，采用相对 Parquet 等价规模 **1.0--5.0 倍** 的规划范围：**约 1.4--7.2 GB（十进制，约 1.3--6.7 GiB）**。这不是官方报价，也不是精确预测；若 HDF5 未启用压缩或块布局差异较大，可能超出范围。
- `.h5` 下载文件不是 ZIP；官方没有说明是否启用 HDF5 内部压缩，所以不能把“压缩下载大小”和“打开后的逻辑数据量”混为一谈。实际下载空间应以 Content-Length 为准；HDF5 通常可原地读取，不需要另行解压副本。

### `/data` 空间规划

本次 `df -B1 /data`：总容量 983,350,071,296 字节，已用 544,798,228,480 字节，可用 **388,524,875,776 字节（约 361.8 GiB）**。按上面的 XS 估算：

|项目|估算范围|计算假设|
|---|---:|---|
|XS HDF5 实际下载文件|1.4--7.2 GB|Parquet 等价 1.43 GB × 1--5；实际值未确认|
|断点续传/下载期间额外空间|通常约 0 GB；若另存备份再加 1.4--7.2 GB|官方代码把未完成内容直接写入目标文件；峰值接近最终文件，不必再乘 2|
|五月筛选后的 HDF5/Parquet 副本|约 0.12--0.61 GB|用 31/365≈8.5% 作为日期占比，再乘 XS 估算范围；实际 XS 每日抽样和存储布局未知|
|带中间索引/转换产物的保守额外空间|约 0.5--2 GB|只作后续处理预留，不是官方需求|
|建议峰值预留|约 **10 GB**（更稳妥 **15 GB**）|覆盖 7.2 GB 主文件、筛选副本、日志/索引和文件系统余量|

即使按 15 GB 峰值，约占 `/data` 可用空间 4.1%，当前磁盘足够容纳 XS。`/tmp` 与根盘此前只有约 96.5 GiB 可用，虽也能容纳 XS 估算上限，但不应把大型下载放在根盘；应把目标和临时文件都放 `/data`。实际下载前仍应检查实时空间和服务器 Content-Length。

### 结论

XS 的官方实际下载大小本次仍未查明；可审计的规划范围是 **1.4--7.2 GB**，建议按 **15 GB 峰值临时空间**准备。XS 内部仍应是按 `/flows/DYYYYMMDD` 的日期表，五月数据需下载后读取 `D20250501`--`D20250531`，没有已确认的单月 XS HDF5 URL。推荐先索取官方 Content-Length/校验和或带标签五月导出，再下载 XS；本阶段未下载 HDF5、未修改数据或模型、未训练。

## 2026-10-19 XS HDF5 下载尝试与停止

- 按用户指定目标 `/data/xiaozj/cesnet-datazoo/CESNET-QUICEXT-25-XS.h5` 检查：目标目录和文件在尝试前均不存在；`/data` 可用 **388,524,875,776 字节**，满足此前 15 GB 规划预留。
- 使用官方 XS URL，以一次 `requests.get(..., stream=True)` 元数据/流式请求尝试连接，超时设置为连接 8 秒、读取 25 秒；结果为 `ReadTimeout`，没有收到响应头或正文，也没有创建目标文件。此前历史调查已经记录 liberouter 端点的超时/503，因此本次按约束停止，不再反复请求。
- 由于下载未开始，**没有实际文件大小、SHA-256/MD5、HDF5 可读性、日期表、实际流数、APP/CATEGORY 分布、缺失值或重复 ID 统计**。这些项目不能用 Parquet 代理或官方 enum 冒充，待服务恢复且成功取得文件后再按日期表和批次读取验证。
- 当前结论：CESNET 官方 XS 下载在本环境不可达；目标目录没有可分析的 HDF5。保留已有估算 **1.4--7.2 GB** 和建议 **15 GB 峰值空间**，但不将其表述为实际大小。原始 Parquet、模型代码和配置均未修改，未转换、未训练。

