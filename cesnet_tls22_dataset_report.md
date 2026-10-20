# CESNET-TLS22 数据集适配前调查

## 范围与文件规模

- 原始目录：`/data/xiaozj/cesnet-tls22`，只读调查；使用 gzip CSV 流式/分块读取，未一次加载数据。
- 一级目录为 `W-2021-40` 与 `W-2021-41`，每周下 7 个日期目录。共有 **14 个 `flows-YYYYMMDD.csv.gz`** 和 **17 个 JSON 统计文件**，无未预期文件类型。
- 14 个 flow 压缩文件合计 **9,482,149,835** 字节（约 **9.48 GB / 8.83 GiB**；目录 `du` 约 8.9G），流 CSV 总行 **141,720,670**。附带 `stats-dataset.json` 的 `global.total-saved` 同为 141,720,670，按日统计 JSON 合计也一致。
- 数据日期为 2021-10-04 至 2021-10-17，连续两周；没有包文件或 PCAP。每条 CSV 行是一个双向 flow 记录，PPI 是其中嵌套的有限包序列，因此属于流级记录 + 部分逐包序列的混合格式。

## 实际字段审计

CSV 表头（30 列）为：`ID, BYTES, BYTES_REV, PACKETS, PACKETS_REV, DURATION, PPI, PPI_LEN, PPI_DURATION, PPI_ROUNDTRIPS, APP, CATEGORY, TCP_FLAGS, TCP_FLAGS_REV, FLAG_CWR, FLAG_CWR_REV, FLAG_ECE, FLAG_ECE_REV, FLAG_URG, FLAG_URG_REV, FLAG_ACK, FLAG_ACK_REV, FLAG_PSH, FLAG_PSH_REV, FLAG_RST, FLAG_RST_REV, FLAG_SYN, FLAG_SYN_REV, FLAG_FIN, FLAG_FIN_REV`。

|字段|实际类型/粒度|缺失率|实测范围或结构|可用性|
|---|---|---:|---|---|
|`ID`|CSV 字符串，内容为整数；流级唯一编号候选|0|0--141,720,669；全量按文件连续递增，无非连续跳变|可作为流标识；不是模型五元组|
|`BYTES`,`BYTES_REV`|CSV 字符串，流级正/反向字节统计|0|165--7,919,407,887；40--26,500,626,268|流统计，非逐包长度|
|`PACKETS`,`PACKETS_REV`|CSV 字符串，流级包数|0|1--5,608,371；1--18,044,162|流统计；合计包数约 29,307,501,175|
|`DURATION`|CSV 字符串转 double，流级秒数|0|0--328.93472；0 时长记录 8,164|流统计|
|`PPI`|CSV 字符串中的 JSON 三数组 `[IPT, DIR, SIZE]`|0|逐流最多 30 个元素；全量 PPI 元素 **2,199,133,576**；IPT 单位毫秒、方向 ±1、SIZE 为传输层 payload 大小（官方定义）|可作为有限逐包序列，但不是完整包轨迹|
|`PPI_LEN`|CSV 字符串转 int，流级 PPI 长度|0|3--30；中位数 13、P90/P99/P999/最大均 30|模型可截取序列长度，但需处理有限/截短语义|
|`PPI_DURATION`|CSV 字符串转 double，PPI 序列时长|0|全量范围 -1.921--327.543；发现 2 条负值|异常字段；不能直接当全流 `DURATION`|
|`PPI_ROUNDTRIPS`|CSV 字符串转 int，流级 PPI 往返统计|0|0--15|流统计|
|`APP`|CSV 字符串，流级应用标签|0|**198** 个非空类别|可作为应用分类标签；与官方 DataZoo 当前元数据的 191 类说明存在版本/发布差异，需以本地文件为准|
|`CATEGORY`|CSV 字符串，流级服务类别标签|0|**21** 个非空类别|可作为服务类别标签；不是用户活动标签|
|`TCP_FLAGS`,`TCP_FLAGS_REV`|CSV 字符串转 int，流级方向 flags 汇总/掩码字段|0|2--255；4--223|不是逐包 flags；需查官方字段定义后决定是否可用于条件特征|
|`FLAG_*`（16 列）|CSV 字符串，流级正/反向 TCP flag 是否出现|0|抽查为 0/1；分别涵盖 CWR/ECE/URG/ACK/PSH/RST/SYN/FIN|流级存在性特征，不能还原逐包 flags|

**缺失字段：**没有 `src_ip`、`dst_ip`、`src_port`、`dst_port`、L4 协议列、逐包 TTL、逐包绝对时间戳。IP、端口和 TTL 不是“缺失值可填 0”，而是整个数据发布未提供。PPI 的 `DIR` 是逐 PPI 元素方向，`IPT` 是逐 PPI 元素相邻时间间隔，`SIZE` 是逐 PPI 元素传输 payload 长度；它们都不提供 IP/端口/TTL。

## 标签与质量统计

- 原始流记录 **141,720,670**；`APP` **198** 类，`CATEGORY` **21** 类，应用—类别组合 **198** 种。所有标签非空，标签缺失流为 **0**。应用类别计数完整保存在 `/data/xiaozj/cesnet-tls22/stats-dataset.json` 的 `apps`，服务类别计数在本报告扫描结果中核对；最高频应用包括 `office-365` 4,926,647、`seznam-media` 4,677,709、`microsoft-update` 4,660,913、`appnexus` 4,369,941、`microsoft-diagnostic` 4,262,593。最小类别为 `apple-authentication` 1、`csas-ib` 4、`csob-ib` 7、`wikimedia-commons` 9。
- 最高频服务类别：`Other services and APIs` 17,571,181、`Advertising` 16,015,204、`Streaming media` 14,461,941、`Social` 11,271,783、`Analytics & Telemetry` 10,918,354。应用—类别组合一对一映射，未发现一个 APP 对应多个 CATEGORY。
- `PPI_LEN` 中位数 13、P90 30；流级 `PACKETS+PACKETS_REV` 中位数 25、P90 83、P99 2,244、P999 27,232、最大 18,907,538。**141,720,226 条**记录的 PPI 长度小于流级包数统计，说明 PPI 只覆盖部分包（官方文档说明 PPI 只描述首 30 个包，且 TCP 零长度/重传/乱序处理会造成差异）；不能用 PPI 长度代表完整 flow length。
- 全量 ID 从 0 到 141,720,669，14 个日文件边界连续递增；按分块扫描未发现非连续 ID 跳变或相邻重复 ID。未建立磁盘去重索引，因此不能把相邻检查扩大为所有字段完全重复的证明。
- 数值解析失败 0、CSV 字段缺失/空值 0、PPI 三数组长度在抽样中一致、方向值为 ±1、首个 IPT 为 0。全量发现 `PPI_DURATION < 0` **2 条**，`PPI_DURATION > DURATION` **8,768 条**；这是有限 PPI 序列与全流时长定义不同/异常值的质量信号，需适配时复核。PPI 时长与全流时长相差超过 1 ms 的记录约 141.67M，不能当作解析错误。
- 本报告的实际扫描样本展示了 `PPI` 数组和标签；没有将附带 stats JSON 的类别计数当成逐包数据。原始记录数与模型可用记录数明确分开。

## 与现有模型输入差异

项目 `pcap_process.py` 和 `CustomDataset` 要求每个类别 JSON flow 具有整数 `src_ip/src_port/dst_ip/dst_port/protocol`，以及逐包 `series` 中的 `time,pkt_len,flags,ttl`。CESNET-TLS22 提供流级 APP/CATEGORY、流级字节/包数/TCP flags 汇总，以及首 30 个 PPI 的 IAT/方向/payload size，但没有任何端点、TTL 或逐包 flags；PPI SIZE 也不是现有 Scapy `len(packet)` 的帧长定义。

因此按现有八字段契约，原始 141,720,670 条中的**直接有效记录数为 0**。即使将 PPI 映射为 `time/pkt_len`，仍缺 `src_ip/dst_ip/src_port/dst_port/ttl/逐包 flags`，不能填零、伪造地址或把流级 TCP flags 当成逐包 flags。APP 可作为标签，CATEGORY 可作为另一种分类标签，但二者不能自动产生活动标签。

## 适配建议

1. 若目标是使用官方标签做分类，可保留 `APP` 或 `CATEGORY` 作为标签，并明确小类样本极少（最小仅 1--9 条）；不要将二者混作应用—活动组合。
2. 若必须兼容当前 NetStateFormer，需获得包含双端 IP、端口、逐包 TTL、完整逐包 TCP flags 和明确帧长定义的原始 PCAP/增强导出；当前 CSV 不足以无损重建。
3. 若设计独立 CESNET 特征适配，应使用官方 PPI 三通道（IPT/DIR/SIZE）和流统计，并将其视为新 schema；对 2 条负 PPI_DURATION、8,768 条 PPI 时长超过全流时长的记录制定明确排除/复核规则。

