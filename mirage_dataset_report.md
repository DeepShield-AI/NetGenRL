# MIRAGE-AppAct-2024 数据集适配前调查

调查日期：2026-10-16；数据路径：`/data/xiaozj/MIRAGE-AppAct-2024`。原始文件只读。此报告是全量扫描结果，未转换数据、改代码或训练。

## 口径与方法

- 逐文件以二进制按行读取 JSON，按顶层 5 元组键切出单条 flow 再解析；内存中只保留当前 flow、计数器和摘要哈希。扫描 2,245/2,245 个文件，无文件读取或记录解析错误。文件结构在扫描中一致：顶层 JSON object，键为 `src_ip,src_port,dst_ip,dst_port,L4协议号`；值含 `packet_data`、`flow_features`、`flow_metadata`。另有 2 个仅含 `{}` 的空文件。
- “原始 flow”指顶层键值对；“包”指 `packet_data.timestamp` 数组元素。字段缺失率：flow 级用 106,157 作分母，逐包数组用 305,825,577 作分母。数组元素 null 数为全量检查，数值类型以每个数组前 4 个元素抽查并结合异常扫描；不能替代所有元素的完整类型证明。
- “核心数组完整”要求有效 IPv4/TCP或UDP 5 元组，非空时间戳，`timestamp`、`src_port`、`dst_port`、`packet_dir`、`IP_packet_bytes`、`iat`、`TCP_flags` 均为同长且无 null 的数组。“可直接输入当前模型”另要求逐包 `ttl` 和按项目规则可得到的 `flags`、`pkt_len`、`time`。
- 重复检测用解码前 flow 值的 BLAKE2b 摘要，不含 5 元组键与文件路径，因此“重复 payload”不等同于重复完整记录；未做跨文件语义去重。

## 文件与规模

- 文件类型：仅 `.json`，2,245 个；总大小 144,240,074,368 字节（144.240 GB，约 134.33 GiB）。一级子目录 20 个。
- 原始 flow 106,157；核心逐包数组完整 106,157；当前八字段模型可直接使用 **0**（逐包 TTL 全部缺失）。包数 305,825,577。
- 空 JSON 文件 2 个，均位于 `Zoom`；其余文件含 flow。

| 一级子目录 | JSON 文件 | 字节 | GB（十进制） | 原始 flow |
|---|---:|---:|---:|---:|
| ClashRoyale | 16 | 613,078,778 | 0.613 | 451 |
| Crunchyroll | 88 | 9,981,810,431 | 9.982 | 5,281 |
| Discord | 290 | 22,778,168,053 | 22.778 | 10,548 |
| GotoMeeting | 70 | 8,164,806,062 | 8.165 | 5,456 |
| JitsiMeet | 106 | 4,459,753,712 | 4.460 | 2,160 |
| KakaoTalk | 6 | 206,006,593 | 0.206 | 192 |
| Line | 58 | 1,378,702,546 | 1.379 | 1,592 |
| Meet | 190 | 11,522,155,934 | 11.522 | 7,410 |
| Messenger | 185 | 6,511,804,327 | 6.512 | 7,483 |
| Omlet | 84 | 8,379,805,326 | 8.380 | 3,081 |
| Signal | 97 | 3,004,716,794 | 3.005 | 2,610 |
| Skype | 138 | 7,871,381,832 | 7.871 | 11,039 |
| Slack | 168 | 9,683,653,086 | 9.684 | 6,903 |
| Teams | 136 | 13,748,130,120 | 13.748 | 9,768 |
| Telegram | 129 | 3,363,228,828 | 3.363 | 4,730 |
| Trueconf | 112 | 4,089,200,029 | 4.089 | 3,998 |
| Twitch | 74 | 9,566,250,519 | 9.566 | 6,524 |
| Webex | 67 | 5,952,322,987 | 5.952 | 6,073 |
| WhatsApp | 55 | 1,258,770,502 | 1.259 | 1,376 |
| Zoom | 176 | 11,706,327,909 | 11.706 | 9,482 |

### 流长度

流长度为逐包 `timestamp` 数组长度，最小 1、中位数 26、P75 45、P90 612、P95 4,178、P99 81,838、最大 1,665,842。

| 包数范围 | flow 数 | 占比 |
|---|---:|---:|
| 1–1 | 891 | 0.84% |
| 2–15 | 21,040 | 19.82% |
| 16–16 | 3,092 | 2.91% |
| 17–100 | 62,198 | 58.59% |
| 101–1,000 | 9,647 | 9.09% |
| 1,001–10,000 | 5,298 | 4.99% |
| 10,001–10,000,000,000 | 3,991 | 3.76% |

当前 `max_seq_len=16` 时，81,134 条 flow（76.43%）会截断；最多保留 1,508,386 个包位置，占原始包数 0.49%。长度超过 16 不视为无效，因为现有数据集类明确截断。

## 真实结构与粒度

样本：`Line/1670407417_2c:ae:2b:fb:3a:67_None_mirage2020dataset_labeled_biflows_all_packets_encryption_metadata.json` 中一条 4 包 flow 的键为 `192.168.20.102,39080,147.92.242.232,443,6`；`packet_data.timestamp` 为 4 个绝对时间戳，`src_port`/`dst_port`/`packet_dir`/`IP_packet_bytes`/`iat`/`TCP_flags` 均有 4 个元素；`flow_metadata.BF_num_packets=4`，`flow_features.packet_length.biflow` 含 min/max/mean 等统计量。这说明记录为**流级容器 + 逐包平行数组 + 流级统计**，不是一行一包。

逐包 `IP_packet_bytes` 是 IP 层长度，不能直接认作现有 `pcap_process.py` 的 `len(packet)`（可能包含链路层头）；`flow_features.packet_length` 是聚合统计，不能恢复逐包长度。逐包 `iat` 与相邻 `timestamp` 差全量核验，无超过 0.01 秒的 flow；第一项为 0。`packet_dir` 观测值仅 0/1；样本中 0 与 5 元组源端口一致、1 与反向端口一致，但若作为生成前方向定义，仍建议验证端口和方向全量一致性。

## 字段审计

表中“缺失”同时给出字段缺失率和已存在数组中的 null 元素数；数组长度不一致单独列出。`可用`是针对当前模型的条件判断。

| 实际字段 | 观测类型 | 粒度 | 缺失 | 可用性与备注 |
|---|---|---|---|---|
| 顶层 5 元组键 | 逗号分隔的字符串 | flow | 0%；0 个无效 5 元组 | IPv4 地址、端口、协议号可解析；协议 6/TCP 88,914 flow，17/UDP 17,243 flow。IP 需转整数。 |
| `packet_data.timestamp` | float 数组（抽查） | packet | 0%；0 null | 绝对 Unix 时间；可差分得到 `time`。 |
| `packet_data.iat` | float 数组（抽查） | packet | 0%；0 null | 逐包间隔；与 timestamp 差一致，首项为 0。 |
| `packet_data.IP_packet_bytes` | int 数组（抽查） | packet | 0%；0 null | IP 层包长；方向符号需由 `packet_dir` 生成；与 Scapy `len(packet)` 定义待对齐。 |
| `packet_data.packet_dir` | int 数组（抽查） | packet | 0%；0 null | 值 0/1，方向可用于有符号长度；0 共 125,096,296 包，1 共 180,729,281 包。 |
| `packet_data.src_port` / `dst_port` | int 数组（抽查） | packet | 各 0%；0 null | 逐包端口，正反向交替；现有模型端口为 flow 级元属性，应使用 5 元组的方向基准。 |
| `packet_data.TCP_flags` | str 数组（抽查） | packet | 0%；0 null；空串 232,909,748 包 | TCP 标志字符串，如 `A`、`PA`、`SA`；需映射成 Scapy 位掩码。UDP 17,243 flow 也有该数组，不能把空串当成观测到的 TCP flags。 |
| 逐包 `ttl` / `TTL` | 不存在 | packet | 100%；305,825,577 个包无值 | 阻断当前八字段模型直接使用；不能填零或伪造。 |
| 逐包 `src_ip` / `dst_ip` | 不存在 | packet | 100% | 5 元组键提供 flow 级 IPv4，可按方向规则使用；不要据此伪造逐包 IP。 |
| `flow_metadata.BF_label` | str 或 null | flow | 字段 0%；null 1 条 | 应用标签；47 个含 null 的取值，含系统进程与背景流，需筛选。 |
| `flow_metadata.BF_activity` | str；仅存在时 | flow | 23.39%；缺失 24,828 条 | 活动标签；缺失与字面 `Unknown` 4,008 条需区分。 |
| `flow_metadata.BF_sublabel` | str 或 null | flow | 0%；null 1 条 | 应用包名/进程子标签，不是活动类别。 |
| `flow_metadata.BF_num_packets` | int | flow | 0% | 与 timestamp 长度全量一致；只是包数，不是逐包特征。 |
| `flow_features.packet_length` | dict（biflow/upstream/downstream） | flow 汇总 | 0% | min/max/mean/std 等统计；不能替代 `IP_packet_bytes`。 |
| `flow_features.iat` | dict（biflow/upstream/downstream） | flow 汇总 | 0% | min/max/mean/std 等统计；不能替代逐包 `packet_data.iat`。 |
| `packet_data.L4_raw_payload` | list；元素为字节列表（抽查） | packet/稀疏 | 0%；0 null；14,336 flow 与时间戳数组不等长 | 非模型字段，不能按索引无条件与逐包数组拼接。 |

其他逐包字段 `IP_header_bytes`、`L4_header_bytes`、`L4_payload_bytes`、`TCP_win_size`、`is_clear`、`heuristic`、`annotations` 全量存在、数组长度均与时间戳相同、null 元素为 0；本阶段未将其认作现有八字段模型输入。`flow_metadata` 的 BF/UF/DF 字节数、持续时间和包数为流级聚合，MSS/WS 中还观察到 NaN；不用于重建缺失逐包属性。

## 标签分布

目录类别和记录内 `BF_label` 不等价。下面的应用类别以记录内 `BF_label` 为准；`<null>` 是 JSON null。`BF_activity` 以记录内原值为准，不将组合标签（如 `audiocall-videocall`）拆分。

### 应用类别（`BF_label`，全量）

| 应用标签 | flow 数 | 占原始 flow |
|---|---:|---:|
| `/system/bin/netd` | 22,681 | 21.37% |
| `Skype` | 8,120 | 7.65% |
| `Teams` | 7,397 | 6.97% |
| `Discord` | 6,867 | 6.47% |
| `Zoom` | 6,749 | 6.36% |
| `Twitch` | 5,949 | 5.60% |
| `Messenger` | 5,280 | 4.97% |
| `Webex` | 4,789 | 4.51% |
| `Slack` | 4,655 | 4.39% |
| `Crunchyroll` | 4,517 | 4.26% |
| `Meet` | 4,305 | 4.06% |
| `GotoMeeting` | 3,873 | 3.65% |
| `Telegram` | 3,800 | 3.58% |
| `Trueconf` | 3,296 | 3.10% |
| `Background` | 2,942 | 2.77% |
| `Omlet` | 2,426 | 2.29% |
| `Signal` | 1,916 | 1.80% |
| `JitsiMeet` | 1,459 | 1.37% |
| `Line` | 1,202 | 1.13% |
| `WhatsApp` | 919 | 0.87% |
| `com.android.networkstack` | 661 | 0.62% |
| `com.zhiliaoapp.musically` | 423 | 0.40% |
| `com.android.networkstack.process` | 369 | 0.35% |
| `Playstore` | 339 | 0.32% |
| `ClashRoyale` | 309 | 0.29% |
| `com.facebook.katana` | 178 | 0.17% |
| `KakaoTalk` | 170 | 0.16% |
| `in.fulldive.shell` | 93 | 0.09% |
| `system_server` | 91 | 0.09% |
| `com.instagram.android` | 79 | 0.07% |
| `com.google.android.youtube` | 58 | 0.05% |
| `com.google.android.contacts` | 53 | 0.05% |
| `Workplace` | 50 | 0.05% |
| `com.cyanogenmod.updater` | 37 | 0.03% |
| `com.android.webview` | 25 | 0.02% |
| `com.google.android.youtube.player` | 18 | 0.02% |
| `com.perfectfreeapps.vrrollercoaster.app` | 15 | 0.01% |
| `com.snapchat.android` | 10 | 0.01% |
| `com.google.android.apps.turbo` | 9 | 0.01% |
| `/system/bin/adbd--root_seclabel=u` | 8 | 0.01% |
| `org.lineageos.jelly` | 8 | 0.01% |
| `android.process.media` | 3 | 0.00% |
| `org.lineageos.lineageparts` | 3 | 0.00% |
| `com.google.android.gm` | 2 | 0.00% |
| `com.google.android.googlequicksearchbox` | 2 | 0.00% |
| `<null>` | 1 | 0.00% |
| `com.google.android.feedback` | 1 | 0.00% |

### 活动类别（`BF_activity`，全量）

| 活动标签 | flow 数 | 占原始 flow |
|---|---:|---:|
| `<null>` | 24,828 | 23.39% |
| `videocall` | 22,021 | 20.74% |
| `audiocall` | 14,236 | 13.41% |
| `video-streaming` | 12,892 | 12.14% |
| `audiocall-videocall` | 6,660 | 6.27% |
| `chat` | 6,607 | 6.22% |
| `videocall-audiocall` | 5,109 | 4.81% |
| `Unknown` | 4,008 | 3.78% |
| `chat-audiocall` | 2,679 | 2.52% |
| `chat-videocall` | 2,668 | 2.51% |
| `audiocall-chat` | 2,090 | 1.97% |
| `videocall-chat` | 1,976 | 1.86% |
| `gaming-online` | 309 | 0.29% |
| `video on-demand` | 74 | 0.07% |

### 应用—活动组合（`BF_label` × `BF_activity`，全量）

共 142 种观测组合，含缺失活动和背景/系统进程标签。

| 应用标签 × 活动标签 | flow 数 | 占原始 flow |
|---|---:|---:|
| `/system/bin/netd | <null>` | 22,681 | 21.37% |
| `Twitch | video-streaming` | 5,949 | 5.60% |
| `Crunchyroll | video-streaming` | 4,517 | 4.26% |
| `Skype | videocall` | 3,164 | 2.98% |
| `Teams | videocall` | 3,147 | 2.96% |
| `Webex | videocall` | 3,064 | 2.89% |
| `Background | Unknown` | 2,942 | 2.77% |
| `GotoMeeting | videocall` | 2,697 | 2.54% |
| `Teams | audiocall` | 2,445 | 2.30% |
| `Omlet | video-streaming` | 2,426 | 2.29% |
| `Discord | videocall` | 2,364 | 2.23% |
| `Zoom | videocall` | 2,300 | 2.17% |
| `Messenger | videocall` | 1,984 | 1.87% |
| `Skype | audiocall` | 1,850 | 1.74% |
| `Webex | audiocall` | 1,725 | 1.62% |
| `Zoom | audiocall` | 1,629 | 1.53% |
| `Messenger | audiocall` | 1,554 | 1.46% |
| `Discord | audiocall` | 1,494 | 1.41% |
| `Meet | videocall` | 1,486 | 1.40% |
| `Zoom | chat` | 1,435 | 1.35% |
| `Skype | chat` | 1,426 | 1.34% |
| `Meet | audiocall-videocall` | 1,234 | 1.16% |
| `GotoMeeting | audiocall` | 1,176 | 1.11% |
| `Slack | audiocall` | 1,048 | 0.99% |
| `Slack | videocall` | 1,028 | 0.97% |
| `Teams | chat` | 1,024 | 0.96% |
| `Messenger | chat` | 958 | 0.90% |
| `Slack | chat` | 909 | 0.86% |
| `Trueconf | chat-audiocall` | 800 | 0.75% |
| `Meet | audiocall` | 795 | 0.75% |
| `Trueconf | audiocall-videocall` | 731 | 0.69% |
| `Meet | videocall-audiocall` | 729 | 0.69% |
| `JitsiMeet | audiocall-videocall` | 721 | 0.68% |
| `Trueconf | chat-videocall` | 708 | 0.67% |
| `JitsiMeet | videocall-audiocall` | 697 | 0.66% |
| `Discord | audiocall-videocall` | 668 | 0.63% |
| `com.android.networkstack | <null>` | 661 | 0.62% |
| `Discord | chat` | 636 | 0.60% |
| `Skype | videocall-audiocall` | 586 | 0.55% |
| `Skype | audiocall-videocall` | 578 | 0.54% |
| `Discord | videocall-audiocall` | 560 | 0.53% |
| `Telegram | audiocall-videocall` | 548 | 0.52% |
| `Zoom | videocall-audiocall` | 547 | 0.52% |
| `Telegram | audiocall` | 520 | 0.49% |
| `Telegram | chat-videocall` | 519 | 0.49% |
| `Telegram | chat-audiocall` | 474 | 0.45% |
| `Zoom | audiocall-videocall` | 458 | 0.43% |
| `Telegram | videocall-chat` | 433 | 0.41% |
| `Telegram | videocall-audiocall` | 426 | 0.40% |
| `Slack | audiocall-videocall` | 424 | 0.40% |
| `com.zhiliaoapp.musically | <null>` | 423 | 0.40% |
| `Messenger | audiocall-videocall` | 413 | 0.39% |
| `Telegram | audiocall-chat` | 402 | 0.38% |
| `Slack | videocall-audiocall` | 394 | 0.37% |
| `Signal | audiocall-videocall` | 373 | 0.35% |
| `Trueconf | videocall-audiocall` | 371 | 0.35% |
| `com.android.networkstack.process | <null>` | 369 | 0.35% |
| `Playstore | Unknown` | 339 | 0.32% |
| `Trueconf | videocall-chat` | 333 | 0.31% |
| `Signal | chat-videocall` | 321 | 0.30% |
| `WhatsApp | videocall` | 311 | 0.29% |
| `ClashRoyale | gaming-online` | 309 | 0.29% |
| `Discord | audiocall-chat` | 298 | 0.28% |
| `Discord | chat-audiocall` | 282 | 0.27% |
| `Signal | audiocall-chat` | 280 | 0.26% |
| `Signal | videocall-chat` | 279 | 0.26% |
| `Discord | chat-videocall` | 272 | 0.26% |
| `Discord | videocall-chat` | 263 | 0.25% |
| `Slack | audiocall-chat` | 252 | 0.24% |
| `Telegram | videocall` | 251 | 0.24% |
| `Trueconf | audiocall-chat` | 247 | 0.23% |
| `Line | audiocall-videocall` | 245 | 0.23% |
| `Messenger | videocall-audiocall` | 229 | 0.22% |
| `Slack | videocall-chat` | 221 | 0.21% |
| `Telegram | chat` | 219 | 0.21% |
| `Signal | chat-audiocall` | 218 | 0.21% |
| `Line | chat-videocall` | 216 | 0.20% |
| `Line | chat-audiocall` | 205 | 0.19% |
| `Teams | audiocall-chat` | 203 | 0.19% |
| `Line | videocall-audiocall` | 200 | 0.19% |
| `Signal | videocall-audiocall` | 199 | 0.19% |
| `Signal | videocall` | 184 | 0.17% |
| `Slack | chat-audiocall` | 179 | 0.17% |
| `Slack | chat-videocall` | 179 | 0.17% |
| `com.facebook.katana | <null>` | 178 | 0.17% |
| `Skype | Unknown` | 176 | 0.17% |
| `WhatsApp | chat-videocall` | 175 | 0.16% |
| `Line | videocall-chat` | 161 | 0.15% |
| `Line | audiocall-chat` | 160 | 0.15% |
| `Teams | chat-audiocall` | 148 | 0.14% |
| `WhatsApp | chat-audiocall` | 144 | 0.14% |
| `Messenger | Unknown` | 142 | 0.13% |
| `WhatsApp | audiocall-videocall` | 113 | 0.11% |
| `Trueconf | Unknown` | 106 | 0.10% |
| `Teams | audiocall-videocall` | 100 | 0.09% |
| `Zoom | audiocall-chat` | 97 | 0.09% |
| `Skype | audiocall-chat` | 93 | 0.09% |
| `in.fulldive.shell | <null>` | 93 | 0.09% |
| `Teams | videocall-audiocall` | 92 | 0.09% |
| `system_server | <null>` | 91 | 0.09% |
| `Teams | videocall-chat` | 90 | 0.08% |
| `Zoom | chat-audiocall` | 90 | 0.08% |
| `Skype | chat-audiocall` | 87 | 0.08% |
| `Skype | videocall-chat` | 81 | 0.08% |
| `Skype | chat-videocall` | 79 | 0.07% |
| `WhatsApp | videocall-audiocall` | 79 | 0.07% |
| `com.instagram.android | <null>` | 79 | 0.07% |
| `Zoom | videocall-chat` | 76 | 0.07% |
| `Teams | chat-videocall` | 74 | 0.07% |
| `Teams | video on-demand` | 74 | 0.07% |
| `KakaoTalk | chat-videocall` | 63 | 0.06% |
| `Signal | Unknown` | 62 | 0.06% |
| `Zoom | chat-videocall` | 62 | 0.06% |
| `Meet | Unknown` | 61 | 0.06% |
| `WhatsApp | audiocall-chat` | 58 | 0.05% |
| `com.google.android.youtube | <null>` | 58 | 0.05% |
| `Zoom | Unknown` | 55 | 0.05% |
| `KakaoTalk | audiocall-videocall` | 54 | 0.05% |
| `com.google.android.contacts | <null>` | 53 | 0.05% |
| `KakaoTalk | chat-audiocall` | 52 | 0.05% |
| `Workplace | Unknown` | 50 | 0.05% |
| `JitsiMeet | videocall` | 41 | 0.04% |
| `WhatsApp | videocall-chat` | 39 | 0.04% |
| `com.cyanogenmod.updater | <null>` | 37 | 0.03% |
| `Discord | Unknown` | 30 | 0.03% |
| `com.android.webview | <null>` | 25 | 0.02% |
| `Slack | Unknown` | 21 | 0.02% |
| `com.google.android.youtube.player | <null>` | 18 | 0.02% |
| `Line | Unknown` | 15 | 0.01% |
| `com.perfectfreeapps.vrrollercoaster.app | <null>` | 15 | 0.01% |
| `com.snapchat.android | <null>` | 10 | 0.01% |
| `com.google.android.apps.turbo | <null>` | 9 | 0.01% |
| `/system/bin/adbd--root_seclabel=u | <null>` | 8 | 0.01% |
| `Telegram | Unknown` | 8 | 0.01% |
| `org.lineageos.jelly | <null>` | 8 | 0.01% |
| `android.process.media | <null>` | 3 | 0.00% |
| `org.lineageos.lineageparts | <null>` | 3 | 0.00% |
| `com.google.android.gm | <null>` | 2 | 0.00% |
| `com.google.android.googlequicksearchbox | <null>` | 2 | 0.00% |
| `<null> | <null>` | 1 | 0.00% |
| `KakaoTalk | Unknown` | 1 | 0.00% |
| `com.google.android.feedback | <null>` | 1 | 0.00% |

## 数据质量与有效数

| 项目 | 数量 | 说明 |
|---|---:|---|
| 原始 flow | 106,157 | 顶层 5 元组记录数；无解析错误 |
| 核心逐包数组完整 | 106,157 | 7 个核心数组同长且无 null，5 元组有效，至少 1 包 |
| 可直接满足现有八字段输入 | 0 | 逐包 TTL 缺失，所有 flow 均受影响 |
| BF_label 缺失 | 1 | JSON null |
| BF_activity 缺失 | 24,828 | 字段不存在；不同于字面 Unknown |
| BF_activity = Unknown | 4,008 | 有字段但活动不明确 |
| 目录名与 BF_label 不一致 | 28,835 | 以记录内标签统计；不要按目录盲目赋标签 |
| 重复 payload | 12 | 相同 flow 值字节摘要；未包含顶层 5 元组键 |
| BF_num_packets 与时间戳长度不一致 | 0 | 全量检查 |
| 时间戳非单调、IAT 非法或与相邻时间差不一致 | 0 | 容差 0.01 秒；全量检查 |
| IP 包长非法或方向值非 0/1 | 0 | 全量检查 |
| UDP flow 带 TCP_flags 数组 | 17,243 | 字段存在但 TCP 语义不适用 |

另有 12 条重复 payload；其可能只是不同 5 元组、时间/内容相同，不能直接当作应删除的完整重复记录。原始数据中没有零包 flow。

## 与现有模型输入的差异和建议

项目约定：每个样本一个双向 flow；`series` 每包含 `time`（相邻包 IAT，首包 0）、`pkt_len`（逆向为负）、`flags`（TCP 位掩码；现有预处理对 UDP 设 0）、`ttl`；flow 元属性为 `src_ip`、`dst_ip`、`src_port`、`dst_port`、`protocol`。`stateFormer/dataset.py` 用 `max_seq_len=16` 截断/补齐，并把八个数值特征 `[time,pkt_len,flags,ttl,src_port,dst_port,src_ip,dst_ip]` 分箱成离散 ID。此前实验已将真实 bin ID 从 1 开始，0 保留给起始占位/padding；不能把缺失字段写作 0。

1. 先确定 TTL 的**真实来源**。若另有对应 PCAP，可用 5 元组、时间戳与方向校验后提取逐包 TTL；没有可验证来源时，应单独设计并评估“不含 TTL”的模型/输入配置。这需要后续明确授权修改模型代码；本阶段不修改。
2. 若可取得 TTL，再设计只读、逐文件适配：解析顶层 5 元组为 flow 元属性，按 `timestamp`/`iat` 和 `packet_dir` 生成序列；核对 IP 层长度与原 Scapy `len(packet)` 的定义，再决定长度特征；TCP flags 按标准位语义映射，UDP 依现有预处理规则处理，明确这是规则值而非原始 TCP flags。
3. 先确定标签任务：应用单标签、活动单标签或应用—活动组合。用 `BF_label`/`BF_activity` 作为来源，单独处理缺失、`Unknown`、后台及系统进程标签；按文件或采集会话分组切分，避免相关流跨训练/测试集。
4. 新适配器应流式逐文件产出，保留审计计数并核验数组长度、方向/端口一致性、标签、数值范围及重复记录；不覆盖原始数据。当前训练读取端会一次性加载类别 JSON，因此正式扩容时还需评估分片读取与内存需求。

调查结论：包级序列信息较完整，但现有八字段输入的 TTL 在全部原始 flow 中缺失；在解决 TTL 来源或正式调整输入设计之前，不应启动转换或训练。
