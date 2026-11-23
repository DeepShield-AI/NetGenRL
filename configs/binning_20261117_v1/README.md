# 等宽、等频正式实验配置（仅准备，未启动）

输入只使用 `data-json/{iscx,ssrc,mirage,utmn,cesnet3}`，不使用其对应训练集。
每个输入各有 equal_width / equal_freq 两份配置，共10项；本轮独立准备。

共同设置：八字段 `[time,pkt_len,flags,ttl,src_port,dst_port,src_ip,dst_ip]`，
max_seq_len=16、batch_size=64、epoch=1000、checkpoint=100、expand_times=1（2倍生成量）、
preencode=true，seed=20261117。训练代码的 beta 保留0.01，模型及 loss 未改动。
seed 固定 Python/NumPy/PyTorch RNG，并为本轮固定按文件名排序的类别编号；不承诺不同硬件间逐位一致。

两种方法共用箱数上限：time=100、pkt_len=500、flags=10、ttl=60、
src_port=250、dst_port=250、src_ip=100、dst_ip=100。
等宽使用配置中的完整取值范围；等频先将数据裁剪至同一范围，再按累计计数计算线性分位点。
序列属性只统计每条 flow 的前16包；meta 属性每条 flow 统计一次；不按类别平衡分位点。
边界按 min_unit 精度舍入，去除重复边界；等频实际箱数可能减少，常量字段保留单箱。
相邻区间共享端点，编码保留现有双闭区间首次匹配规则；反分箱方式保持原实现。
数据集外范围在编码时归入首/末箱，未修改源数据。

输出隔离在 `{bins,model,result}/binning_20261117_v1/{方法}/{数据集相关文件}`。
fresh_start=true：若 bins 文件、模型目录或结果目录已经存在，入口会立即报错，避免复用旧权重或覆盖结果。
未指定 binning 配置时，driver 保持原有 ChiMerge 默认行为；全局 config.json 未更改。

以下为后续启动单项实验的用法，本次没有执行：

```bash
CUDA_VISIBLE_DEVICES=0 /data/xiaozj/anaconda3/envs/exp/bin/python -u driver.py \
  --config configs/binning_20261117_v1/equal_width_iscx.json
```

其余项选择对应 JSON；CLI 也支持 `--binning-method equal_width|equal_freq|chimerge`。
正式 GPU 分配、队列调度和日志重定向在启动时另行确定；本目录不会自动启动或创建后台队列。

验证脚本 `validation/smoke_binning.py` 固定 GPU 0，仅使用临时输入及输出，
五数据集各2类×2条 flow，每种方法训练1轮，再选模、生成和校验，正式配置文件不变。
10项 GPU 0 验证已通过，另通过100组计数分位点与 NumPy 参考对照、常量字段和空输入检查。
这是限定规模功能验证，不代表正式实验结果或生成质量结论。
