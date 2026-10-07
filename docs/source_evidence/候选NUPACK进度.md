# 100,000候选 NUPACK 进度核验

核验时间：2026-09-02。

本地历史目录 `archived/FAD_ThirdStage/data/nupack_batches/` 中存在批次 `0000`–`0030`：

- 31个CSV批次，每批1,000行；
- 合计31,000行、31,000条唯一序列，跨批次重复0；
- 31份summary中的 `nupack_failures` 合计0；
- `structure_failures` 合计0；
- `full_sequence_mfe` 缺失0；
- `nupack_mfe_structure` 缺失0；
- 全部31,000条记录使用NUPACK后端；
- 每个CSV含40个业务字段。

因此旧记录“约完成31,000/100,000”可以精确修正为 **31,000/100,000（31%）**。本地缺少批次 `0031`–`0099`，共69,000条；服务器恢复后应先通过清单确认这些批次是否已经生成，不要直接重算前31批。

这些批次只用于候选阶段的NUPACK计算进度，不是GBSA训练标签，也不能替代新增MD/MMGBSA数据。
