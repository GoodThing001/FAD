# Data contract

`proxy2000_v2/fad_proxy2000_v2_full.csv` 是当前唯一 canonical 数据表。`train.csv`、`val.csv`、`test.csv` 仅保留用于复现历史 seed=42 拆分；正式结论必须重新执行多 seed 拆分。

## Canonical schema

| 字段 | 用途 | 可作输入？ |
|---|---|---|
| Sequence | 141 nt 序列 | 是 |
| MFE / Pre / Aft | NUPACK能量 | 仅明确允许的物理分支 |
| gbsa | ML训练目标 | 否 |
| label | 最终标签 | 否 |
| Gap1 / Gap2 / Gap3 | 标签派生/物理指标 | 否 |
| dock | 昂贵计算结果 | 否 |

所有训练脚本必须显式选择允许字段，不能通过“删除少数列”的方式构造特征。

## NUPACK补充特征

`proxy2000_v2/nupack_features_full.csv` 是服务器回收后生成的安全特征表。它以 `Sequence` 为唯一连接键，包含1192个 `nupack_` / `delta_nupack_` 特征，不包含 `label`、`gbsa`、Gap或其他目标派生列。原始1203列表经序列覆盖、缺失值和基础字段一致性检查后才生成；来源哈希与验证细节见同目录的 `.meta.json`。

正式多seed训练应按 `Sequence` 将该表连接到canonical数据，并在每个训练折内部完成特征筛选。历史 `train_nupack.csv`、`val_nupack.csv`、`test_nupack.csv` 不再作为主数据入口。
