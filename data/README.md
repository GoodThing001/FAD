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

