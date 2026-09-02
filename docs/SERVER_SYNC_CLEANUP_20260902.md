# 服务器同步整理记录（2026-09-02）

## 主项目接收内容

- v8 30-seed完整checkpoint：210行，seed 1–30。
- v8 100-seed完整checkpoint：700行，seed 1–100。
- 100-seed等权秩融合：平均gbsa Spearman 0.374770，bootstrap 95% CI [0.366905, 0.382314]。
- NUPACK增强数据：从服务器原始1203列表生成2000×1193安全表，其中1列为唯一键 `Sequence`，其余1192列均为NUPACK特征。
- NUPACK Python 3.9环境的YAML、explicit和pip freeze记录。
- 旧版604维特征选择stdout与当时脚本快照；由于缺失逐seed CSV，只作补充证据。

## 泄漏与一致性处理

服务器原始 `full_nupack.csv` 的2000条序列与canonical数据一一对应，无缺失。基础数值字段按序列核验一致；`Gap3`最大文本浮点差为3.55e-15，在1e-12容差内一致。

主项目中的 `nupack_features_full.csv` 已主动移除 `Full`、MFE/Pre/Aft、Gap1/2/3、dock、gbsa和label，只保留 `Sequence` 与前缀为 `nupack_` 或 `delta_nupack_` 的特征，防止训练脚本误读目标列。

## 可用冷归档

旧工作区的 `archived/server_sync_20260902/` 保留1,531个内容文件、1,184,423,603字节（约1.10 GiB），包括原始Excel、3K1V PDB、RF00522/Rfam、论文与研究文档、结构结果、早期基线审计，以及RegularizedLinear/SVR/KmerTree的源码、配置和轻量结果。完整SHA256见该目录的 `ARCHIVE_MANIFEST.csv`。可再生 `.joblib`、ZIP运行包、Python缓存和NUPACK源码/构建目录未留在冷归档。

## 待永久删除隔离区

以下内容已从旧工作区移入：

`D:\PyCharm_\FAD_quarantine_20260902\pending_delete_server_sync_20260902`

隔离区最终共27,326个文件、14,908,744,165字节（约13.885 GiB）。其中第一阶段大分组如下：

| 分组 | 文件数 | 字节数 | 原因 |
|---|---:|---:|---|
| artifacts | 17,434 | 10,658,237,131 | 旧RNAfold、固定seed、已关闭路线和可再生训练产物 |
| logs | 85 | 18,433,160 | 正式日志已迁入证据区，其余未注册 |
| data | 279 | 309,775,962 | canonical重复表、历史拆分和派生缓存 |
| FAD_clean | 257 | 40,177,961 | 独立主项目的旧内嵌副本 |
| 临时整理区 | 19 | 24,312,179 | 已完成迁移的中间文件 |
| 可再生joblib | 50 | 105,449,888 | 线性/SVR模型checkpoint |

第二阶段又加入9,202个文件、3,752,357,884字节：KmerTree ZIP与joblib、NUPACK第三方源码/构建目录、Graphify输出、编辑器元数据和Python缓存。详细分组与精确字节数见旧归档中的 `PENDING_DELETE_SUMMARY.json`。

本批尚未永久删除这些文件，因此磁盘空间尚未释放；优点是旧工作区已经降噪且所有动作可恢复。永久删除须再次明确确认该隔离目录。

## 验证

- 服务器证据清单10/10文件字节数和SHA256一致。
- NUPACK安全表SHA256：`eff085664f4377ccba8f8d0e1cd4842bd8c24dc5fa5e56f476b28e736bd8a32d`。
- 主项目8项单元测试全部通过。
- 旧项目根目录最终只保留 `.git`、`.gitignore`、入口 `README/AGENTS` 与 `archived/`。
