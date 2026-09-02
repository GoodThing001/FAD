# Migration map

本表定义旧项目内容的去向。主线材料先复制并验证；精确重复项只有在保留副本和待移动副本SHA256一致时才进入可恢复隔离区。

| 旧位置 | 分类 | 新位置/处理 |
|---|---|---|
| `data/proxy2000_v2/` | mainline | `data/proxy2000_v2/`；只保留 canonical 表和历史拆分 |
| `scripts/tools/baseline_30seed.py` | mainline | 保留为基线入口 |
| `scripts/tools/phase3_fusion_v8.py` | mainline | 保留为 v8 正式复核入口 |
| `scripts/tools/data_registry.py` | mainline | 保留为数据完整性入口 |
| `scripts/tools/feature_selection_pipeline.py` | active | 进入探索实验，完成30 seeds后再决定 |
| `scripts/tools/eval_rank_targets.py` | audit-required | rank变换读取测试标签；修复前不作正式证据 |
| `scripts/tools/eval_final_combo.py` | supplementary | 依赖尚未同步的NUPACK文件 |
| `logs/v10b`–`logs/v13` | supplementary evidence | 回收对应CSV后登记到运行包 |
| `FAD_CNN/`、RNA模型、ESO等 | historical | 只在旧仓保留；新主线写索引，不复制 |
| `artifacts/` | mixed/large | 按实验注册表挑选；禁止整目录迁移 |
| `docs/v1/` | mixed | 提升当前结论，旧版本进入文档归档 |
| NUPACK源码与安装包 | third-party | 移出主线Git，只记录版本和安装方法 |
| MD轨迹、Excel原始数据 | external/raw | 外部只读存储，登记哈希和路径 |
| `graphify-out/` | generated | 外部报告产物，不进入模型主线 |

## 已识别风险

1. 旧仓存在约 2.9 万条 Git 状态变更，不能进行批量删除或重置。
2. 服务器日志引用的多个最终CSV未同步到本地。
3. 根 README 与 v8 文档对“当前推荐模型”的表述不一致。
4. `eval_rank_targets.py` 的 rank-uniform/rank-Gaussian 变换联合排序训练和测试标签；winsor分支不受该问题影响。
5. 旧 `artifacts/` 超过 16 GB，必须按运行登记选择性回收，不能整体复制。
