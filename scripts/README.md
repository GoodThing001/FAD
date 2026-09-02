# Script roles

| 脚本 | 角色 | 状态 |
|---|---|---|
| `run_experiment.py` | 统一运行入口和运行包生成器 | mainline |
| `tools/data_registry.py` | 数据完整性报告 | mainline |
| `tools/baseline_30seed.py` | mutation-only多seed基线 | mainline |
| `tools/phase3_fusion_v8.py` | v8融合正式复核 | mainline |
| `tools/generate_cluster_splits.py` | 聚类拆分生成 | mainline |
| `tools/baseline_cluster_split.py` | 聚类外推评估 | mainline |
| `tools/feature_selection_pipeline.py` | 9月特征选择 | exploratory |
| `tools/eval_final_combo.py` | NUPACK补充消融 | supplementary，当前缺本地NUPACK输入 |

没有出现在本表中的旧脚本不会自动进入clean mainline。

