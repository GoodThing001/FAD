# 服务器同步证据（2026-09-02）

本目录保存从 Linux 服务器回收并经本地核验的关键运行证据。

## 可作为正式证据

- `v8_fusion_100seed/`：100 个随机种子、每个种子 5 个单模型和 2 个融合方法，共 700 行。进度文件记录 `current_seed=100`、`total=700`，日志正常结束。
- `v8_fusion_30seed/`：30 个随机种子，共 210 行，作为 100-seed 运行的早期复核。

100-seed 等权秩融合的平均 gbsa Spearman 为 0.374770，bootstrap 95% CI 为 [0.366905, 0.382314]。相对固定 XGB 方法的配对差值为 +0.002452，95% CI 为 [-0.001483, 0.006524]，54/100 个种子获胜；因此不能宣称融合显著优于 XGB。

## 仅作补充证据

- `legacy_feature_selection_30seed/`：日志显示旧版 604 维特征选择完成 30 seeds，但服务器声明生成的 `artifacts/feat_sel/full_30seed.csv` 未随同步文件出现。
- `source_snapshots/feature_selection_pipeline_legacy.py`：上述旧版运行脚本快照。它与当前严格预注册脚本不同，不应替代当前正式特征选择流程。

所有回收文件的字节数和 SHA256 见 `manifest.csv`。`summary.json` 保存结构化结论。
