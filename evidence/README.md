# Evidence policy

这里保存从旧仓恢复的关键日志、结果和当时脚本快照，用于追踪既有结论。恢复文件均登记SHA256；只有同时包含配置、数据哈希、逐seed指标和完成标记的结果才能升级为正式发布包。

`recovered_runs/` 已包含完整的 phase1、v10、v10b、v11、v12、v13逐seed结果；六组均覆盖seed 1–30并与各自进度记录的行数一致。

`server_sync_20260902/` 已补回并核验v8的30-seed与100-seed完整checkpoint、stdout和运行元数据。100-seed文件覆盖700行，进度达到100/100。服务器回收的NUPACK增强表已去除目标列后登记到 `data/proxy2000_v2/nupack_features_full.csv`。

仍未闭环：

- v8最终命名CSV本身未同步，但完整 `all_results.csv` checkpoint已经足以复核正式汇总；未发现逐样本预测文件。
- 旧版30-seed特征选择只同步到stdout，声明生成的 `full_30seed.csv` 缺失，因此不能升级为正式发布包。
- phase1、v10–v13最终命名CSV仍缺失（本地已有完整checkpoint CSV与主要stdout）。
- 逐帧GBSA与候选批次0031–0099的清单。

具体操作见 `docs/SERVER_HANDOFF.md`。
