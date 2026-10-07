# Evidence policy

这里保存从旧仓恢复的关键日志、结果和当时脚本快照，用于追踪既有结论。恢复文件均登记SHA256；只有同时包含配置、数据哈希、逐seed指标和完成标记的结果才能升级为正式发布包。

`recovered_runs/` 已包含完整的 phase1、v10、v10b、v11、v12、v13逐seed结果；六组均覆盖seed 1–30并与各自进度记录的行数一致。

`server_sync_20260902/` 已补回并核验v8的30-seed与100-seed完整checkpoint、stdout和运行元数据。100-seed文件覆盖700行，进度达到100/100。服务器回收的NUPACK增强表已去除目标列后登记到 `data/proxy2000_v2/nupack_features_full.csv`。

仍未闭环：

- v8最终命名CSV本身未同步，但完整 `all_results.csv` checkpoint已经足以复核正式汇总；未发现逐样本预测文件。
- 旧版30-seed特征选择只同步到stdout，声明生成的 `full_30seed.csv` 缺失，因此不能升级为正式发布包。
- phase1、v10–v13最终命名CSV仍缺失（本地已有完整checkpoint CSV与主要stdout）。
- 逐帧GBSA与候选批次0031–0099的清单。

具体操作见 `docs/服务器/服务器交接.md`。

2026-09/10 主线新增：`loop_20260930/` 保存固定池选样和 96 条待测交接；`search_compare_30seed/` 为 2026-10-02 **同预测上限但不同起点/实际调用数**的历史诊断；`search_compare_matched_30seed/` 保存 2026-10-05 同起点、逐种子按实际预测次数截断的 30-seed 代理诊断，原始逐条档案仍在 `runs/search_compare_matched_30seed/runs/`。三者都不是新序列真实 GBSA 已完成测量的证据。总体口径见 `docs/汇报/9月/九月工作完整汇报_20261005.md`。

2026-10-05 新增[无交叉GA与阶段B收尾证据](search_suite_20261005/README.md)：两代理四策略30开发/100确认、预算与有限消融/稳定性、100种子对接支线、独立运行示例和最终本地验收。主比较对齐共同实际预测次数；不包含新序列真实GBSA测量。


## 2026-10-05 七算法扩展

[七算法证据包](search_seven_20261005/README.md)：ILS、SA、无交叉AdaLead改编版与原四策略，30开发/100确认，2660正式子运行；主/补充预算口径分列，139本地测试与逐次审计。无新真实标签。旧[四算法包](search_suite_20261005/README.md)保留。
