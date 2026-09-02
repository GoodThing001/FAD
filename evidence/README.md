# Evidence policy

这里保存从旧仓恢复的关键日志、结果和当时脚本快照，用于追踪既有结论。恢复文件均登记SHA256；只有同时包含配置、数据哈希、逐seed指标和完成标记的结果才能升级为正式发布包。

`recovered_runs/` 已包含完整的 phase1、v10、v10b、v11、v12、v13逐seed结果；六组均覆盖seed 1–30并与各自进度记录的行数一致。

当前仍需从Linux服务器回收：

- v8 100-seed正式发布包及预测；
- phase1、v10–v13最终命名CSV和stdout（本地已有完整checkpoint CSV）；
- NUPACK增强数据、配置和环境信息；
- 逐帧GBSA与候选批次0031–0099的清单。

具体操作见 `docs/SERVER_HANDOFF.md`。
