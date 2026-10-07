# 同起点、按实际预测次数对齐的搜索比较（2026-10-05）

来源：`runs/search_matched_30seed_final_20261005`（`DONE`；复用同一次 90 个原始子运行，按最终汇总代码重算并逐位一致），配置 `configs/experiments/search_compare_matched_30seed.json`；子运行档案位于 `runs/search_compare_matched_30seed/runs/`。本目录只保存紧凑证据副本：`config.json`、`run_manifest.json`、`DONE`、`runs.csv`、`aligned_runs.csv`、`summary.json`、`completeness.json`、`files.sha256`。原始子运行有逐条 `search_evaluated.jsonl`、快照与分组文件，未复制到本目录。

- canonical 数据 SHA256：`0876C2B25A5571B834EBAC2C0C0F7671E023F6028688E90CCBA884A75FD4322A`；与 `data/registry.csv` 相符。
- 30 seeds（301–330），每 seed 三策略共用 300 条训练数据、同一代理快照、同一组 16 条初始种子、`n_from_wt=4–13`；上限 1024 唯一预测。局部运行后端 `xgb_boot5_sklearn_gbm`。
- 从逐条档案核对初始种子 ID/序列/顺序，以及数据、训练集、排除集、约束与模型快照一致；每 seed 取三策略最少的**实际唯一预测次数**，在各档案前缀内取可导出候选的最低预测 GBSA。共同次数 1021–1024；完整性 90/90，0 失败。
- 平均最低预测 GBSA：匹配种子随机游走 8.4764，爬山 4.7696，束搜索 7.1431。相对随机游走配对差：爬山 −3.7068 [−4.3917, −3.0110]（29/1/0），束 −1.3334 [−1.7765, −0.9080]（26/1/3）。束减爬山 +2.3734 [1.8404, 2.8776]。
- **仅是代理预测的开发诊断**。同一代理既指导搜索又给终点打分，不能证明新序列真实 GBSA 更低。旧服务器 8192 上限比较使用不同后端、不同起点和不同实际预测次数，不能直接拼成算法胜负。真实 GBSA 多独立批次比较仍待协议与预算。

详细解释：[九月工作完整汇报 §9.3](../../docs/汇报/9月/九月工作完整汇报_20261005.md)。
