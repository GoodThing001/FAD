# 2026-10-05 无交叉 GA 与阶段 B 收尾证据

## 范围与读法

GA（genetic algorithm，遗传算法）本轮只有单父本变异，无交叉。目标为独立 GBSA（广义玻恩/表面积计算值）；所有新候选的分数只是预测。没有新 docking（分子对接）、MD（分子动力学）或 GBSA 计算，没有真实回填，也未决定主动学习。

| 包 | 原始本地目录 / 运行器编号 | 数量 | 主要文件 |
| --- | --- | --- | --- |
| [开发](development/summary.json) | `runs/search_suite_development_20261005` / `local_search_suite_development_30seed_20261005` | 30 种子401–430；1140 子运行 | `runs.csv`、`aligned.csv`、`stability.csv`、`ablations.csv`、`fits.csv`、完整性和独立产物审计 |
| [确认](confirmation/summary.json) | `runs/search_suite_confirmation_20261005` / `local_search_suite_confirmation_100seed_20261005` | 100 种子2001–2100；800 子运行 | 同起点、同拟合代理、主指标共同实际预测前缀；参数冻结 |
| [对接支线](dock_confirmation/summary.json) | `runs/local_dock_residual_confirmation_100seed_20261005` | 100 种子3001–3100；100 配对行 | 序列基础、预测对接残差、实际对接残差及三档预筛；原方法种子1逐值回归差为0 |
| [独立GA示例](demo/metrics.csv) | `runs/local_mutation_only_ga_demo_20261005` | 916唯一预测、908预测候选、8代、2组 | 两组代表、前20示例、完整分组与代际记录；不计入确认统计 |
| [最终验证](verification/validation_summary.json) | `runs/verification_search_20261005` | 见实测记录 | 本地测试、编译、链接、数值与结构审计 |

四种搜索算法为同起点随机游走、多起点爬山、多路径束搜索、无交叉 GA。开发预算上限为1024/4096/8192，确认为1024。每种子各代理只训练一次，均为300条训练的仅序列诊断模型；本机`xgb`命令实际使用scikit-learn梯度提升回退，另一个是300树随机森林RF（random forest），两者都不是v8发布秩融合。模型排序能力与候选新标签证据分开报告。

`aligned.csv`记录共同实际次数截面，低预测值不等于低真实GBSA；`runs.csv`记录完整运行实际用量与时间。`stability.csv`在各次搜索停止后的完整前50名单上统计模型变化及其引起的路径/早停敏感性；没有按重训两个运行再对齐次数。所有30/100是随机种子，不是新增生物学实验。

## 原始档案、源码与可复现边界

开发逐条评分原始档案在`runs/search_suite_development_20261005/subruns/`（1140份，2438283条），确认在`runs/search_suite_confirmation_20261005/subruns/`（800份，812265条）；不复制大体积运行目录进Git。本目录保留可共享汇总与审计记录。独立GA的完整候选、预测日志和模型权重在原始运行的`checkpoints/`。`demo/files.sha256`指向原始完整检查点中的文件，本小包只有摘要和示例子集，不能把它当完整检查点恢复。

开发之后、确认之前修复快照中标签/骨架/可变位置的实际落盘与严格核对、源码摘要核对，以及旧静态池不生效参数的恢复兼容。未改模型、搜索提出/选择决策或冻结参数。[开发源码](source_development/manifest.json)与[确认源码](source_confirmation/manifest.json)分别留存。确认启动后，独立入口`search_run.py`又把模型权重保存移到档案校验成功之后，避免拒绝续跑覆盖原权重；该入口未参与搜索套件计算，套件数值无影响，新增回归测试覆盖。[最终源码](source_final/manifest.json)用于当前实践与验证。

运行器的`run_manifest.json`含配置来源、启动时全项目源码摘要、数据登记匹配、Python和库版本；启动快照含未调用脚本的当时版本，不等于它们都参与计算。正式套件所调用核心代码另由`suite_config.json`及各搜索源码指纹绑定。旧搜索未改写，不能编辑指纹强行跨版本续跑。环境与模型训练参数变化需另建运行，不将本机回退与服务器XGBoost结果合并。

本轮只作本地验证，没有声称服务器同步或双端复验。旧96条静态池存档未被改写或回填。数据登记SHA256为`0876C2B25A5571B834EBAC2C0C0F7671E023F6028688E90CCBA884A75FD4322A`。

## 参考文档与重跑

- [完整生工总汇报](../../docs/汇报/9月/GBSA优化项目总汇报_生工团队版_20261005.md)
- [启动前冻结与工程修补记录](../../docs/项目记录/无交叉遗传算法与搜索比较冻结方案_20261005.md)
- [实践操作指南](../../docs/项目记录/无交叉遗传算法运行与阶段B收尾操作指南_20261005.md)

正式实验通过`configs/experiments/`与`scripts/run_experiment.py`执行；审计用`tools/verify_search_suite.py`。重跑应另设运行编号和输出目录；原包保留不覆盖。自举区间逐项报告，未多重比较校正；结果用于本冻结代理的计算诊断，不决定真实算法优胜。
