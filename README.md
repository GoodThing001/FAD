# FAD clean mainline

这是从旧项目安全重建的独立主线。旧目录只作为迁移和审计来源；只有经过校验、可解释、可复现的内容才进入这里。

## 当前正式结论

- 研究对象：PreQ1 / RF00522 riboswitch，2000 条序列，13 个突变位点。
- 当前代理与候选闭环目标：从序列直接预测独立 `gbsa`。`label = 0.961 × Gap2 + 0.039 × gbsa` 只作为现有数据的历史计算字段；权重的优化有效性尚未验证，本阶段不用于选样或候选排序。
- 正式结果：v8 五模型秩融合，100-seed `gbsa Spearman = 0.37477`，bootstrap 95% CI `[0.36691, 0.38231]`；完整服务器checkpoint已回收。
  - vs mutation-only RF 基线（100-seed，0.365）：配对 `+0.010`，CI `[0.006, 0.015]`，66/100 seeds 胜。
  - vs 逐 seed 最优单模型：`-0.006`，44/100——融合是稳定发布配置，不宣称优于最优单模型。
- 2026-09-24 升格（100-seed 正式运行包）：
  - **历史探索：dock 四模型秩融合（303d）= 0.5523，CI [0.5456, 0.5590]**。该数字不代表当前改用 dock；当前目标仍是独立 GBSA。2026-09-30 的折内复核显示，仅靠预测 dock 修正 GBSA 无增益；真实 dock 残差的增益仍依赖候选阶段可获得 docking，详见总体方案 §9。
  - **k-NN Hamming 标签平滑：配对 Δ=+0.0191，CI [0.0146, 0.0237]，82/100 胜**（gbsa 上唯一的有效纯 ML 去噪）。
- `0.395` 是 30-seed 开发候选结果（跨模型×损失的逐 seed 最优 oracle 口径，非单一固定配置）；`0.506` 是历史固定划分离群结果，二者都不作为最终发布指标。
- 「oracle bound 0.412」是经验观察（估计方法未留痕），引用时注明其性质。
- MFE、Pre、Aft、Gap2 必须来自 NUPACK；禁止使用 RNAfold MFE 混合计算。

## 当前阶段 B：搜索接口与无交叉遗传算法已完成

七种同起点策略（随机游走、爬山、束搜索、仅变异遗传算法、迭代局部搜索、模拟退火、无交叉AdaLead改编版）已接通。最新七法30开发/100确认共2660子运行，见[七算法结果](docs/汇报/9月/七算法同起点比较与实施结果_20261005.md)。此前四法的历史比较保留：两固定诊断代理的30种子开发、100独立种子确认与预算/有限消融/重训稳定性已完成，共1940子运行；全部主比较按共同实际预测次数截面。GA优于随机游走的预测结果复现，但未显示稳定优于爬山；没有新序列真实GBSA优胜结论。代理优化本阶段暂停。旧96条为历史静态池存档，不安排回填，主动学习未选择。

- [生工团队完整总汇报与算法选择](docs/汇报/9月/GBSA优化项目总汇报_生工团队版_20261005.md)
- [无交叉GA实践操作指南](docs/项目记录/无交叉遗传算法运行与阶段B收尾操作指南_20261005.md)
- [冻结比较与本地证据](evidence/search_suite_20261005/README.md)

## 目录规则

- `data/proxy2000_v2/`：唯一标准数据入口；NUPACK补充表只含安全特征和序列连接键。
- `configs/experiments/`：实验定义；模型差异通过配置表达。
- `scripts/tools/`：保留的可运行入口。
- `runs/`：每次运行的配置、日志、指标和完成标记；不提交 Git。
- `docs/`：当前结果、迁移映射和方法说明。
- `legacy/`：只存历史索引，不让历史代码进入主线导入路径。

## 快速校验

```bash
python -m unittest discover -s tests -v
python scripts/tools/data_registry.py --data-dir data/proxy2000_v2 --output runs/data_registry.csv
```

## 运行正式基线

先做 2-seed 冒烟测试：

```bash
python scripts/run_experiment.py configs/experiments/baseline_smoke.json
```

100-seed 正式复核应放在 Linux 服务器运行：

```bash
nohup python -u scripts/run_experiment.py configs/experiments/v8_final_100seed.json > v8_final_launcher.log 2>&1 &
```

正式复核前先确认服务器负载，并将并行数控制在 4–8。

## 禁止事项

- 主线序列代理禁止把 `label`、`gbsa`、`Gap1/2/3`、`dock` 或 MMGBSA 输出作为模型输入；独立的折内 dock 辅助探索需注明候选阶段的 dock 可用性，不能混称为序列代理结果。
- 禁止将 ED 全数据目标统计特征用于正式结果。
- 禁止用单一 seed=42 宣称模型提升。
- 禁止在模型目录复制数据集或随意覆盖旧运行结果。
- 禁止把大型日志、模型权重、NUPACK源码和MD轨迹提交到主线 Git。

当前实施顺序与脚本核查见 [GBSA 代理接口与候选闭环实施方案](docs/项目记录/GBSA代理接口与候选闭环实施方案_20260930.md)；迁移状态见 [PROJECT_STATUS.md](PROJECT_STATUS.md)，历史路线判断见 [docs/项目记录/工作流决策.md](docs/项目记录/工作流决策.md)，候选计算进度见 [docs/source_evidence/候选NUPACK进度.md](docs/source_evidence/候选NUPACK进度.md)。
