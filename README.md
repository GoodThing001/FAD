# FAD clean mainline

这是从旧项目安全重建的独立主线。旧目录只作为迁移和审计来源；只有经过校验、可解释、可复现的内容才进入这里。

## 当前正式结论

- 研究对象：PreQ1 / RF00522 riboswitch，2000 条序列，13 个突变位点。
- ML 目标：从序列预测 `gbsa`；最终标签仍按 `0.961 × Gap2 + 0.039 × gbsa` 计算。
- 正式结果：v8 五模型秩融合，100-seed `gbsa Spearman = 0.37477`，bootstrap 95% CI `[0.36691, 0.38231]`；完整服务器checkpoint已回收。
- `0.395` 是 30-seed 开发候选结果；`0.506` 是历史固定划分离群结果，二者都不作为最终发布指标。
- MFE、Pre、Aft、Gap2 必须来自 NUPACK；禁止使用 RNAfold MFE 混合计算。

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

- 禁止把 `label`、`gbsa`、`Gap1/2/3`、`dock` 或 MMGBSA输出作为模型输入。
- 禁止将 ED 全数据目标统计特征用于正式结果。
- 禁止用单一 seed=42 宣称模型提升。
- 禁止在模型目录复制数据集或随意覆盖旧运行结果。
- 禁止把大型日志、模型权重、NUPACK源码和MD轨迹提交到主线 Git。

当前迁移状态见 [PROJECT_STATUS.md](PROJECT_STATUS.md)，完整路线判断见 [docs/工作流决策.md](docs/工作流决策.md)，本次同步整理见 [docs/服务器同步整理_20260902.md](docs/服务器同步整理_20260902.md)，候选计算进度见 [docs/候选NUPACK进度.md](docs/候选NUPACK进度.md)。
