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
| `scripts/loop/interface.py` | GBSA 代理、预测和标签来源接口（`INTERFACE_VERSION=0.2.0`） | mainline |
| `scripts/loop/mutate.py` | 13 位点分层序列变异、WT/父序列溯源、候选池去重 | mainline（v1） |
| `scripts/loop/selection.py` | 六种批次选择策略、零配额/耗尽/方向语义 | mainline（v1） |
| `scripts/loop/surrogates.py` | `xgb` 自举集成、`rf` 诊断代理与 `fusion` 五模型原始预测均值适配器（不是 v8 秩融合）；已拟合身份含训练/权重摘要 | mainline |
| `scripts/loop/oracle.py` | `TableOracle`/`ParkOracle`（待测批次台账、回填 ID/协议/冲突校验、pass/fail 原子持久化）/`CommandOracle` | mainline |
| `scripts/loop/loop.py` | 闭环驱动：每轮唯一性、指标时间点、`PARKED`(rc 3)、检查点指纹与同一批次续跑；`--candidate-source search` 时每轮用搜索层重建候选池（搜索只预测，选样/Oracle/回填流程不变） | mainline |
| `scripts/loop/strategy_compare.py` | 同预算多策略配对比较（含配对 CI、胜率与 `completeness.json` 完整性验收） | mainline |
| `scripts/loop/search.py` | 迭代搜索层接口 `SEARCH_INTERFACE_VERSION=0.1.1`：`SearchContext/ScoredCandidate/CandidateGroup/SearchResult/SearchPolicy/SequenceScorer` + `SurrogateScorer`（模型快照缓存、硬预测预算、训练侧 OOD 标记） | mainline（v1） |
| `scripts/loop/search_archive.py` | 搜索档案：`search_snapshot.json`/`search_state.json`/`predictions.jsonl`/`search_evaluated.jsonl`/成品与 `files.sha256`；恢复指纹强制匹配 + 崩溃边界对账 | mainline（v1） |
| `scripts/loop/search_policies.py` | 七种同起点策略：随机游走、爬山、束搜索、无交叉GA、ILS、SA、无交叉AdaLead改编版；另保留历史分层随机覆盖策略 | mainline（v1） |
| `scripts/loop/search_run.py` | 独立搜索入口（`search_*.json` 配置）：只预测、不查询标签、不写 PARKED；`--model-seed` 与 `--seed` 分离（同一划分换代理种子） | mainline（v1） |
| `scripts/loop/search_compare.py` | 历史比较入口，支持三策略及新增 GA（30 seeds：最优预测值/前 50 多样性/OOD 与训练邻近/可行率/停机分布；跨模型排序稳定性 + 重训一致性 arm B；`completeness.json` 缺行即非零退出） | mainline（v1） |
| `scripts/loop/search_suite.py` | 10 月 5 日冻结四/七算法套件（显式 --policies）：两代理共享拟合、同起点、共同实际预测前缀；预算/稳定性/消融/100种子确认 | mainline |
| `tools/sync_remote.py` | 只推送指定路径到服务器 / 执行远程命令 | 运维 |
| `tools/loop_resume_probe.py` | 服务器端“停机→部分回填→续跑幂等”端到端探针 | 验证 |
| `tools/candidate_pool_report.py` | 候选池报告（分层容量、去重、Hamming 几何、OOD 比例；不读标签） | 验证 |
| `tools/verify_loop_run.py` | 运行包独立复核（§11.1 不变量：批次唯一/增量/时间点/σ/待回填行） | 验证 |
| `tools/launch_loop_jobs.sh` | 服务器礼貌投递脚本（nice 19 / 2 线程 / nohup） | 运维 |
| `tools/check_doc_links.py` | 本地 Markdown 链接离线检查（跳过 http/锚点） | 验证 |
| `reproduction/dock_residual_screen.py` | dock→GBSA 折内残差复核 | 探索性 30-seed；预测 dock 无增益，真实 dock 分支依赖候选 docking 可用性 |
| `reproduction/eval_sigma_calibration.py` | XGB 自举集成 σ 校准研究：σ 是否分层 |误差|、LCB 是否优于纯 greedy（top-5%/10% 召回）、打乱标签负对照；两档 n_train（1600/400）30-seed；uncertainty 采集的放行门槛 | 探索性（方案 §5 门槛） |
| `reproduction/active_learning_strong.py` / `active_learning_deep.py` | 历史主动学习比较 | 已标记实现缺陷，不作为正式策略优劣证据 |
| `reproduction/rank_candidates_demo.py` | 历史三秩候选演示 | 不符合现行 GBSA-only 目标，不接入主闭环 |

现行目标与修复顺序见 [GBSA 代理接口与候选闭环实施方案](../docs/项目记录/GBSA代理接口与候选闭环实施方案_20260930.md)，接口与用法说明见 [候选闭环框架](../docs/论文改进/候选闭环框架_20260930.md)。没有出现在本表中的旧脚本不会自动进入 clean mainline。

## 闭环运行（scripts/loop）

```bash
# 回顾式冒烟（读表内标签，只验证机器可跑通，不是选样效果的证据）
python scripts/run_experiment.py configs/experiments/loop_smoke.json

# 前瞻式交接：生成新候选→导出待测批次→PARKED(rc 3)，等真实 GBSA
python scripts/run_experiment.py configs/experiments/loop_prospective_park.json

# 真实待测批次（96 条=1 板，seed 101；协议串确认后回填）
python scripts/run_experiment.py configs/experiments/loop_handoff_96.json

# 回填后安全续跑（推荐：复用原 run 目录，loop 强制核对检查点指纹）
python scripts/run_experiment.py --resume-run <run_id> --ingest <labels.csv>

# 受控真实回填前的协议钉定（审计记录 protocol_changes.jsonl；勿手改旧运行配置）
python scripts/run_experiment.py --resume-run <run_id> --pin-protocol <协议串>
python scripts/run_experiment.py --resume-run <run_id> --ingest <labels.csv>

# 等价底层命令（同一检查点、同一 batch_id、幂等）
python scripts/loop/loop.py --mode prospective --resume \
  --log-dir runs/<run_id>/checkpoints --ingest <labels.csv> \
  --output runs/<run_id>/resume_metrics.csv --candidate-source mutate \
  --rounds 3 --budget 50 --n-init 400 --n-holdout 300 --pool-size 500 \
  --surrogate xgb --oracle park
```

契约测试：`python -m unittest discover -s tests -p "test_*.py"`（107 项是 10 月 2 日历史计数：原 70 项 + 搜索层 26 项 + 搜索闭环集成 5 项 + 比较驱动 6 项。搜索层覆盖最小化方向/批量顺序/13 位点约束/去重/39 邻居上限/硬预算/同 seed 可复现/模型缓存隔离/边界恢复逐位一致/NaN 与耗尽处理；OOD 辅助与 `loop.py` 实现逐值等价；集成覆盖前瞻式限定/停车流与档案/续跑幂等/配置变更拒续/两轮回填重训与新池；比较驱动覆盖配对 Δ 与 bootstrap CI/胜平负/Jaccard/完整性缺行检测/`--model-seed` 与划分分离）。

运行包复核：`python tools/verify_loop_run.py runs/<run_id>` —— 逐条检查标记与退出码、指纹、批次唯一性与预算、跨轮不重叠、`n_labeled` 增量、指标时间点、σ 合法性、交接清单、OOD 分层与台账待回填/失败行；DONE 运行另核 labeled 行都有 `measured_gbsa` 且 `state.status=retrained`。

比较包派生汇总重算（只重算不改原始运行）：`python tools/recompute_compare_summary.py evidence/loop_20260930/strategy_compare_100seed`。

同预算策略比较（回顾式代理表，可与 random 配对）：

```bash
python scripts/run_experiment.py configs/experiments/loop_strategy_compare_3seed_smoke.json  # 冒烟
python scripts/run_experiment.py configs/experiments/loop_strategy_compare_30seed.json       # 开发尺度
```

## 迭代突变搜索（scripts/loop/search*，当前接口 v0.1.1）

2026-10-05 已支持 `mutation_only_ga`、`iterated_local_search`、`simulated_annealing`、`mutation_only_adalead`；均无交叉。新增三法固定参数、全部表格及可重复实践命令见[七算法实施结果](../docs/汇报/9月/七算法同起点比较与实施结果_20261005.md)。已有七法套件目录需先核验或复制配置另换 out-dir，不能直接写回旧包。
当前七算法的正式比较入口为 `search_suite.py`，共享起点与拟合评分器，主算法比较的最优值、多样性、训练距离与 OOD 按共同实际预测前缀计算；重训稳定性使用各次完整停止档案；
下方原 0.1.0 配置与三策略描述保留作历史运行说明。
当前代理优化可暂停；当前不安排真实 GBSA 回填或主动学习。

```powershell
python scripts/run_experiment.py configs/experiments/search_mutation_only_ga_20261005.json
python scripts/run_experiment.py configs/experiments/search_suite_development_30seed_20261005.json
python scripts/run_experiment.py configs/experiments/search_suite_confirmation_100seed_20261005.json
python tools/verify_search_suite.py runs/search_suite_confirmation_20261005
```

详细实践与参数：[无交叉遗传算法运行指南](../docs/项目记录/无交叉遗传算法运行与阶段B收尾操作指南_20261005.md)。

在「静态候选池选样」之外新增**多代突变搜索层**：内循环只花计算预算（选父代→突变→`Surrogate.predict`→留种），不调用 Oracle、不写真实标签；外循环仍走现有选样/`PARKED`/回填。设计文档：
[docs/项目记录/突变序列迭代搜索算法与代理接口实施方案_20261002.md](../docs/项目记录/突变序列迭代搜索算法与代理接口实施方案_20261002.md)。

```bash
# 256 级工程冒烟（三种同预算策略；产出运行包 + 搜索档案）
python scripts/run_experiment.py configs/experiments/search_smoke_random.json
python scripts/run_experiment.py configs/experiments/search_smoke_hillclimb.json
python scripts/run_experiment.py configs/experiments/search_smoke_beam.json

# 8192 开发预算束搜索（文档 §4.1 的起点参数：beam 32×32、80/20 一跳/两跳、10% 全局重启）
python scripts/run_experiment.py configs/experiments/search_dev_beam_8192.json

# 阶段 B 同预算比较：3-seed 冒烟 / 30-seed 开发（服务器 nice 19）
python scripts/run_experiment.py configs/experiments/search_compare_3seed_smoke.json
python scripts/run_experiment.py configs/experiments/search_compare_30seed.json
```

搜索档案在 `runs/<run_id>/checkpoints/`（独立新入口另存 `fitted_surrogate.pkl`）：`search_snapshot.json`（数据/模型/约束/策略指纹，恢复时强制匹配）、`search_state.json`（每代检查点：父代、RNG 状态、预算计数）、`search_evaluated.jsonl`（全部评分含被淘汰子代）、`search_candidates.csv`/`search_generations.csv`/`search_groups.json`/`search_summary.json`/`files.sha256`。同上下文重跑即续跑（崩溃边界对账到已提交行），完成后重跑幂等返回同一结果。GA 与批量 BO 不在 10 月 2 日第一版范围；10 月 5 日无交叉 GA 已接通，批量 BO 仍未启用；本地无 xgboost 时回退 `sklearn_gbm`（`model_id` 记录后端，不得与服务器后端混比）。

## 搜索层接入闭环（步骤 5）

`--candidate-source search`（仅前瞻式，回溯式直接报错：搜索候选在标准表之外，TableOracle 无标签可揭示）：

```bash
python scripts/run_experiment.py configs/experiments/loop_search_prospective_park.json
```

每轮流程：当前已标注集 → 拟合代理 → 搜索层重建候选池（`round_NNN_search/` 档案，恢复时按已标注集摘要复用/重跑，确定性一致）→ 既有 `select_batch`/Oracle → 批次导出 → `PARKED`(rc 3) → `--ingest` 回填 → 重训 → 下一轮新搜索。搜索的排除集 = 完整标准表 + 已标注/已选/待回填/失败 + 历史批次全部序列；搜索 `seed = loop seed + round`，全部搜索超参写入配置并进入恢复指纹。契约测试见 `tests/test_search_loop_integration.py`（5 项：前瞻式限定、停车流与档案、续跑幂等、配置变更拒续、两轮回填重训与新池）。
