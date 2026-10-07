# evidence/loop_20260930 — 闭环接口（A/B/C）证据清单

本目录记录 2026-09-30 冻结 GBSA 代理↔下游接口（`INTERFACE_VERSION=0.2.0`）后的可复算证据。
`runs/` 不入库，因此这里保存关键的服务器产物副本（小文件）；本地与服务器的完整运行目录分别位于
`runs/local_*` 与服务器 `/home/hzeng/project/FAD_CLEAN/runs/`。

| 文件 | 来源 | 说明 |
|---|---|---|
| `candidate_pool_report.json` | `python tools/candidate_pool_report.py --pool-size 2000 --training-size 400 --seed 1`（本地） | C 包候选池报告：分层容量/填充、与标准表及训练集去重、Hamming 几何、OOD 比例。**结论**：测量库位于 `n_from_wt` 4–13 的壳；同范围生成 2000 条 → 1998 保留（2 条与标准表重复）、OOD=0；默认 1–5 突变生成时 OOD=51% |
| `server_smoke/DONE`、`metrics.csv`、`ledger.csv`、`round_00{1,2}_batch.csv`、`progress.json`、`run_manifest.json` | 服务器 `runs/srv_loop_smoke`（`configs/experiments/loop_smoke.json`） | A 包回顾式 2 轮冒烟：rc 0、`DONE`；holdout Spearman 0.1726 → 0.1849 → 0.3201；每轮 50 条新标签、跨轮无重复。**仅证明机器跑通** |
| `server_park/PARKED`、`park_metrics.csv`、`ledger.csv`、`parked_batch_round_001.csv`、`park_progress.json`、`run_manifest.json` | 服务器 `runs/srv_loop_park_v2_20260930_135716`（`configs/experiments/loop_prospective_park.json`） | B 包前瞻式交接：生成 50 条新候选（`n_from_wt` 4–13）、导出批次、rc 3、`PARKED`（`batch_id=9e3549aa1cc7`、`n_awaiting=50`）；运行器按 `expect_marker=PARKED`/`expect_exit_code=3` 验收 |
| `server_park/resume_probe_metrics.csv` | `python tools/loop_resume_probe.py runs/srv_loop_park_v2_20260930_135716` | 停机→部分回填→续跑：`awaiting 50→45→45`、`labeled 400→405→405`、同一 `batch_id`、无新批次目录、第二次续跑 `duplicate=5` 且指标日志不增长（`tools/loop_resume_probe.py` 打印 `[ok]`） |
| `strategy_compare_3seed/` | `configs/experiments/loop_strategy_compare_3seed_smoke.json`（本地，`DONE`） | D 包驱动冒烟：`runs.csv`（逐 seed/策略/轮次）、`summary.json`（配对 Δregret、Δholdout-Spearman 与胜率）、每 (seed,策略) 的 `metrics.csv` + `ledger.csv` + 逐轮批次 CSV（为控制体积未收录各次 `state.json`，可用同一配置重跑得到）。**3 seeds 不能用于排序策略** |
| `strategy_compare_30seed/` | `configs/experiments/loop_strategy_compare_30seed.json`（服务器，`nice -n 19`，2 线程，约 76 分钟，`DONE`） | 30-seed 同预算配对比较（v1 包）：`runs.csv`（600 行）、`summary.json`、`comparison_README.txt`。**结论**：greedy 末轮 regret 0.451 vs random 2.130（配对 Δ −1.679，CI [−2.637,−0.785]；holdout Spearman +0.031，23/30）；maxmin_coverage 与 random 无差异；greedy 仅在 15/30 seed 更好、中位差≈0，优势在尾部。解释见 `docs/论文改进/策略对比_20260930.md`。**⚠ 本包生成于台账索引修复前：`pred_gbsa/best_pred_in_batch` 作废；发现指标有效。修复后的 v2 重跑进行中** |
| `server_strategy_compare/{DONE,metrics.csv,run_manifest.json}` | 服务器 `runs/srv_strategy_compare_30seed_20260930_135716` | v1 运行包回执（数据哈希、环境、脚本清单、`DONE` 标记） |
| `strategy_compare_30seed_v2/` | `configs/experiments/loop_strategy_compare_30seed_v2.json`（服务器 `srv_strategy_compare_30seed_v2_20260930_182900`，`DONE`） | §7 修复后的 30-seed 重跑：`completeness.json` ok=true（120 子运行/600 行）、`runs.csv`、`summary.json`。发现指标与 v1 逐位一致；**该包的 `pred_gbsa/ood_flag` 台账可信，引用预测台账/OOD 分层一律用本包** |
| `server_strategy_compare_v2/{DONE,metrics.csv,run_manifest.json}` | 服务器 v2 运行目录 | v2 运行包回执 |
| `strategy_compare_100seed/` | `configs/experiments/loop_strategy_compare_100seed.json`（服务器 `srv_strategy_compare_100seed_20260930_201309`，`DONE`） | 预注册的 100-seed 发布候选确认（seeds 1001–1100，400 子运行/2000 行）：`runs.csv`（未改动）、`summary.json`（**2026-10-01 复核后重算**：Δregret CI/中位数/Q25-Q75/**50 胜/37 平/13 负**/**最差 10% +1.763**/**改善最大 10 个 −8.41（占均值差 ≈44.8%）**/Δtop5/Δrecall）、`completeness.json`（ok=true，`observed_runs=400` 语义已修正）、`ood_strata_aggregate.csv`、`metrics_recomputed.csv`、`RECOMPUTED.md`（重算说明：仅尾部标签与子运行数语义改变，主判据与其余指标逐位一致）。判读见 `docs/项目记录/发布候选预注册_100seed_20260930.md` §5 |
| `server_strategy_compare_100seed/{DONE,metrics.csv,run_manifest.json}` | 服务器 100-seed 运行目录 | 100-seed 运行包回执（`metrics.csv` 为该次运行的原始输出，其 `worst10pct_delta_regret_mean` 列是标反的旧值，以 `strategy_compare_100seed/metrics_recomputed.csv` 为准） |
| `handoff_96/` | 服务器 `runs/srv_handoff_20260930_182400`（`configs/experiments/loop_handoff_96.json`） | **真实待测批次**：96 条候选（1 板，seed 101，greedy，`n_from_wt` 5–13）→ `PARKED`(rc 3)；`round_001_batch.csv`（`batch_id/candidate_id/sequence/pred_gbsa/ood_flag/protocol_id`）、`round_001_batch_manifest.json`（模型/协议/数据指纹）、`fingerprint.json`（含 `excluded_canonical_table=2000`）、`round_001_ood_strata.csv`、`ledger.csv`。`tools/verify_loop_run.py` 复核 `[ok]`。**协议串为占位符 `gbsa_pending_protocol`：受控回填前须与团队确认协议并走 `--resume-run … --pin-protocol <协议串>` 审计流程（写 `protocol_changes.jsonl`，勿手改旧运行配置）** |

复现命令与接口细节见 `docs/论文改进/候选闭环框架_20260930.md`；验收项与状态见
`docs/项目记录/GBSA代理接口与候选闭环实施方案_20260930.md` §11。

测试：`python -m unittest discover -s tests -p "test_*.py"` → 65 项通过（闭环契约 54 项，含 P0/P1 全部新用例、附加发现指标、**全批回填→重训→DONE、混合 pass-fail、协议钉定、尾部统计方向**用例）。
运行包复核（修复后的本地重跑 + 服务器交接批次，经扩展后的 `tools/verify_loop_run.py`——指纹/清单/OOD 分层/陈旧标记逐项检查）：

```bash
python tools/verify_loop_run.py runs/verify_smoke_v2_20260930  # [ok] DONE，2 批 100 条台账
python tools/verify_loop_run.py runs/verify_park_v2_20260930   # [ok] PARKED(rc3)，1 批 50 条待回填
# 服务器：runs/srv_handoff_20260930_182400 → [ok] PARKED，96 条待回填，指纹/清单/分层齐全
```

两台机器的后端差异被如实记录：本地无 xgboost 时回退 `sklearn_gbm`（`model_id=xgb_boot5_sklearn_gbm`），服务器记录 `xgb_boot10_xgboost`；`feature_version=seq303_screen100`。§7 修复状态逐行见方案文档 §7.1。
