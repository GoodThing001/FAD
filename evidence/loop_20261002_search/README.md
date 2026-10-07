# evidence/loop_20261002_search — 迭代搜索层服务器证据（8192 开发吞吐）

来源：服务器 `runs/srv_search_dev_beam_8192_20261002_014350`（`configs/experiments/search_dev_beam_8192.json`，nice 19 / 2 线程，`tools/launch_loop_jobs.sh search_8192` 投递，2026-10-02）。

| 文件 | 说明 |
|---|---|
| `metrics.csv` | 1 行：policy=multi_start_beam、seed 1、n_unique_predicted=5598、best_pred_gbsa=5.83842594、stop_reason=stagnation、n_groups=2、model_id=xgb_boot5_xgboost、seconds=108.385 |
| `run_manifest.json` | 服务器运行包回执（数据 SHA256、环境、源码清单、DONE 语义） |
| `DONE` | 运行完成标记 |
| `search_summary.json` / `search_snapshot.json` / `search_generations.csv` / `search_groups.json` / `files.sha256` | 搜索档案成品（`search_candidates.csv`/`search_evaluated.jsonl` 体积较大未回收，完整档案保留在服务器运行目录，哈希见 `files.sha256`） |

**结论边界**：本运行只证明该规模下搜索层吞吐与档案完整性（约 52 预测/秒，全预算 8192 预计 2.5–3 分钟），`pred_gbsa` 的下降**仅为计算过程观察，不构成发现效果证据**（方案 §5 阶段 A）。三种策略的同预算配对开发比较（阶段 B，30 seeds）与真实 GBSA 决策性比较（阶段 C）均未做。
