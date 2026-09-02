# 服务器结果回收与同步清单

核心服务器结果已于2026-09-02回收。以后只增量同步缺失项，不要再次复制整个旧项目或MD轨迹目录。

## 本地已经具备

新增回收：

- `evidence/server_sync_20260902/v8_fusion_100seed/`：700行、100/100 seeds完整checkpoint及日志。
- `evidence/server_sync_20260902/v8_fusion_30seed/`：210行、30/30 seeds完整checkpoint及日志。
- `data/proxy2000_v2/nupack_features_full.csv`：2000条×1192个安全NUPACK特征。
- `environments/`：NUPACK环境记录。

旧仓中的下列检查点已经在本地核验为完整 30-seed，并回收到 `evidence/recovered_runs/`：

- `phase1_v8/all_results.csv`：19,440行，seed 1–30。
- `v10/all_results.csv`：1,080行，seed 1–30。
- `v10b/all_results.csv`：570行，seed 1–30。
- `v11/all_results.csv`：120行，seed 1–30。
- `v12/all_results.csv`：360行，seed 1–30。
- `v13/all_results.csv`：180行，seed 1–30。
- 每组对应的 `progress.json`、`run_meta.json` 和历史运行脚本快照。

这些文件如果服务器哈希相同，不需要重复下载。

## 仍需补回的内容

优先级 A（缺失但不阻塞v8复核）：

1. v8最终命名CSV和逐样本预测文件（如果服务器仍存在）；逐seed checkpoint已经完整回收。
2. 旧版特征选择声明生成的 `artifacts/feat_sel/full_30seed.csv`；缺失时不重建或伪造。
3. 当时服务器Git提交号与主训练环境freeze（NUPACK环境已经回收）。

优先级 B（补齐已完成的 NUPACK 消融证据）：

1. `artifacts/phase1_v8/rank_targets_30seed.csv`
2. `artifacts/v10/nupack_blocks_30seed.csv`
3. `artifacts/v10b/nupack_blocks_motif_split.csv`
4. `artifacts/v11/nupack_combo_30seed.csv`
5. `artifacts/v12/untested_30seed.csv`
6. `artifacts/v13/final_combo_30seed.csv`
7. 对应 `stdout.log`。本地已有 `all_results.csv`，哈希一致时最终 CSV 只用于确认发布命名和完整性。

优先级 C已经完成：历史split NUPACK表已核验并合并成按 `Sequence` 连接的安全全量特征表，不再需要重复同步三个split文件。

优先级 D（进入物理标签阶段）：

- 每个样本的逐帧 GBSA/MMGBSA 表、异常帧记录、轨迹与拓扑的索引清单。
- 先同步小型 CSV/JSON 清单；轨迹本体等确认诊断子集后再传，不要整库搬运。
- 100,000候选的 NUPACK 进度表、已完成序列数、失败列表和合并脚本。

本地已确认批次 `0000`–`0030` 完整，共31,000条且失败0。服务器只需补充批次 `0031`–`0099`、最终合并表或证明它们尚未运行；不要重复传输或重算前31批。

## 服务器上先运行

先把 `tools/server_handoff_inventory.py` 单独传到服务器，然后执行：

```bash
cd /你的/FAD项目根目录
python /tmp/server_handoff_inventory.py \
  --root . \
  --output /tmp/fad_server_handoff_manifest.csv \
  --max-hash-mib 256

git rev-parse HEAD > /tmp/fad_server_git_commit.txt 2>&1 || true
git status --short > /tmp/fad_server_git_status.txt 2>&1 || true
python -V > /tmp/fad_server_python.txt 2>&1
python -m pip freeze > /tmp/fad_server_pip_freeze.txt 2>&1
```

先把以下小文件传回本地，我即可给出精确的第二轮同步命令：

```text
/tmp/fad_server_handoff_manifest.csv
/tmp/fad_server_handoff_manifest.csv.summary.json
/tmp/fad_server_git_commit.txt
/tmp/fad_server_git_status.txt
/tmp/fad_server_python.txt
/tmp/fad_server_pip_freeze.txt
```

不要使用带 `--delete` 的同步命令。不要在核对清单前压缩或传输全部轨迹与 checkpoint。
