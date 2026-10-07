# FAD clean mainline instructions

## Scope

This directory is the curated mainline. Do not import code or data from the parent legacy workspace at runtime. Legacy files may be inspected only as migration evidence.

## Data

- Canonical dataset: `data/proxy2000_v2/fad_proxy2000_v2_full.csv`.
- Verify its SHA256 against `data/registry.csv` before formal experiments.
- Historical train/val/test files reproduce seed=42 only; never use them as the sole publication evaluation.
- MFE, Pre, Aft, and Gap2 must use NUPACK. Never substitute RNAfold MFE.

## Leakage constraints

Never use these as mainline GBSA surrogate inputs: `label`, `total_energy_change`, `r_psp_MMGBSA_dG_Bind`, `dock`, `gbsa`, `Gap1`, `Gap2`, `Gap3`. The exploratory dock correction below is a separate, explicitly labeled experiment; target and derived-label leakage remain prohibited.

`r_i_docking_score` is unavailable for scalable candidate generation and is not a mainline feature. Precomputed full-data ED target statistics are invalid for formal evaluation.

The current optimization and surrogate prediction target is standalone `gbsa`. The historical dock-target experiment (0.5523 four-model rank fusion, 100-seed, 95% CI [0.5456, 0.5590], `runs/dock_fusion_100seed`) does not change that decision. Dock must not be an input to the mainline sequence-only GBSA surrogate. Exploratory fold-clean dock-assisted GBSA correction or prefiltering is permitted under the 2026-09-30 user request only if dock availability at candidate time is stated and results are compared against GBSA-only on paired splits. Prior global fusion and rank replacement failed; a 30-seed predicted-dock residual screen showed no gain, while an observed-dock arm remains an availability upper bound. Sindt et al. JCIM 2025 is a small-molecule cautionary analogue, not direct proof for this RNA dataset. See `docs/项目记录/GBSA代理接口与候选闭环实施方案_20260930.md` §9.

## Evaluation

- Use at least 30 seeds for model development claims and 100 seeds for release claims.
- Perform feature selection inside the training fold.
- Use `rankdata + pearsonr` for Spearman on Windows.
- Treat 0.506 as a historical fixed-split optimization result, not a multi-seed release baseline. Exact 1800/200 replay on 2026-10-06 reproduced 0.5060286507. Frozen five-recipe comparisons completed 30 development and 100 confirmation seeds: full1087 mean 0.343000 versus sequence303 0.378002, paired -0.035002, CI [-0.042662, -0.027231], 20/0/80 wins/ties/losses. This validates the historical fixed-split improvement but does not generalize its advantage. These 1800/200 GBR runs do not replace the 1600/400 v8 release reference; see evidence/supplement_20261006/.
- Release reference: v8 rank fusion, 100-seed gbsa Spearman 0.375, 95% CI [0.367, 0.382].
  - vs mutation-only RF baseline (100-seed, 0.365): paired +0.010, CI [0.006, 0.015], 66/100 seeds win.
  - vs per-seed best single model: -0.006, 44/100 — the fusion is a stable release configuration, not a method proven better than the best single model.
- "Oracle bound 0.412" is an empirical observation whose estimation method is not preserved in the repo; cite it as such, not as a computed quantity.
- Selection strategy (proxy table, pre-registered, 100 seeds 1001–1100, `docs/项目记录/发布候选预注册_100seed_20260930.md`): `greedy` beats `random` in mean paired regret (Δ −1.878, CI [−2.519, −1.261]; criterion met) and in top-10% pool recall (+0.215, 99/100), but the paired median is ≈0 and the win/tie/loss split is 50/37/13 — the mean advantage comes from the 10 biggest-improvement seeds (−8.41, ≈44.8% of the mean), while the 10 seeds where greedy did worst average +1.76. The 30-seed holdout Spearman gain (+0.031) was NOT replicated on the confirmation seeds (Δ ≈ 0). All of this is proxy-table evidence only, not wet-lab evidence. Never cite the "worst 10%" without this direction.

## Current search stage (2026-10-05)

- Supplemental audit (2026-10-06): corrected NUPACK recomputation2000 and44 official checks completed; train-only screening30 and explicitly post hoc forced-inclusion30 showed no stable GBSA ranking gain. Controlled frozen-encoder pretraining30 (no/external/domain) means0.11618/0.13954/0.11927 versus same-split303GBR0.35967 failed the predeclared100-seed advancement gate. These are bounded configurations, not proof all structural or pretrained models are ineffective. The corrected tree-ensemble finite-pool AL100 experiment improved greedy discovery regret (-2.90193 vs random) but not holdout ranking (-0.00370, CI crosses0); do not substitute this for original deep-ensemble/Kriging-Believer AL reproduction. See `docs/汇报/论文复现到七算法搜索_阶段总汇报_PPT素材版_20261006.md` and `evidence/supplement_20261006/`.

- Surrogate optimization is paused for this stage. Implemented `mutation_only_ga` has single-parent mutations and no crossover; do not introduce crossover as an implicit next step.
- Seven matched-start policies (random walk, hill climb, beam, mutation-only GA, iterated local search, simulated annealing, mutation-only AdaLead adaptation) share 8 measured + 8 random starts and the fitted model within each cell. Main comparison outcomes use the per-seed common actual prediction prefix, including starts. Refit stability uses full stopped top-50 archives and is a composite model/search sensitivity diagnostic.
- Latest seven-policy suite: development6001–6030, two proxies, budgets1024/4096,1260subruns including420refits; confirmation8001–8100,budget1024,1400subruns. Generation guard256 for all; primary seven-way common actual-call prefixes and pre-specified pairwise companion are distinct. No crossover in any new policy. AdaLead uses additive signed-minimization tolerance and root-relative rollouts; SA global dedup is an optimization variant, not equilibrium sampling. See `docs/汇报/9月/七算法同起点比较与实施结果_20261005.md` and `evidence/search_seven_20261005/` for actual values.139local tests pass.
- Earlier four-policy suite (retained): development401–430, two sequence-only diagnostic proxies, budgets1024/4096/8192,1140subruns; confirmation2001–2100,budget1024,800subruns. GA beats random walk in proxy minima but does not consistently beat hill climbing. These are predicted-score diagnostics, not new true-GBSA evidence. Local `xgb` is sklearnGBM fallback and RF is a new300-tree diagnostic adapter, neither inherits the v8 release reference.
- The96-sequence historical static-pool export is not a currently authorized calculation/ingest batch. Future measurement interfaces have been tested with artificial labels; real protocol/budget and the decision to use active learning remain external future decisions.
- New search resumes strictly bind fitted model, training sequences and values, WT scaffold, mutable positions, data, policy config and source digest. Preserve historical archives; use fresh output directories when identity changes. Static-pool resumes ignore inactive search settings to retain compatibility without weakening active parameters.
- Current results and limits: `docs/汇报/9月/GBSA优化项目总汇报_生工团队版_20261005.md`, `evidence/search_suite_20261005/README.md`.

## Running experiments

Define experiments in `configs/experiments/` and launch them through `scripts/run_experiment.py`. A completed run must contain config, data hash, environment, metrics, stdout, and a `DONE` marker. A prospective batch awaiting real GBSA labels is marked `PARKED` instead, and this is now enforced: the loop exits 3 whenever any candidate is still awaiting labels, and a config can pin `acceptance.expect_marker` / `acceptance.expect_exit_code` so a run that must park can never be recorded as `DONE`. Contract, artefacts, and evidence: `docs/论文改进/候选闭环框架_20260930.md`.

The closed loop predicts standalone `gbsa` only. The weighted `label = 0.961*Gap2 + 0.039*gbsa` must not be used as a training target, acquisition score, candidate ranking score, or success metric until its weights are validated on this dataset. A smoke run of the loop reads in-table labels and is therefore not evidence that any selection strategy beats `random`.

Resuming a checkpointed run must go through `scripts/run_experiment.py --resume-run <run_id> [--ingest labels.csv] [--pin-protocol <protocol>]`: the loop refuses to continue when the configuration, data hash, pool or holdout no longer matches its checkpoint fingerprint, and a closed batch clears its `PARKED` marker. Ingest enforces `candidate_id`↔sequence, a non-empty `protocol_id` (allow-list via `--approved-protocols`) and persists QC `fail` rows; a fully returned batch is absorbed, retrained and marked `DONE` only after the retrain. The hand-off batch's protocol is a placeholder: pin the real lab protocol through `--pin-protocol` (audited `protocol_changes.jsonl`; rewrites config/fingerprint/ledger/batch manifests) — never hand-edit an old run's config file, and never ingest a real measurement file before pinning.

Long Linux runs must use `nohup`, write checkpoints, and limit shared-server parallelism to 4–8 workers after checking load.

## Repository hygiene

- Do not copy datasets into model directories.
- Do not commit `runs/`, raw/external data, generated feature matrices, checkpoints, or large logs.
- Keep deprecated approaches in `legacy/` documentation, outside the mainline import path.
- Never delete or move parent legacy files without a reviewed SHA256-based deletion manifest.
