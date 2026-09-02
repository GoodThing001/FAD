# FAD clean mainline instructions

## Scope

This directory is the curated mainline. Do not import code or data from the parent legacy workspace at runtime. Legacy files may be inspected only as migration evidence.

## Data

- Canonical dataset: `data/proxy2000_v2/fad_proxy2000_v2_full.csv`.
- Verify its SHA256 against `data/registry.csv` before formal experiments.
- Historical train/val/test files reproduce seed=42 only; never use them as the sole publication evaluation.
- MFE, Pre, Aft, and Gap2 must use NUPACK. Never substitute RNAfold MFE.

## Leakage constraints

Never use these as model inputs: `label`, `total_energy_change`, `r_psp_MMGBSA_dG_Bind`, `dock`, `gbsa`, `Gap1`, `Gap2`, `Gap3`.

`r_i_docking_score` is unavailable for scalable candidate generation and is not a mainline feature. Precomputed full-data ED target statistics are invalid for formal evaluation.

## Evaluation

- Use at least 30 seeds for model development claims and 100 seeds for release claims.
- Perform feature selection inside the training fold.
- Use `rankdata + pearsonr` for Spearman on Windows.
- Treat 0.506 as a historical fixed-split outlier, not a baseline.
- Release reference: v8 rank fusion, 100-seed gbsa Spearman 0.375, 95% CI [0.367, 0.382].

## Running experiments

Define experiments in `configs/experiments/` and launch them through `scripts/run_experiment.py`. A valid run must contain config, data hash, environment, metrics, stdout, and a `DONE` marker.

Long Linux runs must use `nohup`, write checkpoints, and limit shared-server parallelism to 4–8 workers after checking load.

## Repository hygiene

- Do not copy datasets into model directories.
- Do not commit `runs/`, raw/external data, generated feature matrices, checkpoints, or large logs.
- Keep deprecated approaches in `legacy/` documentation, outside the mainline import path.
- Never delete or move parent legacy files without a reviewed SHA256-based deletion manifest.

