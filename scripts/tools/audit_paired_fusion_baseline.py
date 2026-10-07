"""Paired comparison: v8 equal-weight fusion vs fresh mutation-only RF baseline.

Both use the same split protocol (train_test_split random_state=seed,
test_size=0.2) so per-seed results are paired across the two runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

ROOT = Path(__file__).resolve().parents[2]


def _spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


def main():
    v8 = pd.read_csv(
        ROOT / "evidence/server_sync_20260902/v8_fusion_100seed/all_results.csv")
    fusion = v8[v8["method"] == "equal_weight_rank"].set_index("seed")["gbsa_Sp"]
    rf = pd.read_csv(
        ROOT / "evidence/audit_20260924/audit_baseline_rf_100seed/metrics.csv")
    rf = rf[rf["model"] == "RF"].set_index("seed")["gbsa_Sp"]

    common = sorted(set(fusion.index) & set(rf.index))
    f = fusion.loc[common].to_numpy()
    b = rf.loc[common].to_numpy()
    delta = f - b
    rng = np.random.default_rng(20260924)
    boot = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
    result = {
        "n_seeds": len(common),
        "fusion_mean": float(f.mean()),
        "fusion_std": float(f.std()),
        "rf_mutation_only_mean": float(b.mean()),
        "rf_mutation_only_std": float(b.std()),
        "paired_delta_mean": float(delta.mean()),
        "paired_bootstrap_ci95": [float(np.percentile(boot, 2.5)),
                                  float(np.percentile(boot, 97.5))],
        "fusion_wins": int((delta > 0).sum()),
        "fusion_win_rate": float((delta > 0).mean()),
    }
    out = ROOT / "evidence/audit_20260924/paired_fusion_vs_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
