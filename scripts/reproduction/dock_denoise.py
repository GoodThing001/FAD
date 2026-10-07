"""dock-based denoising of gbsa (MM-GBSA).

gbsa (r_psp_MMGBSA_dG_Bind) is MORE accurate but NOISY (std 10.4, +109 outliers,
Spearman 0.34 predictable). dock (r_i_docking_score) is LESS accurate but CLEAN
(std 1.05, Spearman 0.54 predictable). This tests whether fusing dock into gbsa
denoises the label while keeping gbsa's accuracy, i.e. a denoised gbsa that is
MORE predictable than raw gbsa.

Methods (30-seed XGB on 303d + NUPACK):
  1. raw gbsa                              -- baseline
  2. z(gbsa) + z(dock)                     -- z-score average target, eval vs gbsa
  3. rank-average (gbsa, dock)             -- rank-average target, eval vs gbsa
  4. dock-shrunk gbsa: gbsa - a*(gbsa-dock_scaled)  -- outlier shrink, eval vs gbsa

Runs in FAD_env.

Usage:
  python -m scripts.reproduction.dock_denoise --outer-seeds 30
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import rankdata, zscore
from sklearn.model_selection import train_test_split

from .active_learning_strong import build_seq, load_nupack_useful, _spearman
from .rank_objective import make_xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_target(name, gbsa_tr, dock_tr):
    """Fold-internal denoised targets (NO leakage: transforms fit on training rows only).

    The evaluation is always Spearman vs ORIGINAL gbsa on the held-out fold, so
    test-side labels are never observed by any transform.
    """
    g = gbsa_tr.astype(np.float32)
    d = dock_tr.astype(np.float32)
    if name == "raw_gbsa":
        return g
    if name == "zscore_avg":
        zg = zscore(g)
        zd = zscore(d)
        return (zg + zd).astype(np.float32)
    if name == "rank_avg":
        return (rankdata(g) + rankdata(d)).astype(np.float32)
    if name == "shrink_0.5":
        dock_scaled = d * (g.std() / d.std())
        return (g - 0.5 * (g - dock_scaled)).astype(np.float32)
    if name == "shrink_0.3":
        dock_scaled = d * (g.std() / d.std())
        return (g - 0.3 * (g - dock_scaled)).astype(np.float32)
    raise ValueError(name)


def eval_target(X, gbsa, dock, name, seeds):
    """Train on a fold-internal denoised target, evaluate vs original gbsa."""
    sps = []
    for seed in seeds:
        tr, te = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
        m = make_xgb()
        m.fit(X[tr], build_target(name, gbsa[tr], dock[tr]))
        sps.append(_spearman(gbsa[te], m.predict(X[te])))
    return float(np.mean(sps)), float(np.std(sps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/dock_denoise/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    gbsa = df["gbsa"].to_numpy(dtype=np.float32)
    dock = df["dock"].to_numpy(dtype=np.float32)
    X = np.hstack([build_seq(seqs),
                   load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)[0]])

    seeds = list(range(1, args.outer_seeds + 1))

    targets = ["raw_gbsa", "zscore_avg", "rank_avg", "shrink_0.5", "shrink_0.3"]

    results = {}
    print("evaluation is Spearman(prediction, ORIGINAL gbsa) — so gains are real denoising,\n"
          "not just an easier target. All denoising transforms are fit inside the training\n"
          "fold only (no test-label leakage).\n")
    for name in targets:
        mean, std = eval_target(X, gbsa, dock, name, seeds)
        results[name] = {"mean": mean, "std": std}
        print(f"  {name:<14} {mean:.4f} +/- {std:.4f}", flush=True)

    base = results["raw_gbsa"]["mean"]
    print(f"\n{'='*60}")
    print(f"baseline (raw gbsa) : {base:.4f}")
    for name in targets:
        if name == "raw_gbsa":
            continue
        d = results[name]["mean"] - base
        print(f"{name:<14} {results[name]['mean']:.4f}  (Δ {d:+.4f})")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()
