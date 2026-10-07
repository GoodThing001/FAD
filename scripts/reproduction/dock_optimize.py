"""dock (r_i_docking_score) prediction optimization.

dock is a clean binding estimate (std 1.05) but a DIFFERENT quantity than gbsa
(0.289 correlated). Its ceiling is much higher than gbsa's. Optimize dock
prediction to its best, mirroring the gbsa pipeline:

  feature sets: 303d / +NUPACK useful (192d) / +full NUPACK (601d)
  label smoothing: k-NN Hamming (denoise, helped gbsa +0.021)
  model: XGB (500 trees), 30-seed

Runs in FAD_env.

Usage:
  python -m scripts.reproduction.dock_optimize --outer-seeds 30
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from .active_learning_strong import build_seq, _spearman
from .label_smoothing import hamming_matrix, smooth_labels
from .rank_objective import make_xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]

# 7 useful blocks (gbsa) + all 18 blocks
USEFUL = {
    "contact_pca": [r"nupack_contact_"],
    "cluster_pair": [r"nupack_inter_cluster_\d+_\d+$", r"nupack_intra_cluster_\d+$",
                     r"nupack_mut_long_range$"],
    "energy_land": [r"nupack_n_very_stable_pairs$", r"nupack_n_moderate_pairs$",
                    r"nupack_stable_pair_ratio$", r"nupack_eig_ipr$", r"nupack_eig_n_eff_modes$"],
    "pos_entropy": [r"nupack_pos\d+_entropy$"],
    "motif_global": [r"nupack_motif_global_", r"nupack_motif_apt_"],
    "pos_pair_prob": [r"nupack_pos\d+_pair_prob$", r"nupack_pos\d+_likely_paired$",
                      r"nupack_c\d+_mean_pair_prob$", r"nupack_mean_pair_prob$",
                      r"nupack_std_pair_prob$", r"nupack_frac_paired_gt_0_5$"],
    "motif_positional": [r"nupack_motif_pos\d+_is_"],
}


def load_nupack(data_dir, seqs, mode):
    df = pd.read_csv(data_dir / "nupack_features_full.csv")
    df["Sequence"] = df["Sequence"].astype(str)
    cols = [c for c in df.columns if c.startswith("nupack_")]
    if mode == "useful":
        sel = sorted(set(i for pats in USEFUL.values() for i, c in enumerate(cols)
                         if any(re.search(p, c) for p in pats)))
    elif mode == "all":
        sel = list(range(len(cols)))
    else:
        raise ValueError(mode)
    df = df.set_index("Sequence").loc[[str(s) for s in seqs]]
    return df[[cols[i] for i in sel]].to_numpy(dtype=np.float32)


def eval_dock(X, dock, seqs, seeds, smooth=False, k=5, alpha=0.5):
    sps = []
    for seed in seeds:
        tr, te = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
        y_tr = dock[tr].astype(np.float32)
        if smooth:
            tr_seqs = [seqs[i] for i in tr]
            D_tr = hamming_matrix(tr_seqs)
            y_tr = smooth_labels(D_tr, dock[tr], k, alpha).astype(np.float32)
        m = make_xgb()
        m.fit(X[tr], y_tr)
        sps.append(_spearman(dock[te], m.predict(X[te])))
    return float(np.mean(sps)), float(np.std(sps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/dock_optimize/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    dock = df["dock"].to_numpy(dtype=np.float32)
    X_seq = build_seq(seqs)
    X_useful = load_nupack(data_dir, seqs, "useful")
    X_all = load_nupack(data_dir, seqs, "all")
    print(f"features: 303d seq, {X_useful.shape[1]}d useful, {X_all.shape[1]}d full")

    seeds = list(range(1, args.outer_seeds + 1))
    configs = {
        "303d": X_seq,
        "303d+useful192": np.hstack([X_seq, X_useful]),
        "303d+full601": np.hstack([X_seq, X_all]),
    }
    results = {}
    for name, X in configs.items():
        mean, std = eval_dock(X, dock, seqs, seeds)
        results[name] = {"mean": mean, "std": std}
        print(f"  {name:<20} {mean:.4f} +/- {std:.4f}", flush=True)
        # label smoothing on the best (full) config
        if name == "303d+full601":
            ms, ss = eval_dock(X, dock, seqs, seeds, smooth=True)
            results["303d+full601+smooth"] = {"mean": ms, "std": ss}
            print(f"  {'+label_smooth':<20} {ms:.4f} +/- {ss:.4f}", flush=True)

    best = max(results.items(), key=lambda kv: kv[1]["mean"])
    print(f"\n{'='*60}")
    print(f"BEST: {best[0]}  {best[1]['mean']:.4f} +/- {best[1]['std']:.4f}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()
