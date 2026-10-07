"""Replica budget allocation experiment (BTS-RED route, retrospective simulation).

Motivation: the AL≈random finding and the noisy-AL literature (BTS-RED,
NeurIPS 2023; Bellamy et al., JCIM 2022; Sheng et al., KDD 2008) say that when
label noise dominates, measurement budget should buy REPLICAS of existing
labels rather than new sequences. This simulates that regime with synthetic
measurement noise calibrated to the documented gbsa noise (frame-level SD
3-15 kcal/mol, label std 10.4; Chen et al. 2019 / Genheden & Ryde 2015).

Simulation (per seed, 30 seeds):
  - 80/20 split: pool (1600) + holdout (400). y_true = gbsa.
  - initial labeled set: n_init=400 random sequences, 1 noisy measurement each.
    y_meas = y_true + N(0, sigma_meas), sigma_meas in {5, 10} kcal/mol.
  - each round, total budget B=100 measurements, R=4 rounds. Strategies:
      NEW      : B new sequences, 1 measurement each (baseline).
      REP-TOP  : replicate the top n=B/r sequences by predicted mu (r=5
                 measurements each), label = running mean of measurements.
      REP-VAR  : allocate replicas proportional to per-sequence noise-variance
                 estimates (k-NN Hamming label variance at round 0, BTS-RED
                 principle: more replicas where the label is noisiest).
      MIX      : half the budget to NEW (lowest predicted mu), half to REP-VAR.
  - after each round, retrain XGB on current labels; evaluate holdout Spearman
    vs y_true; also track achieved label noise std over the labeled set.

Honest caveat: synthetic noise calibrated to literature; demonstrates the
mechanism in this dataset, does NOT replace a real replica experiment.

Usage (server, FAD_env):
  python -m scripts.reproduction.replica_allocation --outer-seeds 30
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

try:
    from .active_learning_strong import build_seq, load_nupack_useful, _spearman
    from .rank_objective import make_xgb
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from active_learning_strong import build_seq, load_nupack_useful, _spearman
    from rank_objective import make_xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]


def hamming_matrix(seqs):
    n = len(seqs)
    pat = np.zeros((n, len(MUT_POS)), dtype=np.int8)
    for i, s in enumerate(seqs):
        for j, p in enumerate(MUT_POS):
            pat[i, j] = ord(s[p - 1])
    D = np.zeros((n, n), dtype=np.int8)
    for j in range(len(MUT_POS)):
        D += (pat[:, j][:, None] != pat[None, :, j]).astype(np.int8)
    return D


def knn_label_variance(seqs, labels, k=5):
    """Per-sequence noise-variance estimate: variance of the k nearest labeled
    neighbors' labels (Hamming over the 13 mutation positions), computed once
    at round 0 within the labeled set (no leakage)."""
    n = len(seqs)
    D = hamming_matrix(seqs)
    out = np.zeros(n)
    for i in range(n):
        order = np.argsort(D[i])[1:1 + k]
        out[i] = float(np.var(labels[order])) + 1e-8
    return out


def run_seed(X, y_true, seqs, seed, sigma_meas, n_init, budget, rounds, r_rep,
             hetero, mix_frac=0.5):
    rng = np.random.RandomState(1000 + seed)
    pool = np.arange(len(X))
    init = rng.choice(pool, size=n_init, replace=False)
    labeled = init.copy()
    unlabeled = np.setdiff1d(pool, labeled)

    # Heteroscedastic measurement noise: larger at extreme gbsa values,
    # mirroring the 25 physically implausible +109 kcal/mol outliers.
    if hetero:
        med = np.median(y_true)
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        scale = 1.0 + 2.0 * np.abs(y_true - med) / max(iqr, 1e-6)
    else:
        scale = np.ones(len(y_true))

    def measure(idx):
        return y_true[idx] + rng.normal(0.0, sigma_meas, size=len(idx)) * scale[idx]

    labels = measure(labeled)          # 1 measurement each
    counts = np.ones(len(labeled))     # measurement counts
    var_est = knn_label_variance([seqs[i] for i in labeled], labels)

    strategies = {"NEW": [], "REP-TOP": [], "REP-VAR": [], "MIX": []}

    # run each strategy on its own copy of the simulation state
    for sname in ["NEW", "REP-TOP", "REP-VAR", "MIX"]:
        labeled_s = labeled.copy()
        unlabeled_s = unlabeled.copy()
        labels_s = labels.copy()
        counts_s = counts.copy()
        var_s = var_est.copy()
        curve, top_curve, noise_std = [], [], []

        for rnd in range(rounds):
            m = retrain_fit(X, labeled_s, labels_s)
            pred_hold = m.predict(X_hold)
            curve.append(_spearman(y_true_hold, pred_hold))
            # optimization-relevant metric: ranking reliability within the
            # 100 truly-best holdout sequences (evaluation-side selection)
            top100 = np.argsort(y_true_hold)[:100]
            top_curve.append(_spearman(y_true_hold[top100], pred_hold[top100]))
            noise_std.append(float(np.std(labels_s - y_true[labeled_s])))

            mu_pool = m.predict(X[unlabeled_s])
            if sname == "NEW":
                sel = np.argsort(mu_pool)[:budget]           # lowest predicted mu
                chosen = unlabeled_s[sel]
                labels_s = np.concatenate([labels_s, measure(chosen)])
                counts_s = np.concatenate([counts_s, np.ones(len(chosen))])
                labeled_s = np.concatenate([labeled_s, chosen])
                unlabeled_s = np.delete(unlabeled_s, sel)
            elif sname == "REP-TOP":
                n = budget // r_rep
                sel = np.argsort(m.predict(X[labeled_s]))[:n]
                # vectorized: r_rep new measurements each, running-mean update
                new_meas = np.array([measure([labeled_s[i]])[0] for i in sel
                                     for _ in range(r_rep)]).reshape(n, r_rep)
                labels_s[sel] = ((counts_s[sel] * labels_s[sel])[:, None] +
                                 new_meas).sum(axis=1) / (counts_s[sel] + r_rep)
                counts_s[sel] += r_rep
            elif sname == "REP-VAR":
                w = var_s / var_s.sum()
                alloc = np.zeros(len(labeled_s), dtype=int)
                alloc[np.argsort(-w)[:budget // 5]] = 1  # at least top ones get some
                remaining = budget - alloc.sum()
                # distribute remaining measurements proportionally to variance
                order = np.argsort(-w)
                for i in order:
                    if remaining <= 0:
                        break
                    give = min(5, remaining)
                    alloc[i] += give
                    remaining -= give
                for i in np.where(alloc > 0)[0]:
                    extra = alloc[i]
                    meas = np.array([measure([labeled_s[i]])[0] for _ in range(extra)])
                    labels_s[i] = (counts_s[i] * labels_s[i] + meas.sum()) / (counts_s[i] + extra)
                    counts_s[i] += extra
                # update variance estimates from the new labels
                var_s = knn_label_variance([seqs[i] for i in labeled_s], labels_s)
            elif sname == "MIX":
                n_new = max(1, int(budget * mix_frac))
                sel_new = np.argsort(mu_pool)[:n_new]
                chosen = unlabeled_s[sel_new]
                labels_s = np.concatenate([labels_s, measure(chosen)])
                counts_s = np.concatenate([counts_s, np.ones(len(chosen))])
                labeled_s = np.concatenate([labeled_s, chosen])
                unlabeled_s = np.delete(unlabeled_s, sel_new)
                w = var_s / var_s.sum()
                n = budget - n_new
                sel_rep = np.argsort(-w)[:max(1, n // 2)]
                new_meas = np.array([measure([labeled_s[i]])[0] for i in sel_rep
                                     for _ in range(2)]).reshape(len(sel_rep), 2)
                labels_s[sel_rep] = ((counts_s[sel_rep] * labels_s[sel_rep])[:, None] +
                                     new_meas).sum(axis=1) / (counts_s[sel_rep] + 2)
                counts_s[sel_rep] += 2
                var_s = knn_label_variance([seqs[i] for i in labeled_s], labels_s)

        strategies[sname] = {"spearman_curve": curve, "top100_spearman_curve": top_curve,
                             "noise_std_curve": noise_std}
    return strategies


def retrain_fit(X, idx, labels):
    m = make_xgb()
    m.fit(X[idx], labels.astype(np.float32))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--n-init", type=int, default=400)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--rep-k", type=int, default=5)
    ap.add_argument("--mix-frac", type=float, default=0.5,
                    help="fraction of the budget MIX spends on NEW sequences "
                         "(rest goes to variance-allocated replicas)")
    ap.add_argument("--sigmas", default="5,10")
    ap.add_argument("--hetero", action="store_true",
                    help="heteroscedastic measurement noise (larger at extreme gbsa)")
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/replica_allocation/results.json")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    seqs = [str(s) for s in df["Sequence"]]
    y_true = df["gbsa"].to_numpy(dtype=np.float64)
    X = np.hstack([build_seq(seqs),
                   load_nupack_useful(data_dir / "nupack_features_full.csv", seqs)[0]])
    print(f"features: {X.shape[1]}d")

    global y_true_hold, X_hold
    sigmas = [float(s) for s in args.sigmas.split(",")]
    all_res = {}
    for sigma_meas in sigmas:
        print(f"\n== sigma_meas = {sigma_meas} (hetero={args.hetero}) ==")
        agg = {s: {"spearman": [], "top100_spearman": [], "noise_std": []} for s in
               ["NEW", "REP-TOP", "REP-VAR", "MIX"]}
        for seed in range(1, args.outer_seeds + 1):
            tr, te = train_test_split(range(len(X)), test_size=0.2, random_state=seed)
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y_true[tr], y_true[te]
            X_hold, y_true_hold = X_te, y_te

            # rebuild X indexing relative to the train fold
            res = run_seed(X_tr, y_tr, [seqs[i] for i in tr], seed, sigma_meas,
                           args.n_init, args.budget, args.rounds, args.rep_k,
                           args.hetero, args.mix_frac)
            for sname, r in res.items():
                agg[sname]["spearman"].append(r["spearman_curve"])
                agg[sname]["top100_spearman"].append(r["top100_spearman_curve"])
                agg[sname]["noise_std"].append(r["noise_std_curve"])
            print(f"  seed {seed}/{args.outer_seeds} done", flush=True)

        summary = {}
        for sname in agg:
            sp = np.array(agg[sname]["spearman"])
            tp = np.array(agg[sname]["top100_spearman"])
            ns = np.array(agg[sname]["noise_std"])
            summary[sname] = {
                "spearman_mean_by_round": sp.mean(axis=0).tolist(),
                "spearman_std_by_round": sp.std(axis=0).tolist(),
                "top100_spearman_mean_by_round": tp.mean(axis=0).tolist(),
                "final_spearman": float(sp[:, -1].mean()),
                "final_top100_spearman": float(tp[:, -1].mean()),
                "final_noise_std": float(ns[:, -1].mean()),
            }
        all_res[str(sigma_meas)] = summary
        for sname, s in summary.items():
            print(f"  {sname:<8} final_sp={s['final_spearman']:.4f} "
                  f"final_top100={s['final_top100_spearman']:.4f} "
                  f"final_noise={s['final_noise_std']:.2f} "
                  f"curve={[round(v, 4) for v in s['spearman_mean_by_round']]}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_res, open(out, "w"), indent=2)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()
