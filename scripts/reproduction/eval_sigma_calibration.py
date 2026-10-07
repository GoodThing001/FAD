"""Sigma calibration study: does the XGB bootstrap ensemble's sigma stratify
prediction error / hit risk on independent dev folds?

Why this exists (plan §5 gates, docs/项目记录/突变序列迭代搜索算法与代理接口实施方案_20261002.md):
the ensemble sigma is labelled `ensemble_spread_uncalibrated`, and any
uncertainty-based acquisition (LCB mu-beta*sigma, EI, Thompson, batch BO) is
gated on PROOF that sigma relates to error / hit risk on independent folds.
This driver measures exactly that, per seed and at two training sizes (the
closed-loop scale ~400 and the full library scale 1600), with a shuffled-label
negative control.

Per (seed, n_train): split via RandomState(seed) -> train n, test 400; fit the
same XGB bootstrap ensemble the loop uses (screen-k inside the training fold,
winsor 2.5); predict mu/sigma on the test fold; then:
  spearman_mu              Spearman(mu, y)          (reference skill)
  spearman_sigma_error     Spearman(sigma, |mu-y|)  (the calibration question)
  decile table             mean |error| per sigma decile
  top-K recall             K = 5%/10% of test; greedy (lowest mu) vs LCB
                           (mu - beta*sigma, beta in 0.5/1/2); delta = LCB - greedy
  negative control         refit on shuffled labels -> spearman_sigma_error_shuffled

Interpretation gate (written into summary.json): sigma "passes" the first gate
only if spearman_sigma_error CI > 0 while the shuffled control is ~0; an LCB
only earns acquisition use if its recall delta CI > 0 vs plain greedy.

Usage:
  python scripts/reproduction/eval_sigma_calibration.py --outer-seeds 30 \
      --n-trains 1600,400 --output runs/.../metrics.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.loop.interface import TARGET_COLUMN, normalize  # noqa: E402
from scripts.loop.surrogates import make_surrogate  # noqa: E402

BETAS = (0.5, 1.0, 2.0)
K_FRACTIONS = (0.05, 0.10)


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.all(np.isnan(a)) or np.all(np.isnan(b)):
        return float("nan")
    return float(pearsonr(rankdata(a), rankdata(b))[0])


def bootstrap_ci(values, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def top_k_recall(mu: np.ndarray, y: np.ndarray, k: int) -> float:
    """Fraction of the test fold's true best-k captured by the k lowest mu."""
    if k <= 0:
        return float("nan")
    true_top = set(np.argsort(y)[:k])
    hits = sum(1 for i in np.argsort(mu)[:k] if i in true_top)
    return hits / k


def decile_errors(sigma: np.ndarray, err: np.ndarray) -> list[dict]:
    """Mean |error| per sigma decile (duplicated bins merged)."""
    order = np.argsort(sigma)
    n = len(sigma)
    out = []
    for d in range(10):
        lo, hi = int(d * n / 10), int((d + 1) * n / 10)
        idx = order[lo:hi]
        if not len(idx):
            continue
        out.append({
            "decile": d + 1,
            "n": int(len(idx)),
            "mean_sigma": float(np.mean(sigma[idx])),
            "mean_abs_error": float(np.mean(err[idx])),
        })
    return out


def run_seed_arm(a, seed: int, n_train: int) -> dict:
    t0 = time.perf_counter()
    df = pd.read_csv(PROJECT_ROOT / a.data_dir / "fad_proxy2000_v2_full.csv")
    df["Sequence"] = df["Sequence"].map(normalize)
    values = df[TARGET_COLUMN].astype(float).to_numpy()
    perm = np.random.RandomState(seed).permutation(len(df))
    idx_tr = perm[:n_train]
    idx_te = perm[n_train:n_train + a.n_test]
    seqs_tr = df["Sequence"].to_numpy()[idx_tr].tolist()
    y_tr = values[idx_tr]
    seqs_te = df["Sequence"].to_numpy()[idx_te].tolist()
    y_te = values[idx_te]

    model = make_surrogate("xgb", seed=seed, n_members=a.n_members,
                           screen_k=a.screen_k)
    model.fit(seqs_tr, y_tr)
    pred = model.predict(seqs_te)
    mu, sigma = pred.mu, pred.sigma
    err = np.abs(mu - y_te)

    # negative control: same folds, shuffled training labels
    rng = np.random.RandomState(seed)
    y_shuf = y_tr[rng.permutation(len(y_tr))]
    model_shuf = make_surrogate("xgb", seed=seed, n_members=a.n_members,
                                screen_k=a.screen_k)
    model_shuf.fit(seqs_tr, y_shuf)
    pred_shuf = model_shuf.predict(seqs_te)
    spearman_shuf = spearman(pred_shuf.sigma, np.abs(pred_shuf.mu - y_te))

    row = {
        "seed": seed, "n_train": n_train, "n_test": int(len(y_te)),
        "model_id": pred.model_id, "feature_version": pred.feature_version,
        "spearman_mu": spearman(mu, y_te),
        "spearman_sigma_error": spearman(sigma, err),
        "spearman_sigma_error_shuffled": spearman_shuf,
        "mean_abs_error": float(np.mean(err)),
        "mean_sigma": float(np.mean(sigma)),
    }
    for frac in K_FRACTIONS:
        k = max(1, int(round(frac * len(y_te))))
        row[f"recall_greedy_k{frac}"] = top_k_recall(mu, y_te, k)
        for beta in BETAS:
            row[f"delta_recall_lcb{beta}_k{frac}"] = (
                top_k_recall(mu - beta * sigma, y_te, k)
                - row[f"recall_greedy_k{frac}"])
    row["seconds"] = round(time.perf_counter() - t0, 3)
    row["deciles"] = decile_errors(sigma, err)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--n-trains", default="1600,400",
                    help="comma-separated training sizes (arms)")
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-members", type=int, default=5)
    ap.add_argument("--screen-k", type=int, default=100)
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--output", default="runs/sigma_calibration/metrics.csv")
    a = ap.parse_args()

    n_trains = [int(x) for x in a.n_trains.split(",") if x.strip()]
    rows = []
    for seed in range(a.seed_start, a.seed_start + a.outer_seeds):
        for n_train in n_trains:
            row = run_seed_arm(a, seed, n_train)
            rows.append(row)
            print(f"[seed {seed} n_train={n_train}] spearman_mu={row['spearman_mu']:.4f} "
                  f"sigma_err={row['spearman_sigma_error']:.4f} "
                  f"shuf={row['spearman_sigma_error_shuffled']:.4f} "
                  f"drecall_lcb1_k0.05={row['delta_recall_lcb1.0_k0.05']:.4f}")

    flat = []
    for r in rows:
        for d in r.pop("deciles"):
            flat.append({"seed": r["seed"], "n_train": r["n_train"], **d})
    df = pd.DataFrame(rows)
    out = Path(a.output)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    deciles = pd.DataFrame(flat)
    summary_path = out.with_name(out.stem.replace("metrics", "summary") + ".json")
    summary = {"config": vars(a), "n_rows": len(df)}
    for n_train in n_trains:
        sub = df[df["n_train"] == n_train]
        block = {}
        for col in ("spearman_mu", "spearman_sigma_error",
                    "spearman_sigma_error_shuffled", "mean_abs_error"):
            vals = sub[col].to_numpy(float)
            lo, hi = bootstrap_ci(vals)
            block[col] = {"mean": float(np.nanmean(vals)), "ci95": [lo, hi]}
        for frac in K_FRACTIONS:
            for beta in BETAS:
                col = f"delta_recall_lcb{beta}_k{frac}"
                vals = sub[col].to_numpy(float)
                lo, hi = bootstrap_ci(vals)
                block[col] = {"mean": float(np.nanmean(vals)), "ci95": [lo, hi],
                              "n_positive": int((vals > 1e-12).sum())}
        dsub = deciles[deciles["n_train"] == n_train]
        agg = dsub.groupby("decile").agg(
            n=("n", "sum"), mean_sigma=("mean_sigma", "mean"),
            mean_abs_error=("mean_abs_error", "mean")).reset_index()
        block["decile_table"] = agg.to_dict("records")
        block["gate_sigma_tracks_error"] = bool(
            block["spearman_sigma_error"]["ci95"][0] > 0
            and block["spearman_sigma_error_shuffled"]["ci95"][1] < 0.1)
        block["gate_lcb_beats_greedy"] = {
            str(beta): bool(block[f"delta_recall_lcb{beta}_k0.05"]["ci95"][0] > 0)
            for beta in BETAS}
        summary[f"n_train_{n_train}"] = block
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"},
                     ensure_ascii=False, indent=2))
    print(f"[OK] {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
