"""30-Seed Unified Baseline — Phase 0A.

Reproducible multi-seed evaluation of baseline models (ET, RF, GBR, XGB)
on fixed mutation_only features. Generates the canonical baseline registry
with per-seed, per-model results.

Key improvements over historical eval:
  - All models use the SAME train/val/test splits (paired comparison)
  - Reports mean±std across seeds
  - Separates "oracle best" from "single deployable model"
  - Uses nested protocol: feature screening (if any) inside each train fold

Usage:
  python scripts/tools/baseline_30seed.py \
    --models ET,RF,GBR,XGB \
    --seeds 30 \
    --feature-set mutation_only \
    --output artifacts/phase0/baseline_30seed.csv
"""

from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import pearsonr, rankdata, kendalltau
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                               RandomForestRegressor)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASES = "ACGU"
MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]

# ── Metrics ──────────────────────────────────────────────────────────
def _spearman(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]

def _kendall(a, b):
    return kendalltau(a, b)[0]

def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _mae(a, b):
    return float(np.mean(np.abs(a - b)))

# ── Feature builders ─────────────────────────────────────────────────
def build_mutation_only(seqs):
    """13 positions × 4 bases = 52d one-hot."""
    X = np.zeros((len(seqs), 52), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        for j, p in enumerate(MUT_POS):
            idx = p - 1
            if idx < len(s) and s[idx] in BASES:
                X[i, j * 4 + BASES.index(s[idx])] = 1.0
    return X

FEATURE_BUILDERS = {
    "mutation_only": build_mutation_only,
}

# ── Model factory ────────────────────────────────────────────────────
def make_model(name: str):
    if name == "ET":
        return ExtraTreesRegressor(
            n_estimators=500, max_features="log2",
            min_samples_leaf=5, random_state=42, n_jobs=1)
    elif name == "RF":
        return RandomForestRegressor(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=5, random_state=42, n_jobs=1)
    elif name == "GBR":
        return GradientBoostingRegressor(
            n_estimators=500, max_depth=3, min_samples_leaf=3,
            learning_rate=0.03, random_state=42)
    elif name == "XGB":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=500, max_depth=3, learning_rate=0.03,
                random_state=42, verbosity=0, n_jobs=1)
        except ImportError:
            return None
    else:
        raise ValueError(f"Unknown model: {name}")


def run_single_seed(full_df, seed, models, feature_set):
    """Run one seed: split, build features, train all models, evaluate.

    Returns list of per-model result dicts.
    """
    # ── Split ────────────────────────────────────────────────────
    # Use 80/20 split (like the 80/10/10 protocol but without val)
    # This gives 1600 train, 400 test — comparable to historical
    tr_idx, te_idx = train_test_split(
        range(len(full_df)), test_size=0.2, random_state=seed)

    tr_df = full_df.iloc[tr_idx]
    te_df = full_df.iloc[te_idx]

    seq_tr = tr_df["Sequence"].values
    seq_te = te_df["Sequence"].values
    y_tr = tr_df["gbsa"].values
    y_te = te_df["gbsa"].values
    gap2_te = te_df["Gap2"].values
    label_te = te_df["label"].values

    # ── Build features ───────────────────────────────────────────
    builder = FEATURE_BUILDERS[feature_set]
    X_tr = builder(seq_tr)
    X_te = builder(seq_te)

    results = []
    for mname in models:
        model = make_model(mname)
        if model is None:
            results.append({"model": mname, "gbsa_Sp": float("nan"),
                            "error": "ImportError"})
            continue

        try:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)

            gbsa_sp = _spearman(y_te, pred)
            gbsa_kt = _kendall(y_te, pred)
            gbsa_rmse = _rmse(y_te, pred)
            gbsa_mae = _mae(y_te, pred)

            # Hybrid label: 0.961*Gap2 + 0.039*gbsa_pred
            hybrid_pred = 0.961 * gap2_te + 0.039 * pred
            hybrid_sp = _spearman(label_te, hybrid_pred)

            # NUPACK direct baseline (gbsa = mean of training gbsa)
            nupack_sp = _spearman(
                label_te,
                0.961 * gap2_te + 0.039 * y_tr.mean())

            results.append({
                "model": mname,
                "seed": seed,
                "gbsa_Sp": round(gbsa_sp, 6),
                "gbsa_Kendall": round(gbsa_kt, 6),
                "gbsa_RMSE": round(gbsa_rmse, 4),
                "gbsa_MAE": round(gbsa_mae, 4),
                "Hybrid_Sp": round(hybrid_sp, 6),
                "NUPACK_direct_Sp": round(nupack_sp, 6),
                "Hybrid_delta_vs_NUPACK": round(hybrid_sp - nupack_sp, 6),
                "n_train": len(tr_idx),
                "n_test": len(te_idx),
                "feature_set": feature_set,
            })
        except Exception as e:
            results.append({
                "model": mname, "seed": seed,
                "gbsa_Sp": float("nan"),
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="30-Seed Unified Baseline")
    parser.add_argument("--models", default="ET,RF,GBR",
                        help="Comma-separated model names: ET,RF,GBR,XGB")
    parser.add_argument("--seeds", type=int, default=30,
                        help="Number of random seeds (default: 30)")
    parser.add_argument("--seed-start", type=int, default=1,
                        help="Starting seed value (default: 1)")
    parser.add_argument("--feature-set", default="mutation_only",
                        choices=list(FEATURE_BUILDERS.keys()),
                        help="Feature set to use")
    parser.add_argument("--data-dir", default="data/proxy2000_v2",
                        help="Path to v2 data directory")
    parser.add_argument("--output", default="artifacts/phase0/baseline_30seed.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    # ── Load data ──────────────────────────────────────────────────
    data_dir = PROJECT_ROOT / args.data_dir
    full_path = data_dir / "fad_proxy2000_v2_full.csv"
    if not full_path.exists():
        print(f"[ERROR] Full data not found at {full_path}")
        print("  Run data regeneration script first (see CLAUDE.md)")
        sys.exit(1)

    full_df = pd.read_csv(full_path)
    print(f"Data: {len(full_df)} sequences from {full_path}")
    print(f"Models: {models}")
    print(f"Seeds: {seeds[0]}–{seeds[-1]} ({len(seeds)} total)")
    print(f"Feature set: {args.feature_set}")

    # ── Run ────────────────────────────────────────────────────────
    all_results = []
    t0 = time.time()

    for seed in seeds:
        seed_results = run_single_seed(full_df, seed, models, args.feature_set)
        all_results.extend(seed_results)

        # Progress
        done = seed - seeds[0] + 1
        sps = [r["gbsa_Sp"] for r in seed_results if not np.isnan(r["gbsa_Sp"])]
        best_sp = max(sps) if sps else 0
        print(f"  seed={seed:2d}  [{done:2d}/{len(seeds)}]  "
              f"best_gbsa_Sp={best_sp:.4f}  "
              f"elapsed={time.time()-t0:.0f}s")

    # ── Aggregate & save ───────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(PROJECT_ROOT / args.output, index=False)
    print(f"\n[OK] Per-seed results saved to {args.output}")

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"SUMMARY: {args.feature_set} | {len(seeds)} seeds | {len(models)} models")
    print(f"{'='*80}")
    print(f"{'Model':<8} {'Mean Sp':>10} {'Std Sp':>10} {'Min Sp':>10} "
          f"{'Max Sp':>10} {'Mean Hyb':>10} {'Mean RMSE':>10}")

    for mname in models:
        mdf = results_df[
            (results_df["model"] == mname) &
            (results_df["gbsa_Sp"].notna())
        ]
        if len(mdf) == 0:
            continue
        sps = mdf["gbsa_Sp"].values
        hybs = mdf["Hybrid_Sp"].values
        rmses = mdf["gbsa_RMSE"].values
        print(f"{mname:<8} {np.mean(sps):>10.4f} {np.std(sps):>10.4f} "
              f"{np.min(sps):>10.4f} {np.max(sps):>10.4f} "
              f"{np.mean(hybs):>10.4f} {np.mean(rmses):>10.2f}")

    # Best (any model) per seed — oracle upper bound
    best_per_seed = []
    for seed in seeds:
        seed_sps = [
            r["gbsa_Sp"] for r in all_results
            if r.get("seed") == seed and not np.isnan(r.get("gbsa_Sp", float("nan")))
        ]
        if seed_sps:
            best_per_seed.append(max(seed_sps))
    if best_per_seed:
        print(f"\n{'Best (any model)':<8} {np.mean(best_per_seed):>10.4f} "
              f"{np.std(best_per_seed):>10.4f} {np.min(best_per_seed):>10.4f} "
              f"{np.max(best_per_seed):>10.4f}")

    # ── Save summary JSON ──────────────────────────────────────────
    summary = {
        "feature_set": args.feature_set,
        "n_seeds": len(seeds),
        "seed_range": [seeds[0], seeds[-1]],
        "models": models,
        "per_model": {},
        "oracle_best_per_seed": {
            "mean": float(np.mean(best_per_seed)) if best_per_seed else None,
            "std": float(np.std(best_per_seed)) if best_per_seed else None,
        },
    }
    for mname in models:
        mdf = results_df[(results_df["model"] == mname) & results_df["gbsa_Sp"].notna()]
        if len(mdf) == 0:
            continue
        summary["per_model"][mname] = {
            "mean_gbsa_Sp": float(np.mean(mdf["gbsa_Sp"])),
            "std_gbsa_Sp": float(np.std(mdf["gbsa_Sp"])),
            "min_gbsa_Sp": float(np.min(mdf["gbsa_Sp"])),
            "max_gbsa_Sp": float(np.max(mdf["gbsa_Sp"])),
            "mean_Hybrid_Sp": float(np.mean(mdf["Hybrid_Sp"])),
            "mean_RMSE": float(np.mean(mdf["gbsa_RMSE"])),
        }

    summary_path = PROJECT_ROOT / args.output.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
