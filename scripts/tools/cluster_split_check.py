"""Cluster split out-of-cluster generalization check for feature selection.

This is the final acceptance gate for the feature-selection experiment: a
promoted method's *stable* feature subset must not regress more than
``max_drop`` (Spearman) relative to the full-feature control when scored on
mutation-cluster splits (GroupKFold, Hamming <= 1) that were never observed
during selection.

Design
------
The 30-seed run already performs strict in-fold selection and writes a
stability table (``metrics_stability.csv``).  From that table we take each
method's *stable* subset — features selected in >= ``min_frequency`` of seeds,
with the mutation block always force-kept — and treat it as the pre-registered
selection product.  We then score both that subset and the full 1495-d control
on every cluster fold, on the exact same train/test indices, and report the
paired Delta.  A method passes iff its mean Delta >= -``max_drop``.

Reuses the feature builders, model factory, target transform, and Spearman
metric from ``feature_selection_pipeline.py`` so the feature space and
hyperparameters are identical to the main experiment.

Usage
-----
  python scripts/tools/cluster_split_check.py \
    --split-dir data/splits/cluster_hamming_le_1/ \
    --stability-csv runs/<run_id>/metrics_stability.csv \
    --eval-models ET,GBR,XGB --min-frequency 0.5 --max-drop 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from feature_selection_pipeline import (
    PROJECT_ROOT,
    build_mut,
    build_wd,
    build_struct,
    load_nupack_features,
    make_eval_model,
    make_force_keep_mask,
    fit_target_transform,
    _spearman,
)


def build_full_feature_matrix(data_dir: Path, full_df: pd.DataFrame):
    """Return (X, names) for the full 1495-d matrix (303 sequence + 1192 NUPACK)."""
    seqs = full_df["Sequence"].values
    blocks: list[np.ndarray] = []
    all_names: list[str] = []
    for builder in (build_mut, build_wd, build_struct):
        Xg, ng = builder(seqs)
        blocks.append(Xg)
        all_names.extend(ng)
    Xn, nn = load_nupack_features(data_dir, full_df, blocks="all")
    blocks.append(Xn)
    all_names.extend(nn)
    X = np.hstack(blocks).astype(np.float32)
    return X, np.array(all_names)


def load_stable_subsets(
    stability_csv: Path,
    names: np.ndarray,
    force_keep_groups: str,
    min_frequency: float,
):
    """Map method -> boolean mask over ``names`` (stable features + force-keep)."""
    stability = pd.read_csv(stability_csv)
    required = {"method", "feature", "frequency"}
    missing = required - set(stability.columns)
    if missing:
        raise ValueError(f"stability table missing columns: {sorted(missing)}")

    known = set(names.tolist())
    unknown = sorted({f for f in stability["feature"].unique() if f not in known})
    if unknown:
        raise ValueError(
            f"stability table references {len(unknown)} feature(s) not in the "
            f"assembled matrix, e.g. {unknown[:5]}"
        )

    force_keep = make_force_keep_mask(names, force_keep_groups)
    subsets = {"all": np.ones(len(names), dtype=bool)}
    methods = sorted({m for m in stability["method"].unique() if m != "all"})
    for method in methods:
        mdf = stability[stability["method"] == method]
        stable = set(mdf.loc[mdf["frequency"] >= min_frequency, "feature"])
        mask = np.array([n in stable for n in names], dtype=bool)
        mask |= force_keep
        subsets[method] = mask
    return subsets, methods


def main() -> int:
    p = argparse.ArgumentParser(
        description="Cluster split out-of-cluster check for feature selection"
    )
    p.add_argument("--split-dir", default="data/splits/cluster_hamming_le_1/",
                   help="directory written by generate_cluster_splits.py")
    p.add_argument("--stability-csv", required=True,
                   help="metrics_stability.csv from the 30-seed run "
                        "(project-relative or absolute)")
    p.add_argument("--data-dir", default="data/proxy2000_v2")
    p.add_argument("--eval-models", default="ET,GBR,XGB",
                   help="comma-separated: ET,RF,GBR,XGB,LGBM,SVR")
    p.add_argument("--force-keep-groups", default="mut")
    p.add_argument("--min-frequency", type=float, default=0.5,
                   help="minimum cross-seed selection frequency for a feature "
                        "to enter the stable subset")
    p.add_argument("--target", default="winsor",
                   choices=["raw", "winsor", "rank_gaussian"])
    p.add_argument("--loss", default="mae", choices=["mse", "mae"])
    p.add_argument("--max-drop", type=float, default=0.01,
                   help="a method fails if mean Delta (subset - control) < -max_drop")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--output", default=None,
                   help="default: <stability-csv dir>/cluster_split_check.csv")
    args = p.parse_args()

    split_dir = PROJECT_ROOT / args.split_dir
    data_dir = PROJECT_ROOT / args.data_dir
    stability_csv = Path(args.stability_csv)
    if not stability_csv.is_absolute():
        stability_csv = PROJECT_ROOT / stability_csv

    if not (split_dir / "split_metadata.json").is_file():
        raise FileNotFoundError(
            f"cluster split metadata missing at {split_dir}; "
            "run generate_cluster_splits.py first"
        )
    if not stability_csv.is_file():
        raise FileNotFoundError(f"stability table missing: {stability_csv}")
    if not (data_dir / "fad_proxy2000_v2_full.csv").is_file():
        raise FileNotFoundError(f"canonical data missing: {data_dir}")

    meta = json.loads((split_dir / "split_metadata.json").read_text())
    n_splits = meta["n_splits"]

    full_df = pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
    gbsa = full_df["gbsa"].values.astype(np.float64)
    X, names = build_full_feature_matrix(data_dir, full_df)
    print(f"[features] {X.shape[0]} samples x {X.shape[1]} features")

    subsets, methods = load_stable_subsets(
        stability_csv, names, args.force_keep_groups, args.min_frequency
    )
    for method, mask in subsets.items():
        print(f"[subset] {method:<16s} {int(mask.sum()):4d} features")

    eval_models = [m.strip().upper() for m in args.eval_models.split(",") if m.strip()]
    if not eval_models:
        raise ValueError("at least one --eval-models value is required")
    if len(eval_models) != len(set(eval_models)):
        raise ValueError("--eval-models must not repeat")
    for m in eval_models:  # fail fast before the fold loop
        make_eval_model(m, args.loss, args.n_jobs)

    rows: list[dict] = []
    for fold_id in range(n_splits):
        tr_idx = pd.read_csv(
            split_dir / f"fold_{fold_id}" / "train_indices.csv"
        )["index"].to_numpy()
        te_idx = pd.read_csv(
            split_dir / f"fold_{fold_id}" / "test_indices.csv"
        )["index"].to_numpy()
        y_tr = gbsa[tr_idx]
        y_te = gbsa[te_idx]
        y_tr_t, _ = fit_target_transform(y_tr, args.target)

        for method, mask in subsets.items():
            sel = np.where(mask)[0]
            X_tr = X[tr_idx][:, sel]
            X_te = X[te_idx][:, sel]
            for mname in eval_models:
                fm = make_eval_model(mname, args.loss, args.n_jobs)
                fm.fit(X_tr, y_tr_t)
                pred = fm.predict(X_te)
                rows.append({
                    "fold_id": fold_id,
                    "method": method,
                    "model": mname,
                    "n_features": int(mask.sum()),
                    "Spearman": _spearman(y_te, pred),
                })
        print(f"  fold {fold_id + 1}/{n_splits} done")

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / (args.output or str(stability_csv.parent / "cluster_split_check.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    # Paired delta vs the full-feature control on identical folds/models.
    control = (
        res[res["method"] == "all"][["fold_id", "model", "Spearman"]]
        .rename(columns={"Spearman": "control_spearman"})
    )
    merged = res[res["method"] != "all"].merge(control, on=["fold_id", "model"])
    merged["delta"] = merged["Spearman"] - merged["control_spearman"]

    summary = (
        merged.groupby(["method", "model"])
        .agg(
            n_features=("n_features", "first"),
            mean_spearman=("Spearman", "mean"),
            control_mean_spearman=("control_spearman", "mean"),
            mean_delta=("delta", "mean"),
            min_delta=("delta", "min"),
        )
        .reset_index()
    )
    summary["passes_max_drop"] = summary["mean_delta"] >= -args.max_drop
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n{'=' * 100}")
    print(f"CLUSTER SPLIT CHECK  (max_drop = {args.max_drop})")
    print(f"{'=' * 100}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[OK] per-fold results -> {out}")
    print(f"[OK] summary -> {summary_path}")

    all_pass = bool(summary["passes_max_drop"].all())
    n_pass = int(summary["passes_max_drop"].sum())
    print(f"\n[RESULT] {n_pass}/{len(summary)} method-model cells pass "
          f"max_drop={args.max_drop}: {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
