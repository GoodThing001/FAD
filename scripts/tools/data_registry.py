"""Data Registry & Integrity Check — Phase 0A.

Generates a unique data registry CSV with dataset_hash, column checksums,
and split metadata. This is the foundation for all subsequent experiments.

Usage:
  python scripts/tools/data_registry.py \
    --data-dir data/proxy2000_v2 \
    --output artifacts/phase0/data_registry.csv
"""

from __future__ import annotations

import argparse, hashlib, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def hash_file(path: Path) -> str:
    """SHA256 hash of a file's contents."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def hash_dataframe(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    """Deterministic hash of dataframe values (selected columns)."""
    if cols is None:
        cols = sorted(df.columns.tolist())
    vals = df[cols].values
    sha = hashlib.sha256()
    sha.update(vals.tobytes())
    # Also hash column names and dtypes to catch schema drift
    for c in cols:
        sha.update(c.encode())
        sha.update(str(df[c].dtype).encode())
    return sha.hexdigest()[:16]


def compute_column_stats(df: pd.DataFrame) -> dict:
    """Basic statistics for each column for integrity verification."""
    stats = {}
    for c in df.columns:
        col = df[c]
        s = {"dtype": str(col.dtype), "n_missing": int(col.isna().sum())}
        if pd.api.types.is_numeric_dtype(col):
            s.update({
                "min": float(col.min()) if not col.isna().all() else None,
                "max": float(col.max()) if not col.isna().all() else None,
                "mean": float(col.mean()) if not col.isna().all() else None,
                "std": float(col.std()) if not col.isna().all() else None,
            })
        elif col.dtype == object:
            s.update({
                "n_unique": int(col.nunique()),
                "min_len": int(col.str.len().min()) if not col.isna().all() else None,
                "max_len": int(col.str.len().max()) if not col.isna().all() else None,
            })
        stats[c] = s
    return stats


def verify_nupack_consistency(df: pd.DataFrame) -> dict:
    """Verify NUPACK MFE source consistency: Gap2 = Aft - MFE."""
    gap2_recalc = df["Aft"] - df["MFE"]
    max_diff = (df["Gap2"] - gap2_recalc).abs().max()
    return {
        "gap2_recalc_match": bool(np.allclose(df["Gap2"], gap2_recalc, atol=0.1)),
        "gap2_max_diff": float(max_diff),
        "mfe_range": [float(df["MFE"].min()), float(df["MFE"].max())],
        "aft_range": [float(df["Aft"].min()), float(df["Aft"].max())],
        "pre_range": [float(df["Pre"].min()), float(df["Pre"].max())],
    }


def verify_label_consistency(df: pd.DataFrame) -> dict:
    """Verify label formula: label = 0.961*Gap2 + 0.039*gbsa."""
    label_recalc = 0.961 * df["Gap2"] + 0.039 * df["gbsa"]
    max_diff = (df["label"] - label_recalc).abs().max()
    from scipy.stats import pearsonr, rankdata
    def sp(a, b):
        return pearsonr(rankdata(a), rankdata(b))[0]
    return {
        "label_recalc_match": bool(np.allclose(df["label"], label_recalc, atol=0.01)),
        "label_max_diff": float(max_diff),
        "gap2_direct_sp": float(sp(df["label"], df["Gap2"])),
        "gbsa_vs_label_sp": float(sp(df["label"], df["gbsa"])),
        "gap2_vs_gbsa_sp": float(sp(df["Gap2"], df["gbsa"])),
        "label_mean": float(df["label"].mean()),
        "label_std": float(df["label"].std()),
    }


def check_duplicate_sequences(df: pd.DataFrame) -> dict:
    """Check for duplicate or near-duplicate sequences."""
    seqs = df["Sequence"].values
    n_total = len(seqs)
    n_unique_141 = len(set(seqs))

    # 13 mutation-site pattern
    MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]
    patterns = []
    for s in seqs:
        patterns.append("".join(s[p - 1] for p in MUT_POS))
    n_unique_mut = len(set(patterns))

    return {
        "n_total": n_total,
        "n_unique_141nt": n_unique_141,
        "n_unique_mutation_pattern": n_unique_mut,
        "has_duplicate_sequences": n_unique_141 < n_total,
        "has_duplicate_mutations": n_unique_mut < n_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Data Registry & Integrity Check")
    parser.add_argument("--data-dir", default="data/proxy2000_v2",
                        help="Path to v2 data directory")
    parser.add_argument("--output", default="artifacts/phase0/data_registry.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load all data files ──────────────────────────────────────────
    files = {}
    expected = ["train.csv", "val.csv", "test.csv"]
    for fname in expected:
        fp = data_dir / fname
        if not fp.exists():
            print(f"[WARN] {fname} not found — skipping")
            continue
        files[fname] = pd.read_csv(fp)

    full_path = data_dir / "fad_proxy2000_v2_full.csv"
    has_full = full_path.exists()
    if has_full:
        files["full.csv"] = pd.read_csv(full_path)

    # Check for NUPACK-augmented files
    for fname in ["train_nupack.csv", "val_nupack.csv", "test_nupack.csv"]:
        fp = data_dir / fname
        if fp.exists():
            files[fname] = pd.read_csv(fp)

    print(f"Loaded {len(files)} files from {data_dir}")

    # ── Build registry ───────────────────────────────────────────────
    rows = []
    integrity = {}

    for fname, df in files.items():
        print(f"\n{'='*60}")
        print(f"  {fname}: {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"  Columns: {list(df.columns)[:12]}...")

        file_hash = hash_dataframe(df)
        file_path = str(data_dir / fname)
        file_size = (data_dir / fname).stat().st_size if (data_dir / fname).exists() else 0

        cols_str = ",".join(sorted(df.columns.tolist()))

        # Compute a per-row hash for the key columns
        key_cols = [c for c in ["Full", "Sequence", "MFE", "Aft", "gbsa", "label"] if c in df.columns]
        data_hash = hash_dataframe(df, key_cols) if key_cols else ""

        rows.append({
            "filename": fname,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": cols_str,
            "file_hash": file_hash,
            "key_data_hash": data_hash,
            "file_size_bytes": file_size,
            "file_path": file_path,
        })

        # Integrity checks on splits
        if fname in ("train.csv", "val.csv", "test.csv"):
            nupack_ok = verify_nupack_consistency(df)
            label_ok = verify_label_consistency(df)
            dup_ok = check_duplicate_sequences(df)

            rows[-1].update({
                "gap2_recalc_ok": nupack_ok["gap2_recalc_match"],
                "label_recalc_ok": label_ok["label_recalc_match"],
                "gap2_sp_vs_label": label_ok["gap2_direct_sp"],
                "gbsa_sp_vs_label": label_ok["gbsa_vs_label_sp"],
                "n_unique_141nt": dup_ok["n_unique_141nt"],
                "n_unique_mutation_pattern": dup_ok["n_unique_mutation_pattern"],
            })

            integrity[fname] = {**nupack_ok, **label_ok, **dup_ok}

            print(f"  Gap2 recalc: {nupack_ok['gap2_recalc_match']}")
            print(f"  Label recalc: {label_ok['label_recalc_match']}")
            print(f"  Gap2 direct Sp: {label_ok['gap2_direct_sp']:.4f}")
            print(f"  gbsa vs Gap2 Sp: {label_ok['gap2_vs_gbsa_sp']:.4f}")
            print(f"  Unique 141nt: {dup_ok['n_unique_141nt']}/{dup_ok['n_total']}")
            print(f"  Unique mutation: {dup_ok['n_unique_mutation_pattern']}/{dup_ok['n_total']}")

    # ── Cross-file consistency ───────────────────────────────────────
    if "train.csv" in files and "val.csv" in files and "test.csv" in files:
        tv = pd.concat([files["train.csv"], files["val.csv"]])
        te = files["test.csv"]
        # Check no sequence overlap between train+val and test
        tv_seqs = set(tv["Sequence"])
        te_seqs = set(te["Sequence"])
        overlap = tv_seqs & te_seqs
        print(f"\n  Train+Val vs Test overlap: {len(overlap)} sequences")

        # Check Gap2 distribution across splits
        print(f"  Train+Val Gap2: {tv['Gap2'].mean():.2f}±{tv['Gap2'].std():.2f}")
        print(f"  Test Gap2:      {te['Gap2'].mean():.2f}±{te['Gap2'].std():.2f}")

        # Cross-file consistency
        for r in rows:
            if r["filename"] == "train.csv":
                r["train_val_test_overlap"] = len(overlap)

    # ── Save ─────────────────────────────────────────────────────────
    registry_df = pd.DataFrame(rows)
    registry_df.to_csv(output_path, index=False)
    print(f"\n[OK] Registry saved to {output_path}")

    # Also save detailed integrity report as JSON
    integrity_path = output_path.with_suffix(".json")
    with open(integrity_path, "w") as f:
        json.dump(integrity, f, indent=2, default=str)
    print(f"[OK] Integrity report saved to {integrity_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("REGISTRY SUMMARY")
    print(f"  Files registered: {len(rows)}")
    print(f"  File hashes consistent: {len(set(r['file_hash'] for r in rows)) == len(rows)}")
    all_ok = all(
        r.get("gap2_recalc_ok", True) and r.get("label_recalc_ok", True)
        for r in rows if "gap2_recalc_ok" in r
    )
    print(f"  Gap2 consistency: {'PASS' if all_ok else 'FAIL — CHECK DATA'}")
    print(f"  Label consistency: {'PASS' if all_ok else 'FAIL — CHECK DATA'}")


if __name__ == "__main__":
    main()
