"""Mutation Cluster Split Generator — Phase 0B.

Generates Hamming-distance-based cluster splits for evaluating
out-of-cluster generalization. Groups sequences by mutation-site similarity
and creates GroupKFold-compatible split indices.

Protocol:
  1. For each sequence, extract the 13 mutation-site pattern
  2. Compute pairwise Hamming distances on mutation sites
  3. Build groups: sequences with Hamming ≤ threshold form a group
  4. Write GroupKFold splits ensuring no group spans train/test

Usage:
  python scripts/tools/generate_cluster_splits.py \
    --hamming-threshold 1 \
    --n-splits 5 \
    --output data/splits/cluster_hamming_le_1/
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MUT_POS = [30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131]
BASES = "ACGU"


def extract_mutation_pattern(seq: str) -> str:
    """Extract the 13-nt mutation-site pattern from a 141-nt sequence."""
    seq = str(seq).upper().replace("T", "U")
    return "".join(seq[p - 1] for p in MUT_POS)


def hamming_distance(s1: str, s2: str) -> int:
    """Hamming distance between two equal-length strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def build_cluster_groups(patterns: list[str], threshold: int = 1) -> list[set[int]]:
    """Build groups of indices where within-group Hamming ≤ threshold.

    Uses a greedy connected-components approach:
      Two indices are in the same group if there exists a chain of
      sequences connecting them where each adjacent pair has Hamming ≤ threshold.
    """
    n = len(patterns)
    # Build adjacency: O(n²) but n=2000 → 4M comparisons, fast enough
    print(f"  Building adjacency for {n} sequences (threshold={threshold})...")

    # Use union-find for connected components
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Only compare if patterns differ in ≤ threshold positions
    # For threshold=1, pre-group by 12/13 identity to avoid O(n²)
    if threshold <= 1:
        # Hash by first 12 positions — sequences differing by ≤1 must share ≥12
        from collections import defaultdict
        hash_groups = defaultdict(list)
        for i, pat in enumerate(patterns):
            # Key: all 13 positions but mask each one in turn
            for mask_pos in range(13):
                key = (mask_pos, pat[:mask_pos] + "_" + pat[mask_pos + 1:])
                hash_groups[key].append(i)

        seen_pairs = set()
        for indices in hash_groups.values():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a, b = indices[i], indices[j]
                    if (a, b) not in seen_pairs:
                        seen_pairs.add((a, b))
                        if hamming_distance(patterns[a], patterns[b]) <= threshold:
                            union(a, b)
    else:
        # Full O(n²) for threshold > 1
        for i in range(n):
            for j in range(i + 1, n):
                if hamming_distance(patterns[i], patterns[j]) <= threshold:
                    union(i, j)

    # Collect groups
    from collections import defaultdict
    group_map = defaultdict(set)
    for i in range(n):
        group_map[find(i)].add(i)

    groups = list(group_map.values())
    return groups


def create_group_kfold_splits(
    groups: list[set[int]], n_splits: int = 5, total_n: int = 2000
) -> list[dict]:
    """Create GroupKFold-style splits.

    Assigns entire groups to folds, trying to balance fold sizes.
    Returns list of {fold_id, train_indices, test_indices} dicts.
    """
    # Sort groups by size (largest first) for better balancing
    sorted_groups = sorted(groups, key=len, reverse=True)

    # Greedy assignment: assign each group to the currently smallest fold
    fold_sizes = [0] * n_splits
    fold_groups = [[] for _ in range(n_splits)]

    for g in sorted_groups:
        # Find fold with smallest current size
        min_fold = min(range(n_splits), key=lambda f: fold_sizes[f])
        fold_groups[min_fold].append(g)
        fold_sizes[min_fold] += len(g)

    # Generate train/test splits (leave-one-fold-out)
    splits = []
    all_indices = set(range(total_n))
    for fold_id in range(n_splits):
        test_indices = set()
        for g in fold_groups[fold_id]:
            test_indices.update(g)
        train_indices = sorted(all_indices - test_indices)
        test_indices = sorted(test_indices)

        splits.append({
            "fold_id": fold_id,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "test_groups": len(fold_groups[fold_id]),
        })

    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate Cluster Splits")
    parser.add_argument("--hamming-threshold", type=int, default=1,
                        help="Max Hamming distance within a group (default: 1)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of GroupKFold splits (default: 5)")
    parser.add_argument("--data-dir", default="data/proxy2000_v2",
                        help="Path to v2 data directory")
    parser.add_argument("--output", default="data/splits/cluster_hamming_le_1/",
                        help="Output directory for split files")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────
    full_path = data_dir / "fad_proxy2000_v2_full.csv"
    if not full_path.exists():
        print(f"[ERROR] Full data not found at {full_path}")
        sys.exit(1)

    df = pd.read_csv(full_path)
    print(f"Data: {len(df)} sequences")

    # ── Extract mutation patterns ───────────────────────────────────
    patterns = [extract_mutation_pattern(s) for s in df["Sequence"]]
    n_unique = len(set(patterns))
    print(f"Unique mutation patterns: {n_unique}/{len(patterns)}")

    # ── Build groups ───────────────────────────────────────────────
    groups = build_cluster_groups(patterns, args.hamming_threshold)

    n_isolated = sum(1 for g in groups if len(g) == 1)
    n_clustered = sum(len(g) for g in groups if len(g) > 1)
    max_group = max(len(g) for g in groups)
    print(f"Groups: {len(groups)} total")
    print(f"  Isolated (size=1): {n_isolated}")
    print(f"  Clustered (size>1): {len(groups) - n_isolated} groups, {n_clustered} sequences")
    print(f"  Largest group: {max_group} sequences")

    # ── Create splits ──────────────────────────────────────────────
    splits = create_group_kfold_splits(groups, args.n_splits, len(df))
    print(f"\nSplits ({args.n_splits}-fold):")
    for s in splits:
        print(f"  Fold {s['fold_id']}: train={s['n_train']}, "
              f"test={s['n_test']}, test_groups={s['test_groups']}")

    # ── Save ───────────────────────────────────────────────────────
    # Save full split metadata
    meta = {
        "hamming_threshold": args.hamming_threshold,
        "n_splits": args.n_splits,
        "n_sequences": len(df),
        "n_groups": len(groups),
        "n_unique_patterns": n_unique,
        "n_isolated": n_isolated,
        "n_clustered": n_clustered,
        "max_group_size": max_group,
        "splits": splits,
    }
    with open(output_dir / "split_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=list)
    print(f"\n[OK] Metadata saved to {output_dir / 'split_metadata.json'}")

    # Save per-fold train/test indices as CSV
    for s in splits:
        fold_dir = output_dir / f"fold_{s['fold_id']}"
        fold_dir.mkdir(exist_ok=True)
        pd.DataFrame({"index": s["train_indices"]}).to_csv(
            fold_dir / "train_indices.csv", index=False)
        pd.DataFrame({"index": s["test_indices"]}).to_csv(
            fold_dir / "test_indices.csv", index=False)
    print(f"[OK] {args.n_splits} fold splits saved to {output_dir}")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CLUSTER SPLIT SUMMARY")
    print(f"  Hamming threshold: ≤{args.hamming_threshold}")
    print(f"  N splits: {args.n_splits}")
    print(f"  N groups: {len(groups)} ({n_isolated} isolated)")
    print(f"  Group size range: 1–{max_group}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
