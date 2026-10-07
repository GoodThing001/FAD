"""Candidate-pool report (plan §7.1 / package C evidence).

Generates candidate pools with `scripts/loop/mutate.py`, then reports — without
touching any label — the stratum histogram, capacity fill, overlap with the
canonical table and a given training set, Hamming geometry (distance to WT and to
the nearest canonical sequence) and the OOD fraction under the §7.1 rule.

Usage:
  python tools/candidate_pool_report.py --pool-size 2000 --training-size 400 \
      --seed 1 --out evidence/loop_20260930/candidate_pool_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))

from scripts.loop.interface import (MUT_POS, WT_141, candidate_id,  # noqa: E402
                                    n_from_wt, normalize, validate_sequences)
from scripts.loop.mutate import dedupe_against, generate_candidates, stratum_capacity  # noqa: E402
from scripts.loop.selection import _min_distance_to, _mutation_matrix, loo_min_distance  # noqa: E402


def summarise(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), "min": float(values.min()),
            "p05": float(np.percentile(values, 5)), "median": float(np.median(values)),
            "mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-size", type=int, default=2000)
    ap.add_argument("--training-size", type=int, default=400)
    ap.add_argument("--holdout-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-mutations", type=int, default=None,
                    help="default: min n_from_wt observed in the canonical table")
    ap.add_argument("--max-mutations", type=int, default=None,
                    help="default: max n_from_wt observed in the canonical table")
    ap.add_argument("--data-dir", default="data/proxy2000_v2")
    ap.add_argument("--out", default="evidence/loop_20260930/candidate_pool_report.json")
    a = ap.parse_args()

    df = pd.read_csv(ROOT / a.data_dir / "fad_proxy2000_v2_full.csv")
    canonical = [normalize(s) for s in df["Sequence"]]
    validate_sequences(canonical, check_scaffold=False)

    table_n = np.array([n_from_wt(s) for s in canonical])
    table_hist = {str(int(k)): int((table_n == k).sum()) for k in sorted(set(table_n.tolist()))}
    lo_mut = a.min_mutations if a.min_mutations is not None else int(table_n[table_n > 0].min())
    hi_mut = a.max_mutations if a.max_mutations is not None else int(table_n.max())

    rng = np.random.RandomState(a.seed)
    order = rng.permutation(len(canonical))
    training = [canonical[i] for i in order[:a.training_size]]
    holdout = [canonical[i] for i in order[a.training_size:a.training_size + a.holdout_size]]

    pool = generate_candidates(a.pool_size, seed=a.seed,
                               min_mutations=lo_mut, max_mutations=hi_mut)
    after_table = dedupe_against(pool, canonical)
    after_train = dedupe_against(after_table, training)
    dropped_known = after_table.stats.get("deduped_out", 0)
    dropped_train = after_train.stats.get("deduped_out", 0)

    kept2 = list(dict.fromkeys(after_train.sequences))   # internal duplicates
    validate_sequences(kept2)

    n_wt = np.array([n_from_wt(s) for s in kept2])
    P = _mutation_matrix(kept2)
    C = _mutation_matrix(canonical)
    d_canonical = _min_distance_to(P, C)
    train_n = np.array([n_from_wt(s) for s in training])
    lo, hi = int(train_n.min()), int(train_n.max())
    T = _mutation_matrix(training)
    d_train_self = loo_min_distance(T)
    finite = d_train_self[np.isfinite(d_train_self)]
    thresh = float(np.percentile(finite, 95)) if finite.size else float("inf")
    d_train = _min_distance_to(P, T)
    ood = ((n_wt < lo) | (n_wt > hi)) | (d_train > thresh)

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "seed": a.seed,
        "requested_pool_size": a.pool_size,
        "generation_n_from_wt_range": [lo_mut, hi_mut],
        "table_n_from_wt_histogram": table_hist,
        "table_n_from_wt_range": [int(table_n[table_n > 0].min()), int(table_n.max())],
        "generated": len(pool.sequences),
        "kept_after_dedupe": len(kept2),
        "dropped_already_in_canonical_table": dropped_known,
        "dropped_already_in_training_set": dropped_train,
        "stratum_capacity": {str(k): stratum_capacity(k) for k in range(lo_mut, hi_mut + 1)},
        "total_capacity_in_range": sum(stratum_capacity(k) for k in range(lo_mut, hi_mut + 1)),
        "requested_histogram": pool.mutation_histogram(),
        "kept_histogram": {str(k): int((n_wt == k).sum()) for k in sorted(set(n_wt.tolist()))},
        "distance_to_wt": summarise(n_wt),
        "distance_to_nearest_canonical": summarise(d_canonical),
        "distance_to_nearest_training": summarise(d_train),
        "training_n_from_wt_range": [lo, hi],
        "ood_threshold_hamming": thresh,
        "ood_fraction": float(ood.mean()),
        "unique_candidate_ids": len({candidate_id(s) for s in kept2}),
        "changed_position_coverage": {
            str(p): int(sum(1 for s in kept2 if s[p - 1] != WT_141[p - 1])) for p in MUT_POS},
        "notes": ("Label-free report: no gbsa/label/dock/Gap2 value is read. "
                  "`distance_to_nearest_canonical` uses the 13-site Hamming space only."),
    }
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("changed_position_coverage",)}, ensure_ascii=False, indent=2))
    print(f"[ok] {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
