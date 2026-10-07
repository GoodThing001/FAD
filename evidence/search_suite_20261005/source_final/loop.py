"""Closed-loop driver: generate -> predict -> select -> label -> retrain (v0.2.0).

State machine per round:
    selected -> awaiting_labels -> (labels_ingested) -> retrained -> next round

Rules implemented (see docs/项目记录/GBSA代理接口与候选闭环实施方案_20260930.md):
  - the candidate pool is re-filtered every round: labelled, previously selected,
    awaiting-label and failed sequences are never offered again;
  - `n_new_unique` counts only genuinely new, finite labels; the ledger records
    `n_labeled_before/after` and the model's own training size so the metric time
    point is unambiguous;
  - a prospective batch parks as `PARKED` (exit code 3) and no new batch is
    selected until that batch is closed;
  - `--resume` requires the existing checkpoint in the same run directory and
    refuses to start a fresh run silently.

Target: standalone GBSA (see scripts/loop/interface.py for the policy record).

Usage:
  python -m scripts.loop.loop --mode retrospective --rounds 5 --budget 100
  python -m scripts.loop.loop --mode prospective --oracle park --rounds 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

try:
    from .interface import (INTERFACE_VERSION, MUT_POS, PREDICTION_COLUMN,
                            TARGET_COLUMN, Prediction, TargetSpec, WT_141, candidate_id,
                            normalize, validate_sequences)
    from .mutate import generate_candidates
    from .oracle import CommandOracle, ParkOracle, TableOracle
    from .search import SearchContext, SurrogateScorer
    from .search_archive import SearchArchive
    from .search_policies import make_policy as make_search_policy
    from .selection import DEFAULT_RATIOS, STRATEGIES, hamming_novelty, select_batch
    from .surrogates import make_surrogate
except ImportError:  # plain-script invocation (run_experiment.py entrypoint)
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.loop.interface import (INTERFACE_VERSION, MUT_POS, PREDICTION_COLUMN,
                                        TARGET_COLUMN, Prediction, TargetSpec, WT_141, candidate_id,
                                        normalize, validate_sequences)
    from scripts.loop.mutate import generate_candidates
    from scripts.loop.oracle import CommandOracle, ParkOracle, TableOracle
    from scripts.loop.search import SearchContext, SurrogateScorer
    from scripts.loop.search_archive import SearchArchive
    from scripts.loop.search_policies import make_policy as make_search_policy
    from scripts.loop.selection import (DEFAULT_RATIOS, STRATEGIES, hamming_novelty,
                                        select_batch)
    from scripts.loop.surrogates import make_surrogate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXIT_PARKED = 3


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.all(np.isnan(a)) or np.all(np.isnan(b)):
        return float("nan")
    return float(pearsonr(rankdata(a), rankdata(b))[0])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seqs_digest(seqs) -> str:
    return hashlib.sha256("\n".join(sorted(seqs)).encode("utf-8")).hexdigest()


def pool_train_distances(pool_seqs, train_seqs) -> np.ndarray:
    """Min Hamming distance (13 sites) from each pool sequence to the training set."""
    from scripts.loop.selection import _min_distance_to, _mutation_matrix
    if len(train_seqs) == 0:
        return np.full(len(pool_seqs), np.inf)
    return _min_distance_to(_mutation_matrix(pool_seqs), _mutation_matrix(train_seqs))


def ood_threshold(train_seqs) -> dict:
    """§7.1 threshold, computed from the training set only (no labels involved)."""
    if len(train_seqs) == 0:
        return {"n_from_wt_range": None, "hamming_threshold": None}
    train_n = np.array([sum(1 for p in MUT_POS if normalize(s)[p - 1] != WT_141[p - 1])
                        for s in train_seqs])
    from scripts.loop.selection import _mutation_matrix, loo_min_distance
    d_train = loo_min_distance(_mutation_matrix(train_seqs))
    finite = d_train[np.isfinite(d_train)]
    thresh = float(np.percentile(finite, 95)) if finite.size else float("inf")
    return {"n_from_wt_range": [int(train_n.min()), int(train_n.max())],
            "hamming_threshold": thresh}


def compute_ood(pool_seqs, train_seqs) -> np.ndarray:
    """§7.1 OOD rule: n_from_wt outside the training range, or nearest-train
    Hamming distance above the 95th percentile of the training leave-one-out
    nearest-neighbour distances. Training-only, no labels involved.

    Callers must pass the training set **as it was before the current round's
    labels were added**; otherwise already-measured candidates look closer to the
    training set than they were at selection time.
    """
    if len(train_seqs) == 0:
        return np.zeros(len(pool_seqs), dtype=bool)
    info = ood_threshold(train_seqs)
    lo, hi = info["n_from_wt_range"]
    thresh = info["hamming_threshold"]
    pool_n = np.array([sum(1 for p in MUT_POS if normalize(s)[p - 1] != WT_141[p - 1])
                       for s in pool_seqs])
    out_of_range = (pool_n < lo) | (pool_n > hi)

    from scripts.loop.selection import _min_distance_to, _mutation_matrix
    T = _mutation_matrix(train_seqs)
    d_pool = _min_distance_to(_mutation_matrix(pool_seqs), T)
    return out_of_range | (d_pool > thresh)


def select_search_group_batch(pool_seqs, prediction: Prediction, group_ids,
                              *, budget: int, strategy: str, labeled_seqs,
                              ratios, direction: str, seed: int,
                              diverse_multiplier: int) -> dict:
    """Select an explicit, capacity-aware near-even quota from 1–2 groups."""
    if len(pool_seqs) != len(group_ids) or len(pool_seqs) != len(prediction):
        raise ValueError("search group assignment must cover the whole pool")
    groups = list(dict.fromkeys(group_ids))
    if not groups or len(groups) > 2 or any(not g for g in groups):
        raise ValueError("search hand-off needs one or two non-empty groups")
    positions = {g: [i for i, x in enumerate(group_ids) if x == g] for g in groups}
    effective = min(int(budget), len(pool_seqs))
    quotas = {groups[0]: min((effective + 1) // 2, len(positions[groups[0]]))}
    if len(groups) == 2:
        quotas[groups[1]] = min(effective - quotas[groups[0]],
                                len(positions[groups[1]]))
        deficit = effective - sum(quotas.values())
        for g in groups:
            take = min(deficit, len(positions[g]) - quotas[g])
            quotas[g] += take
            deficit -= take
    else:
        quotas[groups[0]] = effective
    chosen: list[int] = []
    roles: dict[int, str] = {}
    for g in groups:
        idx = positions[g]
        if not quotas[g]:
            continue
        sub_pred = Prediction(
            mu=prediction.mu[idx],
            sigma=prediction.sigma[idx] if prediction.has_sigma else None,
            uncertainty_kind=prediction.uncertainty_kind,
            model_id=prediction.model_id,
            feature_version=prediction.feature_version,
            members=prediction.members)
        part = select_batch([pool_seqs[i] for i in idx], sub_pred,
                            budget=quotas[g], strategy=strategy,
                            labeled_seqs=labeled_seqs, ratios=ratios,
                            direction=direction, seed=seed,
                            diverse_multiplier=diverse_multiplier)
        for local in part["indices"]:
            global_pos = idx[int(local)]
            chosen.append(global_pos)
            roles[global_pos] = part["roles"][int(local)]
    chosen.sort()
    return {"indices": chosen, "roles": roles,
            "candidate_ids": [candidate_id(pool_seqs[i]) for i in chosen],
            "effective_budget": len(chosen), "group_quotas": quotas,
            "note": "near-even groups; short group capacity redistributed"}


@dataclass
class LoopState:
    round: int = 0
    status: str = "init"               # init | selected | awaiting_labels | retrained
    open_batch: str | None = None
    labeled_sequences: list[str] = field(default_factory=list)
    labeled_values: list[float] = field(default_factory=list)
    selected_history: list[str] = field(default_factory=list)
    awaiting: dict[str, str] = field(default_factory=dict)   # sequence -> batch_id
    failed: list[str] = field(default_factory=list)
    batches: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    provenance: dict[str, dict] = field(default_factory=dict)

    def to_json(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_json(cls, d: dict) -> "LoopState":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=f"FAD closed loop | interface v{INTERFACE_VERSION}")
    p.add_argument("--mode", default="retrospective", choices=["retrospective", "prospective"])
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--n-init", type=int, default=400)
    p.add_argument("--n-holdout", type=int, default=300)
    p.add_argument("--pool-size", type=int, default=900)
    p.add_argument("--candidate-source", default="canonical",
                   choices=["canonical", "mutate", "search"])
    p.add_argument("--candidate-quotas", default=None,
                   help="mutate mode: 'n_from_wt:count,...' strata, e.g. '8:200,10:200,12:100'")
    # search-layer controls (candidate-source=search; every default must also be
    # written into the on-disk config — no implicit values)
    p.add_argument("--search-policy", default="multi_start_beam",
                   choices=["stratified_random", "matched_seed_random_walk",
                            "multi_start_hill_climb", "multi_start_beam", "mutation_only_ga"])
    p.add_argument("--search-max-unique-predictions", type=int, default=256)
    p.add_argument("--search-max-generations", type=int, default=8)
    p.add_argument("--search-allowed-n-from-wt", default="4:13",
                   help="'lo:hi' mutation shell for the search layer")
    p.add_argument("--search-beam-width", type=int, default=32)
    p.add_argument("--search-offspring-per-parent", type=int, default=32)
    p.add_argument("--search-one-hop-fraction", type=float, default=0.8)
    p.add_argument("--search-restart-fraction", type=float, default=0.1)
    p.add_argument("--search-diversity-pool-factor", type=int, default=4)
    p.add_argument("--search-diversity-distance", type=int, default=2)
    p.add_argument("--search-max-stagnant-generations", type=int, default=None)
    p.add_argument("--search-n-measured-seeds", type=int, default=8)
    p.add_argument("--search-n-random-seeds", type=int, default=8)
    p.add_argument("--search-min-mutual-distance", type=int, default=2)
    p.add_argument("--search-population-size", type=int, default=32)
    p.add_argument("--search-offspring-size", type=int, default=128)
    p.add_argument("--search-tournament-size", type=int, default=2)
    p.add_argument("--search-elite-count", type=int, default=2)
    p.add_argument("--min-mutations", type=int, default=None,
                   help="mutate mode: minimum n_from_wt (default: min over the labeled set)")
    p.add_argument("--max-mutations", type=int, default=None,
                   help="mutate mode: maximum n_from_wt (default: max over the labeled set)")
    p.add_argument("--surrogate", default="xgb", choices=["xgb", "fusion", "rf"])
    p.add_argument("--n-members", type=int, default=10)
    p.add_argument("--screen-k", type=int, default=100)
    p.add_argument("--oracle", default="table", choices=["table", "park", "command"])
    p.add_argument("--oracle-command", default=None)
    p.add_argument("--strategy", default="greedy", choices=list(STRATEGIES))
    p.add_argument("--ratios", default=",".join(str(r) for r in DEFAULT_RATIOS))
    p.add_argument("--direction", default="minimize", choices=["minimize", "maximize"])
    p.add_argument("--diverse-multiplier", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ingest", default=None, help="CSV of returned measurements to absorb before running")
    p.add_argument("--protocol-id", default="unassigned",
                   help="measurement protocol this batch is exported under")
    p.add_argument("--approved-protocols", default="",
                   help="comma-separated protocol ids accepted at ingest; empty = any non-empty")
    p.add_argument("--data-dir", default="data/proxy2000_v2")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--post-retrain-eval", action="store_true", default=True)
    p.add_argument("--no-post-retrain-eval", dest="post_retrain_eval", action="store_false")
    p.add_argument("--output", default="runs/loop/metrics.csv")
    p.add_argument("--log-dir", default="runs/loop/checkpoints")
    args = p.parse_args(argv)
    if args.search_max_stagnant_generations is None:
        args.search_max_stagnant_generations = 4 if args.search_policy == 'mutation_only_ga' else 2
    return args


class ClosedLoop:
    def __init__(self, args):
        self.args = args
        self._validate_args()
        self.target = TargetSpec(direction=args.direction)
        self.log_dir = Path(args.log_dir)
        if not self.log_dir.is_absolute():
            self.log_dir = PROJECT_ROOT / self.log_dir
        self.state_path = self.log_dir / "state.json"
        self.resumed = bool(args.resume)
        if self.resumed and not self.state_path.is_file():
            raise FileNotFoundError(
                f"--resume requested but no checkpoint at {self.state_path}; "
                "refusing to silently start a new run")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ratios = [int(x) for x in str(args.ratios).split(",")]
        self.approved_protocols = [x.strip() for x in str(args.approved_protocols).split(",")
                                   if x.strip()]
        self.fingerprint: dict = {}
        self.pool_stats: dict = {}
        self.pool: list[str] = []
        self.holdout: list[str] = []
        self.holdout_values: np.ndarray | None = None
        self.pool_truth: dict[str, float] = {}
        self.oracle = None
        self.state = LoopState()
        self.ledger: list[dict] = []
        self._canonical_sequences: set[str] = set()
        self._data_sha256: str | None = None
        # argparse stores the range as a string; normalize to a JSON-stable list
        # so the resume fingerprint comparison stays type-consistent
        lo, hi = str(args.search_allowed_n_from_wt).split(":")
        args.search_allowed_n_from_wt = [int(lo), int(hi)]

    def _validate_args(self) -> None:
        a = self.args
        if a.n_init <= 0 or a.n_holdout < 0 or a.pool_size <= 0 or a.budget < 0:
            raise ValueError("n-init>0, n-holdout>=0, pool-size>0, budget>=0 required")
        if len(str(a.ratios).split(",")) != 4:
            raise ValueError("--ratios needs exactly 4 comma-separated values")
        if any(int(x) < 0 for x in str(a.ratios).split(",")):
            raise ValueError("--ratios must be non-negative")
        if a.candidate_source == "search" and a.mode == "retrospective":
            raise ValueError(
                "candidate-source=search is prospective-only: search candidates "
                "live outside the canonical table (the table is part of the "
                "search exclusion set), so only real measurements (ParkOracle) "
                "can label them; TableOracle has nothing to reveal for them")
        if a.search_max_unique_predictions < 1:
            raise ValueError("--search-max-unique-predictions must be >= 1")

    # ---------------- setup ----------------
    def build_setup(self) -> None:
        a = self.args
        rng = np.random.RandomState(a.seed)
        data_path = PROJECT_ROOT / a.data_dir / "fad_proxy2000_v2_full.csv"
        df = pd.read_csv(data_path)
        df["Sequence"] = df["Sequence"].map(normalize)
        if df["Sequence"].duplicated().any():
            raise ValueError("canonical data contains duplicate sequences")
        values = df[TARGET_COLUMN].astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("canonical gbsa contains non-finite values")
        self._canonical_sequences = set(df["Sequence"])
        self._data_sha256 = sha256_file(data_path)
        perm = rng.permutation(len(df))
        idx_init = perm[:a.n_init]
        idx_hold = perm[a.n_init:a.n_init + a.n_holdout]
        init_seqs = df["Sequence"].values[idx_init].tolist()
        init_vals = values[idx_init].tolist()
        self.holdout = df["Sequence"].values[idx_hold].tolist()
        self.holdout_values = values[idx_hold]

        if a.mode == "retrospective":
            n_pool = min(a.pool_size, len(df) - a.n_init - a.n_holdout)
            if n_pool <= 0:
                raise ValueError("n-init + n-holdout leaves no pool")
            idx_pool = perm[a.n_init + a.n_holdout:a.n_init + a.n_holdout + n_pool]
            self.pool = df["Sequence"].values[idx_pool].tolist()
            self.pool_truth = dict(zip(self.pool, values[idx_pool]))
            self.pool_stats = {"requested": a.pool_size, "generated": len(self.pool),
                               "source": "canonical_table_holdout_split",
                               "n_init": a.n_init, "n_holdout": len(self.holdout),
                               "split_seed": a.seed}
            self.oracle = TableOracle(df, self.pool, holdout=self.holdout)
            print(f"[setup] retrospective: init={len(init_seqs)} holdout={len(self.holdout)} "
                  f"pool={len(self.pool)}")
        else:
            size = max(a.pool_size, a.rounds * max(a.budget, 1) * 2)
            # The oracle comes first: its historical batches take part in the
            # deduplication below (P1-a).
            if a.oracle == "park":
                self.oracle = ParkOracle(self.log_dir / "pending_batches",
                                         approved_protocols=self.approved_protocols)
            elif a.oracle == "command":
                if not a.oracle_command:
                    raise ValueError("--oracle command requires --oracle-command")
                self.oracle = CommandOracle(a.oracle_command)
            else:
                raise ValueError("prospective mode supports --oracle park|command")
            if a.candidate_source == "mutate":
                quotas = None
                if a.candidate_quotas:
                    quotas = {}
                    for item in str(a.candidate_quotas).split(","):
                        k, v = item.split(":")
                        quotas[int(k)] = int(v)
                # The measured library sits in a *shell* around WT (this project: 8-11 of
                # the 13 sites mutated).  Generating 1-5 mutation candidates would place
                # the whole pool outside the labeled region, so the default range is taken
                # from the labeled set unless the caller overrides it explicitly.
                labeled_n = [sum(1 for p in MUT_POS if s[p - 1] != WT_141[p - 1])
                             for s in init_seqs]
                lo = a.min_mutations if a.min_mutations is not None else min(labeled_n)
                hi = a.max_mutations if a.max_mutations is not None else max(labeled_n)
                lo = max(1, min(lo, len(MUT_POS)))
                hi = max(lo, min(hi, len(MUT_POS)))
                # Full-library dedup at generation time (P1-a): exclude every sequence
                # we already know — the whole canonical table, the holdout and the
                # initial labelled set.  Historical batches are excluded dynamically in
                # `_available()`, so the generated pool (and therefore the resume
                # fingerprint) does not depend on how many rounds already ran.
                exclude = sorted(set(map(normalize, df["Sequence"]))
                                 | set(map(normalize, init_seqs))
                                 | set(map(normalize, self.holdout)))
                pool = generate_candidates(sum(quotas.values()) if quotas else size,
                                           quotas=quotas,
                                           min_mutations=lo, max_mutations=hi,
                                           exclude=exclude, seed=a.seed)
                self.pool = pool.sequences
                for row in pool.to_rows():
                    self.state.provenance[row["sequence"]] = row
                self.pool_stats = {
                    "requested": sum(quotas.values()) if quotas else size,
                    "generated": len(pool.sequences),
                    "excluded_total": len(exclude),
                    "excluded_canonical_table": len(set(map(normalize, df["Sequence"]))),
                    "excluded_init_labeled": len(set(map(normalize, init_seqs))),
                    "excluded_holdout": len(set(map(normalize, self.holdout))),
                    "n_from_wt_range": [lo, hi],
                }
                print(f"[setup] dedup at generation: excluded {self.pool_stats['excluded_total']} "
                      f"known sequences (canonical {self.pool_stats['excluded_canonical_table']}, "
                      f"init {self.pool_stats['excluded_init_labeled']}, "
                      f"holdout {self.pool_stats['excluded_holdout']}); "
                      f"kept {len(pool.sequences)}")
                n_wt_pool = [int(self.state.provenance[s]["n_from_wt"]) for s in self.pool]
                if n_wt_pool:
                    lo_p, hi_p = min(n_wt_pool), max(n_wt_pool)
                    init_n = [sum(1 for p in MUT_POS if s[p - 1] != WT_141[p - 1])
                              for s in init_seqs]
                    print(f"[setup] pool n_from_wt range [{lo_p},{hi_p}] vs labeled "
                          f"[{min(init_n)},{max(init_n)}]"
                          + ("  [warn] pool is outside the labeled mutation shell: "
                             "expect OOD extrapolation" if lo_p < min(init_n)
                             or hi_p > max(init_n) else ""))
            elif a.candidate_source == "search":
                # The pool is DYNAMIC: rebuilt from the search layer at the start
                # of every round (search on the current labelled set, see
                # `_rebuild_search_pool`).  Setup keeps an empty pool so the
                # resume fingerprint (pool_sha256 of the setup pool) is stable;
                # pool_stats below holds only config-derived values for the same
                # reason — runtime search provenance is written per round.
                self.pool = []
                self.pool_stats = {
                    "requested": a.pool_size, "generated": 0,
                    "source": "search",
                    "search_policy": a.search_policy,
                    "search_max_unique_predictions": a.search_max_unique_predictions,
                    "search_allowed_n_from_wt": a.search_allowed_n_from_wt,
                    "n_init": a.n_init, "n_holdout": len(self.holdout),
                    "split_seed": a.seed,
                }
            else:
                rest = [s for s in df["Sequence"] if s not in set(init_seqs) | set(self.holdout)]
                self.pool = rest[:size]
                self.pool_stats = {"requested": size, "generated": len(self.pool),
                                   "excluded_total": len(set(init_seqs) | set(self.holdout)),
                                   "source": "canonical_table"}
            print(f"[setup] prospective: init={len(init_seqs)} holdout={len(self.holdout)} "
                  f"pool={len(self.pool)} (source={a.candidate_source}"
                  + (f", n_from_wt {lo}-{hi}" if a.candidate_source == "mutate"
                     else f", search_policy={a.search_policy}"
                     if a.candidate_source == "search" else "") + ")")

        if not self.resumed:
            self.state.labeled_sequences = init_seqs
            self.state.labeled_values = init_vals
            self.state.provenance.update({
                s: {"parent": WT_141, "n_from_parent": 0,
                    "n_from_wt": sum(1 for p in MUT_POS if s[p - 1] != WT_141[p - 1]),
                    "changed_positions_from_parent": [], "source": "init_labeled"}
                for s in init_seqs})

    # ---------------- helpers ----------------
    def _known_sequences(self) -> set[str]:
        """Sequences that must never be exported again (P1-a, export-time dedup).

        Generation already removes the canonical table / holdout / init labels; this
        adds everything written into an earlier batch (measured, pending or failed),
        so re-running a round cannot re-offer a sequence an external lab already has.
        """
        known: set[str] = set()
        if hasattr(self.oracle, "all_batch_sequences"):
            known |= {normalize(s) for s in self.oracle.all_batch_sequences()}
        return known

    def _available(self) -> list[int]:
        blocked = (set(self.state.labeled_sequences) | set(self.state.selected_history)
                   | set(self.state.awaiting) | set(self.state.failed)
                   | self._known_sequences())
        return [i for i, s in enumerate(self.pool) if s not in blocked]

    def _rebuild_search_pool(self, model) -> None:
        """Rebuild the candidate pool from the search layer (candidate-source=search).

        Runs (or resumes) the configured search policy on the CURRENT labelled set
        with the just-fitted model as scorer.  The search only predicts — no
        oracle query happens here; batch selection keeps using the ordinary
        select_batch / Oracle flow below.  The per-round archive under the log
        dir is reused verbatim when the labelled set is unchanged, so a resume
        re-derives the identical pool (the search itself is deterministic given
        seed + labelled set).  The canonical table, holdout, labelled / selected /
        awaiting / failed sequences and every historical batch are all excluded
        from search proposals.
        """
        a = self.args
        round_no = self.state.round + 1
        archive_dir = self.log_dir / f"round_{round_no:03d}_search"
        labeled = list(self.state.labeled_sequences)
        labeled_vals = [float(v) for v in self.state.labeled_values]
        forbidden = (self._canonical_sequences
                     | set(labeled)
                     | set(self.state.selected_history)
                     | set(self.state.awaiting)
                     | set(self.state.failed)
                     | self._known_sequences())
        context = SearchContext(
            search_id=f"search_r{round_no}_{a.seed}_{a.search_policy}",
            outer_round=round_no,
            measured_sequences=tuple(labeled),
            measured_gbsa=tuple(labeled_vals),
            forbidden_sequences=frozenset(forbidden),
            seed=a.seed + self.state.round,
            allowed_n_from_wt=tuple(a.search_allowed_n_from_wt),
            max_unique_predictions=a.search_max_unique_predictions,
            max_generations=a.search_max_generations,
            data_sha256=self._data_sha256 or "",
            archive_dir=str(archive_dir),
        )
        policy = make_search_policy(
            a.search_policy,
            beam_width=a.search_beam_width,
            offspring_per_parent=a.search_offspring_per_parent,
            one_hop_fraction=a.search_one_hop_fraction,
            restart_fraction=a.search_restart_fraction,
            diversity_pool_factor=a.search_diversity_pool_factor,
            diversity_distance=a.search_diversity_distance,
            max_stagnant_generations=a.search_max_stagnant_generations,
            n_measured_seeds=a.search_n_measured_seeds,
            n_random_seeds=a.search_n_random_seeds,
            min_mutual_distance=a.search_min_mutual_distance,
            population_size=a.search_population_size,
            offspring_size=a.search_offspring_size,
            tournament_size=a.search_tournament_size,
            elite_count=a.search_elite_count)
        archive = SearchArchive(archive_dir)
        scorer = SurrogateScorer(
            model, context, cache=archive.load_cache(),
            ood_train_sequences=labeled,
            model_id=str(getattr(model, "model_id", None) or "unset"),
            feature_version=str(getattr(model, "feature_version", None) or "unset"))
        result = policy.search(context, scorer)
        ranked = list(result.ranked_candidates)
        group_by_id = {cid: g for g in result.groups for cid in g.member_ids}
        if len(group_by_id) != len(ranked):
            raise ValueError("search groups do not cover every ranked candidate")
        # Reserve half of the pool for each group before batch selection.  A
        # top-N-only pool could otherwise erase the diverse backup group.
        by_group = [[c for c in ranked
                     if group_by_id[c.candidate_id].group_id == g.group_id]
                    for g in result.groups]
        if len(by_group) == 2:
            first = min((a.pool_size + 1) // 2, len(by_group[0]))
            second = min(a.pool_size - first, len(by_group[1]))
            first += min(a.pool_size - first - second,
                         len(by_group[0]) - first)
            second += min(a.pool_size - first - second,
                          len(by_group[1]) - second)
            pool_candidates = by_group[0][:first] + by_group[1][:second]
        else:
            pool_candidates = ranked[:a.pool_size]
        pool_candidates.sort(key=lambda c: (c.pred_gbsa, c.candidate_id))
        self.pool = [c.sequence for c in pool_candidates]
        evaluated = {r["candidate_id"]: r for r in archive.read_evaluated()}
        for c in pool_candidates:
            parent = evaluated.get(c.parent_candidate_id)
            if c.parent_candidate_id and parent is None:
                raise ValueError(f"missing search parent {c.parent_candidate_id}")
            parent_seq = parent["sequence"] if parent else ""
            actual_changed = tuple(p for p in MUT_POS if parent_seq and
                                   c.sequence[p - 1] != parent_seq[p - 1])
            if parent and (len(actual_changed) != c.n_from_parent or
                           actual_changed != c.changed_positions_from_parent):
                raise ValueError(f"search lineage mismatch for {c.candidate_id}")
            g = group_by_id[c.candidate_id]
            self.state.provenance[c.sequence] = {
                "source": "search_pool", "parent": parent_seq,
                "parent_candidate_id": c.parent_candidate_id or "",
                "parent_evidence": c.parent_evidence,
                "n_from_parent": c.n_from_parent,
                "n_from_wt": c.n_from_wt,
                "changed_positions_from_parent": list(c.changed_positions_from_parent),
                "search_id": c.search_id, "search_generation": c.generation,
                "search_operator": c.operator,
                "group_id": g.group_id, "group_role": g.group_role,
            }
        try:
            search_archive_rel = str(archive_dir.relative_to(PROJECT_ROOT))
        except ValueError:  # archive dir outside the project root (e.g. tests)
            search_archive_rel = str(archive_dir)
        self.pool_stats.update({
            "generated": len(self.pool),
            "search_id": context.search_id,
            "search_n_unique_predicted": result.n_unique_predicted,
            "search_ranked": len(ranked),
            "search_stop_reason": result.stop_reason,
            "search_best_pred_gbsa": (float(ranked[0].pred_gbsa) if ranked else None),
            "search_n_groups": len(result.groups),
            "search_pool_group_counts": {g.group_id: sum(
                group_by_id[c.candidate_id].group_id == g.group_id
                for c in pool_candidates)
                for g in result.groups},
            "search_archive": search_archive_rel,
            "search_excluded_total": len(forbidden),
        })
        print(f"[search] round {round_no}: policy={a.search_policy} "
              f"budget={result.n_unique_predicted} stop={result.stop_reason} "
              f"ranked={len(ranked)} pool={len(self.pool)} "
              f"best_pred={self.pool_stats['search_best_pred_gbsa']}")

    def _absorb_returned_labels(self) -> int:
        """Move parked labels (and QC failures) into the state; idempotent.

        Ledger rows written at export time carry `status=awaiting_labels` and no
        `measured_gbsa`; absorbing a returned label must flip them to `labeled`
        with the measured value, otherwise a fully returned batch still looks
        pending in the ledger (and in every downstream export).
        """
        if not self.state.awaiting:
            return 0
        seqs = list(self.state.awaiting)
        if hasattr(self.oracle, "peek"):
            values, available = self.oracle.peek(seqs)
        else:                                                    # pragma: no cover
            values, available = self.oracle.query("__status__", seqs)
        n_new = 0
        for s, v, ok in zip(seqs, values, available):
            if ok and np.isfinite(v) and s not in set(self.state.labeled_sequences):
                self.state.labeled_sequences.append(s)
                self.state.labeled_values.append(float(v))
                self.state.provenance.setdefault(s, {"source": "prospective_return"})
                del self.state.awaiting[s]
                n_new += 1
                for entry in self.ledger:
                    if entry.get("sequence") == s \
                            and str(entry.get("status")) == "awaiting_labels":
                        entry["status"] = "labeled"
                        entry["measured_gbsa"] = float(v)
        # QC failures are persisted by the oracle; absorb them so the candidate is
        # never offered again and the batch can close.
        n_failed = 0
        if hasattr(self.oracle, "failed_sequences"):
            for s in list(self.state.awaiting):
                if s in self.oracle.failed_sequences():
                    self.state.failed.append(s)
                    del self.state.awaiting[s]
                    n_failed += 1
                    for entry in self.ledger:
                        if entry.get("sequence") == s \
                                and str(entry.get("status")) == "awaiting_labels":
                            entry["status"] = "failed"
                            entry["measured_gbsa"] = None
                            entry["selection_role"] = str(
                                entry.get("selection_role", "unassigned")) + ":qc_fail"
        if n_new or n_failed:
            self.state.status = "labels_ingested"
        if n_failed:
            print(f"  [ingest] {n_failed} candidate(s) marked failed by QC; "
                  f"they stay out of the pool")
        if not self.state.awaiting:
            if self.state.open_batch:
                closed = self.state.open_batch
                self.state.open_batch = None
                self.state.batches.append({"batch_id": closed, "round": self.state.round,
                                           "strategy": self.args.strategy,
                                           "n": 0, "n_labeled": 0,
                                           "status": "closed"})
                print(f"  [ingest] batch {closed} closed (no candidates awaiting labels)")
                stale = self.log_dir / "PARKED"
                if stale.is_file():
                    stale.unlink()
                    print("  [ingest] removed the stale PARKED marker")
        return n_new

    def _batch_is_open(self) -> bool:
        return bool(self.state.awaiting)

    def _make_model(self):
        return make_surrogate(self.args.surrogate, seed=self.args.seed,
                              n_members=self.args.n_members,
                              screen_k=self.args.screen_k, target=self.target)

    # ---------------- one round ----------------
    def run_round(self) -> dict:
        a = self.args
        t0 = time.time()
        absorbed = self._absorb_returned_labels()
        if absorbed:
            print(f"  [resume] absorbed {absorbed} returned labels")
        if self._batch_is_open():
            print(f"[stop] batch {self.state.open_batch} still awaiting labels "
                  f"({len(self.state.awaiting)} pending); no new batch selected")
            return self._status_row("awaiting_labels")

        n_before = len(self.state.labeled_sequences)
        model = self._make_model()
        model.fit(self.state.labeled_sequences,
                  np.asarray(self.state.labeled_values, dtype=float))
        if a.candidate_source == "search":
            # dynamic pool: search on the current labelled set (post-absorb),
            # then fall through to the ordinary select/query flow
            self._rebuild_search_pool(model)

        available = self._available()
        if not available:
            print("[stop] candidate pool exhausted")
            return self._status_row("pool_exhausted")
        avail_seqs = [self.pool[i] for i in available]
        mu_all = model.predict(avail_seqs)

        # §7.1: OOD is a *pre-acquisition* statement.  Compute it against the
        # training set as it was before this round's labels were revealed, and keep
        # the threshold so the record can be audited later.
        ood_info = ood_threshold(self.state.labeled_sequences)
        ood_pool = compute_ood(avail_seqs, self.state.labeled_sequences)
        d_pool = pool_train_distances(avail_seqs, self.state.labeled_sequences)

        selection_args = dict(budget=a.budget, strategy=a.strategy,
                              labeled_seqs=self.state.labeled_sequences,
                              ratios=self.ratios, direction=a.direction,
                              seed=a.seed + self.state.round,
                              diverse_multiplier=a.diverse_multiplier)
        if a.candidate_source == "search":
            selection = select_search_group_batch(
                avail_seqs, mu_all,
                [self.state.provenance[s]["group_id"] for s in avail_seqs],
                **selection_args)
        else:
            selection = select_batch(avail_seqs, mu_all, **selection_args)
        # `selection["indices"]` are positions inside `available`/`avail_seqs`, NOT
        # positions inside the batch: every per-candidate lookup below must go through
        # them (writing `mu_all.mu[k]` for batch slot k was the ledger index bug).
        sel_pos = [int(k) for k in selection["indices"]]
        batch_idx = [available[k] for k in sel_pos]
        batch_seqs = [self.pool[i] for i in batch_idx]
        if not batch_seqs:
            print("[stop] selection returned an empty batch")
            return self._status_row("empty_batch")

        batch_id = candidate_id(f"{self.state.round + 1}|{a.strategy}|"
                                f"{'|'.join(selection['candidate_ids'])}", length=12)
        values, available_mask = self.oracle.query(batch_id, batch_seqs)

        n_new = 0
        for s, v, ok in zip(batch_seqs, values, available_mask):
            if ok and np.isfinite(v):
                if s not in set(self.state.labeled_sequences):
                    self.state.labeled_sequences.append(s)
                    self.state.labeled_values.append(float(v))
                    n_new += 1
            else:
                if s not in self.state.awaiting:
                    self.state.awaiting[s] = batch_id

        holdout_pre = float("nan")
        if self.holdout_values is not None and len(self.holdout):
            holdout_pre = spearman(self.holdout_values, model.predict(self.holdout).mu)

        post_retrain = float("nan")
        if a.post_retrain_eval and n_new > 0:
            model2 = self._make_model()
            model2.fit(self.state.labeled_sequences,
                       np.asarray(self.state.labeled_values, dtype=float))
            post_retrain = spearman(self.holdout_values, model2.predict(self.holdout).mu)

        self.state.selected_history.extend(batch_seqs)
        self.state.round += 1
        self.state.open_batch = batch_id if self.state.awaiting else None
        self.state.status = "awaiting_labels" if self.state.awaiting else "retrained"

        # ledger rows (immutable, one per candidate) — every prediction/OOD/role
        # lookup is aligned through `sel_pos` (position within the available pool).
        for k, (pos, i, s) in enumerate(zip(sel_pos, batch_idx, batch_seqs)):
            prov = self.state.provenance.get(s, {"source": "canonical_pool"})
            row = {
                "run_round": self.state.round,
                "batch_id": batch_id,
                "candidate_id": candidate_id(s),
                "sequence": s,
                "parent_sequence": prov.get("parent", WT_141),
                "parent_candidate_id": prov.get("parent_candidate_id", ""),
                "parent_evidence": prov.get("parent_evidence", ""),
                "changed_positions_from_parent": ",".join(
                    str(x) for x in prov.get("changed_positions_from_parent", [])),
                "n_from_parent": prov.get("n_from_parent", ""),
                "n_from_wt": sum(1 for p in MUT_POS if s[p - 1] != WT_141[p - 1]),
                "source": prov.get("source", "canonical_pool"),
                "search_id": prov.get("search_id", ""),
                "search_generation": prov.get("search_generation", ""),
                "search_operator": prov.get("search_operator", ""),
                "group_id": prov.get("group_id", ""),
                "group_role": prov.get("group_role", ""),
                "selection_policy": a.strategy,
                "selection_role": selection["roles"].get(pos, "unassigned"),
                "selection_position": pos,
                "model_id": getattr(model, "model_id", a.surrogate),
                "feature_version": getattr(model, "feature_version", "seq303"),
                "protocol_id": a.protocol_id,
                PREDICTION_COLUMN: float(mu_all.mu[pos]),
                "sigma_gbsa": (float(mu_all.sigma[pos]) if mu_all.has_sigma else None),
                "ood_flag": bool(ood_pool[pos]),
                "ood_hamming_threshold": ood_info["hamming_threshold"],
                "status": "labeled" if np.isfinite(values[k]) else "awaiting_labels",
                "measured_gbsa": (float(values[k]) if np.isfinite(values[k]) else None),
            }
            self.ledger.append(row)

        row = {
            "round": self.state.round,
            "strategy": a.strategy,
            "n_labeled_before": n_before,
            "n_new_unique": n_new,
            "n_labeled_after": len(self.state.labeled_sequences),
            "model_n_train": getattr(model, "n_train", n_before),
            "n_awaiting": len(self.state.awaiting),
            "requested_budget": a.budget,
            "effective_budget": selection["effective_budget"],
            "pool_available": len(available),
            "ood_fraction": round(float(np.mean(ood_pool)), 4) if len(ood_pool) else "",
            "batch_ood_fraction": round(float(np.mean([ood_pool[p] for p in sel_pos])), 4)
            if sel_pos else "",
            "ood_threshold_hamming": ood_info["hamming_threshold"],
            "ood_train_n_from_wt": (f"{ood_info['n_from_wt_range'][0]}-"
                                    f"{ood_info['n_from_wt_range'][1]}"
                                    if ood_info["n_from_wt_range"] else ""),
            "status": self.state.status,
            "holdout_spearman_pre": round(holdout_pre, 6) if np.isfinite(holdout_pre) else "",
            "holdout_spearman_post": round(post_retrain, 6) if np.isfinite(post_retrain) else "",
            "search_stop_reason": self.pool_stats.get("search_stop_reason", ""),
            "search_n_unique_predicted": self.pool_stats.get("search_n_unique_predicted", ""),
            "search_best_pred_gbsa": self.pool_stats.get("search_best_pred_gbsa", ""),
            "search_pool_size": (len(self.pool) if a.candidate_source == "search" else ""),
            "seconds": round(time.time() - t0, 1),
        }
        self.state.history.append(row)
        self._write_ood_strata(available, ood_pool, sel_pos, values, mu_all, d_pool)
        self._flush(batch_seqs, values, batch_id)
        print(f"[round {row['round']}/{a.rounds}] {a.strategy}: "
              f"labeled {n_before}->{row['n_labeled_after']} (new {n_new}) "
              f"awaiting={row['n_awaiting']} holdout_pre={row['holdout_spearman_pre']} "
              f"post={row['holdout_spearman_post']}")
        return row

    def _write_ood_strata(self, available, ood_pool, sel_pos, values, prediction,
                          d_pool) -> None:
        """§4: per-stratum counts plus (retrospective) error/distance diagnostics.

        The binary OOD flag alone is not evidence of an applicable domain, so each
        round also records, per `n_from_wt` stratum: how many pool candidates were
        flagged, their mean distance to the pre-acquisition training set, how many
        were selected, and — where the round revealed real labels — the mean absolute
        error of `pred_gbsa` and whether the stratum's best candidate was picked.
        """
        selected = set(int(p) for p in sel_pos)
        rows = []
        for stratum in sorted({int(sum(1 for p in MUT_POS
                                      if self.pool[i][p - 1] != WT_141[p - 1]))
                               for i in available}):
            pos_in_avail = [k for k, i in enumerate(available)
                            if sum(1 for p in MUT_POS
                                   if self.pool[i][p - 1] != WT_141[p - 1]) == stratum]
            picked = [k for k in pos_in_avail if k in selected]
            errors = [abs(float(prediction.mu[k]) - float(values[sel_pos.index(k)]))
                      for k in picked if np.isfinite(values[sel_pos.index(k)])]
            rows.append({
                "run_round": self.state.round,
                "n_from_wt": stratum,
                "pool_count": len(pos_in_avail),
                "pool_ood_count": int(sum(bool(ood_pool[k]) for k in pos_in_avail)),
                "mean_min_train_hamming": round(
                    float(np.mean([d_pool[k] for k in pos_in_avail])), 4)
                if pos_in_avail else "",
                "selected_count": len(picked),
                "selected_ood_count": int(sum(bool(ood_pool[k]) for k in picked)),
                "mean_min_train_hamming_selected": round(
                    float(np.mean([d_pool[k] for k in picked])), 4) if picked else "",
                "revealed_count": len(errors),
                "mae_pred_vs_measured": round(float(np.mean(errors)), 6) if errors else "",
                "selected_best_in_stratum": (
                    bool(picked) and min(picked, key=lambda k: float(prediction.mu[k]))
                    == min(pos_in_avail, key=lambda k: float(prediction.mu[k]))),
            })
        path = self.log_dir / f"round_{self.state.round:03d}_ood_strata.csv"
        pd.DataFrame(rows).to_csv(path, index=False)

    # ---------------- fingerprints (P0-3) ----------------
    CONFIG_KEYS = ("mode", "candidate_source", "min_mutations", "max_mutations",
                   "candidate_quotas", "rounds", "budget", "n_init", "n_holdout",
                   "pool_size", "surrogate", "n_members", "screen_k", "oracle",
                   "strategy", "ratios", "direction", "diverse_multiplier", "seed",
                   "data_dir", "protocol_id", "approved_protocols",
                   "search_policy", "search_max_unique_predictions",
                   "search_max_generations", "search_allowed_n_from_wt",
                   "search_beam_width", "search_offspring_per_parent",
                   "search_one_hop_fraction", "search_restart_fraction",
                   "search_diversity_pool_factor", "search_diversity_distance",
                   "search_max_stagnant_generations", "search_n_measured_seeds",
                   "search_population_size", "search_offspring_size",
                   "search_tournament_size", "search_elite_count",
                   "search_n_random_seeds", "search_min_mutual_distance")
    # Compared on resume; `init_labeled_sha256` is informational only because the
    # labelled set legitimately grows across rounds.
    FINGERPRINT_KEYS = ("interface_version", "config", "data_sha256", "pool_sha256",
                        "holdout_sha256")

    @staticmethod
    def active_fingerprint_config(config: dict) -> dict:
        """Ignore settings that never affect this candidate source/policy.

        Keeps original stored fingerprints intact, and accepts pre-search-layer
        static-pool records. All active parameters still compare strictly.
        """
        result=dict(config)
        if result.get('candidate_source') != 'search':
            return {k:v for k,v in result.items() if not k.startswith('search_')}
        if result.get('search_policy') != 'mutation_only_ga':
            for k in ('search_population_size','search_offspring_size',
                      'search_tournament_size','search_elite_count'):
                result.pop(k,None)
        return result

    def compute_fingerprint(self) -> dict:
        data_path = PROJECT_ROOT / self.args.data_dir / "fad_proxy2000_v2_full.csv"
        source_files = sorted((PROJECT_ROOT / "scripts" / "loop").glob("*.py"))
        h = hashlib.sha256()
        for path in source_files:
            h.update(path.name.encode("utf-8"))
            h.update(path.read_bytes())
        return {
            "interface_version": INTERFACE_VERSION,
            "config": self.active_fingerprint_config(
                {k: getattr(self.args, k, None) for k in self.CONFIG_KEYS}),
            "data_sha256": sha256_file(data_path) if data_path.is_file() else None,
            "pool_sha256": seqs_digest(self.pool),
            "pool_size": len(self.pool),
            "pool_stats": self.pool_stats,
            "holdout_sha256": seqs_digest(self.holdout),
            "holdout_size": len(self.holdout),
            "init_labeled_sha256": seqs_digest(self.state.labeled_sequences),
            "init_labeled_size": len(self.state.labeled_sequences),
            "source_sha256": h.hexdigest(),
            "source_files": [p.name for p in source_files],
            "created_at": pd.Timestamp.now().isoformat(),
        }
    def check_fingerprint(self) -> None:
        """Write the checkpoint fingerprint, or enforce it on resume."""
        path = self.log_dir / "fingerprint.json"
        current = self.compute_fingerprint()
        if not self.resumed:
            if path.is_file():
                # A fresh run reusing an existing log dir must not look resumable.
                print("[warn] overwriting an existing checkpoint fingerprint "
                      "(log-dir was already used)")
            path.write_text(json.dumps(current, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            self.fingerprint = current
            return
        if not path.is_file():
            raise FileNotFoundError(
                f"--resume requested but {path} is missing; the checkpoint predates "
                "fingerprinting or was not created by this driver")
        stored = json.loads(path.read_text(encoding="utf-8"))
        diffs = {}
        for key in self.FINGERPRINT_KEYS:
            old_value=stored.get(key)
            new_value=current.get(key)
            if key=='config':
                old_value=self.active_fingerprint_config(old_value or {})
                new_value=self.active_fingerprint_config(new_value or {})
            if old_value != new_value:
                diffs[key] = {"stored": stored.get(key), "current": current.get(key)}
        if diffs:
            detail = "; ".join(f"{k} differs" for k in diffs)
            raise ValueError(
                f"resume refuses to continue: checkpoint fingerprint mismatch ({detail}). "
                "Re-run with the original configuration, or start a new run in a new "
                "--log-dir. Differences: " + json.dumps(diffs, ensure_ascii=False)[:2000])
        if stored.get("source_sha256") != current.get("source_sha256"):
            print("[warn] source files changed since this checkpoint was written "
                  f"({stored.get('source_sha256', '')[:12]} -> "
                  f"{current.get('source_sha256', '')[:12]}); the record keeps the "
                  "original fingerprint and both hashes stay auditable")
        self.fingerprint = stored
        print(f"[resume] fingerprint matched (data={str(stored['data_sha256'])[:12]}, "
              f"pool={str(stored['pool_sha256'])[:12]}, n_pool={stored['pool_size']})")

    def _final_retrain(self, absorbed: int, exit_code: int) -> int:
        """Retrain after the last batch closed (rounds exhausted), then record it.

        Returns the (unchanged) exit code; the side effect is a `retrained`
        terminal metrics row with a fresh holdout score and `state.status`.
        """
        t0 = time.time()
        model = self._make_model()
        model.fit(self.state.labeled_sequences,
                  np.asarray(self.state.labeled_values, dtype=float))
        holdout_pre = float("nan")
        if self.args.post_retrain_eval and self.holdout_values is not None \
                and len(self.holdout):
            holdout_pre = spearman(self.holdout_values, model.predict(self.holdout).mu)
        self.state.status = "retrained"
        row = self._status_row("retrained")
        row["n_new_unique"] = absorbed
        row["model_n_train"] = getattr(model, "n_train", len(self.state.labeled_sequences))
        row["holdout_spearman_post"] = (round(holdout_pre, 6)
                                        if np.isfinite(holdout_pre) else "")
        row["seconds"] = round(time.time() - t0, 1)
        self._persist_state()
        print(f"[retrain] batch closed and {absorbed} labels absorbed; final retrain "
              f"holdout_spearman={row['holdout_spearman_post']} "
              f"(n_train={row['model_n_train']})")
        return exit_code

    def _persist_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state.to_json(), indent=2),
                                   encoding="utf-8")
        if self.ledger:                       # keep the ledger consistent with absorbed
            pd.DataFrame(self.ledger).to_csv(  # labels / QC failures
                self.log_dir / "ledger.csv", index=False)
        if self.state.history:
            pd.DataFrame(self.state.history).to_csv(
                self.log_dir / "history.csv", index=False)
        (self.log_dir / "progress.json").write_text(json.dumps(
            {"interface_version": INTERFACE_VERSION, "completed_rounds": self.state.round,
             "requested_rounds": self.args.rounds, "status": self.state.status,
             "n_labeled": len(self.state.labeled_sequences),
             "n_failed": len(self.state.failed),
             "awaiting_labels": len(self.state.awaiting),
             "open_batch": self.state.open_batch,
             "pool_stats": self.pool_stats}, indent=2), encoding="utf-8")

    def _status_row(self, status: str) -> dict:
        row = {"round": self.state.round + 1, "strategy": self.args.strategy,
               "n_labeled_before": len(self.state.labeled_sequences),
               "n_new_unique": 0,
               "n_labeled_after": len(self.state.labeled_sequences),
               "model_n_train": len(self.state.labeled_sequences),
               "n_awaiting": len(self.state.awaiting), "status": status,
               "requested_budget": self.args.budget,
               "effective_budget": 0,
               "pool_available": len(self._available()),
               "ood_fraction": "", "batch_ood_fraction": "",
               "ood_threshold_hamming": "", "ood_train_n_from_wt": "",
               "holdout_spearman_pre": "", "holdout_spearman_post": "",
               "search_stop_reason": self.pool_stats.get("search_stop_reason", ""),
               "search_n_unique_predicted": self.pool_stats.get("search_n_unique_predicted", ""),
               "search_best_pred_gbsa": self.pool_stats.get("search_best_pred_gbsa", ""),
               "search_pool_size": (len(self.pool)
                                    if self.args.candidate_source == "search" else ""),
               "seconds": 0.0}
        last = self.state.history[-1] if self.state.history else None
        if (last and str(last.get("status")) == status
                and last.get("n_awaiting") == row["n_awaiting"]
                and last.get("n_labeled_after") == row["n_labeled_after"]):
            # Re-entering the same waiting state (e.g. a repeated resume) must not
            # fabricate an extra round-like row in the metrics log.
            self.state.status = status
            self._persist_state()
            return row
        self.state.history.append(row)
        self.state.status = status
        self._persist_state()          # absorbed labels / status must survive a restart
        return row

    def _flush(self, batch_seqs, values, batch_id) -> None:
        self.state.batches.append({"batch_id": batch_id, "round": self.state.round,
                                   "strategy": self.args.strategy,
                                   "n": len(batch_seqs),
                                   "n_labeled": int(np.isfinite(values).sum()),
                                   "status": self.state.status})
        pd.DataFrame(self.ledger).to_csv(self.log_dir / "ledger.csv", index=False)
        batch_rows = [r for r in self.ledger if r["batch_id"] == batch_id]
        pd.DataFrame(batch_rows).to_csv(
            self.log_dir / f"round_{self.state.round:03d}_batch.csv", index=False)
        if self.args.candidate_source == "search" and hasattr(self.oracle, "batch_path"):
            # Oracle's batch.csv is its minimal state/ingest table. Give the
            # measurement team the full search lineage and group allocation too.
            pd.DataFrame(batch_rows).to_csv(
                Path(self.oracle.batch_path(batch_id)) / "batch_handoff.csv",
                index=False)
        self._write_batch_manifest(batch_id, batch_rows)
        self._persist_state()

    def _write_batch_manifest(self, batch_id: str, batch_rows: list[dict]) -> None:
        """Hand-off sidecar: the batch CSV alone is not enough to reproduce a run."""
        payload = {
            "batch_id": batch_id,
            "run_round": self.state.round,
            "strategy": self.args.strategy,
            "n_candidates": len(batch_rows),
            "export_columns": ["batch_id", "candidate_id", "sequence",
                               PREDICTION_COLUMN, "ood_flag", "protocol_id"],
            "model_id": sorted({str(r.get("model_id", "")) for r in batch_rows}),
            "feature_version": sorted({str(r.get("feature_version", "")) for r in batch_rows}),
            "protocol_id": self.args.protocol_id,
            "approved_protocols": list(self.approved_protocols),
            "target": {"column": TARGET_COLUMN, "direction": self.args.direction},
            "interface_version": INTERFACE_VERSION,
            "fingerprint": self.fingerprint,
            "ingest_schema": ["batch_id", "candidate_id", "sequence", TARGET_COLUMN,
                              "measurement_status", "protocol_id"],
            "measurement_status_values": ["pass", "fail"],
            "created_at": pd.Timestamp.now().isoformat(),
        }
        if self.args.candidate_source == "search":
            payload["export_columns"] += [
                "parent_sequence", "parent_candidate_id", "parent_evidence",
                "changed_positions_from_parent", "n_from_parent",
                "search_id", "search_generation", "search_operator",
                "group_id", "group_role"]
            payload["search_handoff"] = {
                "search_id": self.pool_stats.get("search_id"),
                "archive": self.pool_stats.get("search_archive"),
                "handoff_file": "batch_handoff.csv",
                "group_allocation_rule": "near_even_with_capacity_redistribution",
                "pool_group_counts": self.pool_stats.get("search_pool_group_counts"),
                "batch_group_counts": {g: sum(r.get("group_id") == g
                                              for r in batch_rows)
                                       for g in sorted({r.get("group_id")
                                                        for r in batch_rows})},
            }
        targets = [self.log_dir / f"round_{self.state.round:03d}_batch_manifest.json"]
        if hasattr(self.oracle, "batch_path"):
            targets.append(Path(self.oracle.batch_path(batch_id)) / "batch_manifest.json")
        for path in targets:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    # ---------------- driver ----------------
    def run(self) -> tuple[pd.DataFrame, int]:
        a = self.args
        if self.resumed:
            self.state = LoopState.from_json(json.loads(self.state_path.read_text(encoding="utf-8")))
            print(f"[resume] round={self.state.round} status={self.state.status} "
                  f"awaiting={len(self.state.awaiting)}")
        self.build_setup()
        self.check_fingerprint()          # P0-3: config/data/pool/holdout must match
        ledger_path = self.log_dir / "ledger.csv"
        if ledger_path.is_file():
            self.ledger = pd.read_csv(ledger_path).to_dict("records")

        if a.ingest and hasattr(self.oracle, "ingest"):
            report = self.oracle.ingest(a.ingest)
            print(f"[ingest] accepted={report['accepted']} duplicate={report['duplicate']} "
                  f"failed={report.get('failed', 0)} rejected={len(report['rejected'])}"
                  + (f" closed={report['closed_batches']}" if report.get("closed_batches")
                     else ""))
            if report["rejected"]:
                for item in report["rejected"][:5]:
                    print(f"  [ingest][reject] {item['reason']}")
            if report["accepted"] and not self.approved_protocols:
                print("[warn] no approved-protocols allow-list is pinned: any non-empty "
                      "protocol_id is accepted. Pin the lab protocol "
                      "(--approved-protocols / runner --pin-protocol) before a "
                      "controlled real ingest.")

        # Absorb returned labels even when no further round is requested: otherwise a
        # resume that only delivers measurements would silently leave them parked and
        # could exit 0 (i.e. look DONE) while the batch is still open.
        labeled_before = len(self.state.labeled_sequences)
        failed_before = len(self.state.failed)
        absorbed = self._absorb_returned_labels()
        if absorbed:
            print(f"[resume] absorbed {absorbed} returned labels before the round loop")
            self._persist_state()

        rows = list(self.state.history)
        exit_code = 0
        while self.state.round < a.rounds:
            row = self.run_round()
            if row["status"] in ("awaiting_labels", "pool_exhausted", "empty_batch"):
                if row["status"] == "awaiting_labels":
                    exit_code = EXIT_PARKED
                    (self.log_dir / "PARKED").write_text(json.dumps(
                        {"reason": "awaiting_labels", "batch_id": self.state.open_batch,
                         "n_awaiting": len(self.state.awaiting)}, indent=2),
                        encoding="utf-8")
                break
            rows = list(self.state.history)

        if self.state.awaiting and exit_code == 0:
            # Requested rounds are exhausted but the batch is still unresolved: this is
            # PARKED (waiting for real measurements), never DONE.
            self._status_row("awaiting_labels")
            exit_code = EXIT_PARKED
            (self.log_dir / "PARKED").write_text(json.dumps(
                {"reason": "awaiting_labels_after_requested_rounds",
                 "batch_id": self.state.open_batch,
                 "n_awaiting": len(self.state.awaiting),
                 "completed_rounds": self.state.round,
                 "requested_rounds": a.rounds}, indent=2), encoding="utf-8")
            print(f"[park] {len(self.state.awaiting)} labels still pending after "
                  f"{self.state.round} round(s); exit {EXIT_PARKED}")

        if not self.state.awaiting:
            # Batch closed (or never opened): a leftover PARKED marker would wrongly
            # claim the run is still waiting for measurements.
            stale = self.log_dir / "PARKED"
            if stale.is_file():
                stale.unlink()
                print("[cleanup] removed PARKED: no candidates are awaiting labels")

        # A resume that closes the last batch must actually RETRAIN: otherwise the
        # checkpoint keeps `labels_ingested`, the ledger rows stay pending and no
        # post-ingest holdout score exists, while the runner would still mark DONE.
        # `--no-post-retrain-eval` only skips the holdout *evaluation*, never the
        # retrain itself — the terminal state must be `retrained` either way.
        resume_activity = ((len(self.state.labeled_sequences) - labeled_before)
                           + (len(self.state.failed) - failed_before))
        if self.resumed and resume_activity > 0 and exit_code == 0 \
                and not self.state.awaiting:
            exit_code = self._final_retrain(absorbed, exit_code)

        rows = list(self.state.history)      # include the terminal status row
        out = Path(a.output)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[OK] {len(rows)} rows -> {out} (exit_code={exit_code})")
        return pd.DataFrame(rows), exit_code


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"FAD closed loop | interface v{INTERFACE_VERSION} | "
          f"target={TargetSpec().name} ({TargetSpec().label_policy}) | "
          f"strategy={args.strategy}")
    loop = ClosedLoop(args)
    _, code = loop.run()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
