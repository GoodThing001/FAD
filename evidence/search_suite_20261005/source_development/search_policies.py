"""Search policies for the iterative mutation search layer (interface v0.1.0).

Implemented (plan §1, §4):
  stratified_random       no generational feedback; stratified random candidates
  multi_start_hill_climb  one-hop local hill climbing from shared seeds, restart
                          on local optima
  multi_start_beam        multi-parent beam search with diversity constraint,
                          one/two-hop offspring mix, global-restart quota, and
                          stagnation stop
  mutation_only_ga       tournament selection, single-parent mutations,
                          elitist population update; no crossover

Batch BO is not implemented. Mutation-only GA requires no two-parent schema.

Fairness rules all policies share (plan §4.1): same context (measured set,
forbidden set, allowed mutation shell), same seed, same surrogate snapshot, and
the same hard `max_unique_predictions` budget; measured gbsa decides seed
*eligibility* only and is never mixed into predicted rankings.  All randomness
flows through one `np.random.RandomState` whose state is checkpointed after
every generation, so a resume at a generation boundary reproduces the
uninterrupted run exactly (except wall-clock `score_time` / timestamps).
"""

from __future__ import annotations

import hashlib
import time
from math import comb
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from .interface import BASES, normalize, validate_sequences
    from .search import (SEARCH_INTERFACE_VERSION, BudgetExceededError,
                         CandidateGroup, ScoredCandidate, SearchContext,
                         SearchResult, SurrogateScorer)
    from .search_archive import SearchArchive
except ImportError:  # plain-script invocation
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.loop.interface import BASES, normalize, validate_sequences
    from scripts.loop.search import (SEARCH_INTERFACE_VERSION, BudgetExceededError,
                                     CandidateGroup, ScoredCandidate, SearchContext,
                                     SearchResult, SurrogateScorer)
    from scripts.loop.search_archive import SearchArchive

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_N_MEASURED_SEEDS = 8
DEFAULT_N_RANDOM_SEEDS = 8
DEFAULT_MIN_MUTUAL_DISTANCE = 2
DEFAULT_BEAM_WIDTH = 32
DEFAULT_OFFSPRING_PER_PARENT = 32
DEFAULT_ONE_HOP_FRACTION = 0.8
DEFAULT_RESTART_FRACTION = 0.1
DEFAULT_DIVERSITY_POOL_FACTOR = 4
DEFAULT_DIVERSITY_DISTANCE = 2
DEFAULT_MAX_STAGNANT_GENERATIONS = 2
DEFAULT_GROUP_TOP_N = 50
DEFAULT_GROUP_CLUSTER_RADIUS = 1
DEFAULT_GROUP_SEPARATION = 3
RANDOM_CHUNK = 128


# --------------------------------------------------------------------------
# shared helpers (sequence geometry, generation, seeds)
# --------------------------------------------------------------------------

def mutation_distance(a: str, b: str,
                      positions: Sequence[int] | None = None) -> int:
    """Hamming distance over the mutable positions (all sites by default)."""
    a, b = normalize(a), normalize(b)
    pos = list(positions) if positions is not None else list(range(1, len(a) + 1))
    return sum(1 for p in pos if a[p - 1] != b[p - 1])


def n_from_wt(seq: str, context: SearchContext) -> int:
    return sum(1 for p in context.mutable_positions_1based
               if normalize(seq)[p - 1] != context.wt_sequence[p - 1])


def _waterfill(ks: Sequence[int], total: int, caps: dict[int, int]) -> dict[int, int]:
    plan = {int(k): 0 for k in ks}
    remaining = int(total)
    active = [int(k) for k in ks]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for k in list(active):
            room = caps[k] - plan[k]
            give = min(room, share, remaining)
            if give > 0:
                plan[k] += give
                remaining -= give
                progressed = True
            if plan[k] >= caps[k]:
                active.remove(k)
        if not progressed:
            break
    return plan


def stratified_random_seqs(n: int, context: SearchContext, rng,
                           exclude: Sequence[str] = ()) -> list[str]:
    """Up to `n` new random sequences, waterfilled across the allowed n_from_wt
    strata. Returns fewer than `n` only when the space is exhausted (callers
    treat that as a graceful stop, not an error)."""
    if n <= 0:
        return []
    lo, hi = context.allowed_n_from_wt
    strata = list(range(lo, hi + 1))
    caps = {k: comb(len(context.mutable_positions_1based), k) * (len(BASES) - 1) ** k
            for k in strata}
    excluded = {normalize(s) for s in exclude}
    out: list[str] = []
    seen: set[str] = set()
    wt = context.wt_sequence
    pos = context.mutable_positions_1based
    # Re-waterfill after every round: a stratum that produced nothing (e.g. k=0,
    # capacity 1, already seen) is marked exhausted and its share redistributed
    # instead of being silently lost.  Deterministic: all randomness via `rng`.
    active = list(strata)
    while len(out) < n and active:
        plan = _waterfill(active, n - len(out), caps)
        for k, want in list(plan.items()):
            got = 0
            attempts = 0
            limit = max(1000, want * 200)
            while got < want and attempts < limit:
                attempts += 1
                sites = rng.choice(len(pos), size=k, replace=False)
                s = list(wt)
                for pi in sites:
                    p = pos[int(pi)] - 1
                    choices = [b for b in BASES if b != s[p]]
                    s[p] = choices[int(rng.randint(len(choices)))]
                cand = "".join(s)
                if cand in excluded or cand in seen:
                    continue
                seen.add(cand)
                out.append(cand)
                got += 1
            if want > 0 and got == 0:
                active.remove(k)
    return out


def pick_initial_seeds(context: SearchContext, *, n_measured: int, n_random: int,
                       min_mutual_distance: int, rng) -> list[tuple[str, str]]:
    """Shared seed set: measured low-gbsa sequences (mutually distant) then
    stratified random seeds. Returns [(sequence, parent_evidence), ...] in a
    deterministic order. Measured gbsa decides eligibility only."""
    pos = context.mutable_positions_1based
    ranked = sorted(zip(context.measured_sequences, context.measured_gbsa),
                    key=lambda t: (float(t[1]), t[0]))
    measured_seeds: list[str] = []
    for s, _ in ranked:
        if all(mutation_distance(s, k, pos) >= min_mutual_distance
               for k in measured_seeds):
            measured_seeds.append(s)
        if len(measured_seeds) >= n_measured:
            break
    exclude = set(context.forbidden_sequences) | set(context.measured_sequences)
    random_seeds = stratified_random_seqs(n_random, context, rng, exclude=exclude)
    return [(s, "measured") for s in measured_seeds] + \
           [(s, "none") for s in random_seeds]


def one_hop_neighbors(seq: str, context: SearchContext) -> list[tuple[str, tuple[int, ...]]]:
    """Every feasible 1-hop neighbour of `seq` (at most 13*3 = 39), ordered by
    position then base. Neighbours whose n_from_wt leaves the allowed shell are
    dropped, so the 39-neighbour cap is an upper bound, never exceeded."""
    s = normalize(seq)
    out: list[tuple[str, tuple[int, ...]]] = []
    for p in context.mutable_positions_1based:
        i = p - 1
        for b in BASES:
            if b == s[i]:
                continue
            cand = s[:i] + b + s[i + 1:]
            k = n_from_wt(cand, context)
            if context.allowed_n_from_wt[0] <= k <= context.allowed_n_from_wt[1]:
                out.append((cand, (p,)))
    return out


def two_hop_neighbors(seq: str, context: SearchContext, rng,
                      limit: int | None = None) -> list[tuple[str, tuple[int, ...]]]:
    """Sampled feasible 2-hop neighbours (full set is C(13,2)*9 = 702). The
    enumeration order is deterministic; the sample is drawn from `rng`."""
    s = normalize(seq)
    pos = list(context.mutable_positions_1based)
    lo, hi = context.allowed_n_from_wt
    feasible: list[tuple[str, tuple[int, ...]]] = []
    for ai in range(len(pos)):
        for bi in range(ai + 1, len(pos)):
            a, b = pos[ai], pos[bi]
            ia, ib = a - 1, b - 1
            for ba in BASES:
                if ba == s[ia]:
                    continue
                for bb in BASES:
                    if bb == s[ib]:
                        continue
                    cand = s[:ia] + ba + s[ia + 1:]
                    cand = cand[:ib] + bb + cand[ib + 1:]
                    k = n_from_wt(cand, context)
                    if lo <= k <= hi:
                        feasible.append((cand, (a, b)))
    if limit is None or len(feasible) <= limit:
        return feasible
    idx = rng.choice(len(feasible), size=int(limit), replace=False)
    return [feasible[int(i)] for i in idx]


def build_groups(ranked: Sequence[ScoredCandidate], context: SearchContext, *,
                 top_n: int = DEFAULT_GROUP_TOP_N,
                 cluster_radius: int = DEFAULT_GROUP_CLUSTER_RADIUS,
                 separation: int = DEFAULT_GROUP_SEPARATION) -> tuple[CandidateGroup, ...]:
    """Partition every exportable candidate into one or two auditable groups.

    The top-ranked candidate is prototype 1.  The first top-N candidate at
    distance >= separation is prototype 2, if one exists.  All candidates,
    including those beyond top-N, go to the nearer prototype.  A radius-only
    group left most of the batch without group membership.
    """
    if not ranked:
        return ()
    rule = (f"top{top_n}_prototype_sep{separation}_nearest_prototype_"
            "all_ranked_ties_to_best")
    pos = context.mutable_positions_1based
    pool = list(ranked[:top_n])
    proto1 = pool[0]
    proto2 = next((c for c in pool[1:]
                   if mutation_distance(c.sequence, proto1.sequence, pos)
                   >= separation), None)
    members1: list[ScoredCandidate] = []
    members2: list[ScoredCandidate] = []
    for c in ranked:
        if proto2 is None or mutation_distance(c.sequence, proto1.sequence, pos) \
                <= mutation_distance(c.sequence, proto2.sequence, pos):
            members1.append(c)
        else:
            members2.append(c)
    groups = [_make_group(context.search_id, 1, "predicted_best", members1,
                          rule, pos)]
    if proto2 is not None:
        groups.append(_make_group(context.search_id, 2, "diverse_backup",
                                  members2, rule, pos))
    return tuple(groups)


def _make_group(search_id: str, idx: int, role: str, members: list[ScoredCandidate],
                rule: str, pos) -> CandidateGroup:
    if not members:
        raise ValueError("a candidate group must have at least one member")
    # O(n) even for a search archive with thousands of candidates.
    ds = [mutation_distance(members[0].sequence, c.sequence, pos)
          for c in members]
    summary = (f"n={len(members)} distance_to_prototype "
               f"min={min(ds)} mean={np.mean(ds):.3f} max={max(ds)}")
    return CandidateGroup(
        group_id=f"{search_id}_g{idx}",
        group_role=role,
        prototype_id=members[0].candidate_id,
        member_ids=tuple(c.candidate_id for c in members),
        within_group_distance_summary=summary,
        group_selection_rule=rule,
    )


# --------------------------------------------------------------------------
# RNG state (de)serialization
# --------------------------------------------------------------------------

def rng_to_state(rng: np.random.RandomState) -> list:
    s = rng.get_state()
    return [s[0], np.asarray(s[1]).tolist(), int(s[2]), int(s[3]), float(s[4])]


def rng_from_state(payload: list) -> tuple:
    return (str(payload[0]), np.asarray(payload[1], dtype=np.uint32),
            int(payload[2]), int(payload[3]), float(payload[4]))


def _source_sha256() -> str:
    h = hashlib.sha256()
    for name in ("search.py", "search_policies.py", "search_archive.py"):
        path = PROJECT_ROOT / "scripts" / "loop" / name
        h.update(name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def _is_measured_seed(c: ScoredCandidate) -> bool:
    """Only the measured seeds themselves are non-exportable; their children are
    brand-new predicted sequences (parent_evidence records the PARENT's status)."""
    return c.operator == "seed" and c.parent_evidence == "measured"


def _best_exportable(cands: list[ScoredCandidate]) -> ScoredCandidate | None:
    exportable = [c for c in cands if not _is_measured_seed(c)]
    if not exportable:
        return None
    return min(exportable, key=lambda c: (c.pred_gbsa, c.candidate_id))


# --------------------------------------------------------------------------
# base policy: snapshot / checkpoint / resume / finalize plumbing
# --------------------------------------------------------------------------

class BaseSearchPolicy:
    name = "base"

    #: hyperparameters that enter policy_config (and therefore the snapshot)
    CONFIG_FIELDS: tuple[str, ...] = ()

    def __init__(self, **cfg):
        self.cfg = dict(cfg)
        # test-only interruption hook; NOT part of policy_config so an
        # interrupted run and its resuming run share the same snapshot
        self.hard_stop_generations: int | None = self.cfg.pop(
            "hard_stop_generations", None)

    def policy_config(self) -> dict:
        return {k: self.cfg[k] for k in self.CONFIG_FIELDS if k in self.cfg}

    # ---- driver -----------------------------------------------------------
    def search(self, context: SearchContext, scorer: SurrogateScorer) -> SearchResult:
        if not context.archive_dir:
            raise ValueError("SearchContext.archive_dir is required for checkpointing")
        archive = SearchArchive(context.archive_dir)
        snapshot = self._snapshot_payload(context, scorer)
        existing = archive.read_snapshot()
        if existing is None:
            archive.write_snapshot(snapshot)
        else:
            archive.check_snapshot(snapshot)

        state = archive.read_state()
        if state and state.get("done"):
            return self._load_done(archive, state)

        rng = np.random.RandomState(context.seed)
        if state is not None:
            archive.reconcile_to_state(state)          # mid-generation crash guard
            scorer.cache = archive.load_cache()        # replay predictions
            rng.set_state(rng_from_state(state["rng_state"]))
            resumed = True
        else:
            state = self._fresh_state(context)
            resumed = False

        try:
            return self._execute(context, scorer, archive, state, resumed, rng)
        except BudgetExceededError as exc:             # contract: never silent
            raise ValueError(
                f"search policy {self.name} exceeded the prediction budget: {exc}"
            ) from exc

    # ---- shared pieces ----------------------------------------------------
    def _snapshot_payload(self, context: SearchContext, scorer: SurrogateScorer) -> dict:
        fp = context.fingerprint()
        return {
            "search_interface_version": SEARCH_INTERFACE_VERSION,
            "search_id": context.search_id,
            "outer_round": context.outer_round,
            "seed": context.seed,
            "data_sha256": context.data_sha256,
            "measured_sha256": fp["measured_sha256"],
            "measured_count": len(context.measured_sequences),
            "forbidden_sha256": fp["forbidden_sha256"],
            "forbidden_count": len(context.forbidden_sequences),
            "allowed_n_from_wt": list(context.allowed_n_from_wt),
            "max_unique_predictions": context.max_unique_predictions,
            "max_generations": context.max_generations,
            "constraints_version": context.constraints_version,
            "policy": self.name,
            "policy_config": self.policy_config(),
            "model_snapshot": {"model_id": scorer.model_id,
                               "feature_version": scorer.feature_version},
            "source_sha256": _source_sha256(),
        }

    def _fresh_state(self, context: SearchContext) -> dict:
        return {
            "search_id": context.search_id,
            "policy": self.name,
            "generation": 0,
            "parents": [],
            "best": None,
            "stagnant_generations": 0,
            "walk_index": 0,
            "current_parent": None,
            "generation_rows": [],
            "stats": {"proposed": 0, "rejected_forbidden": 0,
                      "rejected_duplicate": 0, "rejected_constraint": 0,
                      "scored": 0},
            "evaluated_rows_committed": 0,
            "rng_state": None,
            "started_at": pd.Timestamp.now().isoformat(),
            "done": False,
            "stop_reason": None,
            "ranked_candidates": [],
            "groups": [],
            "n_unique_predicted": 0,
        }

    def _checkpoint(self, archive: SearchArchive, scorer: SurrogateScorer,
                    state: dict, pending_evaluated: list[ScoredCandidate],
                    rng) -> None:
        """Order: cache -> evaluated log -> state (state write last so a crash
        window can be reconciled to the committed row count on resume)."""
        archive.save_cache(scorer.cache)
        archive.append_evaluated([c.to_dict() for c in pending_evaluated])
        state["evaluated_rows_committed"] = int(state.get("evaluated_rows_committed", 0)) \
            + len(pending_evaluated)
        state["rng_state"] = rng_to_state(rng)
        state["n_unique_predicted"] = scorer.n_used
        archive.write_state(state)

    def _finalize(self, context: SearchContext, scorer: SurrogateScorer,
                  archive: SearchArchive, state: dict, *,
                  stop_reason: str) -> SearchResult:
        archive.save_cache(scorer.cache)          # ensure the cache file exists
        rows = archive.read_evaluated()
        cands = [ScoredCandidate.from_dict(r) for r in rows]
        exportable = [c for c in cands if not _is_measured_seed(c)]
        exportable.sort(key=lambda c: (c.pred_gbsa, c.candidate_id))
        groups = build_groups(exportable, context)
        if stop_reason != "interrupted":
            state["done"] = True
            state["stop_reason"] = stop_reason
            state["ranked_candidates"] = [c.to_dict() for c in exportable]
            state["groups"] = [g.to_dict() for g in groups]
            state["n_unique_predicted"] = scorer.n_used
            archive.write_state(state)
            summary = self._summary(context, scorer, state, exportable, groups,
                                    stop_reason)
            archive.finalize(
                candidates=[c.to_dict() for c in exportable],
                generations=list(state["generation_rows"]),
                groups=[g.to_dict() for g in groups],
                summary=summary,
            )
        return SearchResult(
            ranked_candidates=tuple(exportable),
            groups=groups,
            n_unique_predicted=scorer.n_used,
            stop_reason=stop_reason,
            search_state_path=str(archive._path(SearchArchive.STATE)),
        )

    def _summary(self, context, scorer, state, ranked, groups, stop_reason) -> dict:
        top = list(ranked[:10])
        started = state.get("started_at") or pd.Timestamp.now().isoformat()
        seconds = (pd.Timestamp.now() - pd.Timestamp(started)).total_seconds()
        return {
            "search_interface_version": SEARCH_INTERFACE_VERSION,
            "search_id": context.search_id,
            "policy": self.name,
            "policy_config": self.policy_config(),
            "seed": context.seed,
            "outer_round": context.outer_round,
            "data_sha256": context.data_sha256,
            "max_unique_predictions": context.max_unique_predictions,
            "n_unique_predicted": scorer.n_used,
            "n_candidates": len(ranked),
            "best_pred_gbsa": (float(ranked[0].pred_gbsa) if ranked else None),
            "best_candidate_id": (ranked[0].candidate_id if ranked else None),
            "stop_reason": stop_reason,
            "generations_run": int(state["generation"]),
            "n_groups": len(groups),
            "group_roles": [g.group_role for g in groups],
            "model_id": scorer.model_id,
            "feature_version": scorer.feature_version,
            "top10_ood_fraction": (
                round(float(np.mean([c.ood_flag for c in top])), 4) if top else None),
            "top10_min_train_hamming": (
                round(float(min(c.nearest_train_hamming for c in top)), 4)
                if top else None),
            "seconds_total": round(float(seconds), 3),
            "rejection_stats": dict(state["stats"]),
        }

    def _load_done(self, archive, state) -> SearchResult:
        ranked = tuple(ScoredCandidate.from_dict(d) for d in state["ranked_candidates"])
        groups = tuple(CandidateGroup.from_dict(d) for d in state["groups"])
        return SearchResult(
            ranked_candidates=ranked,
            groups=groups,
            n_unique_predicted=int(state.get("n_unique_predicted", 0)),
            stop_reason=state["stop_reason"],
            search_state_path=str(archive._path(SearchArchive.STATE)),
        )

    # ---- proposal filtering / scoring shared by all policies ----------------
    def _filter_proposals(self, context: SearchContext, scorer: SurrogateScorer,
                          proposals: list, state: dict) -> list:
        """proposals: [(seq, meta)]; drops forbidden / already-scored /
        in-batch duplicates / invalid. Counts every rejection in state['stats'].
        Returns the surviving [(seq, meta)] list in first-seen order."""
        forbidden = context.forbidden_sequences
        scored = set(scorer.cache.sequences())
        seen: set[str] = set()
        out = []
        for seq, meta in proposals:
            s = normalize(seq)
            state["stats"]["proposed"] += 1
            if s in forbidden:
                state["stats"]["rejected_forbidden"] += 1
                continue
            if s in scored or s in seen:
                state["stats"]["rejected_duplicate"] += 1
                continue
            try:
                validate_sequences([s], "proposal", check_scaffold=True)
            except ValueError:
                state["stats"]["rejected_constraint"] += 1
                continue
            seen.add(s)
            out.append((s, meta))
        return out

    def _score_proposals(self, scorer: SurrogateScorer, context: SearchContext,
                         proposals: list, *, generation: int,
                         state: dict) -> list[ScoredCandidate]:
        """Score the surviving proposals and build ScoredCandidate records.
        `proposals`: [(seq, (parent_candidate_id, parent_evidence, operator,
        n_from_parent, changed_positions))]."""
        if not proposals:
            return []
        scorer.score([p[0] for p in proposals])     # BudgetExceededError propagates
        out = []
        for s, meta in proposals:
            pid, pev, op, nfp, chg = meta
            out.append(scorer.candidate_for(
                s, generation=generation, parent_candidate_id=pid,
                parent_evidence=pev, operator=op, n_from_parent=nfp,
                changed_positions_from_parent=chg, rng_seed=context.seed))
        state["stats"]["scored"] += len(out)
        return out

    def _score_seeds(self, scorer: SurrogateScorer, context: SearchContext,
                     seeds: list[tuple[str, str]], state: dict) -> list[ScoredCandidate]:
        scorer.score([s for s, _ in seeds])
        out = []
        for s, pev in seeds:
            out.append(scorer.candidate_for(
                s, generation=0, parent_candidate_id=None, parent_evidence=pev,
                operator="seed", n_from_parent=0, changed_positions_from_parent=(),
                rng_seed=context.seed))
        state["stats"]["scored"] += len(out)
        return out

    @staticmethod
    def _gen_row(state: dict, *, generation: int, n_parents: int,
                 n_proposed_gen: int, n_rejected_gen: int, n_new: int,
                 best: ScoredCandidate | None, seconds: float) -> dict:
        return {
            "generation": generation,
            "n_parents": n_parents,
            "n_proposed": n_proposed_gen,
            "n_rejected": n_rejected_gen,
            "n_new_scored": n_new,
            "n_cumulative_scored": int(state["stats"]["scored"]),
            "best_pred_gbsa": (best.pred_gbsa if best else ""),
            "best_candidate_id": (best.candidate_id if best else ""),
            "seconds": round(seconds, 3),
        }


# --------------------------------------------------------------------------
# stratified random baseline (no generational feedback)
# --------------------------------------------------------------------------

class RandomSearchPolicy(BaseSearchPolicy):
    name = "stratified_random"
    CONFIG_FIELDS: tuple[str, ...] = ()

    def _execute(self, context, scorer, archive, state, resumed, rng):
        gen_rows = list(state["generation_rows"])
        best = (ScoredCandidate.from_dict(state["best"]) if state.get("best")
                else None)
        generation = 1
        while scorer.remaining > 0:
            t0 = time.perf_counter()
            exclude = set(context.forbidden_sequences) | set(scorer.cache.sequences())
            batch = stratified_random_seqs(min(scorer.remaining, RANDOM_CHUNK),
                                           context, rng, exclude=exclude)
            proposals = [(s, (None, "none", "global_restart", 0, ())) for s in batch]
            proposed_before = int(state["stats"]["proposed"])
            kept = self._filter_proposals(context, scorer, proposals, state)
            candidates = self._score_proposals(scorer, context, kept,
                                               generation=generation, state=state)
            if candidates:
                cand_best = _best_exportable(candidates)
                if best is None or (cand_best is not None
                                    and cand_best.pred_gbsa < best.pred_gbsa):
                    best = cand_best or best
            state["best"] = best.to_dict() if best else None
            gen_rows.append(self._gen_row(
                state, generation=generation, n_parents=0,
                n_proposed_gen=int(state["stats"]["proposed"]) - proposed_before,
                n_rejected_gen=(int(state["stats"]["proposed"]) - proposed_before)
                - len(candidates),
                n_new=len(candidates), best=best,
                seconds=time.perf_counter() - t0))
            state["generation_rows"] = gen_rows
            self._checkpoint(archive, scorer, state, candidates, rng)
            if not candidates and not batch:
                break
        stop = "budget_exhausted" if scorer.remaining == 0 else "space_exhausted"
        return self._finalize(context, scorer, archive, state, stop_reason=stop)


class MatchedSeedRandomWalkPolicy(BaseSearchPolicy):
    """Score-guidance-free random walk from the hill/beam initial seed set."""

    name = "matched_seed_random_walk"
    CONFIG_FIELDS = ("n_measured_seeds", "n_random_seeds", "min_mutual_distance")

    def __init__(self, *, n_measured_seeds=DEFAULT_N_MEASURED_SEEDS,
                 n_random_seeds=DEFAULT_N_RANDOM_SEEDS,
                 min_mutual_distance=DEFAULT_MIN_MUTUAL_DISTANCE, **cfg):
        super().__init__(n_measured_seeds=n_measured_seeds,
                         n_random_seeds=n_random_seeds,
                         min_mutual_distance=min_mutual_distance, **cfg)

    def _execute(self, context, scorer, archive, state, resumed, rng):
        if not resumed:
            seeds = pick_initial_seeds(
                context, n_measured=self.cfg["n_measured_seeds"],
                n_random=self.cfg["n_random_seeds"],
                min_mutual_distance=self.cfg["min_mutual_distance"], rng=rng)
            seeds = seeds[:scorer.remaining]
            if not seeds:
                return self._finalize(context, scorer, archive, state,
                                      stop_reason="no_seeds")
            scored = self._score_seeds(scorer, context, seeds, state)
            state["best"] = (best.to_dict() if
                             (best := _best_exportable(scored)) else None)
            self._checkpoint(archive, scorer, state, scored, rng)
        generation = int(state["generation"])
        while scorer.remaining and generation < context.max_generations:
            if self.hard_stop_generations is not None and \
                    generation >= self.hard_stop_generations:
                return self._finalize(context, scorer, archive, state,
                                      stop_reason="interrupted")
            start = time.perf_counter()
            archive_rows = [ScoredCandidate.from_dict(r)
                            for r in archive.read_evaluated()]
            seen = set(scorer.cache.sequences())
            batch_seen: set[str] = set()
            chunk = min(scorer.remaining,
                        max(RANDOM_CHUNK, (context.max_unique_predictions +
                                           context.max_generations - 1) //
                            context.max_generations))
            proposals = []
            attempts = 0
            while len(proposals) < chunk and attempts < chunk * 100:
                attempts += 1
                parent = archive_rows[int(rng.randint(len(archive_rows)))]
                neighbors = one_hop_neighbors(parent.sequence, context)
                if not neighbors:
                    continue
                seq, changed = neighbors[int(rng.randint(len(neighbors)))]
                if seq in seen or seq in batch_seen or seq in context.forbidden_sequences:
                    continue
                batch_seen.add(seq)
                proposals.append((seq, (parent.candidate_id,
                                        "measured" if _is_measured_seed(parent)
                                        else "predicted_only", "one_hop", 1,
                                        changed)))
            before = int(state["stats"]["proposed"])
            kept = self._filter_proposals(context, scorer, proposals, state)
            kids = self._score_proposals(scorer, context, kept,
                                         generation=generation + 1, state=state)
            previous = (ScoredCandidate.from_dict(state["best"])
                        if state.get("best") else None)
            best = _best_exportable(kids)
            if best and (previous is None or best.pred_gbsa < previous.pred_gbsa):
                previous = best
            generation += 1
            state["generation"] = generation
            state["best"] = previous.to_dict() if previous else None
            state["generation_rows"].append(self._gen_row(
                state, generation=generation, n_parents=len(archive_rows),
                n_proposed_gen=int(state["stats"]["proposed"]) - before,
                n_rejected_gen=len(proposals) - len(kids), n_new=len(kids),
                best=previous, seconds=time.perf_counter() - start))
            self._checkpoint(archive, scorer, state, kids, rng)
            if not kids:
                break
        stop = ("budget_exhausted" if scorer.remaining == 0 else
                "max_generations" if generation >= context.max_generations else
                "neighbors_exhausted")
        return self._finalize(context, scorer, archive, state, stop_reason=stop)


# --------------------------------------------------------------------------
# multi-start local hill climbing (one-hop walk per seed)
# --------------------------------------------------------------------------

class LocalHillClimbPolicy(BaseSearchPolicy):
    name = "multi_start_hill_climb"
    CONFIG_FIELDS = ("n_measured_seeds", "n_random_seeds", "min_mutual_distance")

    def __init__(self, *, n_measured_seeds=DEFAULT_N_MEASURED_SEEDS,
                 n_random_seeds=DEFAULT_N_RANDOM_SEEDS,
                 min_mutual_distance=DEFAULT_MIN_MUTUAL_DISTANCE, **cfg):
        super().__init__(n_measured_seeds=n_measured_seeds,
                         n_random_seeds=n_random_seeds,
                         min_mutual_distance=min_mutual_distance, **cfg)

    def _execute(self, context, scorer, archive, state, resumed, rng):
        t0 = time.perf_counter()
        gen_rows = list(state["generation_rows"])
        if resumed:
            parents = [ScoredCandidate.from_dict(d) for d in state["parents"]]
            best = (ScoredCandidate.from_dict(state["best"]) if state.get("best")
                    else None)
        else:
            seeds = pick_initial_seeds(
                context, n_measured=self.cfg["n_measured_seeds"],
                n_random=self.cfg["n_random_seeds"],
                min_mutual_distance=self.cfg["min_mutual_distance"], rng=rng)
            if not seeds:
                return self._finalize(context, scorer, archive, state,
                                      stop_reason="no_seeds")
            seeds_cands = self._score_seeds(scorer, context, seeds, state)
            parents = list(seeds_cands)
            best = _best_exportable(seeds_cands)
            state["parents"] = [c.to_dict() for c in parents]
            state["best"] = best.to_dict() if best else None
            gen_rows.append(self._gen_row(state, generation=0, n_parents=len(parents),
                                          n_proposed_gen=0, n_rejected_gen=0,
                                          n_new=len(seeds_cands), best=best,
                                          seconds=time.perf_counter() - t0))
            state["generation_rows"] = gen_rows
            state["generation"] = 0
            self._checkpoint(archive, scorer, state, seeds_cands, rng)

        completed = int(state["walk_index"])
        work: list[ScoredCandidate] = []
        if state.get("current_parent") is not None:
            work.append(ScoredCandidate.from_dict(state["current_parent"]))
            work += parents[completed + 1:]
        else:
            work += parents[completed:]

        pending: list[ScoredCandidate] = []
        last_proposed = int(state["stats"]["proposed"])
        generation = int(state["generation"])
        walks_ended_optimum = 0

        for start in work:
            current = start
            while scorer.remaining > 0 and generation < context.max_generations:
                neighbours = one_hop_neighbors(current.sequence, context)
                # cap BEFORE proposing: a walk step must never propose more than
                # the budget can score (deterministic truncation point)
                proposals = [(s, (current.candidate_id,
                                  "measured" if _is_measured_seed(current)
                                  else "predicted_only",
                                  "one_hop", 1, chg)) for s, chg in neighbours]
                kept = self._filter_proposals(context, scorer, proposals, state)
                if len(kept) > scorer.remaining:
                    indices = np.sort(rng.choice(len(kept), scorer.remaining,
                                                  replace=False))
                    kept = [kept[int(i)] for i in indices]
                if not kept:
                    walks_ended_optimum += 1
                    break
                t0_step = time.perf_counter()
                kids = self._score_proposals(scorer, context, kept,
                                             generation=generation + 1, state=state)
                pending = kids
                best_kid = min(kids, key=lambda c: (c.pred_gbsa, c.candidate_id)) \
                    if kids else None
                if best_kid is None or best_kid.pred_gbsa >= current.pred_gbsa:
                    walks_ended_optimum += 1
                    break
                generation += 1
                current = best_kid
                if best is None or current.pred_gbsa < best.pred_gbsa:
                    best = current
                gen_rows.append(self._gen_row(
                    state, generation=generation, n_parents=1,
                    n_proposed_gen=int(state["stats"]["proposed"]) - last_proposed,
                    n_rejected_gen=(int(state["stats"]["proposed"]) - last_proposed)
                    - len(kids),
                    n_new=len(kids), best=best,
                    seconds=time.perf_counter() - t0_step))
                state.update({
                    "parents": [c.to_dict() for c in parents],
                    "walk_index": completed,
                    "current_parent": current.to_dict(),
                    "generation": generation,
                    "best": best.to_dict() if best else None,
                    "generation_rows": gen_rows,
                    "stop_reason": None,
                })
                self._checkpoint(archive, scorer, state, pending, rng)
                pending = []
                last_proposed = int(state["stats"]["proposed"])
                if self.hard_stop_generations is not None \
                        and generation >= self.hard_stop_generations:
                    state["stop_reason"] = "interrupted"
                    self._checkpoint(archive, scorer, state, [], rng)
                    return self._finalize(context, scorer, archive, state,
                                          stop_reason="interrupted")
            completed += 1
            state.update({
                "walk_index": completed,
                "current_parent": None,
                "generation": generation,
                "best": best.to_dict() if best else None,
                "generation_rows": gen_rows,
                "stop_reason": None,
            })
            self._checkpoint(archive, scorer, state, pending, rng)
            pending = []
            last_proposed = int(state["stats"]["proposed"])
            if scorer.remaining == 0:
                break
            if generation >= context.max_generations:
                break

        if scorer.remaining == 0:
            stop = "budget_exhausted"
        elif generation >= context.max_generations:
            stop = "max_generations"
        elif walks_ended_optimum == len(work) and work:
            stop = "local_optima"
        else:
            stop = "neighbors_exhausted"
        return self._finalize(context, scorer, archive, state, stop_reason=stop)


# --------------------------------------------------------------------------
# multi-start beam search
# --------------------------------------------------------------------------

class BeamSearchPolicy(BaseSearchPolicy):
    name = "multi_start_beam"
    CONFIG_FIELDS = ("beam_width", "offspring_per_parent", "one_hop_fraction",
                     "restart_fraction", "diversity_pool_factor",
                     "diversity_distance", "max_stagnant_generations",
                     "n_measured_seeds", "n_random_seeds", "min_mutual_distance")

    def __init__(self, *, beam_width=DEFAULT_BEAM_WIDTH,
                 offspring_per_parent=DEFAULT_OFFSPRING_PER_PARENT,
                 one_hop_fraction=DEFAULT_ONE_HOP_FRACTION,
                 restart_fraction=DEFAULT_RESTART_FRACTION,
                 diversity_pool_factor=DEFAULT_DIVERSITY_POOL_FACTOR,
                 diversity_distance=DEFAULT_DIVERSITY_DISTANCE,
                 max_stagnant_generations=DEFAULT_MAX_STAGNANT_GENERATIONS,
                 n_measured_seeds=DEFAULT_N_MEASURED_SEEDS,
                 n_random_seeds=DEFAULT_N_RANDOM_SEEDS,
                 min_mutual_distance=DEFAULT_MIN_MUTUAL_DISTANCE, **cfg):
        if not 0.0 <= one_hop_fraction <= 1.0:
            raise ValueError("one_hop_fraction must be within [0, 1]")
        if not 0.0 <= restart_fraction <= 1.0:
            raise ValueError("restart_fraction must be within [0, 1]")
        if beam_width < 1 or offspring_per_parent < 1:
            raise ValueError("beam_width and offspring_per_parent must be >= 1")
        super().__init__(beam_width=beam_width, offspring_per_parent=offspring_per_parent,
                         one_hop_fraction=one_hop_fraction,
                         restart_fraction=restart_fraction,
                         diversity_pool_factor=diversity_pool_factor,
                         diversity_distance=diversity_distance,
                         max_stagnant_generations=max_stagnant_generations,
                         n_measured_seeds=n_measured_seeds,
                         n_random_seeds=n_random_seeds,
                         min_mutual_distance=min_mutual_distance, **cfg)

    def _execute(self, context, scorer, archive, state, resumed, rng):
        t0 = time.perf_counter()
        gen_rows = list(state["generation_rows"])
        pending: list[ScoredCandidate] = []
        if resumed:
            parents = [ScoredCandidate.from_dict(d) for d in state["parents"]]
            generation = int(state["generation"])
            stagnant = int(state.get("stagnant_generations", 0))
            best = (ScoredCandidate.from_dict(state["best"]) if state.get("best")
                    else None)
        else:
            seeds = pick_initial_seeds(
                context, n_measured=self.cfg["n_measured_seeds"],
                n_random=self.cfg["n_random_seeds"],
                min_mutual_distance=self.cfg["min_mutual_distance"], rng=rng)
            if not seeds:
                return self._finalize(context, scorer, archive, state,
                                      stop_reason="no_seeds")
            seeds_cands = self._score_seeds(scorer, context, seeds, state)
            pending = list(seeds_cands)
            parents = self._diverse_select(seeds_cands, context)
            generation = 0
            stagnant = 0
            best = _best_exportable(seeds_cands)
            state["parents"] = [c.to_dict() for c in parents]
            state["best"] = best.to_dict() if best else None
            gen_rows.append(self._gen_row(state, generation=0, n_parents=len(parents),
                                          n_proposed_gen=0, n_rejected_gen=0,
                                          n_new=len(seeds_cands), best=best,
                                          seconds=time.perf_counter() - t0))
            state["generation_rows"] = gen_rows
            state["generation"] = generation
            self._checkpoint(archive, scorer, state, pending, rng)
            pending = []

        stop = None
        while stop is None:
            if scorer.remaining == 0:
                stop = "budget_exhausted"
                break
            if generation >= context.max_generations:
                stop = "max_generations"
                break
            if self.hard_stop_generations is not None \
                    and generation >= self.hard_stop_generations:
                state["stop_reason"] = "interrupted"
                self._checkpoint(archive, scorer, state, [], rng)
                return self._finalize(context, scorer, archive, state,
                                      stop_reason="interrupted")

            g_start = time.perf_counter()
            n_parents = len(parents)
            slots = min(scorer.remaining, n_parents * self.cfg["offspring_per_parent"])
            n_restart = int(round(slots * self.cfg["restart_fraction"]))
            n_off = slots - n_restart

            proposals: list[tuple[str, tuple]] = []
            base, extra = divmod(n_off, n_parents) if n_parents else (0, 0)
            for i, parent in enumerate(parents):
                quota = base + (1 if i < extra else 0)
                if quota <= 0:
                    continue
                n_1h = int(round(quota * self.cfg["one_hop_fraction"]))
                ones = one_hop_neighbors(parent.sequence, context)
                n_take_1h = min(len(ones), n_1h)
                # sample the one-hop quota instead of position-truncating: the
                # first-k sites must not structurally dominate the search.
                # Deterministic (drawn from the checkpointed RNG); order kept
                # canonical for the evaluated log.
                if len(ones) > n_take_1h:
                    idx = np.sort(rng.choice(len(ones), size=n_take_1h,
                                             replace=False))
                else:
                    idx = np.arange(len(ones))
                for i in idx:
                    s, chg = ones[int(i)]
                    proposals.append((s, (parent.candidate_id,
                                          "measured" if _is_measured_seed(parent)
                                          else "predicted_only",
                                          "one_hop", 1, chg)))
                need = quota - n_take_1h
                if need > 0:
                    twos = two_hop_neighbors(parent.sequence, context, rng,
                                             limit=need)
                    for s, chg in twos:
                        proposals.append((s, (parent.candidate_id,
                                              "measured" if _is_measured_seed(parent)
                                              else "predicted_only",
                                              "two_hop", 2, chg)))
            if n_restart > 0:
                exclude = set(context.forbidden_sequences) \
                    | set(scorer.cache.sequences())
                for s in stratified_random_seqs(n_restart, context, rng,
                                                exclude=exclude):
                    proposals.append((s, (None, "none", "global_restart", 0, ())))

            proposed_before = int(state["stats"]["proposed"])
            kept = self._filter_proposals(context, scorer, proposals, state)
            kids = self._score_proposals(scorer, context, kept,
                                         generation=generation + 1, state=state)
            pending = kids
            if kids:
                new_best = _best_exportable(kids)
                if best is None or (new_best is not None
                                    and new_best.pred_gbsa < best.pred_gbsa):
                    best = new_best or best
                    stagnant = 0
                else:
                    stagnant += 1
            else:
                stagnant += 1
            parents = self._diverse_select(parents + kids, context)
            generation += 1
            gen_rows.append(self._gen_row(
                state, generation=generation, n_parents=n_parents,
                n_proposed_gen=int(state["stats"]["proposed"]) - proposed_before,
                n_rejected_gen=(int(state["stats"]["proposed"]) - proposed_before)
                - len(kids),
                n_new=len(kids), best=best,
                seconds=time.perf_counter() - g_start))
            state.update({
                "parents": [c.to_dict() for c in parents],
                "generation": generation,
                "stagnant_generations": stagnant,
                "best": best.to_dict() if best else None,
                "generation_rows": gen_rows,
                "stop_reason": None,
            })
            self._checkpoint(archive, scorer, state, pending, rng)
            pending = []
            if stagnant >= self.cfg["max_stagnant_generations"]:
                stop = "stagnation"
            elif not kids:
                stop = "neighbors_exhausted"
        return self._finalize(context, scorer, archive, state, stop_reason=stop)

    def _diverse_select(self, cands: list[ScoredCandidate],
                        context: SearchContext) -> list[ScoredCandidate]:
        """Keep beam_width parents: rank by pred_gbsa, look at the top
        diversity_pool_factor*beam_width, greedily keep candidates that differ
        from every kept parent at >= diversity_distance mutable positions, fill
        by rank when short. diversity_distance=0 disables the rule (pure
        top-mu selection)."""
        bw = int(self.cfg["beam_width"])
        order = sorted(cands, key=lambda c: (c.pred_gbsa, c.candidate_id))
        head = order[: bw * int(self.cfg["diversity_pool_factor"])]
        pos = context.mutable_positions_1based
        kept: list[ScoredCandidate] = []
        for c in head:
            if all(mutation_distance(c.sequence, k.sequence, pos)
                   >= int(self.cfg["diversity_distance"]) for k in kept):
                kept.append(c)
            if len(kept) >= bw:
                break
        if len(kept) < bw:
            for c in head:
                if c not in kept:
                    kept.append(c)
                if len(kept) >= bw:
                    break
        return kept


class MutationOnlyGeneticPolicy(BeamSearchPolicy):
    """Single-parent elitist GA with tournament selection; never recombines.

    Parents are sampled by fitness tournaments, unlike beam's equal quotas.
    The shared initial set is scored without adding free population members.
    Population diversity is a soft rule; best elites always survive.
    """

    name = "mutation_only_ga"
    CONFIG_FIELDS = ("population_size", "offspring_size", "tournament_size",
                     "elite_count", "one_hop_fraction", "restart_fraction",
                     "diversity_pool_factor", "diversity_distance",
                     "max_stagnant_generations", "n_measured_seeds",
                     "n_random_seeds", "min_mutual_distance")

    def __init__(self, *, population_size=32, offspring_size=128,
                 tournament_size=2, elite_count=2, one_hop_fraction=0.8,
                 restart_fraction=0.1, max_stagnant_generations=4,
                 **cfg):
        if population_size < 1 or offspring_size < 1 or tournament_size < 1:
            raise ValueError("GA sizes must be >= 1")
        if not 1 <= elite_count <= population_size:
            raise ValueError("GA elite_count must lie within the population")
        cfg.pop("beam_width", None)
        cfg.pop("offspring_per_parent", None)
        super().__init__(beam_width=population_size, one_hop_fraction=one_hop_fraction,
                         restart_fraction=restart_fraction,
                         max_stagnant_generations=max_stagnant_generations, **cfg)
        self.cfg.update(population_size=population_size, offspring_size=offspring_size,
                        tournament_size=tournament_size, elite_count=elite_count)

    def _survive(self, candidates, context):
        unique = {c.candidate_id: c for c in candidates}
        ordered = sorted(unique.values(), key=lambda c: (c.pred_gbsa, c.candidate_id))
        elite = ordered[:self.cfg["elite_count"]]
        diverse = self._diverse_select(ordered, context)
        merged = elite + [c for c in diverse if c not in elite]
        return merged[:self.cfg["population_size"]]

    def _execute(self, context, scorer, archive, state, resumed, rng):
        if resumed:
            parents = [ScoredCandidate.from_dict(d) for d in state["parents"]]
        else:
            seeds = pick_initial_seeds(context,
                n_measured=self.cfg["n_measured_seeds"],
                n_random=self.cfg["n_random_seeds"],
                min_mutual_distance=self.cfg["min_mutual_distance"], rng=rng)
            seeds = seeds[:scorer.remaining]
            if not seeds:
                return self._finalize(context, scorer, archive, state, stop_reason="no_seeds")
            scored = self._score_seeds(scorer, context, seeds, state)
            parents = self._survive(scored, context)
            best = _best_exportable(scored)
            state["parents"] = [c.to_dict() for c in parents]
            state["best"] = best.to_dict() if best else None
            self._checkpoint(archive, scorer, state, scored, rng)
        while True:
            generation = int(state["generation"])
            if scorer.remaining == 0:
                stop = "budget_exhausted"
                break
            if generation >= context.max_generations:
                stop = "max_generations"
                break
            if self.hard_stop_generations is not None and generation >= self.hard_stop_generations:
                return self._finalize(context, scorer, archive, state, stop_reason="interrupted")
            t0 = time.perf_counter()
            slots = min(self.cfg["offspring_size"], scorer.remaining)
            n_restart = int(round(slots * self.cfg["restart_fraction"]))
            proposals = []
            for _ in range(slots - n_restart):
                contestants = rng.choice(len(parents), self.cfg["tournament_size"], replace=True)
                parent = min((parents[int(i)] for i in contestants),
                             key=lambda c: (c.pred_gbsa, c.candidate_id))
                one = rng.rand() < self.cfg["one_hop_fraction"]
                neighbors = (one_hop_neighbors(parent.sequence, context) if one else
                             two_hop_neighbors(parent.sequence, context, rng, limit=1))
                if not neighbors:
                    continue
                seq, changed = neighbors[int(rng.randint(len(neighbors)))]
                proposals.append((seq, (parent.candidate_id,
                    "measured" if _is_measured_seed(parent) else "predicted_only",
                    "one_hop" if one else "two_hop", len(changed), changed)))
            if n_restart:
                exclude = set(context.forbidden_sequences) | set(scorer.cache.sequences())
                proposals += [(s, (None, "none", "global_restart", 0, ())) for s in
                              stratified_random_seqs(n_restart, context, rng, exclude=exclude)]
            before = int(state["stats"]["proposed"])
            kept = self._filter_proposals(context, scorer, proposals, state)
            kids = self._score_proposals(scorer, context, kept,
                                         generation=generation + 1, state=state)
            previous = ScoredCandidate.from_dict(state["best"]) if state.get("best") else None
            new_best = _best_exportable(kids)
            improved = new_best is not None and (previous is None or new_best.pred_gbsa < previous.pred_gbsa)
            if improved:
                previous = new_best
            state["stagnant_generations"] = 0 if improved else state["stagnant_generations"] + 1
            parents = self._survive(parents + kids, context)
            state.update(generation=generation + 1,
                         parents=[c.to_dict() for c in parents],
                         best=previous.to_dict() if previous else None)
            state["generation_rows"].append(self._gen_row(state,
                generation=generation + 1, n_parents=len(parents),
                n_proposed_gen=int(state["stats"]["proposed"]) - before,
                n_rejected_gen=len(proposals) - len(kids), n_new=len(kids),
                best=previous, seconds=time.perf_counter() - t0))
            self._checkpoint(archive, scorer, state, kids, rng)
            if state["stagnant_generations"] >= self.cfg["max_stagnant_generations"]:
                stop = "stagnation"
                break
            if not kids:
                stop = "neighbors_exhausted"
                break
        return self._finalize(context, scorer, archive, state, stop_reason=stop)


POLICIES = {
    RandomSearchPolicy.name: RandomSearchPolicy,
    MatchedSeedRandomWalkPolicy.name: MatchedSeedRandomWalkPolicy,
    LocalHillClimbPolicy.name: LocalHillClimbPolicy,
    BeamSearchPolicy.name: BeamSearchPolicy,
    MutationOnlyGeneticPolicy.name: MutationOnlyGeneticPolicy,
}


def make_policy(name: str, **cfg):
    if name not in POLICIES:
        raise ValueError(f"unknown search policy {name!r}; choices: {sorted(POLICIES)}")
    return POLICIES[name](**cfg)
