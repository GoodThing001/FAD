"""Contract tests for the iterative mutation search layer (v0.1.0).

Covers plan §5 阶段A items 1-3 (docs/项目记录/突变序列迭代搜索算法与代理接口实施方案_20261002.md):
minimisation direction, batch order, 13-site constraint, dedup, the 39-neighbour
cap, the hard prediction budget, same-seed reproducibility, model cache
isolation, boundary resume bit-identity, and clear failures for NaN / error /
exhaustion cases.  All "landscape" tests use deterministic mock surrogates; they
prove algorithm mechanics, NOT GBSA discovery.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.loop.interface import Prediction, WT_141, candidate_id, normalize
from scripts.loop.search import (SEARCH_INTERFACE_VERSION, BudgetExceededError,
                                 ScoredCandidate, SearchContext, SurrogateScorer,
                                 nearest_train_hamming, ood_flags, training_ood_info)
from scripts.loop.search_archive import PredictionCache, SearchArchive
from scripts.loop.search_policies import (build_groups, make_policy,
                                          mutation_distance, one_hop_neighbors,
                                          pick_initial_seeds, stratified_random_seqs,
                                          two_hop_neighbors)

BASES = "ACGU"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

class MockSurrogate:
    def __init__(self, f, *, model_id="mock_v1", feature_version="mock",
                 with_sigma=False, fail=False, wrong_length=False):
        self.f = f
        self.model_id = model_id
        self.feature_version = feature_version
        self.with_sigma = with_sigma
        self.fail = fail
        self.wrong_length = wrong_length

    def fit(self, sequences, y):
        return self

    def predict(self, sequences):
        if self.fail:
            raise ValueError("boom: surrogate failed")
        mu = np.array([float(self.f(normalize(s))) for s in sequences])
        if self.wrong_length:
            mu = mu[:-1] if len(mu) > 1 else mu
        if self.with_sigma:
            return Prediction(mu=mu, sigma=np.full(len(mu), 0.5),
                              uncertainty_kind="mock_spread_uncalibrated",
                              model_id=self.model_id, feature_version=self.feature_version)
        return Prediction(mu=mu, model_id=self.model_id,
                          feature_version=self.feature_version)


def target_landscape(target: str):
    return lambda s: float(mutation_distance(s, target))


def flat_landscape():
    return lambda s: 1.0


def mutated_seq(ctx: SearchContext, n: int, base_idx: int = 0) -> str:
    """Deterministic sequence at exactly n mutations from WT."""
    s = list(ctx.wt_sequence)
    for p in list(ctx.mutable_positions_1based)[:n]:
        base = s[p - 1]
        alt = [b for b in BASES if b != base][base_idx]
        s[p - 1] = alt
    return "".join(s)


def make_context(tmpdir, *, measured=None, forbidden=(), budget=256, seed=0,
                 allowed=(0, 13), max_generations=8, search_id="test_search",
                 data_sha256="DEADBEEF") -> SearchContext:
    if measured is None:
        measured = [(WT_141, 0.0)]
    return SearchContext(
        search_id=search_id, outer_round=1,
        measured_sequences=tuple(s for s, _ in measured),
        measured_gbsa=tuple(float(v) for _, v in measured),
        forbidden_sequences=frozenset(forbidden),
        seed=seed, allowed_n_from_wt=allowed,
        max_unique_predictions=budget, max_generations=max_generations,
        data_sha256=data_sha256, archive_dir=str(tmpdir))


def make_scorer(f, ctx, *, model_id="mock_v1", with_sigma=False, cache=None,
                fail=False, wrong_length=False) -> SurrogateScorer:
    sur = MockSurrogate(f, model_id=model_id, with_sigma=with_sigma, fail=fail,
                        wrong_length=wrong_length)
    return SurrogateScorer(sur, ctx, cache=cache,
                           ood_train_sequences=list(ctx.measured_sequences))


def read_evaluated(tmpdir) -> list[dict]:
    return SearchArchive(tmpdir).read_evaluated()


def read_state(tmpdir) -> dict:
    return SearchArchive(tmpdir).read_state()


def strip_clock(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "score_time"} for r in rows]


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


# --------------------------------------------------------------------------
# interface / context
# --------------------------------------------------------------------------

class InterfaceContractTest(TempDirTestCase):
    def test_version_and_context_validation(self):
        self.assertEqual(SEARCH_INTERFACE_VERSION, "0.1.1")
        with self.assertRaises(ValueError):
            make_context(self.tmpdir, allowed=(9, 4))
        with self.assertRaises(ValueError):
            make_context(self.tmpdir, measured=[(WT_141, 0.0), (WT_141,)])
        with self.assertRaises(ValueError):
            make_context(self.tmpdir, budget=0)
        ctx = make_context(self.tmpdir)
        fp = ctx.fingerprint()
        for key in ("search_id", "seed", "measured_sha256", "forbidden_sha256",
                    "allowed_n_from_wt", "max_unique_predictions",
                    "max_generations", "data_sha256"):
            self.assertIn(key, fp)

    def test_context_round_trip(self):
        ctx = make_context(self.tmpdir, measured=[(WT_141, 0.0), (mutated_seq(
            make_context(self.tmpdir), 3), 1.5)])
        ctx2 = SearchContext.from_dict(ctx.to_dict())
        self.assertEqual(ctx2.fingerprint(), ctx.fingerprint())


# --------------------------------------------------------------------------
# scorer contract
# --------------------------------------------------------------------------

class ScorerContractTest(TempDirTestCase):
    def test_batch_order_alignment_and_minimisation_direction(self):
        ctx = make_context(self.tmpdir)
        tgt = mutated_seq(ctx, 4)
        scorer = make_scorer(target_landscape(tgt), ctx)
        seqs = [mutated_seq(ctx, k) for k in (2, 5, 1)]
        pred = scorer.score(seqs)
        self.assertEqual(list(pred.mu), [2.0, 1.0, 3.0])
        # minimisation direction: lowest mu ranks first
        self.assertEqual(min(range(3), key=lambda i: pred.mu[i]), 1)

    def test_dedup_and_hard_budget(self):
        ctx = make_context(self.tmpdir, budget=3)
        scorer = make_scorer(flat_landscape(), ctx)
        a, b = mutated_seq(ctx, 1), mutated_seq(ctx, 2)
        pred = scorer.score([a, a, b, a])
        self.assertEqual(len(pred.mu), 4)
        self.assertEqual(scorer.n_used, 2)
        self.assertEqual(scorer.remaining, 1)
        scorer.score([mutated_seq(ctx, 3)])
        self.assertEqual(scorer.n_used, 3)
        with self.assertRaises(BudgetExceededError):
            scorer.score([mutated_seq(ctx, 4)])

    def test_misaligned_and_nonfinite_predictions_rejected(self):
        ctx = make_context(self.tmpdir)
        scorer = make_scorer(flat_landscape(), ctx, wrong_length=True)
        with self.assertRaises(ValueError):
            scorer.score([mutated_seq(ctx, 1), mutated_seq(ctx, 2)])
        bad = MockSurrogate(lambda s: float("nan"))
        scorer2 = SurrogateScorer(bad, ctx)
        with self.assertRaises(ValueError):
            scorer2.score([mutated_seq(ctx, 1)])

    def test_sigma_passthrough(self):
        ctx = make_context(self.tmpdir)
        scorer = make_scorer(flat_landscape(), ctx, with_sigma=True)
        pred = scorer.score([mutated_seq(ctx, 1)])
        self.assertTrue(pred.has_sigma)
        self.assertEqual(pred.uncertainty_kind, "mock_spread_uncalibrated")

    def test_cache_isolation_by_model_snapshot(self):
        # in-process: two scorers never share a cache
        ctx = make_context(self.tmpdir, budget=5)
        scorer_a = make_scorer(flat_landscape(), ctx, model_id="model_A")
        scorer_b = make_scorer(flat_landscape(), ctx, model_id="model_B")
        seq = mutated_seq(ctx, 1)
        scorer_a.score([seq])
        self.assertEqual(scorer_b.n_used, 0)
        # one cache can never mix two models
        cache = PredictionCache()
        cache.put(seq, 1.0, None, None, "model_A", "mock", 0.0)
        with self.assertRaises(ValueError):
            cache.put(mutated_seq(ctx, 2), 1.0, None, None, "model_B", "mock", 0.0)

    def test_surrogate_failure_propagates_clearly(self):
        ctx = make_context(self.tmpdir, budget=16)
        policy = make_policy("multi_start_beam", beam_width=2,
                             offspring_per_parent=2, n_measured_seeds=1,
                             n_random_seeds=0)
        scorer = make_scorer(flat_landscape(), ctx, fail=True)
        with self.assertRaises(ValueError) as cm:
            policy.search(ctx, scorer)
        self.assertIn("boom", str(cm.exception))


# --------------------------------------------------------------------------
# sequence geometry
# --------------------------------------------------------------------------

class GeometryTest(TempDirTestCase):
    def test_one_hop_neighbors_cap_and_range(self):
        ctx = make_context(self.tmpdir, allowed=(4, 13))
        interior = mutated_seq(ctx, 8)
        ones = one_hop_neighbors(interior, ctx)
        self.assertEqual(len(ones), 39)              # 13 sites x 3 bases
        for s, chg in ones:
            self.assertEqual(len(chg), 1)
            self.assertTrue(4 <= mutation_distance(s, ctx.wt_sequence) <= 13)
        # boundary parent: fewer feasible neighbours, deterministic order
        boundary = mutated_seq(ctx, 4)
        ones_b = one_hop_neighbors(boundary, ctx)
        self.assertLess(len(ones_b), 39)
        self.assertEqual(ones_b, one_hop_neighbors(boundary, ctx))
        # order is position-major, base-minor
        pos_first = [chg[0] for _, chg in ones]
        self.assertEqual(pos_first, sorted(pos_first))

    def test_two_hop_neighbors_sampled_deterministic(self):
        ctx = make_context(self.tmpdir, allowed=(0, 13))
        parent = mutated_seq(ctx, 5)
        rng1 = np.random.RandomState(3)
        rng2 = np.random.RandomState(3)
        s1 = two_hop_neighbors(parent, ctx, rng1, limit=10)
        s2 = two_hop_neighbors(parent, ctx, rng2, limit=10)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1), 10)
        for s, chg in s1:
            self.assertEqual(len(chg), 2)
            self.assertTrue(0 <= mutation_distance(s, ctx.wt_sequence) <= 13)
        full = two_hop_neighbors(parent, ctx, np.random.RandomState(0))
        self.assertLessEqual(len(full), 702)

    def test_stratified_random_strata_and_exclude(self):
        ctx = make_context(self.tmpdir, allowed=(4, 13))
        forbidden = {mutated_seq(ctx, 5)}
        rng = np.random.RandomState(0)
        out = stratified_random_seqs(50, ctx, rng, exclude=forbidden)
        self.assertEqual(len(out), 50)
        self.assertEqual(len(set(out)), 50)
        self.assertTrue(forbidden.isdisjoint(out))
        ns = {mutation_distance(s, ctx.wt_sequence) for s in out}
        self.assertEqual(ns, set(range(4, 14)))      # waterfill covers every stratum
        out2 = stratified_random_seqs(50, ctx, np.random.RandomState(0),
                                      exclude=forbidden)
        self.assertEqual(out, out2)


# --------------------------------------------------------------------------
# seed selection
# --------------------------------------------------------------------------

class SeedSelectionTest(TempDirTestCase):
    def test_shared_deterministic_seeds(self):
        ctx = make_context(
            self.tmpdir,
            measured=[(mutated_seq(make_context(self.tmpdir), k), float(k))
                      for k in (1, 2, 3, 4, 5, 6)])
        rng1, rng2 = np.random.RandomState(0), np.random.RandomState(0)
        s1 = pick_initial_seeds(ctx, n_measured=3, n_random=4,
                                min_mutual_distance=2, rng=rng1)
        s2 = pick_initial_seeds(ctx, n_measured=3, n_random=4,
                                min_mutual_distance=2, rng=rng2)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1), 7)
        measured_seeds = [s for s, ev in s1 if ev == "measured"]
        self.assertEqual(len(measured_seeds), 3)
        # lowest-gbsa measured sequence comes first
        self.assertEqual(measured_seeds[0], mutated_seq(ctx, 1))
        pos = ctx.mutable_positions_1based
        for a in measured_seeds:
            for b in measured_seeds:
                if a != b:
                    self.assertGreaterEqual(mutation_distance(a, b, pos), 2)
        for s, ev in s1:
            if ev == "none":
                self.assertNotIn(s, ctx.forbidden_sequences)


# --------------------------------------------------------------------------
# policies: mechanics, reproducibility, budget, stops
# --------------------------------------------------------------------------

class RandomPolicyTest(TempDirTestCase):
    def test_reproducible_budget_and_validity(self):
        ctx_a = make_context(self.tmpdir / "a", budget=200, seed=11)
        ctx_b = make_context(self.tmpdir / "b", budget=200, seed=11)
        tgt = mutated_seq(ctx_a, 9)
        r1 = make_policy("stratified_random").search(
            ctx_a, make_scorer(target_landscape(tgt), ctx_a))
        r2 = make_policy("stratified_random").search(
            ctx_b, make_scorer(target_landscape(tgt), ctx_b))
        self.assertEqual(r1.stop_reason, "budget_exhausted")
        self.assertEqual(r1.n_unique_predicted, 200)
        self.assertEqual([c.candidate_id for c in r1.ranked_candidates],
                         [c.candidate_id for c in r2.ranked_candidates])
        self.assertEqual([c.pred_gbsa for c in r1.ranked_candidates],
                         [c.pred_gbsa for c in r2.ranked_candidates])
        preds = [c.pred_gbsa for c in r1.ranked_candidates]
        self.assertEqual(preds, sorted(preds))
        self.assertEqual(len({c.sequence for c in r1.ranked_candidates}), 200)
        for c in r1.ranked_candidates:
            self.assertNotIn(c.sequence, ctx_a.forbidden_sequences)
            self.assertEqual(c.parent_evidence, "none")

    def test_space_exhausted_graceful(self):
        ctx = make_context(self.tmpdir, budget=100, allowed=(1, 1))
        r = make_policy("stratified_random").search(
            ctx, make_scorer(flat_landscape(), ctx))
        self.assertEqual(r.stop_reason, "space_exhausted")
        self.assertEqual(r.n_unique_predicted, 39)    # whole 1-mutation stratum


class HillClimbPolicyTest(TempDirTestCase):
    def test_reaches_target_on_smooth_landscape(self):
        ctx = make_context(self.tmpdir, budget=400, seed=1)
        tgt = mutated_seq(ctx, 8)
        policy = make_policy("multi_start_hill_climb", n_measured_seeds=1,
                             n_random_seeds=0, min_mutual_distance=1)
        r = policy.search(ctx, make_scorer(target_landscape(tgt), ctx))
        self.assertEqual(r.ranked_candidates[0].pred_gbsa, 0.0)
        self.assertEqual(r.ranked_candidates[0].sequence, tgt)
        self.assertIn(r.stop_reason, ("local_optima", "budget_exhausted", "max_generations"))
        # every child is a 1-hop operator
        for c in r.ranked_candidates:
            if c.operator != "seed":
                self.assertEqual(c.operator, "one_hop")
                self.assertEqual(c.n_from_parent, 1)

    def test_budget_is_never_exceeded(self):
        ctx = make_context(self.tmpdir, budget=90, seed=2)
        policy = make_policy("multi_start_hill_climb", n_measured_seeds=1,
                             n_random_seeds=1)
        r = policy.search(ctx, make_scorer(flat_landscape(), ctx))
        self.assertLessEqual(r.n_unique_predicted, 90)
        self.assertIn(r.stop_reason, ("budget_exhausted", "local_optima",
                                      "neighbors_exhausted"))


class BeamPolicyTest(TempDirTestCase):
    def test_mechanism_vs_baselines_same_budget(self):
        tgt = mutated_seq(make_context(self.tmpdir), 10)
        ctx_r = make_context(self.tmpdir / "r", budget=400, seed=5)
        ctx_h = make_context(self.tmpdir / "h", budget=400, seed=5, max_generations=32)
        ctx_b = make_context(self.tmpdir / "b", budget=400, seed=5)
        cfg = dict(n_measured_seeds=2, n_random_seeds=2, min_mutual_distance=2)
        r_rand = make_policy("stratified_random").search(
            ctx_r, make_scorer(target_landscape(tgt), ctx_r))
        r_hill = make_policy("multi_start_hill_climb", **cfg).search(
            ctx_h, make_scorer(target_landscape(tgt), ctx_h))
        r_beam = make_policy("multi_start_beam", beam_width=8,
                             offspring_per_parent=8, **cfg).search(
            ctx_b, make_scorer(target_landscape(tgt), ctx_b))
        best = lambda r: r.ranked_candidates[0].pred_gbsa
        # hill climbing follows the smooth gradient to the exact optimum;
        # beam (truncated deterministic 1-hop proposals + random 2-hop/restart
        # quota) must beat random at the same budget but is not promised to
        # match hill climbing on this toy landscape
        self.assertEqual(best(r_hill), 0.0)
        self.assertLess(best(r_beam), best(r_rand))
        state_b = read_state(ctx_b.archive_dir)
        self.assertGreater(state_b["generation"], 1)  # genuinely multi-generation

    def test_diversity_rule_switchable(self):
        ctx = make_context(self.tmpdir)
        policy_on = make_policy("multi_start_beam", beam_width=3,
                                diversity_pool_factor=4, diversity_distance=2)
        policy_off = make_policy("multi_start_beam", beam_width=3,
                                 diversity_pool_factor=4, diversity_distance=0)

        def cand(s, pred):
            return ScoredCandidate(
                candidate_id=candidate_id(s), sequence=s, search_id="t",
                outer_round=1, generation=0, parent_candidate_id=None,
                parent_evidence="none", operator="seed", n_from_parent=0,
                n_from_wt=mutation_distance(s, ctx.wt_sequence),
                changed_positions_from_parent=(), pred_gbsa=pred,
                sigma_gbsa=None, uncertainty_kind=None, model_id="m",
                feature_version="f", ood_flag=False, nearest_train_hamming=1.0,
                constraints_version="v1", rng_seed=0, score_time=0.0)

        cands = [cand(mutated_seq(ctx, k), float(20 - k)) for k in range(1, 11)]
        kept_on = policy_on._diverse_select(cands, ctx)
        kept_off = policy_off._diverse_select(cands, ctx)
        self.assertEqual(len(kept_on), 3)
        self.assertEqual(len(kept_off), 3)
        pos = ctx.mutable_positions_1based
        for a in kept_on:
            for b in kept_on:
                if a is not b:
                    self.assertGreaterEqual(
                        mutation_distance(a.sequence, b.sequence, pos), 2)
        # distance 0 disables the rule -> pure top-mu selection
        top3 = sorted(cands, key=lambda c: c.pred_gbsa)[:3]
        self.assertEqual([c.sequence for c in kept_off],
                         [c.sequence for c in top3])

    def test_stagnation_stop(self):
        ctx = make_context(self.tmpdir, budget=300, seed=3)
        policy = make_policy("multi_start_beam", beam_width=4,
                             offspring_per_parent=4, n_measured_seeds=1,
                             n_random_seeds=1, max_stagnant_generations=2)
        r = policy.search(ctx, make_scorer(flat_landscape(), ctx))
        self.assertEqual(r.stop_reason, "stagnation")
        self.assertEqual(read_state(ctx.archive_dir)["generation"], 2)

    def test_no_seeds_stop(self):
        ctx = make_context(self.tmpdir, measured=[], budget=64)
        policy = make_policy("multi_start_beam", n_measured_seeds=1,
                             n_random_seeds=0)
        r = policy.search(ctx, make_scorer(flat_landscape(), ctx))
        self.assertEqual(r.stop_reason, "no_seeds")


# --------------------------------------------------------------------------
# checkpoint / resume
# --------------------------------------------------------------------------

class ResumeTest(TempDirTestCase):
    def test_immediate_parent_evidence_after_multiple_generations(self):
        for name in ("multi_start_hill_climb", "multi_start_beam",
                     "matched_seed_random_walk"):
            with self.subTest(policy=name):
                ctx = make_context(self.tmpdir / name, budget=350, seed=7)
                target = mutated_seq(ctx, 6)
                policy = make_policy(name, n_measured_seeds=1, n_random_seeds=0)
                policy.search(ctx, make_scorer(target_landscape(target), ctx))
                rows = read_evaluated(ctx.archive_dir)
                by_id = {r["candidate_id"]: r for r in rows}
                parent_types = set()
                for row in rows:
                    parent_id = row["parent_candidate_id"]
                    if parent_id is None:
                        continue
                    parent = by_id[parent_id]
                    expected = ("measured" if parent["operator"] == "seed"
                                and parent["parent_evidence"] == "measured"
                                else "predicted_only")
                    self.assertEqual(row["parent_evidence"], expected)
                    parent_types.add(expected)
                self.assertEqual(parent_types, {"measured", "predicted_only"})

    def test_matched_random_walk_resume_and_shared_seed_order(self):
        cfg = dict(n_measured_seeds=2, n_random_seeds=2,
                   min_mutual_distance=2)
        def run(path, stop=None):
            ctx = make_context(path, budget=160, seed=17, max_generations=4)
            policy = make_policy("matched_seed_random_walk",
                                 hard_stop_generations=stop, **cfg)
            result = policy.search(ctx, make_scorer(flat_landscape(), ctx))
            return ctx, result
        ctx_full, full = run(self.tmpdir / "full")
        _, interrupted = run(self.tmpdir / "interrupted", stop=1)
        self.assertEqual(interrupted.stop_reason, "interrupted")
        ctx_resumed = make_context(self.tmpdir / "interrupted", budget=160,
                                   seed=17, max_generations=4)
        resumed = make_policy("matched_seed_random_walk", **cfg).search(
            ctx_resumed, make_scorer(flat_landscape(), ctx_resumed,
                                     cache=SearchArchive(
                                         self.tmpdir / "interrupted").load_cache()))
        self.assertEqual(resumed.n_unique_predicted, full.n_unique_predicted)
        self.assertEqual(strip_clock(read_evaluated(self.tmpdir / "full")),
                         strip_clock(read_evaluated(self.tmpdir / "interrupted")))
        ctx_h = make_context(self.tmpdir / "hill", budget=160,
                             seed=17, max_generations=4)
        make_policy("multi_start_hill_climb", **cfg).search(
            ctx_h, make_scorer(flat_landscape(), ctx_h))
        seeds = lambda path: [(r["candidate_id"], r["sequence"])
                              for r in read_evaluated(path) if r["operator"] == "seed"]
        self.assertEqual(seeds(self.tmpdir / "full"),
                         seeds(self.tmpdir / "hill"))

    def _beam_runs(self, tmpdir, *, hard_stop=None):
        ctx = make_context(tmpdir, budget=512, seed=7)
        tgt = mutated_seq(ctx, 6)
        cfg = dict(beam_width=8, offspring_per_parent=8, n_measured_seeds=2,
                   n_random_seeds=2, min_mutual_distance=2, max_generations=6)
        policy = make_policy("multi_start_beam", hard_stop_generations=hard_stop,
                             **cfg)
        r = policy.search(ctx, make_scorer(target_landscape(tgt), ctx))
        return ctx, cfg, r

    def test_beam_resume_is_bit_identical(self):
        full_dir = self.tmpdir / "full"
        int_dir = self.tmpdir / "interrupted"
        ctx_full, cfg, r_full = self._beam_runs(full_dir)
        _, _, r_int = self._beam_runs(int_dir, hard_stop=2)
        self.assertEqual(r_int.stop_reason, "interrupted")

        ctx_int = make_context(int_dir, budget=512, seed=7)
        policy2 = make_policy("multi_start_beam", **cfg)
        cache = SearchArchive(int_dir).load_cache()
        r_resumed = policy2.search(
            ctx_int, make_scorer(target_landscape(mutated_seq(ctx_int, 6)),
                                 ctx_int, cache=cache))
        self.assertEqual(r_resumed.stop_reason, r_full.stop_reason)
        self.assertEqual(r_resumed.n_unique_predicted, r_full.n_unique_predicted)
        self.assertEqual([c.candidate_id for c in r_resumed.ranked_candidates],
                         [c.candidate_id for c in r_full.ranked_candidates])
        self.assertEqual([g.to_dict() for g in r_resumed.groups],
                         [g.to_dict() for g in r_full.groups])
        self.assertEqual(strip_clock(read_evaluated(int_dir)),
                         strip_clock(read_evaluated(full_dir)))

    def test_hill_climb_resume_is_bit_identical(self):
        full_dir = self.tmpdir / "full"
        int_dir = self.tmpdir / "interrupted"
        cfg = dict(n_measured_seeds=2, n_random_seeds=0, min_mutual_distance=2)

        def run(tmpdir, hard_stop=None):
            ctx = make_context(tmpdir, budget=300, seed=9)
            policy = make_policy("multi_start_hill_climb",
                                 hard_stop_generations=hard_stop, **cfg)
            return ctx, policy.search(
                ctx, make_scorer(target_landscape(mutated_seq(ctx, 5)), ctx))

        ctx_full, r_full = run(full_dir)
        _, r_int = run(int_dir, hard_stop=2)
        self.assertEqual(r_int.stop_reason, "interrupted")
        ctx_int = make_context(int_dir, budget=300, seed=9)
        policy2 = make_policy("multi_start_hill_climb", **cfg)
        cache = SearchArchive(int_dir).load_cache()
        r_resumed = policy2.search(
            ctx_int, make_scorer(target_landscape(mutated_seq(ctx_int, 5)),
                                 ctx_int, cache=cache))
        self.assertEqual(r_resumed.stop_reason, r_full.stop_reason)
        self.assertEqual(r_resumed.n_unique_predicted, r_full.n_unique_predicted)
        self.assertEqual([c.candidate_id for c in r_resumed.ranked_candidates],
                         [c.candidate_id for c in r_full.ranked_candidates])
        self.assertEqual(strip_clock(read_evaluated(int_dir)),
                         strip_clock(read_evaluated(full_dir)))

    def test_resume_refuses_snapshot_mismatch(self):
        int_dir = self.tmpdir / "interrupted"
        _, cfg, _ = self._beam_runs(int_dir, hard_stop=1)
        # changed seed -> refuse
        ctx_bad = make_context(int_dir, budget=512, seed=8)
        with self.assertRaises(ValueError) as cm:
            make_policy("multi_start_beam", **cfg).search(
                ctx_bad, make_scorer(flat_landscape(), ctx_bad,
                                     cache=SearchArchive(int_dir).load_cache()))
        self.assertIn("snapshot mismatch", str(cm.exception))
        # changed policy config -> refuse
        ctx_ok = make_context(int_dir, budget=512, seed=7)
        with self.assertRaises(ValueError):
            make_policy("multi_start_beam", beam_width=16, **{k: v for k, v in cfg.items()
                                                              if k != "beam_width"}).search(
                ctx_ok, make_scorer(flat_landscape(), ctx_ok,
                                    cache=SearchArchive(int_dir).load_cache()))

    def test_rerun_after_done_is_idempotent(self):
        ctx = make_context(self.tmpdir, budget=200, seed=4)
        cfg = dict(beam_width=4, offspring_per_parent=4, n_measured_seeds=1,
                   n_random_seeds=1)
        r1 = make_policy("multi_start_beam", **cfg).search(
            ctx, make_scorer(flat_landscape(), ctx))
        n_rows = len(read_evaluated(self.tmpdir))
        r2 = make_policy("multi_start_beam", **cfg).search(
            ctx, make_scorer(flat_landscape(), ctx,
                             cache=SearchArchive(self.tmpdir).load_cache()))
        self.assertEqual(r2.stop_reason, r1.stop_reason)
        self.assertEqual(r2.n_unique_predicted, r1.n_unique_predicted)
        self.assertEqual([c.candidate_id for c in r2.ranked_candidates],
                         [c.candidate_id for c in r1.ranked_candidates])
        self.assertEqual(len(read_evaluated(self.tmpdir)), n_rows)


# --------------------------------------------------------------------------
# OOD helper equivalence + candidate groups
# --------------------------------------------------------------------------

class OodAndGroupsTest(TempDirTestCase):
    def test_ood_helpers_match_loop_implementations(self):
        from scripts.loop import loop as loop_mod
        ctx = make_context(self.tmpdir, allowed=(4, 13))
        rng = np.random.RandomState(0)
        train = stratified_random_seqs(30, ctx, rng)
        pool = stratified_random_seqs(25, ctx, rng)
        self.assertEqual(training_ood_info(train), loop_mod.ood_threshold(train))
        self.assertTrue(np.array_equal(ood_flags(pool, train),
                                       loop_mod.compute_ood(pool, train)))
        self.assertTrue(np.allclose(nearest_train_hamming(pool, train),
                                    loop_mod.pool_train_distances(pool, train)))

    def test_groups_one_or_two_clusters(self):
        ctx = make_context(self.tmpdir)

        def cand(s, pred):
            return ScoredCandidate(
                candidate_id=candidate_id(s), sequence=s, search_id="t",
                outer_round=1, generation=1, parent_candidate_id=None,
                parent_evidence="none", operator="global_restart", n_from_parent=0,
                n_from_wt=mutation_distance(s, ctx.wt_sequence),
                changed_positions_from_parent=(), pred_gbsa=pred, sigma_gbsa=None,
                uncertainty_kind=None, model_id="m", feature_version="f",
                ood_flag=False, nearest_train_hamming=1.0, constraints_version="v1",
                rng_seed=0, score_time=0.0)

        # one cluster: everything within radius 1 of the best
        c1 = [cand(mutated_seq(ctx, 4), 1.0), cand(mutated_seq(ctx, 5), 2.0)]
        g1 = build_groups(c1, ctx)
        self.assertEqual(len(g1), 1)
        self.assertEqual(g1[0].group_role, "predicted_best")
        # two clusters: a second basin far from the first
        far = list(ctx.wt_sequence)
        for p in list(ctx.mutable_positions_1based)[:10]:
            far[p - 1] = [b for b in BASES if b != far[p - 1]][1]
        c2 = [cand(mutated_seq(ctx, 4), 1.0), cand(mutated_seq(ctx, 5), 2.0),
              cand("".join(far), 3.0)]
        g2 = build_groups(c2, ctx)
        self.assertEqual(len(g2), 2)
        self.assertEqual([g.group_role for g in g2],
                         ["predicted_best", "diverse_backup"])
        self.assertEqual({cid for g in g2 for cid in g.member_ids},
                         {c.candidate_id for c in c2})
        self.assertEqual(sum(len(g.member_ids) for g in g2), len(c2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
