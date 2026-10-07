"""Mechanism, lineage, budget and recovery of single-parent GA."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from test_search_contract import (TempDirTestCase, make_context, make_scorer,
    target_landscape, mutated_seq, read_evaluated, strip_clock, flat_landscape)
from scripts.loop.search_policies import make_policy, mutation_distance
from scripts.loop.search_archive import SearchArchive
from scripts.loop.surrogates import make_surrogate


class GeneticContractTest(TempDirTestCase):
    def test_single_parent_mutations_and_shared_start(self):
        ctx = make_context(self.tmpdir, budget=350, seed=7, max_generations=32)
        target = mutated_seq(ctx, 6)
        policy = make_policy("mutation_only_ga", n_measured_seeds=1,
                             n_random_seeds=2, restart_fraction=0)
        result = policy.search(ctx, make_scorer(target_landscape(target), ctx))
        rows = read_evaluated(ctx.archive_dir)
        lookup = {r["candidate_id"]: r for r in rows}
        self.assertLessEqual(result.n_unique_predicted, 350)
        self.assertLess(result.ranked_candidates[0].pred_gbsa, 6)
        self.assertEqual(len(rows), len({r["sequence"] for r in rows}))
        for row in rows:
            if row["operator"] == "seed":
                continue
            self.assertIn(row["operator"], ("one_hop", "two_hop"))
            parent = lookup[row["parent_candidate_id"]]
            self.assertEqual(mutation_distance(row["sequence"], parent["sequence"]),
                             row["n_from_parent"])
            self.assertEqual(row["parent_evidence"], "measured" if
                parent["operator"] == "seed" and parent["parent_evidence"] == "measured"
                else "predicted_only")

    def test_resume_matches_uninterrupted(self):
        cfg = dict(n_measured_seeds=1, n_random_seeds=2, offspring_size=32)
        def run(where, stop=None):
            ctx = make_context(where, budget=250, seed=5, max_generations=32)
            return make_policy("mutation_only_ga", hard_stop_generations=stop,
                **cfg).search(ctx, make_scorer(target_landscape(mutated_seq(ctx, 7)), ctx))
        run(self.tmpdir / "full")
        interrupted = run(self.tmpdir / "part", stop=2)
        self.assertEqual(interrupted.stop_reason, "interrupted")
        run(self.tmpdir / "part")
        self.assertEqual(strip_clock(read_evaluated(self.tmpdir / "full")),
                         strip_clock(read_evaluated(self.tmpdir / "part")))

    def test_elite_monotonicity_and_stagnation(self):
        ctx = make_context(self.tmpdir, budget=400, max_generations=32)
        result = make_policy("mutation_only_ga", n_measured_seeds=1,
            n_random_seeds=2, max_stagnant_generations=2).search(
            ctx, make_scorer(flat_landscape(), ctx))
        self.assertEqual(result.stop_reason, "stagnation")
        state = SearchArchive(ctx.archive_dir).read_state()
        values = [r['best_pred_gbsa'] for r in state['generation_rows']]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_hill_generation_cap_is_enforced(self):
        ctx = make_context(self.tmpdir, budget=400, max_generations=1)
        result = make_policy("multi_start_hill_climb", n_measured_seeds=1,
            n_random_seeds=0).search(ctx, make_scorer(
            target_landscape(mutated_seq(ctx, 8)), ctx))
        self.assertEqual(result.stop_reason, "max_generations")
        self.assertEqual(SearchArchive(ctx.archive_dir).read_state()['generation'], 1)

    def test_parameter_validation(self):
        for kwargs in (dict(population_size=0), dict(offspring_size=0),
                       dict(tournament_size=0), dict(elite_count=33)):
            with self.assertRaises(ValueError):
                make_policy("mutation_only_ga", **kwargs)

    def test_fitted_model_identity_and_real_cache_rejection(self):
        ctx = make_context(self.tmpdir, budget=40)
        seqs = [mutated_seq(ctx, n) for n in range(1, 13)]
        y = [float(n) for n in range(1, 13)]
        a = make_surrogate('rf', seed=1).fit(seqs, y)
        b = make_surrogate('rf', seed=2).fit(seqs, y)
        self.assertNotEqual(a.model_id, b.model_id)
        from scripts.loop.search import SurrogateScorer
        score = SurrogateScorer(a, ctx)
        score.score(seqs[:2])
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            SurrogateScorer(b, ctx, cache=score.cache, model_id=b.model_id)


if __name__ == '__main__':
    unittest.main()
