"""New optimizer mechanics, budget, genealogy, and exact checkpoint recovery."""
import math
import json
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from test_search_contract import (TempDirTestCase, make_context, make_scorer,
    target_landscape, mutated_seq, read_evaluated, strip_clock, flat_landscape)
from scripts.loop.search_policies import (make_policy, SimulatedAnnealingPolicy,
                                         mutation_distance)
from scripts.loop.search_archive import SearchArchive
from scripts.loop.loop import ClosedLoop
from scripts.loop import search_run

NEW = ('iterated_local_search', 'simulated_annealing', 'mutation_only_adalead')
OLD = ('matched_seed_random_walk', 'multi_start_hill_climb',
       'multi_start_beam', 'mutation_only_ga')


class ExtensionsTest(TempDirTestCase):
    def test_all_new_search_sources_park_with_complete_group_handoff(self):
        import pandas as pd
        from test_search_loop_integration import _args_for
        for name in NEW:
            with self.subTest(policy=name):
                directory=self.tmpdir/name
                args=_args_for(directory, **{'search-policy':name,
                    'search-max-generations':64})
                engine=ClosedLoop(args)
                self.assertEqual(engine.run()[1],3)
                ckpt=directory/'checkpoints'
                batch=pd.read_csv(ckpt/'round_001_batch.csv')
                self.assertEqual(len(batch),10)
                self.assertTrue(batch.group_id.notna().all())
                manifest=json.loads((ckpt/'round_001_batch_manifest.json').read_text('utf-8'))
                self.assertEqual(sum(manifest['search_handoff']['batch_group_counts'].values()),10)
                again=ClosedLoop(_args_for(directory, **{'search-policy':name,
                    'search-max-generations':64,'resume':None}))
                self.assertEqual(again.run()[1],3)
                self.assertEqual(pd.read_csv(ckpt/'round_001_batch.csv').candidate_id.tolist(),
                                 batch.candidate_id.tolist())

    def test_all_seven_share_ordered_starts(self):
        signatures = []
        for name in OLD + NEW:
            ctx = make_context(self.tmpdir / name, budget=100, seed=13, max_generations=64)
            make_policy(name, n_measured_seeds=1, n_random_seeds=3).search(
                ctx, make_scorer(flat_landscape(), ctx))
            signatures.append([(r['sequence'], r['parent_evidence']) for r in
                               read_evaluated(ctx.archive_dir) if r['operator'] == 'seed'])
        self.assertTrue(all(s == signatures[0] for s in signatures))
        self.assertEqual(len(signatures[0]), 4)

    def test_lineage_budget_groups_and_generation_cap(self):
        for name in NEW:
            with self.subTest(policy=name):
                ctx = make_context(self.tmpdir / name, budget=145, seed=5, max_generations=128)
                result = make_policy(name, n_measured_seeds=1, n_random_seeds=2).search(
                    ctx, make_scorer(target_landscape(mutated_seq(ctx, 8)), ctx))
                rows = read_evaluated(ctx.archive_dir)
                self.assertEqual(len(rows), result.n_unique_predicted)
                self.assertLessEqual(len(rows), 145)
                self.assertEqual(len({r['sequence'] for r in rows}), len(rows))
                self.assertEqual(len([c for g in result.groups for c in g.member_ids]),
                                 len(result.ranked_candidates))
                lookup = {}
                for row in rows:
                    if row['parent_candidate_id']:
                        parent = lookup[row['parent_candidate_id']]
                        self.assertGreater(row['generation'], parent['generation'])
                        self.assertEqual(mutation_distance(row['sequence'], parent['sequence']),
                                         row['n_from_parent'])
                        self.assertIn(row['operator'], ('one_hop', 'two_hop'))
                    lookup[row['candidate_id']] = row
                self.assertLessEqual(SearchArchive(ctx.archive_dir).read_state()['generation'], 128)

    def test_checkpoint_resume_bit_identity_for_each_new_policy(self):
        for name in NEW:
            with self.subTest(policy=name):
                def run(path, stop=None):
                    ctx = make_context(path, budget=600, seed=27, max_generations=256)
                    return make_policy(name, n_measured_seeds=1, n_random_seeds=2,
                        hard_stop_generations=stop).search(ctx,
                        make_scorer(target_landscape(mutated_seq(ctx, 7)), ctx))
                full = self.tmpdir / name / 'full'; part = self.tmpdir / name / 'part'
                run(full)
                interrupted = run(part, 2)
                self.assertEqual(interrupted.stop_reason, 'interrupted')
                run(part)
                self.assertEqual(strip_clock(read_evaluated(full)), strip_clock(read_evaluated(part)))

    def test_ils_actually_escapes_flat_local_neighborhood_by_two_sites(self):
        ctx = make_context(self.tmpdir, budget=140, seed=7, max_generations=32)
        make_policy(NEW[0], n_measured_seeds=1, n_random_seeds=0).search(
            ctx, make_scorer(flat_landscape(), ctx))
        rows = read_evaluated(ctx.archive_dir)
        self.assertTrue(any(r['operator'] == 'two_hop' for r in rows))
        waves = SearchArchive(ctx.archive_dir).read_state()['generation_rows']
        self.assertGreater(sum(w['perturbations'] for w in waves), 0)

    def test_sa_probability_and_actual_worse_steps(self):
        self.assertEqual(SimulatedAnnealingPolicy.acceptance_probability(-1, 2), 1)
        self.assertAlmostEqual(SimulatedAnnealingPolicy.acceptance_probability(1, 2), math.exp(-0.5))
        base = make_context(self.tmpdir)
        ctx = make_context(self.tmpdir, measured=[(base.wt_sequence, -100),
            (mutated_seq(base, 13), 100)], budget=100, seed=0, max_generations=128)
        make_policy(NEW[1], n_measured_seeds=1, n_random_seeds=0,
            sa_temperature_scale=100).search(ctx,
            make_scorer(target_landscape(ctx.wt_sequence), ctx))
        waves = SearchArchive(ctx.archive_dir).read_state()['generation_rows']
        self.assertGreater(sum(w['accepted_worse'] for w in waves), 0)
        temperatures = [w['temperature'] for w in waves]
        self.assertEqual(temperatures, sorted(temperatures, reverse=True))

    def test_adalead_signed_minimum_uses_additive_threshold(self):
        base = make_context(self.tmpdir)
        ctx = make_context(self.tmpdir, measured=[(base.wt_sequence, -10),
            (mutated_seq(base, 13), -9)], budget=70, max_generations=64)
        make_policy(NEW[2], n_measured_seeds=1, n_random_seeds=0,
            adalead_rollouts=1).search(ctx,
            make_scorer(lambda s: -10 if s == ctx.wt_sequence else -9, ctx))
        waves = SearchArchive(ctx.archive_dir).read_state()['generation_rows']
        self.assertTrue(waves)
        self.assertTrue(all(w['eligible_roots'] == 1 for w in waves))
        self.assertTrue(all(abs(w['tolerance'] - 0.025) < 1e-10 for w in waves))

    def test_active_extension_parameter_changes_reject_resume(self):
        for name, parameter, changed in ((NEW[0], 'ils_perturb_sites', 3),
                (NEW[1], 'sa_temperature_scale', 0.5),
                (NEW[2], 'adalead_tolerance_scale', 0.1)):
            with self.subTest(policy=name):
                ctx = make_context(self.tmpdir / name, budget=60, max_generations=64)
                make_policy(name, n_measured_seeds=1, n_random_seeds=1).search(
                    ctx, make_scorer(flat_landscape(), ctx))
                with self.assertRaises(ValueError):
                    make_policy(name, n_measured_seeds=1, n_random_seeds=1,
                        **{parameter: changed}).search(ctx, make_scorer(flat_landscape(), ctx))

    def test_invalid_parameters(self):
        for name, kwargs in ((NEW[0], {'ils_perturb_sites': 1}),
                (NEW[1], {'sa_temperature_scale': 0}),
                (NEW[1], {'sa_temperature_scale': float('nan')}),
                (NEW[1], {'sa_final_temperature_ratio': 0}),
                (NEW[2], {'adalead_parent_cap': 0}),
                (NEW[2], {'adalead_rollouts': 0}),
                (NEW[2], {'adalead_tolerance_scale': -1})):
            with self.subTest(policy=name, kwargs=kwargs), self.assertRaises(ValueError):
                make_policy(name, **kwargs)

    def test_loop_fingerprint_preserves_only_active_extension_settings(self):
        cfg = {'candidate_source': 'search', 'search_policy': NEW[1],
               'search_sa_temperature_scale': 0.25, 'search_ils_perturb_sites': 2,
               'search_adalead_rollouts': 16}
        active = ClosedLoop.active_fingerprint_config(cfg)
        self.assertIn('search_sa_temperature_scale', active)
        self.assertNotIn('search_ils_perturb_sites', active)
        self.assertNotIn('search_adalead_rollouts', active)
        cfg['candidate_source'] = 'canonical'
        self.assertEqual(ClosedLoop.active_fingerprint_config(cfg), {'candidate_source': 'canonical'})

    def test_standalone_cli_exposes_all_three(self):
        for name in NEW:
            self.assertEqual(search_run.parse_args(['--policy', name]).policy, name)


if __name__ == '__main__':
    unittest.main()
