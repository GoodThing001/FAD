"""Resume must not silently reuse a changed label or sequence domain."""
from dataclasses import replace
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).parent))
from test_search_contract import TempDirTestCase,make_context,make_scorer,flat_landscape
from scripts.loop.search_policies import make_policy

class ContextIdentityTest(TempDirTestCase):
    def test_labels_reference_and_mutable_domain_are_bound(self):
        ctx=make_context(self.tmpdir,budget=64,max_generations=4)
        cfg=dict(n_measured_seeds=1,n_random_seeds=2,offspring_size=16)
        make_policy('mutation_only_ga',**cfg).search(ctx,make_scorer(flat_landscape(),ctx))
        snapshot=json.loads((Path(ctx.archive_dir)/'search_snapshot.json').read_text('utf-8'))
        self.assertEqual(snapshot['measured_values_sha256'],ctx.fingerprint()['measured_values_sha256'])
        self.assertEqual(snapshot['wt_sequence'],ctx.wt_sequence)
        self.assertEqual(snapshot['mutable_positions_1based'],list(ctx.mutable_positions_1based))
        changed_ref=list(ctx.wt_sequence)
        p=ctx.mutable_positions_1based[0]-1
        changed_ref[p]='A' if changed_ref[p]!='A' else 'C'
        variants=[replace(ctx,measured_gbsa=(1.,)),
                  replace(ctx,wt_sequence=''.join(changed_ref)),
                  replace(ctx,mutable_positions_1based=tuple(reversed(ctx.mutable_positions_1based)))]
        for changed in variants:
            with self.subTest(fingerprint=changed.fingerprint()):
                with self.assertRaisesRegex(ValueError,'snapshot'):
                    make_policy('mutation_only_ga',**cfg).search(
                        changed,make_scorer(flat_landscape(),changed))
        snapshot['source_sha256']={'older_test_source':'different'}
        (Path(ctx.archive_dir)/'search_snapshot.json').write_text(json.dumps(snapshot),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'source_sha256'):
            make_policy('mutation_only_ga',**cfg).search(ctx,make_scorer(flat_landscape(),ctx))
