"""Synthetic GA proposals exercise the optional future file/state path only."""
import json
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

import test_loop_contract as contract
ROOT=contract.ROOT

class MutationOnlyGAFutureTest(contract.ResumeRetrainTest):
    ARGS = dict(contract.ResumeRetrainTest.ARGS, **{
        'candidate-source': 'search', 'search-policy': 'mutation_only_ga',
        'search-max-unique-predictions': 64, 'search-max-generations': 8,
        'search-population-size': 8, 'search-offspring-size': 16,
        'search-n-measured-seeds': 2, 'search-n-random-seeds': 2,
        'n-init': 40, 'n-holdout': 20, 'pool-size': 30})

    def test_ga_pin_protocol_mixed_return_and_done(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import run_experiment as runner
        run_id='test_ga_future_protocol'
        run_dir=self._make_run(run_id,tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(run_dir,ignore_errors=True))
        self.assertEqual(runner.resume_run(run_id,None,pin='synthetic_ga_test_v1'),3)
        batch=pd.read_csv(run_dir/'checkpoints/round_001_batch.csv')
        self.assertTrue(batch['group_id'].notna().all())
        self.assertEqual(batch['group_id'].nunique(),2)
        labels=self._labels(batch, list(np.linspace(-10,-5,len(batch)-1))+[np.nan],
                            ['pass']*(len(batch)-1)+['fail'],'synthetic_ga_test_v1')
        self.addCleanup(lambda: shutil.rmtree(labels.parent,ignore_errors=True))
        self.assertEqual(runner.resume_run(run_id,str(labels)),0)
        self.assertTrue((run_dir/'DONE').exists())
        state=json.loads((run_dir/'checkpoints/state.json').read_text('utf-8'))
        self.assertEqual(state['status'],'retrained')
        self.assertEqual(len(state['failed']),1)
        ledger=pd.read_csv(run_dir/'checkpoints/ledger.csv')
        self.assertEqual(sum(ledger.status=='labeled'),len(batch)-1)
