"""New search-only settings must not block a legacy static-pool resume."""
from pathlib import Path
import json
import sys
import unittest
import tempfile
sys.path.insert(0,str(Path(__file__).parent))
import test_search_loop_integration as fixtures
from scripts.loop.loop import ClosedLoop

_args_for=fixtures._args_for

class LegacyFingerprintTest(unittest.TestCase):
    def setUp(self):
        self._tmp=tempfile.TemporaryDirectory()
        self.tmpdir=Path(self._tmp.name)
        self.ckpt=self.tmpdir/'checkpoints'

    def tearDown(self):
        self._tmp.cleanup()

    def test_legacy_static_fingerprint_resumes_without_rewriting(self):
        args=_args_for(self.tmpdir,**{'candidate-source':'mutate'})
        self.assertEqual(ClosedLoop(args).run()[1],3)
        path=self.ckpt/'fingerprint.json'
        old=json.loads(path.read_text('utf-8'))
        old['config']={k:v for k,v in old['config'].items() if not k.startswith('search_')}
        path.write_text(json.dumps(old,indent=2),encoding='utf-8')
        before=path.read_bytes()
        self.assertEqual(ClosedLoop(_args_for(self.tmpdir,**{'candidate-source':'mutate','resume':None})).run()[1],3)
        self.assertEqual(path.read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            ClosedLoop(_args_for(self.tmpdir,**{'candidate-source':'mutate','resume':None,'budget':'9'})).run()
