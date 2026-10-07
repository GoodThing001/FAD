"""Small CPU contracts only; random tensors here are not scientific results."""
import numpy as np
import unittest
try:
    import torch
except ImportError:
    torch=None
if torch is not None:
    from scripts.reproduction.supplement_paper_20261006 import heads_fit,in_domain
    from scripts.reproduction.new_features import WT_141

def check_regression_heads_do_not_use_holdout_targets():
    torch.set_num_threads(1)
    E=torch.tensor(np.random.default_rng(4).normal(size=(20,128)),dtype=torch.float32)
    y=np.random.default_rng(5).normal(size=20);tr=np.arange(16)
    first=heads_fit(E,y,tr,42,'cpu',2,2)
    altered=y.copy();altered[16:]+=100000
    second=heads_fit(E,altered,tr,42,'cpu',2,2)
    assert np.array_equal(first[0],second[0]) and np.array_equal(first[1],second[1])

def check_domain_corpus_excludes_known_sequences_and_preserves_fixed_sites():
    result=in_domain([WT_141],{WT_141},50,42)
    variable={p-1 for p in [30,31,32,33,89,90,91,92,127,128,129,130,131]}
    assert len(set(result))==50 and WT_141 not in result
    for s in result:
        assert len(s)==141 and all(s[i]==WT_141[i] for i in range(141) if i not in variable)

@unittest.skipIf(torch is None,'PyTorch is not installed in this interpreter')
class PaperContract(unittest.TestCase):
    def test_training_targets(self):check_regression_heads_do_not_use_holdout_targets()
    def test_corpus_exclusion(self):check_domain_corpus_excludes_known_sequences_and_preserves_fixed_sites()
