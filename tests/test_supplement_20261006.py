import numpy as np
from scripts.reproduction.supplement_cpu_20261006 import extra
from scripts.reproduction.new_features import WT_141, build_seq
from scripts.reproduction.supplement_active_20261006 import select

def test_recovered_extra_features_wt_and_one_transition():
    x=extra([WT_141])[0]
    assert np.all(x[:12]==0)
    seq=list(WT_141);p=30;wt=seq[p-1];seq[p-1]={'A':'G','G':'A','C':'U','U':'C'}[wt]
    z=extra([''.join(seq)])[0]
    assert z[0]==1 and z[1]==1 and z[4]==1 and z[5]==0
    assert z[10]==1 and z[11]==0

def test_acquisition_direction_and_batch_uniqueness():
    available=np.arange(200);mu=np.arange(200,dtype=float);sigma=mu.copy()
    X=np.arange(220,dtype=float)[:,None]
    assert np.array_equal(select('greedy',mu,sigma,X,np.arange(200,220),available,1),np.arange(100))
    assert set(select('uncertainty',mu,sigma,X,np.arange(200,220),available,1))==set(range(100,200))
    mixed=select('mixed',mu,sigma,X,np.arange(200,220),available,1)
    assert len(mixed)==len(set(mixed))==100
    assert set(mixed)<=set(available)

def test_bounded_coverage_matches_full_float32_reduction():
    from scripts.reproduction.supplement_active_20261006 import coverage
    from scripts.reproduction.supplement_active_bounded_20261006 import coverage as bounded
    rng=np.random.default_rng(42);X=rng.normal(size=(211,303)).astype(np.float32)
    available=np.arange(150);labeled=np.arange(150,211);chosen=[0,4,90]
    assert coverage(X,labeled,available,25,chosen)==bounded(X,labeled,available,25,chosen)
