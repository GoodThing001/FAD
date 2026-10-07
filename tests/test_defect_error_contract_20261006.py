"""API failure contracts using a stub; not a replacement for real NUPACK."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

def load_with_stub(monkeypatch,probability):
    matrix=np.eye(141);matrix[:2,:2]=[[.3,.7],[.7,.3]]
    stub=SimpleNamespace(config=SimpleNamespace(threads=0),Model=lambda **kw:object(),
        Strand=lambda sequence,**kw:sequence,Complex=lambda strands:strands,
        pairs=lambda *a,**kw:SimpleNamespace(to_array=lambda:matrix),
        structure_probability=probability)
    monkeypatch.setitem(sys.modules,'nupack',stub)
    path=Path(__file__).parents[1]/'scripts/reproduction/ensemble_defect.py'
    spec=importlib.util.spec_from_file_location('scripts.reproduction._defect_contract_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod._TARGET_PARTNER=[1,0]+[-1]*139;mod._WT_STRUCT='()'+'.'*139
    return mod

def test_unexpected_probability_error_is_persisted_not_zero(monkeypatch):
    def failure(**kw):raise RuntimeError('unexpected API failure')
    mod=load_with_stub(monkeypatch,failure)
    with pytest.raises(RuntimeError,match='unexpected API failure'):mod.compute_defect('GC'+'A'*139)
    row=mod._worker('GC'+'A'*139)
    assert row['error_type']=='RuntimeError' and 'unexpected API failure' in row['error']

def test_invalid_target_pair_is_explicit_zero_without_api_call(monkeypatch):
    def forbidden(**kw):raise AssertionError('probability API should not run for GG target pair')
    mod=load_with_stub(monkeypatch,forbidden)
    defect,_,prob,status=mod.compute_defect('GG'+'A'*139)
    assert prob==0 and status=='invalid_target_pair' and defect==pytest.approx(.6)
