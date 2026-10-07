from pathlib import Path
import sys
import json
import pickle
import hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.verify_search_suite import audit_subrun
from scripts.loop.interface import normalize
from scripts.reproduction.eval_fusion_smoothing import build
known=set(pd.read_csv(ROOT/'data/proxy2000_v2/fad_proxy2000_v2_full.csv').Sequence.map(normalize))
out=[];models=[];starts=[]
for key in ('ils','sa','adalead'):
    root=ROOT/f'runs/local_search_{key}_demo_20261005';arch=root/'checkpoints'
    count,signature,_=audit_subrun(arch,known)
    s=json.loads((arch/'search_summary.json').read_text('utf-8'))
    model=pickle.loads((arch/'fitted_surrogate.pkl').read_bytes())
    assert model['model_id']==s['model_id']
    rows=[json.loads(line) for line in (arch/'search_evaluated.jsonl').read_text('utf-8').splitlines()]
    matrix=build([r['sequence'] for r in rows])
    if model['top_idx'] is not None:matrix=matrix[:,model['top_idx']]
    predicted=np.stack([m.predict(matrix) for m in model['models']]).mean(axis=0)
    np.testing.assert_allclose(predicted,[r['pred_gbsa'] for r in rows],rtol=0,atol=1e-12)
    assert json.loads((root/'DONE').read_text('utf-8'))['acceptance_ok']
    models.append(s['model_id']);starts.append(signature)
    out.append(dict(policy=s['policy'],run=root.name,actual_predictions=count,
        n_candidates=s['n_candidates'],best_pred_gbsa=s['best_pred_gbsa'],
        stop_reason=s['stop_reason'],n_groups=s['n_groups'],
        fitted_weights_sha256=hashlib.sha256((arch/'fitted_surrogate.pkl').read_bytes()).hexdigest()))
assert len(set(models))==1 and all(x==starts[0] for x in starts)
result=dict(ok=True,checks=['artifact integrity','budget and genealogy','groups',
    'saved fitted weights reproduce archived scores','same fitted model and ordered starts'],runs=out)
(Path(__file__).parent/'demo_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
