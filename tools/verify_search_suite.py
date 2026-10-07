"""Independent artifact audit: budgets, exact scores, genealogy and group coverage."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.loop.interface import MUT_POS,WT_141,candidate_id,normalize,validate_sequences

def audit_subrun(path,known):
    summary=json.loads((path/'search_summary.json').read_text('utf-8'))
    hashes=json.loads((path/'files.sha256').read_text('utf-8'))
    for name,digest in hashes.items():
        assert hashlib.sha256((path/name).read_bytes()).hexdigest()==digest,(path,name,'digest')
    snapshot=json.loads((path/'search_snapshot.json').read_text('utf-8'))
    rows=[json.loads(line) for line in (path/'search_evaluated.jsonl').read_text('utf-8').splitlines()]
    cached=[json.loads(line) for line in (path/'predictions.jsonl').read_text('utf-8').splitlines()]
    assert len(rows)==len(cached)==summary['n_unique_predicted']<=summary['max_unique_predictions']
    assert len({r['sequence'] for r in rows})==len(rows)
    validate_sequences([r['sequence'] for r in rows])
    cache={r['sequence']:r for r in cached}
    history={};export={}
    for r in rows:
        assert r['candidate_id']==candidate_id(r['sequence'])
        assert np.isfinite(r['pred_gbsa'])
        assert r['pred_gbsa']==cache[r['sequence']]['mu']
        assert r['model_id']==cache[r['sequence']]['model_id']==summary['model_id']
        n=sum(r['sequence'][p-1]!=WT_141[p-1] for p in MUT_POS)
        assert n==r['n_from_wt'] and 4<=n<=13
        parent_id=r['parent_candidate_id']
        if parent_id:
            assert parent_id in history,(path,'parent missing')
            parent=history[parent_id]
            changed=[p for p in MUT_POS if r['sequence'][p-1]!=parent['sequence'][p-1]]
            assert changed==r['changed_positions_from_parent']
            assert len(changed)==r['n_from_parent']
            assert r['operator'] in ('one_hop','two_hop')
            assert len(changed)==(1 if r['operator']=='one_hop' else 2)
            expected='measured' if parent['operator']=='seed' and parent['parent_evidence']=='measured' else 'predicted_only'
            assert r['parent_evidence']==expected
            assert r['generation']>parent['generation']
        else:
            assert r['operator'] in ('seed','global_restart')
        measured_seed=r['operator']=='seed' and r['parent_evidence']=='measured'
        if measured_seed:
            assert r['sequence'] in known
        else:
            assert r['sequence'] not in known
            export[r['candidate_id']]=r
        history[r['candidate_id']]=r
    cands=pd.read_csv(path/'search_candidates.csv',float_precision='round_trip')
    assert set(cands.candidate_id)==set(export)
    assert list(cands.candidate_id)==sorted(export,key=lambda c:(export[c]['pred_gbsa'],c))
    groups=json.loads((path/'search_groups.json').read_text('utf-8'))
    members=[cid for g in groups for cid in g['member_ids']]
    assert len(members)==len(set(members))==len(export) and set(members)==set(export)
    assert len(groups) in (1,2)
    for g in groups:
        assert g['prototype_id'] in g['member_ids']
    assert float(cands.pred_gbsa.iloc[0])==summary['best_pred_gbsa']
    assert snapshot['model_snapshot']['model_id']==summary['model_id']
    scores=[float('inf') if r['operator']=='seed' and r['parent_evidence']=='measured'
            else r['pred_gbsa'] for r in rows]
    assert max(r['generation'] for r in rows)<=snapshot['max_generations']
    return len(rows),[(r['candidate_id'],r['sequence'],r['parent_evidence']) for r in rows if r['operator']=='seed'],scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('suite');a=ap.parse_args()
    root=Path(a.suite).resolve();config=json.loads((root/'suite_config.json').read_text('utf-8'))
    complete=json.loads((root/'completeness.json').read_text('utf-8'));assert complete['ok']
    canonical=ROOT/'data/proxy2000_v2/fad_proxy2000_v2_full.csv'
    assert hashlib.sha256(canonical.read_bytes()).hexdigest().upper()==config['data_sha256']
    known=set(pd.read_csv(canonical).Sequence.map(normalize))
    runs=pd.read_csv(root/'runs.csv');assert len(runs)==complete['observed_subruns']
    starts={};total=0;main_scores={}
    for i,row in enumerate(runs.to_dict('records')):
        count,signature,scores=audit_subrun(root/row['archive'],known);total+=count
        assert count==row['actual_predictions']
        if row['arm']=='main':
            key=(row['seed'],row['proxy'],row['budget'])
            if key in starts: assert starts[key]==(signature,row['model_id'])
            else: starts[key]=(signature,row['model_id'])
            main_scores[key+(row['policy'],)]=scores
        if (i+1)%100==0:print(f'[audit] {i+1}/{len(runs)} subruns',flush=True)
    aligned=pd.read_csv(root/'aligned.csv',float_precision='round_trip')
    for row in aligned.to_dict('records'):
        key=(row['seed'],row['proxy'],row['budget'])
        common=min(len(main_scores[key+(p,)]) for p in config['policies'])
        assert row['common_n']==common
        assert row['best_pred_gbsa']==min(main_scores[key+(row['policy'],)][:common])
    if (root/'pairwise.csv').exists() and (root/'pairwise.csv').stat().st_size>1:
        pairs=pd.read_csv(root/'pairwise.csv',float_precision='round_trip')
        for row in pairs.to_dict('records'):
            key=(row['seed'],row['proxy'],row['budget'])
            left=main_scores[key+(row['policy'],)];right=main_scores[key+(row['baseline'],)]
            common=min(len(left),len(right))
            assert row['common_n']==common
            assert row['policy_best']==min(left[:common])
            assert row['baseline_best']==min(right[:common])
            assert abs(row['delta']-(min(left[:common])-min(right[:common])))<1e-12
    result=dict(ok=True,subruns=len(runs),evaluated_rows=total,
        checks=['file digests','budget/cache/count identity','finite aligned scores',
                'RNA scaffold and mutation shell','single-parent genealogy/evidence',
                'canonical-table exclusion','sorted export','complete unique groups',
                'same starts and fitted model per comparison cell',
                'generation cap','seven-way and pairwise actual-prefix scores'])
    (root/'artifact_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
