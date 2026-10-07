"""Verify four DONE execution cohorts of the frozen experiment, then combine."""
import argparse,json,hashlib,shutil
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--output',required=True)
for key in ['seeds','seed-start','pretrain-epochs','head-epochs','members','corpus-size']:p.add_argument('--'+key,type=int,required=True)
a=p.parse_args();out=root/a.output;pred=out.parent/'predictions';pred.mkdir(exist_ok=True)
frames=[];manifests=[];children=[]
canonical_hash=hashlib.sha256((root/'data/proxy2000_v2/fad_proxy2000_v2_full.csv').read_bytes()).hexdigest()
for i in range(1,5):
    child=root/f'runs/srv_supplement_paper_cohort{i}_20261006'
    assert (child/'DONE').exists(),f'incomplete child {child}'
    m=json.loads((child/'run_manifest.json').read_text());assert m['canonical_data']['sha256'].lower()==canonical_hash and m['canonical_data']['registry_match']
    c=json.loads((child/'config.json').read_text());assert c['entrypoint']=='scripts/reproduction/supplement_paper_20261006.py'
    for key in ['pretrain-epochs','head-epochs','members','corpus-size']:assert c['arguments'][key]==getattr(a,key.replace('-','_'))
    for rel in ['scripts/reproduction/'+n+'.py' for n in ['supplement_paper_20261006','new_features','encoder','surrogate','pretrain','data','tokenizer']]:
        assert m['source_manifest']['files'][rel]==hashlib.sha256((root/rel).read_bytes()).hexdigest(),rel
    f=pd.read_csv(child/'metrics.csv');assert len(f)==c['arguments']['seeds']*4
    assert set(f.seed)==set(range(c['arguments']['seed-start'],c['arguments']['seed-start']+c['arguments']['seeds']))
    for file in (child/'predictions').iterdir():
        assert not (pred/file.name).exists(),f'duplicate artifact {file.name}'
        shutil.copy2(file,pred/file.name)
    frames.append(f);manifests.append(json.loads((child/'paper_manifest.json').read_text()))
    children.append({'run_id':child.name,'config':c,'run_manifest':m,'done':json.loads((child/'DONE').read_text())})
f=pd.concat(frames,ignore_index=True);assert len(f)==120 and not f.duplicated(['seed','variant']).any()
assert set(f.seed)==set(range(a.seed_start,a.seed_start+a.seeds))
assert set(f.variant)=={'no_pretrain','foreign_pretrain','in_domain_pretrain','sequence303_GBR_control'}
assert f.groupby('seed').size().eq(4).all()
for key in ['foreign_used_sha256','foreign_corpus_sha256','initial_encoder_seed','n_members','head_epochs','pretrain_epochs']:
    assert len({m[key] for m in manifests})==1,key
metadata=manifests[0];metadata['execution_cohorts']=4;metadata['foreign_reused']='within each cohort; same initial weights, corpus, epochs and seed across cohorts'
metadata['shared_foreign_pretrain_and_embedding_seconds']=sum(m['shared_foreign_pretrain_and_embedding_seconds'] for m in manifests)
metadata['random_encoder_embedding_seconds']=sum(m['random_encoder_embedding_seconds'] for m in manifests)
(out.parent/'paper_manifest.json').write_text(json.dumps(metadata,indent=2))
(out.parent/'cohort_manifest.json').write_text(json.dumps(children,indent=2))
f.to_csv(out,index=False);print('verified 4 DONE cohorts, 30 unique seeds, 120 metrics rows',flush=True)
