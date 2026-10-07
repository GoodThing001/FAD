"""Independently verify predictions and summarize completed supplementary runs."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,rankdata
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'evidence/supplement_20261006';DEST.mkdir(exist_ok=True)

def boot(a):
    a=np.asarray(a);rng=np.random.default_rng(20261006)
    b=rng.choice(a,(20000,len(a)),replace=True).mean(1)
    return [float(x) for x in np.quantile(b,[.025,.975])]

def collect_local():
    for p in (ROOT/'runs').glob('local_supplement_*_20261006'):
        if (p/'DONE').exists():
            target=DEST/p.name
            shutil.copytree(p,target,dirs_exist_ok=True)
            manifest=json.loads((target/'run_manifest.json').read_text())
            config=json.loads((target/'config.json').read_text())
            snapshot=target/'source_snapshots';snapshot.mkdir(exist_ok=True)
            for rel in [config['entrypoint'],'scripts/reproduction/new_features.py']:
                source=ROOT/rel;digest=manifest['source_manifest']['files'][rel]
                assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,f'changed actual model source: {rel}'
                shutil.copy2(source,snapshot/source.name)

def snapshot_remote(p):
    manifest=json.loads((p/'run_manifest.json').read_text())
    config=json.loads((p/'config.json').read_text())
    rels=[config['entrypoint']]
    if 'paper' in p.name:
        rels += ['scripts/reproduction/'+n+'.py' for n in ['supplement_paper_20261006','new_features','encoder','surrogate','pretrain','data','tokenizer']]
    elif 'compute' in p.name:rels+=['scripts/reproduction/structure_metrics.py']
    else:rels+=['scripts/reproduction/new_features.py']
    snapshot=p/'source_snapshots';snapshot.mkdir(exist_ok=True)
    for rel in rels:
        source=ROOT/rel
        assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_manifest']['files'][rel],f'changed remote model source: {rel}'
        shutil.copy2(source,snapshot/source.name)

def verify_computation(p):
    manifest=json.loads((p/'run_manifest.json').read_text());assert manifest['canonical_data']['registry_match']
    f=pd.read_csv(p/'metrics.csv');canonical=pd.read_csv(ROOT/'data/proxy2000_v2/fad_proxy2000_v2_full.csv')
    assert len(f)==len(canonical)==2000 and f.Sequence.is_unique and f.Sequence.tolist()==canonical.Sequence.tolist()
    assert np.isfinite(f.select_dtypes('number')).all().all()
    assert f.ensemble_defect.between(0,141).all() and f.wt_struct_prob.between(0,1).all()
    v=json.loads((p/'nupack_validation.json').read_text())
    assert v['n_errors']==0 and len(v['checks'])==44
    for c in v['checks']:
        assert np.isclose(abs(c['official_count']-c['ours_count']),c['absolute_error'],atol=1e-14)
        assert c['absolute_error']<=1e-5
    assert f.probability_status.value_counts().to_dict()==v['status_counts']
    assert (f.loc[f.probability_status.eq('invalid_target_pair'),'wt_struct_prob']==0).all()
    old=pd.read_csv(ROOT/'evidence/audit_20260924/ensemble_defect_features.csv').set_index('Sequence').loc[f.Sequence]
    return {'run_id':p.name,'rows':len(f),'official_checks':len(v['checks']),'official_max_error':max(c['absolute_error'] for c in v['checks']),'status_counts':v['status_counts'],'correct_defect_range':[float(f.ensemble_defect.min()),float(f.ensemble_defect.max())],'old_defect_range':[float(old.ensemble_defect.min()),float(old.ensemble_defect.max())],'mean_absolute_change_from_old':float(np.abs(f.ensemble_defect.to_numpy()-old.ensemble_defect.to_numpy()).mean()),'elapsed_seconds':v['elapsed_seconds']}

def summarize(p):
    assert (p/'DONE').exists(),f'not complete {p}'
    manifest=json.loads((p/'run_manifest.json').read_text());assert manifest['canonical_data']['registry_match']
    canonical=pd.read_csv(ROOT/'data/proxy2000_v2/fad_proxy2000_v2_full.csv').gbsa.to_numpy()
    f=pd.read_csv(p/'metrics.csv');s={'run_id':p.name,'n_rows':len(f),'predictions_verified':0,'summary':{},'paired':[],'discovery_rounds_verified':0,'sigma_calibration':{}}
    keys=['seed','variant'] if 'variant' in f else ['seed','policy','round']
    assert not f.duplicated(keys).any()
    config=json.loads((p/'config.json').read_text());a=config['arguments']
    observed=set(f.seed.astype(str))-{'fixed42'}
    expected={str(i) for i in range(a['seed-start'],a['seed-start']+a['seeds'])}
    assert observed==expected
    if 'variant' in f:
        if 'fixed42' in f.seed.astype(str).tolist():
            s['fixed_split']=f[f.seed.astype(str)=='fixed42'].to_dict('records');f=f[f.seed.astype(str)!='fixed42'].copy()
        for row in pd.read_csv(p/'metrics.csv').itertuples():
            pred=p/'predictions'/f'prediction_{row.seed}_{row.variant}.csv';q=pd.read_csv(pred)
            assert len(q)==row.n_test and q.canonical_index.is_unique and np.isfinite(q.select_dtypes('number')).all().all()
            assert np.allclose(q.truth,canonical[q.canonical_index.to_numpy()],rtol=0,atol=1e-10)
            rho=pearsonr(rankdata(q.truth),rankdata(q.prediction))[0]
            assert np.isclose(rho,row.spearman,atol=1e-12)
            assert np.isclose(np.mean(np.abs(q.truth-q.prediction)),row.mae,atol=1e-10)
            assert np.isclose(np.sqrt(np.mean((q.truth-q.prediction)**2)),row.rmse,atol=1e-10)
            s['predictions_verified']+=1
        if 'paper' in p.name:
            from sklearn.model_selection import train_test_split
            known=pd.read_csv(ROOT/'data/proxy2000_v2/fad_proxy2000_v2_full.csv').Sequence.tolist();known_set=set(known)
            mut=[30,31,32,33,89,90,91,92,127,128,129,130,131];corpora_verified=0
            for seed in sorted(f.seed.unique()):
                split=json.loads((p/'predictions'/f'split_{seed}.json').read_text());tr,te=train_test_split(np.arange(2000),test_size=.2,random_state=int(seed))
                assert split['train']==tr.tolist() and split['test']==te.tolist()
                corpus=pd.read_csv(p/'predictions'/f'in_domain_corpus_{seed}.csv').Sequence.tolist()
                rng=np.random.default_rng(int(seed));expected_corpus=[];seen=set(known)
                while len(expected_corpus)<a['corpus-size']:
                    seq=list(known[int(tr[int(rng.integers(len(tr)))])])
                    for pos in mut:
                        if rng.random()<.5:seq[pos-1]='ACGU'[int(rng.integers(4))]
                    seq=''.join(seq)
                    if seq not in seen:expected_corpus.append(seq);seen.add(seq)
                assert corpus==expected_corpus and set(corpus).isdisjoint(known_set)
                assert hashlib.sha256('\n'.join(corpus).encode()).hexdigest()==split['corpus_sha256']
                for variant in f.variant.unique():
                    q=pd.read_csv(p/'predictions'/f'prediction_{seed}_{variant}.csv')
                    assert q.canonical_index.tolist()==te.tolist()
                corpora_verified+=1
            s['train_parent_corpora_verified']=corpora_verified
            manifest=json.loads((p/'paper_manifest.json').read_text())
            s['paper_budget']={k:manifest[k] for k in ['device','torch','foreign_n','foreign_length_stats','n_members','head_epochs','pretrain_epochs','budget_matching','shared_foreign_pretrain_and_embedding_seconds','random_encoder_embedding_seconds']}
            domain_times=[json.loads(q.read_text())['pretrain_and_embedding_seconds'] for q in (p/'predictions').glob('domain_timing_*.json')]
            assert len(domain_times)==a['seeds']
            s['paper_budget']['mean_domain_pretrain_and_embedding_seconds']=float(np.mean(domain_times))
            s['paper_budget']['mean_arm_training_prediction_seconds']={str(n):float(g.seconds.mean()) for n,g in f.groupby('variant')}
            if (p/'cohort_manifest.json').exists():
                cohorts=json.loads((p/'cohort_manifest.json').read_text());assert len(cohorts)==4
                all_seeds=[]
                for c in cohorts:
                    cfg=c['config']['arguments'];all_seeds+=list(range(cfg['seed-start'],cfg['seed-start']+cfg['seeds']))
                    assert c['done']['acceptance_ok'] and c['done']['return_code']==0
                    assert c['run_manifest']['canonical_data']['registry_match']
                assert len(all_seeds)==len(set(all_seeds))==a['seeds'] and set(all_seeds)==set(map(int,expected))
                s['execution_cohorts_verified']=4
        if 'defect_dev' in p.name or 'defect_force' in p.name:
            inp=json.loads((p/'defect_input.json').read_text());file=DEST/Path(inp['path']).parts[1]/'metrics.csv'
            assert hashlib.sha256(file.read_bytes()).hexdigest()==inp['sha256']
            presence={}
            feature_map=[('plus_defect',{'ensemble_defect':303}),('plus_probability',{'wt_struct_prob':303}),('plus_both',{'ensemble_defect':303,'wt_struct_prob':304})] if 'defect_dev' in p.name else [('forced_defect',{'ensemble_defect':303}),('forced_probability',{'wt_struct_prob':303}),('forced_both',{'ensemble_defect':303,'wt_struct_prob':304})]
            for variant,features in feature_map:
                selected=[json.loads((p/'predictions'/f'selection_{seed}_{variant}.json').read_text()) for seed in f.seed.unique()]
                assert all(len(ix)==len(set(ix))==100 for ix in selected)
                presence[variant]={name:int(sum(index in ix for ix in selected)) for name,index in features.items()}
            s['added_feature_selected_counts']=presence
            if 'defect_force' in p.name:
                assert all(n==a['seeds'] for features in presence.values() for n in features.values())
                first=pd.read_csv(DEST/'srv_supplement_defect_dev_20261006/metrics.csv')
                for seed in f.seed.unique():
                    old=pd.read_csv(DEST/'srv_supplement_defect_dev_20261006/predictions'/f'prediction_{seed}_base303.csv')
                    new=pd.read_csv(p/'predictions'/f'prediction_{seed}_base303.csv')
                    assert np.array_equal(old.canonical_index,new.canonical_index) and np.allclose(old.prediction,new.prediction,atol=1e-10,rtol=0)
                s['baseline_prediction_equality_verified']=int(f.seed.nunique())
        key='variant';ref='seq303' if 'seq303' in f.variant.tolist() else 'base303' if 'base303' in f.variant.tolist() else 'no_pretrain'
    else:
        for seed in f.seed.unique():
            first=pd.read_csv(p/'predictions'/f'holdout_{seed}_random_0.csv')
            for policy in ['greedy','uncertainty','mixed']:
                same=pd.read_csv(p/'predictions'/f'holdout_{seed}_{policy}_0.csv')
                assert np.array_equal(first.canonical_index,same.canonical_index)
                assert np.array_equal(first.prediction,same.prediction) and np.array_equal(first.sigma,same.sigma)
        s['initial_prediction_equality_verified']=int(f.seed.nunique())
        calibration=[]
        for row in f.itertuples():
            q=pd.read_csv(p/'predictions'/f'holdout_{row.seed}_{row.policy}_{row.round}.csv')
            assert np.allclose(q.truth,canonical[q.canonical_index.to_numpy()],rtol=0,atol=1e-10)
            assert np.isclose(pearsonr(rankdata(q.truth),rankdata(q.prediction))[0],row.spearman,atol=1e-12)
            sp=json.loads((p/'predictions'/f'split_{row.seed}.json').read_text())
            hold=set(sp['holdout']);initial=set(sp['initial']);pool=set(sp['pool'])
            assert len(hold)==len(initial)==400 and len(pool)==1200 and not hold&initial and not hold&pool and not initial&pool
            assert set(q.canonical_index)==hold
            revealed=[]
            for rnd in range(row.round):
                queries=pd.read_csv(p/'predictions'/f'queries_{row.seed}_{row.policy}_{rnd}.csv');ids=queries.canonical_index.to_list()
                assert len(ids)==len(set(ids))==100 and set(ids)<=pool and not set(ids)&set(revealed)
                assert np.allclose(queries.truth_revealed,canonical[ids],rtol=0,atol=1e-10)
                revealed+=ids
            assert row.n_train==400+row.round*100
            if revealed:
                best=float(np.min(canonical[revealed]));top=set(sorted(pool,key=lambda i:canonical[i])[:120])
                assert np.isclose(best,row.query_best,atol=1e-10)
                assert np.isclose(best-np.min(canonical[list(pool)]),row.regret,atol=1e-10)
                assert np.isclose(len(set(revealed)&top)/120,row.top10_recall,atol=1e-12)
                s['discovery_rounds_verified']+=1
            if row.round==5:
                cal=float(pearsonr(rankdata(q.sigma),rankdata(np.abs(q.truth-q.prediction)))[0])
                calibration.append({'seed':row.seed,'policy':row.policy,'spearman_sigma_absolute_error':cal})
            s['predictions_verified']+=1
        for policy,g in pd.DataFrame(calibration).groupby('policy'):
            a=g.spearman_sigma_absolute_error.to_numpy();s['sigma_calibration'][policy]={'mean':float(a.mean()),'ci95':boot(a),'n':len(a)}
        pd.DataFrame(calibration).to_csv(p/'sigma_calibration.csv',index=False)
        s['random_sampling_sanity']={'pool_size':1200,'queries':500,'top10_count':120,'expected_uniform_top10_recall':500/1200,'observed_mean_top10_recall':float(f[f.policy.eq('random') & f['round'].eq(5)].top10_recall.mean())}
        f=f[f['round']==5].copy();key='policy';ref='random'
    for name,a in f.groupby(key):
        s['summary'][name]={metric:{'mean':float(a[metric].mean()),'sd':float(a[metric].std(ddof=1)),'ci95':boot(a[metric])} for metric in ['spearman','mae','rmse']+(['regret','top10_recall'] if key=='policy' else [])}
    metrics=['spearman','mae','rmse']+(['regret','top10_recall'] if key=='policy' else [])
    for metric in metrics:
        q=f.pivot(index='seed',columns=key,values=metric)
        for name in q:
            if name==ref:continue
            d=(q[name]-q[ref]).to_numpy();better=d>1e-10 if metric in ['spearman','top10_recall'] else d< -1e-10
            s['paired'].append({'metric':metric,'variant':name,'reference':ref,'delta':float(d.mean()),'median_delta':float(np.median(d)),'ci95':boot(d),'wins':int(better.sum()),'ties':int((np.abs(d)<=1e-10).sum()),'losses':int(len(d)-better.sum()-(np.abs(d)<=1e-10).sum()),'n':len(d)})
        if ref=='seq303':
            for name,reference in [('historical1087','seq_nup904'),('seq303','mut52')]:
                d=(q[name]-q[reference]).to_numpy();better=d>1e-10 if metric=='spearman' else d< -1e-10
                s['paired'].append({'metric':metric,'variant':name,'reference':reference,'delta':float(d.mean()),'median_delta':float(np.median(d)),'ci95':boot(d),'wins':int(better.sum()),'ties':int((abs(d)<=1e-10).sum()),'losses':int(len(d)-better.sum()-(abs(d)<=1e-10).sum()),'n':len(d)})
    (p/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    if 'paper' in p.name:
        q=f.pivot(index='seed',columns='variant',values='spearman');checks={}
        for reference in ['no_pretrain','foreign_pretrain','sequence303_GBR_control']:
            delta=(q.in_domain_pretrain-q[reference]).to_numpy();ci=boot(delta)
            checks[reference]={'delta':float(delta.mean()),'ci95':ci,'criterion_met':ci[0]>0,'wins':int(sum(delta>1e-10)),'ties':int(sum(abs(delta)<=1e-10)),'losses':int(sum(delta< -1e-10))}
        s['confirmation_gate']={'comparisons':checks,'advance_to_100_seeds':all(d['criterion_met'] for d in checks.values())}
        (p/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    return s

def main():
    collect_local();summaries=[];computations=[]
    for p in sorted(DEST.iterdir()):
        if p.is_dir() and (p/'DONE').exists():
            if p.name.startswith('srv_'):snapshot_remote(p)
            if (p/'predictions').exists():summaries.append(summarize(p))
            elif (p/'nupack_validation.json').exists():computations.append(verify_computation(p))
    lines=['# 2026-10-06 补实验证据','','本轮重新训练和逐预测核对；不将旧结果复算记为新训练。真实GBSA、docking、MD新增均为0。','','|运行|指标行|预测文件核验|发现轮次核验|状态|','|---|---:|---:|---:|---|']
    lines += [f"|{s['run_id']}|{s['n_rows']}|{s['predictions_verified']}|{s['discovery_rounds_verified']}|DONE|" for s in summaries]
    lines += ['','完整逐种子、配对CI、胜平负、分歧校准及文件SHA256见 [index.json](index.json)。','冻结配方与种子见 [协议](../../docs/项目记录/补实验冻结协议_20261006.md)。','用户随后明确授权服务器上传与运行，首个NUPACK官方核验失败包保留；修订原因及范围见冻结协议§2.1。','8项默认环境测试通过；另2项PyTorch CPU契约测试通过（人工张量仅用于程序检查，不是实验结果）。']
    lines += ['强制保留结构诊断为首轮结果后提出的消融，见协议§2.2；两份FAILED包不纳入正式统计。','预训练四组DONE聚合，域内对三对照的晋级条件未达，不追加这组配方的100种子确认。','服务器终态只读审计见 [记录](supplement_remote_validation_20261006.json)，本轮实验无遗留进程；原论文完整外域语料来源未重新认证。']
    for c in computations:lines.append(f"NUPACK正式重算：{c['run_id']}，{c['rows']}行，{c['official_checks']}个官方对照，最大绝对误差{c['official_max_error']:.3g}。")
    (DEST/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name!='index.json'}
    (DEST/'index.json').write_text(json.dumps({'new_real_GBSA':0,'runs':summaries,'feature_computations':computations,'retained_failed_runs':[p.name for p in DEST.iterdir() if p.is_dir() and (p/'FAILED').exists()],'file_hashes':hashes},ensure_ascii=False,indent=2),encoding='utf-8')
    print([(s['run_id'],s['n_rows'],s['predictions_verified']) for s in summaries])
if __name__=='__main__':main()
