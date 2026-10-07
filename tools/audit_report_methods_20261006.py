"""Recompute saved experimental statistics; never fit a surrogate or run MD."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'evidence/report_review_20261006'
CHECKS = []
SOURCES = {}


def read(path, csv=False):
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p) if csv else json.loads(p.read_text('utf-8-sig'))


def check(name, actual, expected, tol=1e-10):
    assert np.allclose(actual, expected, atol=tol, rtol=0), (name, actual, expected)
    CHECKS.append(name)


def boot(delta, seed=20260924):
    rng = np.random.default_rng(seed)
    return np.quantile(rng.choice(delta, (20000, len(delta)), replace=True).mean(axis=1), [.025, .975])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Read-only migration evidence: copy documentation, not code/data for runtime.
    legacy = Path('D:/PyCharm_/FAD/archived/server_sync_20260902')
    migration = OUT / 'seed42_migration_evidence'
    migration.mkdir(exist_ok=True)
    provenance = []
    for source, name in [
        ('legacy_source/scripts/README.md', '历史优化路径_README.md'),
        ('supplementary_artifacts/phase0/audit_report.md', '历史基线混用审计.md'),
        ('legacy_docs/v1/97_BestModel/0.5060/README.md', '历史0506交付说明.md'),
    ]:
        p = legacy / source
        dest = migration / name
        if p.exists():
            shutil.copyfile(p, dest)
            provenance.append({'original':str(p), 'copy':str(dest.relative_to(ROOT)),
                               'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (migration / 'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')

    v8 = read('evidence/server_sync_20260902/v8_fusion_100seed/all_results.csv', True)
    assert len(v8) == 700 and set(v8.seed) == set(range(1, 101))
    f = v8[v8.method.eq('equal_weight_rank')].set_index('seed').gbsa_Sp.sort_index()
    singles = v8[v8.method.eq('single')].pivot(index='seed', columns='model',values='gbsa_Sp').sort_index()
    baseline = read('evidence/audit_20260924/audit_baseline_rf_100seed/metrics.csv', True)
    basecol = next(c for c in ['gbsa_spearman','gbsa_Sp','spearman'] if c in baseline)
    b = baseline.set_index('seed')[basecol].sort_index()
    summary = read('evidence/audit_20260924/paired_fusion_vs_baseline.json')
    check('v8 vs RF paired mean', (f-b).mean(), summary['paired_delta_mean'])
    check('v8 vs RF wins', int((f>b).sum()), summary['fusion_wins'])
    v8refs = {}
    for name, ref in [('mutation_only_RF',b), ('fixed_XGB',singles.XGB), ('posthoc_best_single',singles.max(axis=1))]:
        delta = (f-ref).to_numpy()
        v8refs[name] = {'v8_mean':float(f.mean()),'reference_mean':float(ref.mean()),
                       'paired_delta':float(delta.mean()),'ci95_recomputed':boot(delta).tolist(),
                       'wins':int((delta>0).sum()),'n':len(delta)}
    fresh = read('evidence/audit_20260924/audit_v8_10seed/metrics.csv', True)
    # Fresh audit is a full 7-method table; compare only the matching frozen arm.
    if 'method' in fresh:
        fresh = fresh[fresh.method.eq('equal_weight_rank')]
    col = next(c for c in ['gbsa_Sp','gbsa_spearman','spearman'] if c in fresh.columns)
    fs = fresh.set_index('seed')[col]
    check('fresh v8 10-seed matches historical same seeds',fs.to_numpy(),f.loc[fs.index].to_numpy())

    path = 'evidence/server_sync_20260903/feature_selection_all_safe_30seed/'
    metrics = read(path+'metrics.csv',True)
    paired = read(path+'metrics_paired.csv',True)
    assert len(metrics)==360 and not metrics.duplicated(['seed','method','model']).any()
    control = metrics[metrics.method.eq('all')].set_index(['seed','model']).Spearman
    rng = np.random.default_rng(20260902)
    for (method, model), sub in metrics[~metrics.method.eq('all')].groupby(['method','model']):
        delta = sub.Spearman.to_numpy()-np.array([control.loc[(s,model)] for s in sub.seed])
        row = paired[paired.method.eq(method)&paired.model.eq(model)].iloc[0]
        check(f'FS {method}/{model} mean',sub.Spearman.mean(),row.mean_spearman)
        check(f'FS {method}/{model} delta',delta.mean(),row.mean_delta_vs_all)
        check(f'FS {method}/{model} wins',int((delta>0).sum()),row.wins_vs_all)
        ci=np.quantile(rng.choice(delta,(20000,len(delta)),replace=True).mean(axis=1),[.025,.975])
        check(f'FS {method}/{model} CI',ci,[row.bootstrap_ci95_low,row.bootstrap_ci95_high])
    selected = read(path+'metrics_selected_features.csv',True)
    forced={f'mut_p{p}_{b}' for p in [30,31,32,33,89,90,91,92,127,128,129,130,131] for b in 'ACGU'}
    for r in selected.itertuples():
        names=r.selected_features.split('|')
        assert forced <= set(names)
        assert len(names)==r.n_selected
        assert hashlib.sha256('\n'.join(names).encode()).hexdigest()==r.selected_sha256
    CHECKS.append('120 selected masks: forced mutation features, counts and hashes')
    # Features remain forced; actual post-stability counts are not exactly 100.
    counts = metrics[~metrics.method.eq('all')].groupby('method').n_selected.agg(['min','median','max']).to_dict('index')

    for file, base_key, delta_key in [
        ('evidence/audit_20260924/ml_squeeze/smoothing_variants.json','raw','paired_delta_vs_raw'),
        ('evidence/audit_20260924/ml_squeeze/rank_objectives2.json','xgb_mse','paired_delta_vs_mse'),
        ('evidence/audit_20260924/eval_defect_features_results.json','base','paired_delta_vs_base'),
    ]:
        j=read(file)
        baseline=np.asarray(j['per_seed'][base_key])
        for name, arr in j['per_seed'].items():
            arr=np.asarray(arr)
            s=j['summary'][name]
            check(f'{Path(file).stem}/{name} mean',arr.mean(),s['mean'])
            check(f'{Path(file).stem}/{name} std',arr.std(),s['std'])
            d=arr-baseline
            check(f'{Path(file).stem}/{name} delta',d.mean(),s[delta_key])
            check(f'{Path(file).stem}/{name} wins',int((d>0).sum()),s['wins'])
            check(f'{Path(file).stem}/{name} CI',boot(d),s['paired_ci95'])
    for file in ['negative_controls','sparse_epistasis']:
        j=read(f'evidence/audit_20260924/ml_squeeze/{file}.json')
        for name,s in j.items():
            if isinstance(s,dict) and 'per_seed' in s:
                a=np.array(s['per_seed'])
                check(f'{file}/{name} mean',a.mean(),s['mean'])
                check(f'{file}/{name} std',a.std(),s['std'])
    sm=read('evidence/audit_20260924/audit_label_smoothing_100seed_results.json')
    smcsv=read('evidence/audit_20260924/formal_label_smoothing_100seed/metrics.csv',True)
    for k in ['original','smoothed']:
        check(f'100-seed smoothing {k} CSV vs JSON',smcsv[k].to_numpy(),sm['per_seed'][k])
        check(f'100-seed smoothing {k} mean',smcsv[k].mean(),sm[k]['mean'])
    ds=(smcsv.smoothed-smcsv.original).to_numpy()
    check('100-seed smoothing delta',ds.mean(),sm['paired_delta_mean'])
    check('100-seed smoothing CI',boot(ds),sm['paired_bootstrap_ci95'])
    check('100-seed smoothing wins',int((ds>0).sum()),sm['wins'])
    fusion=read('evidence/audit_20260924/ml_squeeze/fusion_smoothing.json')
    for k,a in fusion['per_seed'].items():
        if k in fusion['summary'] and 'mean' in fusion['summary'][k]:
            check('fusion smoothing mean '+k,np.mean(a),fusion['summary'][k]['mean'])
    d=np.array(fusion['per_seed']['smooth_fusion'])-np.array(fusion['per_seed']['raw_fusion'])
    check('fusion smoothing delta',d.mean(),fusion['summary']['paired_delta']['mean'])
    check('fusion smoothing CI',boot(d),fusion['summary']['paired_delta']['ci95'])

    features=read('evidence/audit_20260924/ensemble_defect_features.csv',True)
    defect_data={'rows':len(features),'unique':int(features.Sequence.nunique()),
                 'missing_values':int(features.isna().sum().sum()),
                 'wt_struct_prob_zero_count':int(features.wt_struct_prob.eq(0).sum()),
                 'old_implementation_invalid_unpaired_probability':True,
                 'old_results_not_valid_test_of_correct_ensemble_defect':True,
                 'full_nupack_feature_recomputation_pending':not (ROOT/'evidence/supplement_20261006/srv_supplement_defect_compute_v2_20261006/DONE').exists()}
    ext=read('evidence/audit_20260924/ml_squeeze/extrapolation.json')
    check('extrapolation random mean',np.mean(ext['random_control']['per_seed']),ext['random_control']['spearman_mean'])
    data=read('data/proxy2000_v2/fad_proxy2000_v2_full.csv',True)
    from scripts.loop.interface import MUT_POS, WT_141
    nc=data.Sequence.map(lambda s:sum(s[p-1]!=WT_141[p-1] for p in MUT_POS))
    for k,mask in [('low_out',nc.le(9)),('mid_out',nc.between(10,11)),('high_out',nc.ge(12))]:
        check('extrapolation test size '+k,int(mask.sum()),ext[k]['n_test'])
        check('extrapolation train size '+k,int((~mask).sum()),ext[k]['n_train'])
    fw=read('evidence/audit_20260924/server_results/feature_weighting.json')
    for k in ['weighted','baseline','hard_useful7']:
        col='weighted_fusion' if k=='weighted' else k
        check('feature weighting '+k,np.mean([r[col] for r in fw['results']]),fw[k+'_mean'])
    paper=read('evidence/audit_20260924/server_results/paper_fusion.json')
    check('paper fusion saved 5-seed mean',np.mean([r['spearman'] for r in paper['results']]),paper['spearman_mean'])
    check('paper fusion saved 5-seed std',np.std([r['spearman'] for r in paper['results']]),paper['spearman_std'])
    gate=read('evidence/audit_20260924/server_results/feature_gating.json')
    gate_values=[r['spearman'] for r in gate['results']]
    check('feature gating saved 3-seed mean',np.mean(gate_values),gate['spearman_mean'])
    check('feature gating saved 3-seed std',np.std(gate_values),gate['spearman_std'])
    for part in ['val','test']:
        p=read(f'evidence/audit_20260924/ml_squeeze/release_package/locked_{part}_predictions.csv',True)
        score=float(pearsonr(rankdata(p.gbsa_true),rankdata(p.fused_rank)).statistic)
        check('locked '+part+' reported Spearman to 3 decimals',round(score,3),.316 if part=='val' else .451)

    # Use the same frozen static-pool summary function, but do not launch any runs.
    from scripts.loop.strategy_compare import bootstrap_ci
    static=read('evidence/loop_20260930/strategy_compare_100seed/runs.csv',True)
    assert len(static)==2000 and not static.duplicated(['seed','strategy','round']).any()
    assert set(static.seed)==set(range(1001,1101))
    last=static[static['round'].eq(5)]
    p=last.pivot(index='seed',columns='strategy',values='regret')
    d=(p.greedy-p.random).to_numpy()
    staticstats={'mean_delta_regret':float(d.mean()),'ci95':list(bootstrap_ci(d)),
        'median_delta':float(np.median(d)),'wins':int((d< -1e-10).sum()),
        'ties':int((abs(d)<=1e-10).sum()),'losses':int((d>1e-10).sum()),
        'worst10_mean':float(np.sort(d)[-10:].mean()),'best10_mean':float(np.sort(d)[:10].mean())}
    check('static regret mean',staticstats['mean_delta_regret'],-1.878069633233386,1e-8)
    from scripts.loop.strategy_compare import paired_summary
    staticsummary=read('evidence/loop_20260930/strategy_compare_100seed/summary.json')
    recalculated=paired_summary(static,staticsummary['strategies'],None)
    for strategy,row in recalculated.items():
        for k,v in row.items():
            if isinstance(v,(float,int,list)):
                check(f'static {strategy}/{k}',v,staticsummary['paired_vs_random_final_round'][strategy][k])
    for row in staticsummary['per_round']:
        block=static[static.strategy.eq(row['strategy'])&static['round'].eq(row['round'])]
        for key,col in [('mean_regret','regret'),('mean_best_selected','best_selected'),('mean_holdout_post','holdout_spearman_post')]:
            check(f"static round {row['round']} {row['strategy']}/{key}",block[col].mean(),row[key])
    # Summary-only files cannot support independent per-seed recomputation.
    summary_only=['evidence/audit_20260924/server_results/reproduce.json',
        'evidence/audit_20260924/server_results/new_features.json',
        'evidence/audit_20260924/audit_dock_denoise_fixed_results.json',
        'evidence/audit_20260924/audit_dock_rank_corr_fixed_results.json']
    for p in summary_only:read(p)
    result={'ok':True,'statistical_assertions':len(CHECKS),'checks':CHECKS,
        'v8_comparators':v8refs,'feature_selection_actual_counts':counts,
        'defect_feature_audit':defect_data,'static_pool_recomputed':staticstats,
        'summary_only_not_independently_recomputed':summary_only,
        'legacy_seed42_documentation_recovered':provenance,
        'source_sha256':SOURCES,'new_model_training_runs':0,'new_GBSA':0,
        'verification_boundary':'Saved-row statistics and source review; does not reproduce all fitting or validate physical labels.'}
    (OUT/'method_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','source_sha256','legacy_seed42_documentation_recovered']},ensure_ascii=False))


if __name__=='__main__':main()
