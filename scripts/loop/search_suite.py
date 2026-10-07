"""Frozen-model, matched-start mutation-search suite; no Oracle or new labels.

Each split/model is fitted once, then independently cached policy runs share it.
All paired outcome and top-50 diagnostics use the same actual-call prefix.
Sub-search archives are checkpointed by the ordinary search layer. Completion
requires every planned arm, equal ordered starts, and equal fitted model IDs.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.loop.interface import TARGET_COLUMN, normalize
from scripts.loop.search import SearchContext, SurrogateScorer, SEARCH_INTERFACE_VERSION
from scripts.loop.search_archive import SearchArchive, sha256_file
from scripts.loop.search_policies import make_policy
from scripts.loop.surrogates import make_surrogate
from scripts.loop.search_compare import bootstrap_ci, spearman, top50_diversity, jaccard

POLICIES = ('matched_seed_random_walk', 'multi_start_hill_climb',
            'multi_start_beam', 'mutation_only_ga')
EXTENDED_POLICIES = POLICIES + ('iterated_local_search', 'simulated_annealing',
                               'mutation_only_adalead')
BASELINE = POLICIES[0]
FROZEN_CONFIG = dict(beam_width=32, offspring_per_parent=32,
    one_hop_fraction=0.8, restart_fraction=0.1, diversity_pool_factor=4,
    diversity_distance=2, n_measured_seeds=8, n_random_seeds=8,
    min_mutual_distance=2, population_size=32, offspring_size=128,
    tournament_size=2, elite_count=2, ils_perturb_sites=2,
    sa_temperature_scale=0.25, sa_final_temperature_ratio=0.02,
    adalead_tolerance_scale=0.05, adalead_parent_cap=32, adalead_rollouts=16)
ABLATIONS = {'ga_tournament1': {'tournament_size': 1},
             'ga_tournament4': {'tournament_size': 4},
             'ga_single_hop': {'one_hop_fraction': 1.0}}

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def exportable(rows):
    return [r for r in rows if not (r['operator'] == 'seed' and
                                  r['parent_evidence'] == 'measured')]

def diagnostics(rows):
    pool = exportable(rows)
    if not pool:
        raise ValueError('no exportable sequence in aligned prefix')
    top = pd.DataFrame(sorted(pool, key=lambda r: (r['pred_gbsa'], r['candidate_id']))[:50])
    return dict(best_pred_gbsa=float(top.iloc[0]['pred_gbsa']),
        best_candidate_id=top.iloc[0]['candidate_id'],
        diversity_top50=top50_diversity(top),
        ood_fraction_top50=float(top['ood_flag'].mean()),
        min_train_hamming_top50=float(top['nearest_train_hamming'].mean()),
        n_exportable=len(pool), top=top)

def run_search(a, seed, proxy, model, train_s, train_y, forbidden, data_hash,
               budget, name, arm='main', overrides=None):
    path = Path(a['out_dir']) / 'subruns' / f's{seed}_{proxy}_{budget}_{name}_{arm}'
    context = SearchContext(search_id=path.name, outer_round=1,
        measured_sequences=tuple(train_s), measured_gbsa=tuple(train_y),
        forbidden_sequences=forbidden, seed=seed, max_unique_predictions=budget,
        max_generations=a['max_generations'], allowed_n_from_wt=(4,13),
        data_sha256=data_hash, archive_dir=str(path))
    cfg = dict(FROZEN_CONFIG)
    actual_policy = name if name in EXTENDED_POLICIES else 'mutation_only_ga'
    cfg['max_stagnant_generations'] = 4 if actual_policy == 'mutation_only_ga' else 2
    cfg.update(overrides or {})
    scorer = SurrogateScorer(model, context, cache=SearchArchive(path).load_cache(),
        model_id=model.model_id, feature_version=model.feature_version)
    t0 = time.perf_counter()
    result = make_policy(actual_policy, **cfg).search(context, scorer)
    elapsed = time.perf_counter()-t0
    rows = [json.loads(x) for x in (path/'search_evaluated.jsonl').read_text('utf-8').splitlines()]
    if len(rows) != result.n_unique_predicted or len({r['sequence'] for r in rows}) != len(rows):
        raise ValueError('actual prediction/evaluated archive count mismatch')
    if len(rows) > budget:
        raise ValueError('prediction budget violated')
    d = diagnostics(rows)
    d.pop('top')
    summary = json.loads((path/'search_summary.json').read_text('utf-8'))
    stats = summary['rejection_stats']
    child_count = sum(r['operator'] != 'seed' for r in rows)
    row = dict(seed=seed, proxy=proxy, budget=budget, policy=name, arm=arm,
        actual_predictions=len(rows), stop_reason=result.stop_reason,
        model_id=model.model_id, seconds=elapsed,
        prediction_seconds=scorer.prediction_seconds, prediction_calls=scorer.prediction_calls,
        proposed=stats['proposed'], scored_children=child_count,
        feasibility=(child_count/stats['proposed'] if stats['proposed'] else None),
        n_groups=len(result.groups), archive=str(path.relative_to(Path(a['out_dir']))), **d)
    return row, rows

def align(archives, seed, proxy, budget):
    seeds = {p: [(r['candidate_id'],r['sequence'],r['parent_evidence'])
                 for r in rs if r['operator']=='seed'] for p,rs in archives.items()}
    first = next(iter(seeds))
    if any(s != seeds[first] for s in seeds.values()):
        raise ValueError(f'seed {seed}: unequal initial seed IDs/order')
    models = {r['model_id'] for rs in archives.values() for r in rs}
    if len(models) != 1:
        raise ValueError(f'seed {seed}: unequal fitted models')
    common_n = min(map(len, archives.values()))
    out=[]
    for policy,rs in archives.items():
        d=diagnostics(rs[:common_n]); d.pop('top')
        out.append(dict(seed=seed,proxy=proxy,budget=budget,policy=policy,
            common_n=common_n,initial_seed_count=len(seeds[first]),
            actual_predictions=len(rs),**d))
    return out

def worker(a, seed):
    done = Path(a['out_dir'])/'seed_results'/f'{seed}.json'
    if done.exists():
        return json.loads(done.read_text('utf-8'))
    df=pd.read_csv(ROOT/'data/proxy2000_v2/fad_proxy2000_v2_full.csv')
    seqs=df['Sequence'].map(normalize).tolist()
    vals=df[TARGET_COLUMN].to_numpy(float)
    idx=np.random.RandomState(seed).permutation(len(df))
    tr=idx[:a['n_init']]; te=idx[a['n_init']:]
    train_s=[seqs[i] for i in tr]; train_y=vals[tr].tolist()
    forbidden=frozenset(seqs)
    models={}; fits=[]; runs=[]; aligned=[]; stability=[]; ablations=[]; pairwise=[]
    for proxy in a['proxies']:
        t0=time.perf_counter()
        model=make_surrogate(proxy,seed=seed,n_members=5,screen_k=100)
        model.fit(train_s,np.asarray(train_y))
        train_seconds=time.perf_counter()-t0
        fits.append(dict(seed=seed,proxy=proxy,model_id=model.model_id,
            train_seconds=train_seconds,n_train=len(tr),n_holdout=len(te),
            holdout_spearman=spearman(vals[te],model.predict([seqs[i] for i in te]).mu)))
        models[proxy]=model
    all_archives={}
    for proxy,model in models.items():
        for budget in a['budgets']:
            archives={}
            for policy in a['policies']:
                row,rs=run_search(a,seed,proxy,model,train_s,train_y,forbidden,
                                  a['data_sha256'],budget,policy)
                runs.append(row); archives[policy]=rs
            aligned.extend(align(archives,seed,proxy,budget))
            for policy, baseline in comparison_pairs(a['policies']):
                pair = align({policy: archives[policy], baseline: archives[baseline]},
                             seed, proxy, budget)
                pairwise.append(dict(seed=seed, proxy=proxy, budget=budget,
                    policy=policy, baseline=baseline, common_n=pair[0]['common_n'],
                    policy_best=pair[0]['best_pred_gbsa'],
                    baseline_best=pair[1]['best_pred_gbsa'],
                    delta=pair[0]['best_pred_gbsa']-pair[1]['best_pred_gbsa']))
            all_archives[(proxy,budget)]=archives
        budget=a['budgets'][0]
        archives=all_archives[(proxy,budget)]
        if a['stability']:
            alt=make_surrogate(proxy,seed=seed+1000,n_members=5,screen_k=100)
            alt.fit(train_s,np.asarray(train_y))
            for policy,rs in archives.items():
                top=diagnostics(rs)['top']
                other=models[next(p for p in models if p!=proxy)] if len(models)>1 else alt
                pred_alt=alt.predict(top.sequence.tolist()).mu
                pred_cross=other.predict(top.sequence.tolist()).mu
                altrow,altrs=run_search(a,seed,proxy,alt,train_s,train_y,forbidden,
                    a['data_sha256'],budget,policy,arm='refit')
                runs.append(altrow)
                top_alt=diagnostics(altrs)['top']
                stability.append(dict(seed=seed,proxy=proxy,policy=policy,budget=budget,
                    original_model_id=model.model_id,refit_model_id=alt.model_id,
                    cross_model_id=other.model_id,
                    refit_rank_spearman=spearman(top.pred_gbsa,pred_alt),
                    cross_proxy_rank_spearman=spearman(top.pred_gbsa,pred_cross),
                    refit_top50_jaccard=jaccard(set(top.candidate_id),set(top_alt.candidate_id)),
                    own_best_rescored_refit=float(pred_alt[0]),
                    own_best_rescored_other_proxy=float(pred_cross[0])))
        if a['ablations']:
            variants={'mutation_only_ga':archives['mutation_only_ga']}
            for name,cfg in ABLATIONS.items():
                row,rs=run_search(a,seed,proxy,model,train_s,train_y,forbidden,
                    a['data_sha256'],budget,name,arm='ablation',overrides=cfg)
                runs.append(row); variants[name]=rs
            ablations.extend(align(variants,seed,proxy,budget))
    result=dict(seed=seed,runs=runs,aligned=aligned,stability=stability,
                ablations=ablations,fits=fits,pairwise=pairwise)
    dump(done,result)
    return result

def paired(frame, policy, baseline, metric='best_pred_gbsa'):
    pivot=frame.pivot(index='seed',columns='policy',values=metric)
    d=(pivot[policy]-pivot[baseline]).to_numpy(float)
    lo,hi=bootstrap_ci(d)
    return dict(policy=policy,baseline=baseline,n=len(d),mean_delta=float(np.mean(d)),
        median_delta=float(np.median(d)),ci95=[lo,hi],
        wins=int(np.sum(d < -1e-10)),ties=int(np.sum(abs(d)<=1e-10)),
        losses=int(np.sum(d>1e-10)))


def comparison_pairs(policies):
    pairs = [(p, BASELINE) for p in policies if p != BASELINE]
    for policy in ('mutation_only_ga', 'iterated_local_search',
                   'simulated_annealing', 'mutation_only_adalead'):
        if policy in policies and 'multi_start_hill_climb' in policies:
            pairs.append((policy, 'multi_start_hill_climb'))
    if 'mutation_only_ga' in policies and 'multi_start_beam' in policies:
        pairs.append(('mutation_only_ga', 'multi_start_beam'))
    return pairs

def summarize(results,a):
    frames={k:pd.DataFrame([row for res in results for row in res[k]])
            for k in ('runs','aligned','stability','ablations','fits','pairwise')}
    for k,df in frames.items():
        df.to_csv(Path(a['out_dir'])/f'{k}.csv',index=False)
    n_policies = len(a['policies'])
    expected_main=a['seeds']*len(a['proxies'])*len(a['budgets'])*n_policies
    expected_refit=a['seeds']*len(a['proxies'])*n_policies if a['stability'] else 0
    expected_abl=a['seeds']*len(a['proxies'])*3 if a['ablations'] else 0
    checks=dict(seeds_complete=len(results)==a['seeds'],
        main_rows=len(frames['aligned'])==expected_main,
        total_subruns=len(frames['runs'])==expected_main+expected_refit+expected_abl,
        no_duplicate_runs=not frames['runs'].duplicated(['seed','proxy','budget','policy','arm']).any(),
        finite_results=bool(np.isfinite(frames['aligned'].best_pred_gbsa).all()),
        stability_rows=len(frames['stability'])==expected_refit,
        pairwise_rows=len(frames['pairwise'])==a['seeds']*len(a['proxies'])*len(a['budgets'])*len(comparison_pairs(a['policies'])),
        ablation_rows=len(frames['ablations'])==(expected_abl*4//3 if expected_abl else 0))
    completeness=dict(ok=all(checks.values()),checks=checks,expected_subruns=expected_main+expected_refit+expected_abl,
        observed_subruns=len(frames['runs']),observed_seeds=len(results))
    dump(Path(a['out_dir'])/'completeness.json',completeness)
    summary=dict(config=a,search_interface=SEARCH_INTERFACE_VERSION,policies=a['policies'],
        outcome='predicted GBSA only; not true GBSA discovery',
        alignment='same ordered 16 starts, same fitted model; all metrics at per-seed common actual prediction prefix',
        cells=[],stability=[],ablations=[],pairwise_prefix=[])
    for (proxy,budget),df in frames['aligned'].groupby(['proxy','budget']):
        means=df.groupby('policy')[['best_pred_gbsa','diversity_top50','ood_fraction_top50',
            'min_train_hamming_top50','actual_predictions']].mean().reset_index().to_dict('records')
        comparisons=[paired(df,p,b) for p,b in comparison_pairs(a['policies'])]
        summary['cells'].append(dict(proxy=proxy,budget=int(budget),means=means,
            common_n=[int(df.common_n.min()),float(df.common_n.median()),int(df.common_n.max())],
            paired=comparisons))
    for (proxy,budget,policy,baseline),df in frames['pairwise'].groupby(
            ['proxy','budget','policy','baseline']):
        d=df.delta.to_numpy(float);lo,hi=bootstrap_ci(d)
        summary['pairwise_prefix'].append(dict(proxy=proxy,budget=int(budget),
            policy=policy,baseline=baseline,n=len(d),mean_delta=float(d.mean()),
            median_delta=float(np.median(d)),ci95=[lo,hi],
            wins=int(np.sum(d < -1e-10)),ties=int(np.sum(abs(d)<=1e-10)),
            losses=int(np.sum(d > 1e-10)),
            common_n=[int(df.common_n.min()),float(df.common_n.median()),int(df.common_n.max())]))
    if not frames['stability'].empty:
        summary['stability']=frames['stability'].groupby(['proxy','policy'])[
            ['refit_rank_spearman','cross_proxy_rank_spearman','refit_top50_jaccard']].mean().reset_index().to_dict('records')
    if not frames['ablations'].empty:
        for proxy,df in frames['ablations'].groupby('proxy'):
            summary['ablations'].append(dict(proxy=proxy,paired=[
                paired(df,p,'mutation_only_ga') for p in ABLATIONS]))
    summary['fits']=frames['fits'].groupby('proxy')[['holdout_spearman','train_seconds']].mean().reset_index().to_dict('records')
    summary['stop_reasons']=frames['runs'].groupby(['proxy','budget','policy','arm','stop_reason']).size().reset_index(name='n').to_dict('records')
    dump(Path(a['out_dir'])/'summary.json',summary)
    metrics=frames['aligned'].groupby(['proxy','budget','policy']).agg(
        n_seeds=('seed','size'),mean_best_pred_gbsa=('best_pred_gbsa','mean'),
        mean_actual_predictions=('actual_predictions','mean')).reset_index()
    metrics.to_csv(a['output'],index=False)
    return completeness

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=30)
    p.add_argument('--seed-start',type=int,default=401)
    p.add_argument('--budgets',default='1024,4096,8192')
    p.add_argument('--proxies',default='xgb,rf')
    p.add_argument('--policies',default=','.join(POLICIES))
    p.add_argument('--n-init',type=int,default=300)
    p.add_argument('--max-generations',type=int,default=64)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--stability',action='store_true')
    p.add_argument('--ablations',action='store_true')
    p.add_argument('--out-dir',required=True)
    p.add_argument('--output',required=True)
    a=vars(p.parse_args())
    if not 1<=a['workers']<=2:
        p.error('this resource-checked suite supports one or two workers')
    if a['seeds']<1 or not 1<=a['n_init']<2000:
        p.error('invalid seeds/n-init')
    a['budgets']=[int(x) for x in a['budgets'].split(',')]
    a['proxies']=a['proxies'].split(',')
    a['policies']=a['policies'].split(',')
    if (not set(a['policies']) <= set(EXTENDED_POLICIES)
            or len(set(a['policies']))!=len(a['policies']) or BASELINE not in a['policies']):
        p.error('unique supported policies including matched_seed_random_walk required')
    if a['ablations'] and 'mutation_only_ga' not in a['policies']:
        p.error('GA ablations require mutation_only_ga')
    if min(a['budgets'])<32 or len(set(a['budgets']))!=len(a['budgets']):
        p.error('unique budgets >=32 required')
    if not set(a['proxies']) <= {'xgb','rf'} or len(set(a['proxies']))!=len(a['proxies']):
        p.error('proxies must be unique xgb and/or rf')
    a['out_dir']=str((ROOT/a['out_dir']).resolve())
    out=Path(a['out_dir']);out.mkdir(parents=True,exist_ok=True)
    a['data_sha256']=sha256_file(ROOT/'data/proxy2000_v2/fad_proxy2000_v2_full.csv').upper()
    a['policy_config']=FROZEN_CONFIG
    a['source_hashes']={str(f.relative_to(ROOT)):sha256_file(f) for f in (
        Path(__file__), ROOT/'scripts/loop/search.py',ROOT/'scripts/loop/search_policies.py',
        ROOT/'scripts/loop/surrogates.py',ROOT/'scripts/loop/search_archive.py',
        ROOT/'scripts/loop/interface.py',ROOT/'scripts/loop/search_compare.py')}
    manifest=out/'suite_config.json'
    if manifest.exists() and json.loads(manifest.read_text('utf-8'))!=a:
        raise ValueError('existing suite config/source fingerprint changed; use fresh out-dir')
    dump(manifest,a)
    results=[]
    seeds=range(a['seed_start'],a['seed_start']+a['seeds'])
    with ProcessPoolExecutor(max_workers=a['workers']) as executor:
        futures={executor.submit(worker,a,seed):seed for seed in seeds}
        for future in as_completed(futures):
            res=future.result();results.append(res)
            dump(out/'progress.json',dict(completed=len(results),total=a['seeds'],last_seed=res['seed']))
            print(f"[suite] {len(results)}/{a['seeds']} seed={res['seed']} complete",flush=True)
    check=summarize(sorted(results,key=lambda r:r['seed']),a)
    print(json.dumps(check),flush=True)
    return 0 if check['ok'] else 2

if __name__=='__main__':
    raise SystemExit(main())
