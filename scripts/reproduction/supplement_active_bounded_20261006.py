"""Frozen confirmation copy of paired AL; coverage rows chunked to reduce RAM. Algorithms and float32 reductions unchanged."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata,pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import Parallel,delayed
from scripts.reproduction.new_features import build_seq
ROOT=Path(__file__).resolve().parents[2]

def coverage(X,labeled,available,k,chosen):
    dist=np.concatenate([((X[available[j:j+32],None,:]-X[labeled][None,:,:])**2).sum(2).min(1) for j in range(0,len(available),32)])
    for i in chosen:dist=np.minimum(dist,((X[available]-X[i])**2).sum(1))
    ids=[]
    for _ in range(k):
        mask=np.isin(available,list(chosen)+ids); ix=int(np.argmax(np.where(mask,-np.inf,dist)));id=int(available[ix]);ids.append(id);dist=np.minimum(dist,((X[available]-X[id])**2).sum(1))
    return ids

def select(policy,mu,sigma,X,labeled,available,seed):
    rng=np.random.default_rng(seed)
    if policy=='random':return rng.choice(available,100,replace=False)
    if policy=='greedy':return available[np.argsort(mu,kind='stable')[:100]]
    if policy=='uncertainty':return available[np.argsort(-sigma,kind='stable')[:100]]
    chosen=list(available[np.argsort(-sigma,kind='stable')[:45]])
    chosen+=coverage(X,labeled,available,25,chosen)
    chosen+=list(i for i in available[np.argsort(mu,kind='stable')] if i not in chosen)[:20]
    chosen+=list(rng.choice([i for i in available if i not in chosen],10,replace=False))
    assert len(set(chosen))==100
    return np.array(chosen)

def one(seed,X,y,out):
    perm=np.random.RandomState(seed).permutation(len(y));hold=perm[:400];init=perm[400:800];pool=perm[800:];top=set(pool[np.argsort(y[pool])[:120]])
    (out/f'split_{seed}.json').write_text(json.dumps({'holdout':hold.tolist(),'initial':init.tolist(),'pool':pool.tolist()}))
    rows=[]
    for policy in ['random','greedy','uncertainty','mixed']:
        labeled=init.copy();queried=[];available=pool.copy()
        for round in range(6):
            t=time.time();sc=StandardScaler().fit(X[labeled]);Z=sc.transform(X)
            preds=[]
            for member in range(5):
                rng=np.random.default_rng(seed*100+member);sample=rng.choice(labeled,len(labeled),replace=True)
                m=HistGradientBoostingRegressor(max_iter=200,max_depth=3,max_leaf_nodes=31,min_samples_leaf=20,learning_rate=.03,l2_regularization=1,random_state=seed*100+member).fit(Z[sample],y[sample])
                preds.append(m.predict(Z))
            preds=np.array(preds);mu=preds.mean(0);sigma=preds.std(0,ddof=1)
            pd.DataFrame({'canonical_index':hold,'truth':y[hold],'prediction':mu[hold],'sigma':sigma[hold]}).to_csv(out/f'holdout_{seed}_{policy}_{round}.csv',index=False)
            rows.append(dict(seed=seed,policy=policy,round=round,n_train=len(labeled),spearman=float(pearsonr(rankdata(y[hold]),rankdata(mu[hold]))[0]),mae=float(np.abs(y[hold]-mu[hold]).mean()),rmse=float(np.sqrt(np.mean((y[hold]-mu[hold])**2))),query_best=float(np.min(y[queried])) if queried else np.nan,regret=float(np.min(y[queried])-np.min(y[pool])) if queried else np.nan,top10_recall=len(set(queried)&top)/120,seconds=time.time()-t))
            if round==5:break
            selected=select(policy,mu[available],sigma[available],Z,labeled,available,seed*10+round)
            pd.DataFrame({'canonical_index':selected,'truth_revealed':y[selected],'mu_before_reveal':mu[selected],'sigma_before_reveal':sigma[selected]}).to_csv(out/f'queries_{seed}_{policy}_{round}.csv',index=False)
            queried.extend(selected.tolist());labeled=np.r_[labeled,selected];available=available[~np.isin(available,selected)]
        print(seed,policy,rows[-1]['spearman'],rows[-1]['regret'],flush=True)
    pd.DataFrame(rows).to_csv(out/f'metrics_{seed}.csv',index=False)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seeds',type=int,default=30);p.add_argument('--seed-start',type=int,default=11001);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    out=ROOT/a.output;art=out.parent/'predictions';art.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(ROOT/'data/proxy2000_v2/fad_proxy2000_v2_full.csv');X=build_seq(df.Sequence.tolist());y=df.gbsa.to_numpy()
    # Process chunks create resumable per-seed files and stream accumulated metrics.
    for start in range(a.seed_start,a.seed_start+a.seeds,a.workers):
        seeds=list(range(start,min(start+a.workers,a.seed_start+a.seeds)))
        result=Parallel(n_jobs=a.workers)(delayed(one)(seed,X,y,art) for seed in seeds)
        pd.DataFrame([r for rows in result for r in rows]).to_csv(out,mode='a',header=not out.exists(),index=False)
if __name__=='__main__':main()
