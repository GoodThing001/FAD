"""Frozen recipe replay and fold-clean structural ablations; no legacy imports."""
from __future__ import annotations
import argparse, json, sys, hashlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from joblib import Parallel, delayed
from scripts.reproduction.new_features import build_seq, MUT_POS, WT_141
ROOT = Path(__file__).resolve().parents[2]

def extra(seq):
    out = np.zeros((len(seq), 14), np.float32)
    clusters = [MUT_POS[:4], MUT_POS[4:8], MUT_POS[8:]]
    for k,s in enumerate(seq):
        changed=[p for p in MUT_POS if s[p-1]!=WT_141[p-1]]
        cm=[sum(p in changed for p in cl) for cl in clusters]
        transitions=sum((WT_141[p-1],s[p-1]) in [('A','G'),('G','A'),('C','U'),('U','C')] for p in changed)
        cats=[sum((WT_141[p-1] in 'AG')==a and (s[p-1] in 'AG')==b for p in changed) for a,b in [(True,False),(False,True),(True,True),(False,False)]]
        gc=sum(s[p-1] in 'GC' for p in MUT_POS)
        out[k]=[len(changed),*cm,transitions,len(changed)-transitions,*cats,max(cm),float(all(cm)),gc/13,(gc-sum(WT_141[p-1] in 'GC' for p in MUT_POS))/13]
    return out

def fit_one(name, X, y, tr, te, seed, out):
    t=time.time()
    sc=StandardScaler().fit(X[tr]); a=sc.transform(X[tr]); b=sc.transform(X[te])
    # A separate scaler per feature block is algebraically column-wise identical.
    screen=ExtraTreesRegressor(n_estimators=200,max_features='log2',min_samples_leaf=5,random_state=42,n_jobs=1).fit(a,y[tr])
    ix=np.argsort(-screen.feature_importances_)[:100]
    m=GradientBoostingRegressor(n_estimators=500,max_depth=3,min_samples_leaf=3,learning_rate=.03,random_state=42).fit(a[:,ix],y[tr])
    pred=m.predict(b[:,ix])
    pd.DataFrame({'canonical_index':te,'truth':y[te],'prediction':pred}).to_csv(out/f'prediction_{seed}_{name}.csv',index=False)
    (out/f'selection_{seed}_{name}.json').write_text(json.dumps(ix.tolist()))
    return dict(seed=seed,variant=name,n_train=len(tr),n_test=len(te),n_features=X.shape[1],spearman=float(pearsonr(rankdata(y[te]),rankdata(pred))[0]),mae=float(np.abs(y[te]-pred).mean()),rmse=float(np.sqrt(np.mean((y[te]-pred)**2))),seconds=time.time()-t)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=['history','defect'],default='history');p.add_argument('--seeds',type=int,default=30);p.add_argument('--seed-start',type=int,default=11001);p.add_argument('--workers',type=int,default=4);p.add_argument('--defect-csv',default='');p.add_argument('--fixed',action='store_true');args=p.parse_args()
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True); artifacts=out.parent/'predictions';artifacts.mkdir(exist_ok=True)
    df=pd.read_csv(ROOT/'data/proxy2000_v2/fad_proxy2000_v2_full.csv'); seq=df.Sequence.tolist(); y=df.gbsa.to_numpy(); base=build_seq(seq)
    if args.mode=='history':
        oh=base[:,:52].reshape(-1,13,4); epi=np.einsum('nik,njk->nij',oh,oh).reshape(-1,169); ext=extra(seq)
        f=pd.read_csv(ROOT/'data/proxy2000_v2/nupack_features_full.csv').set_index('Sequence').loc[seq]; cols=[c for c in f if c.startswith('nupack_')];assert len(cols)==601
        nup=f[cols].fillna(0).to_numpy(np.float32)
        variants={'mut52':base[:,:52],'seq303':base,'seq_mt_epi486':np.hstack([base,ext,epi]),'seq_nup904':np.hstack([base,nup]),'historical1087':np.hstack([base,ext,epi,nup])}
        (out.parent/'features.json').write_text(json.dumps({'nupack_columns':cols,'nupack_sha256':hashlib.sha256((ROOT/'data/proxy2000_v2/nupack_features_full.csv').read_bytes()).hexdigest(),'models_fixed_random_state':42,'target':'gbsa','no_winsorization':True,'screen_train_only':True},indent=2))
    else:
        f=pd.read_csv(ROOT/args.defect_csv).set_index('Sequence').loc[seq]; z=f[['ensemble_defect','wt_struct_prob']].to_numpy();assert np.isfinite(z).all()
        variants={'base303':base,'plus_defect':np.hstack([base,z[:,:1]]),'plus_probability':np.hstack([base,z[:,1:]]),'plus_both':np.hstack([base,z])}
    splits=[]
    if args.fixed:
        lookup={s:i for i,s in enumerate(seq)}
        train=pd.concat([pd.read_csv(ROOT/'data/proxy2000_v2/train.csv'),pd.read_csv(ROOT/'data/proxy2000_v2/val.csv')]); test=pd.read_csv(ROOT/'data/proxy2000_v2/test.csv')
        tr=np.array([lookup[s] for s in train.Sequence]);te=np.array([lookup[s] for s in test.Sequence]);assert len(set(tr)&set(te))==0 and len(tr)+len(te)==2000
        splits.append(('fixed42',tr,te))
    for seed in range(args.seed_start,args.seed_start+args.seeds):
        tr,te=train_test_split(np.arange(len(y)),test_size=.1 if args.mode=='history' else .2,random_state=seed);splits.append((seed,tr,te))
    (out.parent/'splits.json').write_text(json.dumps({str(s):{'train':tr.tolist(),'test':te.tolist()} for s,tr,te in splits}))
    for seed,tr,te in splits:
        rows=Parallel(n_jobs=args.workers)(delayed(fit_one)(name,X,y,tr,te,seed,artifacts) for name,X in variants.items())
        pd.DataFrame(rows).to_csv(out,mode='a',header=not out.exists(),index=False)
        print(seed,[(r['variant'],round(r['spearman'],5)) for r in rows],flush=True)
if __name__=='__main__':main()
