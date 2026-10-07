"""Controlled encoder pretraining migration, explicit seeds and test exclusion."""
from __future__ import annotations
import argparse,json,sys,hashlib,time,random,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata,pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from scripts.reproduction.new_features import build_seq
from scripts.reproduction.encoder import SequenceEncoder
from scripts.reproduction.surrogate import RegressionHead
from scripts.reproduction.pretrain import pretrain
from scripts.reproduction.data import tokenize_batch
from scripts.reproduction import tokenizer as tok
ROOT=Path(__file__).resolve().parents[2]
MUT=[30,31,32,33,89,90,91,92,127,128,129,130,131]

def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def in_domain(seqs,excluded,n,seed):
    rng=np.random.default_rng(seed);result=[];seen=set(excluded)
    while len(result)<n:
        seq=list(seqs[int(rng.integers(len(seqs)))])
        for p in MUT:
            if rng.random()<.5:seq[p-1]='ACGU'[int(rng.integers(4))]
        seq=''.join(seq)
        if seq not in seen:result.append(seq);seen.add(seq)
    return result

def embed(encoder,seqs,device):
    encoder.eval();parts=[]
    with torch.no_grad():
        for i in range(0,len(seqs),32):
            ids,mask,_=tokenize_batch(seqs[i:i+32]);parts.append(encoder(ids.to(device),mask.to(device)).detach())
    return torch.cat(parts)

def heads_fit(E,y,tr,seed,device,epochs,n):
    seed_all(seed);heads=torch.nn.ModuleList([RegressionHead(d_model=128).to(device) for _ in range(n)])
    z=torch.as_tensor(y[tr],device=device,dtype=torch.float32);mean=z.mean();sd=z.std();z=(z-mean)/(sd+1e-8)
    opt=torch.optim.Adam(heads.parameters(),lr=1e-4);gen=torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm=torch.randperm(len(tr),generator=gen)
        for start in range(0,len(tr),128):
            ix=perm[start:start+128];x=E[tr[ix.numpy()]];opt.zero_grad()
            loss=sum(F.mse_loss(h(x).squeeze(-1),z[ix]) for h in heads)
            loss.backward();opt.step()
    with torch.no_grad():
        predictions=torch.stack([h(E).squeeze(-1) for h in heads]).cpu().numpy()*sd.item()+mean.item()
    return predictions.mean(0),predictions.std(0,ddof=1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seeds',type=int,default=30);p.add_argument('--seed-start',type=int,default=11001);p.add_argument('--corpus',default='data/ngs_sequences.txt');p.add_argument('--pretrain-epochs',type=int,default=5);p.add_argument('--head-epochs',type=int,default=50);p.add_argument('--members',type=int,default=10);p.add_argument('--corpus-size',type=int,default=2000);a=p.parse_args()
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
    out=ROOT/a.output;art=out.parent/'predictions';art.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(ROOT/'data/proxy2000_v2/fad_proxy2000_v2_full.csv');seqs=df.Sequence.tolist();y=df.gbsa.to_numpy();excluded=set(seqs)
    X_sequence=build_seq(seqs)
    foreign=[]
    for line in (ROOT/a.corpus).read_text().splitlines():
        s=line.strip().upper().replace('T','U')
        if s and len(s)<=300 and set(s)<=set('ACGU') and s not in excluded and s not in foreign:foreign.append(s)
        if len(foreign)==a.corpus_size:break
    assert len(foreign)==a.corpus_size,'insufficient valid foreign sequences'
    # Same initial random encoder for all arms/splits. Foreign corpus is independent
    # of FAD labels/splits, so its pretrained encoder can be reused without leakage.
    init_t=time.time()
    seed_all(13000);initial=SequenceEncoder(tok.VOCAB_SIZE).to(device);state=copy.deepcopy(initial.state_dict())
    E_none=embed(initial,seqs,device)
    initial_embedding_seconds=time.time()-init_t
    foreign_t=time.time()
    seed_all(13000);external=SequenceEncoder(tok.VOCAB_SIZE).to(device);external.load_state_dict(state)
    pretrain(external,foreign,device,a.pretrain_epochs,32,8e-4,.98,13000);E_foreign=embed(external,seqs,device)
    manifest={'device':device,'torch':torch.__version__,'foreign_corpus_sha256':hashlib.sha256((ROOT/a.corpus).read_bytes()).hexdigest(),'foreign_used_sha256':hashlib.sha256('\n'.join(foreign).encode()).hexdigest(),'foreign_n':len(foreign),'foreign_reused':True,'initial_encoder_seed':13000,'n_members':a.members,'head_epochs':a.head_epochs,'pretrain_epochs':a.pretrain_epochs,'head_batch_size':128,'pretrain_batch_size':32,'supervision':'GBSA MSE, target standardized on train only','physics_features':False,'encoder_frozen':True}
    manifest['foreign_length_stats']={'min':min(map(len,foreign)),'mean':float(np.mean(list(map(len,foreign)))),'max':max(map(len,foreign))}
    manifest['budget_matching']='equal sequence count and epochs, not equal tokens (foreign sequence lengths differ)'
    manifest['shared_foreign_pretrain_and_embedding_seconds']=time.time()-foreign_t
    manifest['random_encoder_embedding_seconds']=initial_embedding_seconds
    (out.parent/'paper_manifest.json').write_text(json.dumps(manifest,indent=2))
    for seed in range(a.seed_start,a.seed_start+a.seeds):
        t=time.time();tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed)
        corpus=in_domain([seqs[i] for i in tr],excluded,a.corpus_size,seed)
        assert set(corpus).isdisjoint(excluded)
        pd.DataFrame({'Sequence':corpus}).to_csv(art/f'in_domain_corpus_{seed}.csv',index=False)
        (art/f'split_{seed}.json').write_text(json.dumps({'train':tr.tolist(),'test':te.tolist(),'corpus_sha256':hashlib.sha256('\n'.join(corpus).encode()).hexdigest(),'corpus_overlap_all_known':0}))
        domain_t=time.time()
        seed_all(seed);enc=SequenceEncoder(tok.VOCAB_SIZE).to(device);enc.load_state_dict(state)
        pretrain(enc,corpus,device,a.pretrain_epochs,32,8e-4,.98,seed);E_domain=embed(enc,seqs,device)
        (art/f'domain_timing_{seed}.json').write_text(json.dumps({'pretrain_and_embedding_seconds':time.time()-domain_t}))
        rows=[]
        for name,E in [('no_pretrain',E_none),('foreign_pretrain',E_foreign),('in_domain_pretrain',E_domain)]:
            arm_t=time.time()
            mu,sd=heads_fit(E,y,tr,seed,device,a.head_epochs,a.members)
            pd.DataFrame({'canonical_index':te,'truth':y[te],'prediction':mu[te],'sigma':sd[te]}).to_csv(art/f'prediction_{seed}_{name}.csv',index=False)
            rows.append(dict(seed=seed,variant=name,spearman=float(pearsonr(rankdata(y[te]),rankdata(mu[te]))[0]),mae=float(np.abs(y[te]-mu[te]).mean()),rmse=float(np.sqrt(np.mean((y[te]-mu[te])**2))),n_train=len(tr),n_test=len(te),seconds=time.time()-arm_t))
            print(seed,name,rows[-1]['spearman'],flush=True)
        arm_t=time.time()
        m=GradientBoostingRegressor(n_estimators=500,max_depth=3,min_samples_leaf=3,learning_rate=.03,random_state=42).fit(X_sequence[tr],y[tr])
        pred=m.predict(X_sequence[te]);name='sequence303_GBR_control'
        pd.DataFrame({'canonical_index':te,'truth':y[te],'prediction':pred}).to_csv(art/f'prediction_{seed}_{name}.csv',index=False)
        rows.append(dict(seed=seed,variant=name,spearman=float(pearsonr(rankdata(y[te]),rankdata(pred))[0]),mae=float(np.abs(y[te]-pred).mean()),rmse=float(np.sqrt(np.mean((y[te]-pred)**2))),n_train=len(tr),n_test=len(te),seconds=time.time()-arm_t))
        pd.DataFrame(rows).to_csv(out,mode='a',header=not out.exists(),index=False)
        del enc,E_domain
if __name__=='__main__':main()
