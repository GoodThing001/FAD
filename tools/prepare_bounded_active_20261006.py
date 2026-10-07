"""Freeze a memory bounded, numerically identical confirmation entrypoint."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
src=(ROOT/'scripts/reproduction/supplement_active_20261006.py').read_text('utf-8')
old="    dist=((X[available,None,:]-X[labeled][None,:,:])**2).sum(2).min(1)"
new="    dist=np.concatenate([((X[available[j:j+32],None,:]-X[labeled][None,:,:])**2).sum(2).min(1) for j in range(0,len(available),32)])"
assert src.count(old)==1
src=src.replace(old,new).replace('Paired table-only AL: independent learning and discovery endpoints.','Frozen confirmation copy of paired AL; coverage rows chunked to reduce RAM. Algorithms and float32 reductions unchanged.')
(ROOT/'scripts/reproduction/supplement_active_bounded_20261006.py').write_text(src,encoding='utf-8')
p=ROOT/'configs/experiments/supplement_active_confirmation_20261006.json';d=json.loads(p.read_text());d['entrypoint']='scripts/reproduction/supplement_active_bounded_20261006.py';p.write_text(json.dumps(d,indent=2))
