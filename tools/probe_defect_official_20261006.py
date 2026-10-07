"""Diagnose official defect API compatibility; no GBSA performance selection."""
import json,inspect
from pathlib import Path
import numpy as np,pandas as pd,nupack
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.reproduction import ensemble_defect as e
from scripts.reproduction.structure_metrics import target_structure_metrics
root=Path(__file__).resolve().parents[1]
structure,partner=e.compute_wt_target()
seqs=pd.read_csv(root/'data/proxy2000_v2/fad_proxy2000_v2_full.csv').Sequence.tolist()
allowed={'AU','UA','GC','CG','GU','UG'}
compatible=[s for s in seqs if all(s[i]+s[j] in allowed for i,j in enumerate(partner) if j>i)]
rows=[]
for name,seq,target in [('WT',e.WT_141,structure),('first',seqs[0],structure),('first_unpaired',seqs[0],'.'*141)]+[('compatible',s,structure) for s in compatible[:3]]:
    row={'name':name,'Sequence':seq,'target':target}
    try:
        official=float(nupack.defect(strands=[seq],structure=target,model=e._get_model()))*141
        ours=target_structure_metrics(nupack.pairs(strands=[seq],model=e._get_model()).to_array(),e._target_from_dotbracket(target),e.MUT_POS)[0]
        row.update(official=official,ours=ours,error=abs(official-ours))
    except Exception as ex:row.update(error_type=type(ex).__name__,message=str(ex))
    rows.append(row)
result={'compatible_count':len(compatible),'checks':rows,'official_source':inspect.getsource(nupack.defect)}
out=root/'runs/defect_official_probe_20261006.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
