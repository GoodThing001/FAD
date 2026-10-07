"""Read-only post-run input and process audit, excludes SSH credentials."""
import pathlib,json,hashlib
import pandas as pd
root=pathlib.Path(__file__).resolve().parents[1]
paper=root/'runs/srv_supplement_paper_aggregate_20261006'
manifest=json.loads((paper/'paper_manifest.json').read_text())
known=set(pd.read_csv(root/'data/proxy2000_v2/fad_proxy2000_v2_full.csv').Sequence)
corpus=root/'data/ngs_sequences.txt';used=[];seen=set()
for line in corpus.read_text().splitlines():
    s=line.strip().upper().replace('T','U')
    if s and len(s)<=300 and set(s)<=set('ACGU') and s not in known and s not in seen:used.append(s);seen.add(s)
    if len(used)==2000:break
actual={'foreign_corpus_sha256':hashlib.sha256(corpus.read_bytes()).hexdigest(),'foreign_used_sha256':hashlib.sha256('\n'.join(used).encode()).hexdigest(),'foreign_n':len(used)}
assert all(manifest[k]==v for k,v in actual.items())
live=[]
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:args=(p/'cmdline').read_bytes().decode().split('\0')
    except (FileNotFoundError,PermissionError,UnicodeDecodeError):continue
    if any(a.startswith('runs/srv_supplement_') and a.endswith('/metrics.csv') for a in args) and any(a.endswith('.py') for a in args):live.append(int(p.name))
result={'independent_foreign_input_check':actual,'known_sequence_overlap':len(set(used)&known),'active_supplement_experiment_pids':live,'source_provenance_boundary':'historical server corpus, original-paper full accession not independently recertified'}
out=root/'runs/supplement_remote_validation_20261006.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert not live,'supplement experiments still running'
