from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
validation=Path(__file__).parent
evidence=ROOT/'evidence/search_seven_20261005'
summaries={}
for name,n,first,subruns in [('development',30,6001,1260),('confirmation',100,8001,1400)]:
    root=ROOT/f'runs/search_seven_{name}_20261005'
    s=json.loads((root/'summary.json').read_text('utf-8'));summaries[name]=s
    assert json.loads((root/'artifact_audit.json').read_text('utf-8'))['ok']
    aligned=pd.read_csv(root/'aligned.csv')
    assert set(aligned.seed)==set(range(first,first+n))
    assert len(pd.read_csv(root/'runs.csv'))==subruns
    assert json.loads((ROOT/f'runs/local_search_seven_{name}_20261005/DONE').read_text('utf-8'))['acceptance_ok']
    for path,digest in s['config']['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
        assert hashlib.sha256((evidence/'source_snapshots'/Path(path).name).read_bytes()).hexdigest()==digest
assert summaries['development']['config']['policy_config']==summaries['confirmation']['config']['policy_config']
assert summaries['development']['config']['source_hashes']==summaries['confirmation']['config']['source_hashes']

images={};anchors={};diagrams={};links=[]
docs=[ROOT/'docs/汇报/9月/GBSA优化项目总汇报_生工团队版_20261005.md',
      ROOT/'docs/汇报/9月/七算法同起点比较与实施结果_20261005.md',evidence/'README.md']
for p in docs:
    text=p.read_text('utf-8');assert '{{' not in text
    diagrams[p.name]=len(re.findall(r'```mermaid\n',text))
    images[p.name]=len(re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text))
    ids=set(re.findall(r'<a id="([^"]+)"',text));anchors[p.name]=len(ids)
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
        if target.startswith(('https://','http://','mailto:')):continue
        if target.startswith('#'):
            assert target[1:] in ids,(p,target)
        else:
            path,_,anchor=target.partition('#')
            dest=(p.parent/path).resolve();assert dest.exists(),(p,target)
            if anchor:assert f'id="{anchor}"' in dest.read_text('utf-8'),(p,target)
        links.append(target)
assert diagrams[docs[0].name]==9 and diagrams[docs[1].name]==3
result=subprocess.run([sys.executable,str(ROOT/'tools/check_doc_links.py')],
                      capture_output=True,text=True,cwd=ROOT)
assert result.returncode==0,result.stdout+result.stderr

check=dict(ok=True,docs_checked=3,local_links_and_anchors=len(links),
    all_document_link_check=result.stdout.strip(),images=images,mermaid_sources=diagrams,
    explicit_anchors=anchors,tests=139,formal_subruns=2660,
    formal_evaluated_rows=2190086+1364671,
    source_and_parameter_freeze_unchanged=True,
    development_seed_range=[6001,6030],confirmation_seed_range=[8001,8100])
(validation/'final_audit.json').write_text(json.dumps(check,indent=2,ensure_ascii=False),encoding='utf-8')
meta=json.loads((validation/'validation.json').read_text('utf-8'))
meta['confirmation_artifact_audit']=json.loads((ROOT/'runs/search_seven_confirmation_20261005/artifact_audit.json').read_text('utf-8'))
meta['final_document_audit']=check
(validation/'validation.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
for name in ['final_audit.json','validation.json','final_audit.py','audit_demos.py','update_main.py','update_indexes.py','render_new.cjs']:
    dest=evidence/name if name.endswith('.json') else evidence/'source_snapshots'/name
    shutil.copy2(validation/name,dest)
hashes={str(p.relative_to(evidence)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in evidence.rglob('*') if p.is_file() and p.name!='files.sha256.json'}
(evidence/'files.sha256.json').write_text(json.dumps(hashes,indent=2,ensure_ascii=False),encoding='utf-8')
assert all(hashlib.sha256((evidence/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
print(json.dumps(check,ensure_ascii=True));print('evidence hash manifest files',len(hashes))
