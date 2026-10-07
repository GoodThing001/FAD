"""Validate the report's local destinations, assets and saved numeric evidence."""
from pathlib import Path
import hashlib
import json
import re
import sys
from urllib.parse import unquote
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.check_doc_links import PAT

DOC=ROOT/'docs/汇报/论文复现到七算法搜索_阶段总汇报_PPT素材版_20261006.md'
OUT=ROOT/'docs/汇报/figures/阶段总汇报_20261006'

def slug(s):
    s=re.sub(r'<[^>]*>','',s).strip().lower()
    return re.sub(r'[^\w\-\s]','',s).replace(' ','-')

def main():
    s=DOC.read_text('utf-8')
    manifest=json.loads((OUT/'report_sources.json').read_text('utf-8'))
    for path,digest in manifest['source_files'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    links=[];images=[];anchors=[]
    for m in PAT.finditer(s):
        dest=unquote(m.group(1).strip().strip('<>'))
        if dest.startswith(('https://','http://','mailto:')):continue
        path,_,anchor=dest.partition('#')
        p=(DOC.parent/path).resolve() if path else DOC
        if p==OUT/'report_validation.json':
            links.append(dest)  # This output is created only after every check.
            continue
        assert p.exists(),dest
        links.append(dest)
        if anchor and p.is_file():
            target=p.read_text('utf-8-sig')
            headings=set(slug(h) for h in re.findall(r'^#{1,6}\s+(.+)$',target,re.M))
            headings.update(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)',target))
            assert anchor in headings,(dest,sorted(headings))
            anchors.append(dest)
        if m.start()>0 and s[m.start()-1]=='!':
            with Image.open(p) as im:
                im.load()
                assert im.width>300 and im.height>120
                assert np.asarray(im.convert('L')).std()>2
                images.append({'path':p.relative_to(ROOT).as_posix(),'width':im.width,'height':im.height,
                               'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    codes=re.findall(r'```mermaid\r?\n([\s\S]*?)```',s)
    assert len(codes)==6
    for p in OUT.glob('*.mmd'):
        assert p.read_text('utf-8').strip() in [c.strip() for c in codes]
        ET.parse(p.with_suffix('.svg'))
        assert p.with_suffix('.png').exists()
    for p in OUT.glob('*.svg'):ET.parse(p)
    supplement=ROOT/'evidence/supplement_20261006/index.json'
    supplementary_hashes=0
    if supplement.exists():
        for rel,digest in json.loads(supplement.read_text('utf-8'))['file_hashes'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
            supplementary_hashes+=1
    assert '{{' not in s
    from scripts.loop.interface import WT_141,MUT_POS
    dataset=pd.read_csv(ROOT/'data/proxy2000_v2/fad_proxy2000_v2_full.csv')
    counts=dataset.Sequence.map(lambda seq:sum(seq[p-1]!=WT_141[p-1] for p in MUT_POS))
    for label,mask in [('0–3',counts.le(3))]+[(str(i),counts.eq(i)) for i in range(4,14)]:
        assert f'| {label} | {int(mask.sum())} | {mask.mean()*100:.2f}% |' in s
    summary=json.loads((ROOT/'evidence/search_seven_20261005/confirmation/summary.json').read_text('utf-8'))
    aligned=pd.read_csv(ROOT/'evidence/search_seven_20261005/confirmation/aligned.csv')
    pairwise=pd.read_csv(ROOT/'evidence/search_seven_20261005/confirmation/pairwise.csv')
    paired_count=0
    for cell in summary['cells']:
        pivot=aligned[aligned.proxy.eq(cell['proxy'])].pivot(index='seed',columns='policy',values='best_pred_gbsa')
        for row in cell['paired']:
            delta=(pivot[row['policy']]-pivot[row['baseline']]).to_numpy()
            assert np.isclose(delta.mean(),row['mean_delta'],rtol=0,atol=1e-12)
            assert np.isclose(np.median(delta),row['median_delta'],rtol=0,atol=1e-12)
            assert (sum(delta< -1e-10),sum(abs(delta)<=1e-10),sum(delta>1e-10))==(row['wins'],row['ties'],row['losses'])
            paired_count+=1
    for row in summary['pairwise_prefix']:
        delta=pairwise[pairwise.proxy.eq(row['proxy'])&pairwise.policy.eq(row['policy'])&pairwise.baseline.eq(row['baseline'])].delta.to_numpy()
        assert np.isclose(delta.mean(),row['mean_delta'],rtol=0,atol=1e-12)
        assert (sum(delta< -1e-10),sum(abs(delta)<=1e-10),sum(delta>1e-10))==(row['wins'],row['ties'],row['losses'])
        paired_count+=1
    result={'ok':True,'report':DOC.relative_to(ROOT).as_posix(),
            'characters':len(s),'lines':s.count('\n'),
            'markdown_tables':len(re.findall(r'^\| ---',s,re.M)),
            'inline_images':len(images),'new_data_figures':8+manifest.get('supplemental_figures',0),'new_flowcharts':6,
            'mermaid_sources':len(codes),'local_destinations':len(links),'verified_anchors':len(anchors),
            'input_sha256_checked':len(manifest['source_files']),
            'seven_confirmation_paired_mean_median_win_tie_loss_checks':paired_count,
            'new_model_experiments':manifest.get('new_model_experiments',0),'new_GBSA':0,
            'supplementary_files_sha256_checked':supplementary_hashes,
            'report_sha256':hashlib.sha256(DOC.read_bytes()).hexdigest(),
            'images':images}
    (OUT/'report_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='images'},ensure_ascii=False))

if __name__=='__main__':main()
