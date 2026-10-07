"""Build the seven-policy report from completed, independently audited tables."""
from pathlib import Path
import json
import re
import shutil
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/汇报/9月/七算法同起点比较与实施结果_20261005.md'
NAMES={'matched_seed_random_walk':'同起点随机游走','multi_start_hill_climb':'多起点爬山',
       'multi_start_beam':'多路径束搜索','mutation_only_ga':'无交叉遗传算法',
       'iterated_local_search':'迭代局部搜索','simulated_annealing':'模拟退火',
       'mutation_only_adalead':'无交叉 AdaLead 改编版'}
PROXIES={'rf':'随机森林','xgb':'五成员梯度提升（本机后端）'}
NEW=['iterated_local_search','simulated_annealing','mutation_only_adalead']

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

def load(name):
    root=ROOT/f'runs/search_seven_{name}_20261005'
    summary=json.loads((root/'summary.json').read_text('utf-8'))
    audit=json.loads((root/'artifact_audit.json').read_text('utf-8'))
    assert audit['ok'] and json.loads((root/'completeness.json').read_text('utf-8'))['ok']
    return root,summary,audit

def result_tables(summary):
    means=[];contrasts=[];common=[]
    for cell in sorted(summary['cells'],key=lambda c:(c['budget'],c['proxy'])):
        common.append([PROXIES[cell['proxy']],cell['budget'],*cell['common_n']])
        m={r['policy']:r for r in cell['means']}
        for p in NAMES:
            r=m[p]
            means.append([PROXIES[cell['proxy']],cell['budget'],NAMES[p],f"{r['best_pred_gbsa']:.4f}",
                f"{r['diversity_top50']:.3f}",f"{100*r['ood_fraction_top50']:.2f}%",
                f"{r['min_train_hamming_top50']:.3f}",f"{r['actual_predictions']:.1f}"])
        for r in cell['paired']:
            contrasts.append([PROXIES[cell['proxy']],cell['budget'],NAMES[r['policy']]+' − '+NAMES[r['baseline']],
                f"{r['mean_delta']:+.4f}",f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]",
                f"{r['median_delta']:+.4f}",f"{r['wins']}/{r['ties']}/{r['losses']}"])
    return (table(['代理','预算上限','算法','最低预测均值','前50距离','分布外比例','最近训练距离','各法实际次数均值'],means),
            table(['代理','上限','配对：前者减后者','均值差','逐项95%区间','中位差','胜/平/负'],contrasts),
            table(['代理','预算上限','七者共同次数最少','中位数','最多'],common))

def main():
    dev,ds,da=load('development');conf,cs,ca=load('confirmation')
    assert ds['config']['source_hashes']==cs['config']['source_hashes']
    assert ds['config']['policy_config']==cs['config']['policy_config']
    assert ds['config']['policies']==cs['config']['policies']
    assert ds['config']['max_generations']==cs['config']['max_generations']==256
    assert cs['config']['seed_start']==8001 and cs['config']['seeds']==100
    dm,dc,dn=result_tables(ds);cm,cc,cn=result_tables(cs)
    tests=(ROOT/'runs/search_seven_validation_20261005/tests.log').read_text('utf-8')
    count=int(re.search(r'Ran (\d+) tests',tests).group(1));assert re.search(r'\nOK\s*\n',tests)
    stability=table(['代理','算法','重训重评分相关','跨代理重评分相关','独立重跑前50交并比'],[
        [PROXIES[r['proxy']],NAMES[r['policy']],f"{r['refit_rank_spearman']:.3f}",
         f"{r['cross_proxy_rank_spearman']:.3f}",f"{r['refit_top50_jaccard']:.3f}"] for r in ds['stability']])
    pairtable=table(['代理','比较','共同次数最少/中位/最多','均值差','逐项95%区间','胜/平/负'],[
        [PROXIES[r['proxy']],NAMES[r['policy']]+' − '+NAMES[r['baseline']],
         '/'.join(str(x) for x in r['common_n']),f"{r['mean_delta']:+.4f}",
         f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]",f"{r['wins']}/{r['ties']}/{r['losses']}"]
        for r in cs['pairwise_prefix']])
    run=pd.read_csv(conf/'runs.csv')
    timing=table(['代理','算法','总搜索秒数均值','其中预测秒数','预测批调用均值','实际次数均值'],[
        [PROXIES[proxy],NAMES[p],f'{v.seconds.mean():.3f}',f'{v.prediction_seconds.mean():.3f}',
         f'{v.prediction_calls.mean():.1f}',f'{v.actual_predictions.mean():.1f}']
        for (proxy,p),v in run.groupby(['proxy','policy'])])
    stops=table(['代理','上限','算法','停止原因','子运行数'],[
        [PROXIES[r['proxy']],r['budget'],NAMES[r['policy']],r['stop_reason'],r['n']]
        for r in ds['stop_reasons'] if r['arm']=='main'])
    fits=table(['包','代理','留出集排序相关均值','每种子训练秒数均值'],[
        [kind,PROXIES[r['proxy']],f"{r['holdout_spearman']:.4f}",f"{r['train_seconds']:.3f}"]
        for kind,s in [('开发',ds),('确认',cs)] for r in s['fits']])
    verdict=[];overview=[]
    for cell in cs['cells']:
        best=min(cell['means'],key=lambda r:r['best_pred_gbsa'])
        overview.append([PROXIES[cell['proxy']],NAMES[best['policy']],
                         f"{best['best_pred_gbsa']:.4f}",
                         '同一代理下七法共同次数的描述性均值；具体配对差和稳定性见§5–§6'])
        for p in NEW:
            r=next(r for r in cell['paired'] if r['policy']==p and r['baseline']=='multi_start_hill_climb')
            direction=('本设置下均值更低，逐项区间也在零以下' if r['ci95'][1]<0 else
                       '本设置下均值更高，逐项区间也在零以上' if r['ci95'][0]>0 else
                       '区间跨零，不能确认本设置优于爬山')
            verdict.append(f"- **{PROXIES[cell['proxy']]}／{NAMES[p]}对爬山**：差 {r['mean_delta']:+.4f}，"
                f"95% 区间 [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]，胜/平/负 "
                f"{r['wins']}/{r['ties']}/{r['losses']}；{direction}。")
    demos=json.loads((Path(__file__).parent/'demo_audit.json').read_text('utf-8'))
    assert demos['ok']
    demotable=table(['已实际完成的运行','算法','实际预测数','可导出候选','组数','最低预测','停止原因'],[
        [r['run'],NAMES[r['policy']],r['actual_predictions'],r['n_candidates'],r['n_groups'],
         f"{r['best_pred_gbsa']:.6f}",r['stop_reason']] for r in demos['runs']])
    text=(Path(__file__).parent/'report_template.md').read_text('utf-8')
    values={'CONF_MEANS':cm,'CONF_CONTRASTS':cc,'CONF_N':cn,'DEV_MEANS':dm,'DEV_CONTRASTS':dc,
        'DEV_N':dn,'PAIR_TABLE':pairtable,'STABILITY':stability,'TIMING':timing,'STOPS':stops,
        'FITS':fits,'VERDICT':'\n'.join(verdict),'TEST_COUNT':str(count),'DEMOS':demotable,
        'OVERVIEW':table(['固定诊断评分器','平均最低预测的方法','该均值','如何解释'],overview),
        'DEV_RECORDS':str(da['evaluated_rows']),'CONF_RECORDS':str(ca['evaluated_rows'])}
    for key,value in values.items():text=text.replace('{{'+key+'}}',value)
    assert '{{' not in text
    DOC.write_text(text,encoding='utf-8')
    out=ROOT/'evidence/search_seven_20261005';out.mkdir(exist_ok=True)
    for label,root in [('development',dev),('confirmation',conf),('smoke',ROOT/'runs/search_seven_smoke_20261005')]:
        dest=out/label;dest.mkdir(exist_ok=True)
        for filename in ['suite_config.json','summary.json','completeness.json','artifact_audit.json',
                         'runs.csv','aligned.csv','pairwise.csv','stability.csv','fits.csv']:
            source=root/filename
            if source.exists():shutil.copy2(source,dest/filename)
        wrap=ROOT/f'runs/local_search_seven_{label}_20261005'
        for filename in ['config.json','run_manifest.json','environment.json','data_hash.json','DONE','stdout.log','metrics.csv']:
            source=wrap/filename
            if source.exists():shutil.copy2(source,dest/f'runner_{filename}')
    sources=out/'source_snapshots';sources.mkdir(exist_ok=True)
    for name in ['search_policies.py','search.py','search_run.py','search_compare.py','search_suite.py',
                 'search_archive.py','loop.py','surrogates.py','interface.py']:
        shutil.copy2(ROOT/'scripts/loop'/name,sources/name)
    for source in [ROOT/'tools/verify_search_suite.py',ROOT/'tools/plot_search_seven.py',
                   ROOT/'tests/test_search_extensions.py',Path(__file__),
                   Path(__file__).parent/'report_template.md']:
        shutil.copy2(source,sources/source.name)
    for name in ['eval_fusion_smoothing.py','label_smoothing.py']:
        shutil.copy2(ROOT/'scripts/reproduction'/name,sources/name)
    for source in (ROOT/'configs/experiments').glob('search_*20261005.json'):
        if source.name.startswith(('search_seven_','search_ils_','search_sa_','search_adalead_')):
            shutil.copy2(source,out/source.name)
    shutil.copy2(ROOT/'docs/项目记录/七算法同起点比较冻结方案_20261005.md',out/'冻结方案.md')
    shutil.copy2(Path(__file__).parent/'tests.log',out/'tests.log')
    shutil.copy2(Path(__file__).parent/'protocol_amendment.json',out/'protocol_amendment.json')
    shutil.copy2(Path(__file__).parent/'validation.json',out/'validation.json')
    shutil.copy2(Path(__file__).parent/'demo_audit.json',out/'demo_audit.json')
    for r in demos['runs']:
        src=ROOT/'runs'/r['run'];dest=out/'demos'/r['run'];dest.mkdir(parents=True,exist_ok=True)
        for filename in ['config.json','run_manifest.json','metrics.csv','DONE']:
            shutil.copy2(src/filename,dest/filename)
        for filename in ['search_summary.json','search_snapshot.json','search_generations.csv','search_groups.json','search_candidates.csv']:
            shutil.copy2(src/'checkpoints'/filename,dest/filename)
    (out/'README.md').write_text(f'''# 七算法同起点比较证据（2026-10-05）

主结果见[七算法实施结果](../../docs/汇报/9月/七算法同起点比较与实施结果_20261005.md)与[生工团队总汇报](../../docs/汇报/9月/GBSA优化项目总汇报_生工团队版_20261005.md)。

- 开发：30 种子 6001–6030，两代理、两档预算、1260 子运行，{da['evaluated_rows']} 条实际评分档案通过独立审计。
- 确认：100 种子 8001–8100，两代理、1024 上限、1400 子运行，{ca['evaluated_rows']} 条档案通过审计。
- 工程小试：2 种子、56 子运行、7168 条档案通过审计。
- 七策略均使用相同顺序起点及同一拟合代理；主表 `aligned.csv` 用七者共同实际次数。`pairwise.csv` 是预设的两两共同次数补充表。
- {count} 项本地测试通过。人工回传与暂停测试不构成真实测量安排。无服务器同步声明。

本包仅包含紧凑表、核验、源码与冻结方案。完整逐次档案保存在本地 `runs/search_seven_*_20261005/subruns/`；未复制数据集或大型逐条日志。旧四算法包 `evidence/search_suite_20261005/` 保留。`files.sha256.json` 为本包内容摘要。

**边界**：无新序列真实 GBSA、无新对接/分子动力学、无湿实验；AdaLead 是无交叉、固定评分器、加法阈值改编版；两个代理均为算法诊断模型。
''',encoding='utf-8')
    hashes={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob('*') if p.is_file() and p.name!='files.sha256.json'}
    (out/'files.sha256.json').write_text(json.dumps(hashes,indent=2,ensure_ascii=False),encoding='utf-8')
    print(DOC);print('evidence files',len(hashes))

if __name__=='__main__':main()
