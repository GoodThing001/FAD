"""Build an evidence-only Chinese report and figures; no fitting or label requests.

Saved experiment summaries retain their original uncertainty conventions. This
builder additionally recomputes descriptive dataset statistics and one explicitly
labelled bootstrap figure from the saved v8 rows. Inputs are content-hashed.
"""
from pathlib import Path
import hashlib
import json
import re
import sys
from report_supplement_sections_20261006 import build as supplement_sections

import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.loop.interface import WT_141, MUT_POS

DOC = ROOT/'docs/汇报/论文复现到七算法搜索_阶段总汇报_PPT素材版_20261006.md'
OUT = ROOT/'docs/汇报/figures/阶段总汇报_20261006'
DATAOUT = OUT/'图表数据'
OUT.mkdir(parents=True, exist_ok=True)
DATAOUT.mkdir(parents=True, exist_ok=True)
SOURCES = {}
PARTS = {}
POLICIES = ['matched_seed_random_walk', 'multi_start_hill_climb', 'multi_start_beam',
            'mutation_only_ga', 'iterated_local_search', 'simulated_annealing', 'mutation_only_adalead']
NAMES = dict(zip(POLICIES, ['同起点随机游走','多起点爬山','多路径束搜索','无交叉 GA','ILS','SA','无交叉 AdaLead']))
PROXIES = {'rf':'RF','xgb':'本地 GBM（配置名 xgb）'}
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'axes.unicode_minus':False, 'font.size':11, 'svg.fonttype':'none'})

def register(p):
    p = ROOT/p if not isinstance(p, Path) else p
    SOURCES[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def csv(p):
    return pd.read_csv(register(p))

def js(p):
    return json.loads(register(p).read_text('utf-8-sig'))

def text(p):
    return register(p).read_text('utf-8-sig')

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v).replace('|','／').replace('\n',' ') for v in row)+' |' for row in rows])

def fmt(v, n=4):
    return f'{float(v):.{n}f}'

def interval(ci, n=4):
    return f'[{fmt(ci[0],n)}, {fmt(ci[1],n)}]'

def signed(v, n=4):
    return f'{float(v):+.{n}f}'

def cite(path, label='源结果'):
    register(path)
    return f'[{label}](../../{Path(path).as_posix()})'

def dump_csv(frame, name):
    frame.to_csv(DATAOUT/f'{name}.csv', index=False, encoding='utf-8-sig')

def save(fig, name):
    fig.savefig(OUT/f'{name}.png',dpi=240,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)

def bootstrap(a):
    a=np.asarray(a,dtype=float)
    rng=np.random.default_rng(20260924)
    return np.quantile(a[rng.integers(0,len(a),(20000,len(a)))].mean(axis=1),[.025,.975])

def sp(a,b):
    return float(pearsonr(rankdata(a),rankdata(b)).statistic)

def diagram(name, title, source):
    (OUT/f'{name}.mmd').write_text(source.strip()+'\n',encoding='utf-8')
    return (f'![{title}](figures/阶段总汇报_20261006/{name}.png)\n\n'
            f'[矢量版](figures/阶段总汇报_20261006/{name}.svg) · '
            f'[可编辑流程源](figures/阶段总汇报_20261006/{name}.mmd)\n\n'
            f'<details>\n<summary>流程图可编辑源代码</summary>\n\n```mermaid\n{source.strip()}\n```\n\n</details>')

def main():
    audit=js('evidence/report_review_20261006/method_verification.json')
    for p,digest in audit['source_sha256'].items():
        assert hashlib.sha256(register(p).read_bytes()).hexdigest()==digest,p
    d=csv('data/proxy2000_v2/fad_proxy2000_v2_full.csv')
    reg=csv('data/registry.csv')
    digest=hashlib.sha256((ROOT/'data/proxy2000_v2/fad_proxy2000_v2_full.csv').read_bytes()).hexdigest()
    assert digest in reg.sha256.str.lower().tolist()
    seq=d.Sequence.astype(str)
    assert seq.str.len().eq(141).all() and seq.nunique()==2000
    counts=seq.map(lambda s:sum(s[p-1]!=WT_141[p-1] for p in MUT_POS))
    assert all(all(s[i]==WT_141[i] for i in range(141) if i+1 not in MUT_POS) for s in seq)
    PARTS['DATA_CONTRACT']=table(['项目','本次读取结果'],[
        ['标准表行/列',f'{d.shape[0]} / {d.shape[1]}'],['`Sequence` 长度/唯一序列','141 / 2000'],
        ['`Full` 与 `Sequence`','`Full` 是较长历史字段；搜索使用 141 nt 的 `Sequence`，不混淆两列'],
        ['13 位点之外骨架','2000 条全部与代码 WT 一致'],['实际数据 SHA256',f'`{digest.upper()}`'],
        ['登记表核验','匹配'],['2000 / 理论全空间比例',f'{2000/4**13*100:.6f}%（不是合法壳覆盖率）']])
    mt=pd.DataFrame({'n_from_wt':range(14),'count':[int(counts.eq(i).sum()) for i in range(14)]})
    mt['fraction_pct']=mt['count']/len(d)*100
    dump_csv(mt,'A01_mutation_counts')
    dump_csv(d[['gbsa','dock']], 'A01_observed_labels')
    PARTS['MUTATION_TABLE']=table(['相对 WT 突变数','条数','占 2000 条比例'],[
        ['0–3',int(counts.le(3).sum()),f'{counts.le(3).mean()*100:.2f}%']]+
        [[i,int(counts.eq(i).sum()),f'{counts.eq(i).mean()*100:.2f}%'] for i in range(4,14)])+f'\n\n平均突变数 **{counts.mean():.4f}**；数据 4/5 突变分别只有 {counts.eq(4).sum()}/{counts.eq(5).sum()} 条。0–3组中的1条为0突变WT参考，1–3突变各为0条；不能把参考条目当作这些突变区间已有覆盖。'
    PARTS['LABEL_AUDIT']=table(['核查量','本次重算数值','解释'],[
        ['复合公式最大绝对残差',f'{np.abs(d.label-(.961*d.Gap2+.039*d.gbsa)).max():.3e}','公式成立，不等于权重有效'],
        ['Spearman(Gap2, GBSA)',fmt(sp(d.Gap2,d.gbsa)),'表内弱关联'],
        ['Spearman(MFE, GBSA)',fmt(sp(d.MFE,d.gbsa)),'不作因果判断'],
        ['Spearman(docking, GBSA)',fmt(sp(d.dock,d.gbsa)),'两个评分不能简单互换'],
        ['GBSA：最小 / 最大 / SD',f'{fmt(d.gbsa.min())} / {fmt(d.gbsa.max())} / {fmt(d.gbsa.std(ddof=1))}','SD 为样本标准差'],
        ['docking：最小 / 最大 / SD',f'{fmt(d.dock.min())} / {fmt(d.dock.max())} / {fmt(d.dock.std(ddof=1))}','不同量纲下方差不等于噪声大小'],
        ['GBSA >0 的行数',int(d.gbsa.gt(0).sum()),'仅描述符号；不把正值直接判为错误']])
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    axs[0].bar(mt.n_from_wt,mt['count'],color='#0072B2')
    axs[0].set(xlabel='相对参考序列的突变位点数',ylabel='历史表序列数',title='2000 条序列覆盖的突变壳')
    axs[0].set_xticks(range(14));axs[0].grid(axis='y',alpha=.2)
    for ax,col,c in zip(axs[1:],['gbsa','dock'],['#D55E00','#009E73']):
        ax.hist(d[col],bins=40,color=c,edgecolor='white')
        ax.set(xlabel='GBSA（kcal/mol）' if col=='gbsa' else '历史 docking 分数（程序标度）',ylabel='条数',title=f'{col} 列：各自横轴')
    fig.suptitle('既有标准表的实际统计；没有新增测量')
    save(fig,'A01_实际数据分布')

    rep=js('evidence/audit_20260924/server_results/reproduce.json')
    pf=js('evidence/audit_20260924/server_results/paper_fusion.json')
    deep=js('evidence/audit_20260924/server_results/active_deep.json')
    ptable=[['异域语料编码器历史单次',fmt(rep['spearman']),'1 次 80/20 划分；1600 训练；30 成员','探索单次，非 30/100 确认'],
        ['同域突变语料+物理融合',fmt(pf['spearman_mean']),f"5 种子；SD {fmt(pf['spearman_std'])}；192 物理维",'探索结果，转导/筛选口径受审计限制'],
        ['历史深度集成 active 曲线',f"起点 {fmt(deep['curve_active'][0])}；末点 {fmt(deep['curve_active'][-1])}",'初始 400，每轮 100，10 成员','仅留历史，不用于 AL 优劣'],
        ['同包 random 曲线',f"起点 {fmt(deep['curve_random'][0])}；末点 {fmt(deep['curve_random'][-1])}",'同包历史对照','旧有效性结论撤回']]
    PARTS['PAPER_RESULTS']=table(['运行/方法','GBSA Spearman','实际条件','证据等级'],ptable)+'\n\n'+cite('evidence/audit_20260924/server_results/reproduce.json')+'；'+cite('evidence/audit_20260924/server_results/paper_fusion.json')+'；'+cite('evidence/audit_20260924/server_results/active_deep.json')
    pfig=pd.DataFrame({'condition':['异域历史单次','同域融合历史 5 种子均值'],'spearman':[rep['spearman'],pf['spearman_mean']]})
    dump_csv(pfig,'A02_historical_replication')
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    ax.barh(pfig.condition,pfig.spearman,color=['#999999','#0072B2'])
    for i,v in enumerate(pfig.spearman):ax.text(v+.005,i,fmt(v),va='center')
    ax.set(xlim=(0,.35),xlabel='GBSA Spearman（历史协议不同，不作配对增益）',title='论文方法迁移：现存历史结果与评价边界')
    ax.text(.34,.02,'单次与 5 种子，均未达到发布确认口径',transform=ax.transAxes,ha='right',fontsize=10)
    save(fig,'A02_论文迁移历史结果')

    v8=csv('evidence/server_sync_20260902/v8_fusion_100seed/all_results.csv')
    base=csv('evidence/audit_20260924/audit_baseline_rf_100seed/metrics.csv')
    paired=js('evidence/audit_20260924/paired_fusion_vs_baseline.json')
    fusion=v8[v8.method.eq('equal_weight_rank')].set_index('seed').gbsa_Sp.sort_index()
    singles=v8[v8.method.eq('single')].pivot(index='seed',columns='model',values='gbsa_Sp').sort_index()
    # The baseline column is intentionally identified by the saved schema.
    basecol=next(c for c in ['gbsa_spearman','gbsa_Sp','spearman'] if c in base.columns)
    baseline=base.set_index('seed')[basecol].sort_index()
    delta=(fusion-baseline).to_numpy()
    assert len(delta)==100 and np.isclose(delta.mean(),paired['paired_delta_mean'])
    rows=[[m,fmt(singles[m].mean()),fmt(singles[m].std(ddof=1)),'100 种子，同 v8 表'] for m in ['RF','ET','GBR','XGB','SVR']]
    rows += [['mutation-only RF',fmt(paired['rf_mutation_only_mean']),fmt(paired['rf_mutation_only_std']),'100 种子；正式 CI [0.3557, 0.3725]'],
        ['v8 等权秩融合',fmt(fusion.mean()),fmt(fusion.std(ddof=1)),'100 种子；正式 CI [0.3669, 0.3823]']]
    best=singles.max(axis=1)
    PARTS['V8_RESULTS']=table(['固定方法','均值 ρ','跨种子 SD','条件/区间'],rows)+'\n\n'+table(['配对参照','平均 Δρ','95% CI','胜种子'],[
        ['v8−mutation-only RF',signed(paired['paired_delta_mean']),interval(paired['paired_bootstrap_ci95']),f"{paired['fusion_wins']}/100"],
        ['v8−固定 XGB（可部署参照）',signed((fusion-singles.XGB).mean()),'[−0.00148, +0.00652]（历史审计区间）','54/100'],
        ['v8−每种子最佳单模型（事后参照）',signed((fusion-best).mean()),'[−0.0082, −0.0033]','44/100']])+'\n\n'+cite('evidence/server_sync_20260902/v8_fusion_100seed/all_results.csv')+'；'+cite('evidence/audit_20260924/paired_fusion_vs_baseline.json')
    PARTS['V8_RESULTS']+='\n\n本次独立重算 v8−固定 XGB 的配对均值 **+0.00245161**、胜 **54/100**；用固定自举种子 20260924、20000 次重采样得到 CI **'+interval(audit['v8_comparators']['fixed_XGB']['ci95_recomputed'],6)+'**。与历史区间端点的小差别是自举抽样实现/种子不同造成的数值差，两个区间均跨零；不改写旧发布区间。'+cite('evidence/report_review_20261006/method_verification.json','本次方法数值核验')
    assert sum((fusion-best)>0)==44
    means=[baseline.mean(),singles.XGB.mean(),fusion.mean()]
    cis=[bootstrap(baseline),bootstrap(singles.XGB),bootstrap(fusion)]
    dump_csv(pd.DataFrame({'seed':fusion.index,'v8':fusion.values,'baseline_rf':baseline.values,'delta':delta}),'A03_v8_paired')
    dump_csv(pd.DataFrame({'method':['mutation-only RF','单 XGB','v8 秩融合'],'mean':means,'ci_low':[c[0] for c in cis],'ci_high':[c[1] for c in cis]}),'A03_means_recomputed_ci')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    axs[0].errorbar(means,range(3),xerr=np.array([[m-c[0] for m,c in zip(means,cis)],[c[1]-m for m,c in zip(means,cis)]]),fmt='o',capsize=5,color='#0072B2')
    axs[0].set_yticks(range(3),['mutation-only RF','单 XGB','v8 秩融合']);axs[0].set(xlabel='GBSA Spearman；100 种子均值',title='固定方法（图中区间重新描述统计）')
    axs[0].grid(axis='x',alpha=.2)
    axs[1].bar(range(100),np.sort(delta),color=['#0072B2' if x>0 else '#D55E00' for x in np.sort(delta)])
    axs[1].axhline(0,color='black',lw=1);axs[1].set(xlabel='100 个配对差按值排序',ylabel='v8−mutation-only RF 的 Δρ',title='配对 +0.0104；66/100 胜')
    save(fig,'A03_正式GBSA代理基准')

    fs=csv('evidence/server_sync_20260903/feature_selection_all_safe_30seed/metrics_paired.csv')
    PARTS['FEATURE_SELECTION']=table(['筛选方法','模型','对照均值','筛选均值','配对增量','95% CI','胜/30'],[
        [r.method,r.model,fmt(r.control_mean_spearman),fmt(r.mean_spearman),signed(r.mean_delta_vs_all),interval([r.bootstrap_ci95_low,r.bootstrap_ci95_high]),r.wins_vs_all]
        for r in fs.itertuples()])+'\n\n'+cite('evidence/server_sync_20260903/feature_selection_all_safe_30seed/metrics_paired.csv')
    cl=csv('evidence/server_sync_20260903/feature_selection_all_safe_30seed/cluster_split_check_summary.csv')
    PARTS['CLUSTER_TABLE']=table(['聚类划分：筛选/模型','筛选均值','对照均值','平均差','最差折差','记录中平均判据通过'],[
        [f'{r.method}/{r.model}',fmt(r.mean_spearman),fmt(r.control_mean_spearman),signed(r.mean_delta),signed(r.min_delta),str(r.passes_max_drop)] for r in cl.itertuples()])+'\n\n'+cite('evidence/server_sync_20260903/feature_selection_all_safe_30seed/cluster_split_check_summary.csv')
    PARTS['FEATURE_COUNTS']=table(['筛选方法','最终特征数：最小 / 中位 / 最大','为什么不是统一 100 维'],[
        [k,f"{int(v['min'])} / {v['median']:g} / {int(v['max'])}",'内折 top-100 后按 3/5 一致性保留，再强制并入 52 个突变特征']
        for k,v in audit['feature_selection_actual_counts'].items()])
    sm=js('evidence/audit_20260924/ml_squeeze/smoothing_variants.json')
    rk=js('evidence/audit_20260924/ml_squeeze/rank_objectives2.json')
    sm100=js('evidence/audit_20260924/audit_label_smoothing_100seed_results.json')
    fsm=js('evidence/audit_20260924/ml_squeeze/fusion_smoothing.json')
    PARTS['SMOOTH_RANK_TABLES']='**30 种子平滑配对：**\n\n'+table(['配置','均值','Δ vs raw','95% CI','胜/30'],[
        [k,fmt(v['mean']),signed(v['paired_delta_vs_raw']),interval(v['paired_ci95']),v['wins']] for k,v in sm['summary'].items()])+'\n\n**30 种子损失比较：**\n\n'+table(['配置','均值','Δ vs MSE','95% CI','胜/30'],[
        [k,fmt(v['mean']),signed(v['paired_delta_vs_mse']),interval(v['paired_ci95']),v['wins']] for k,v in rk['summary'].items()])+'\n\n**确认与组合：**\n\n'+table(['实验','原始/试验均值','配对差','95% CI','胜/种子'],[
        ['100 种子近邻平滑',f"{fmt(sm100['original']['mean'])} / {fmt(sm100['smoothed']['mean'])}",signed(sm100['paired_delta_mean']),interval(sm100['paired_bootstrap_ci95']),f"{sm100['wins']}/100"],
        ['30 种子 v8×平滑',f"{fmt(fsm['summary']['raw_fusion']['mean'])} / {fmt(fsm['summary']['smooth_fusion']['mean'])}",signed(fsm['summary']['paired_delta']['mean']),interval(fsm['summary']['paired_delta']['ci95']),'13/30']])+'\n\n'+ '；'.join(cite(p) for p in ['evidence/audit_20260924/ml_squeeze/smoothing_variants.json','evidence/audit_20260924/ml_squeeze/rank_objectives2.json','evidence/audit_20260924/audit_label_smoothing_100seed_results.json','evidence/audit_20260924/ml_squeeze/fusion_smoothing.json'])
    neg=js('evidence/audit_20260924/ml_squeeze/negative_controls.json')
    epi=js('evidence/audit_20260924/ml_squeeze/sparse_epistasis.json')
    ext=js('evidence/audit_20260924/ml_squeeze/extrapolation.json')
    PARTS['NEGATIVE_EPI_EXTRAP']='**负对照，10 种子：**\n\n'+table(['配置','Spearman 均值','SD'],[[k,fmt(v['mean']),fmt(v['std'])] for k,v in neg.items()])+'\n\n**稀疏交互，30 种子：**\n\n'+table(['模型/特征','均值','SD'],[[k,fmt(epi[k]['mean']),fmt(epi[k]['std'])] for k in ['rf_52d','lasso_epistasis','xgb_495d']])+'\n\n**突变数域留出（前三行是指定域单次协议）：**\n\n'+table(['协议','训练/测试数','测试突变数','Spearman'],[
        [k,f"{ext[k]['n_train']}/{ext[k]['n_test']}",str(ext[k]['test_count_range']),fmt(ext[k]['spearman'])] for k in ['high_out','mid_out','low_out']]+[['随机对照（30 种子）','1600/400','混合域',fmt(ext['random_control']['spearman_mean'])]])+'\n\n'+ '；'.join(cite(p) for p in ['evidence/audit_20260924/ml_squeeze/negative_controls.json','evidence/audit_20260924/ml_squeeze/sparse_epistasis.json','evidence/audit_20260924/ml_squeeze/extrapolation.json'])
    fig,axs=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    panels=[('标签平滑：30 种子',[(k,v['mean']) for k,v in sm['summary'].items()]),
            ('损失目标：30 种子',[(k,v['mean']) for k,v in rk['summary'].items()]),
            ('突变数域外推：单次域/随机均值',[(k,ext[k]['spearman']) for k in ['high_out','mid_out','low_out']]+[('random30',ext['random_control']['spearman_mean'])])]
    figure_names={'raw':'未平滑','LS_k5_a0.5':'近邻平滑 k=5，α=0.5',
        'NLS_a0.25':'负平滑 α=0.25','NLS_a0.5':'负平滑 α=0.5',
        'xgb_mse':'XGB 均方误差','xgb_quantile50':'XGB 中位数分位损失',
        'lgbm_lambdarank':'LightGBM 整组排序','xgbranker_subsample':'XGB 成对排序（500条）',
        'high_out':'留出12–13突变','mid_out':'留出10–11突变','low_out':'留出≤9突变','random30':'随机切分30种子'}
    for ax,(title,rs) in zip(axs,panels):
        ax.barh([figure_names[r[0]] for r in rs],[r[1] for r in rs],color='#0072B2')
        for i,(_,v) in enumerate(rs):ax.text(v+.006,i,fmt(v,3),va='center',fontsize=10)
        ax.set(xlim=(0,.48),xlabel='GBSA Spearman',title=title);ax.grid(axis='x',alpha=.2)
    save(fig,'A04_代理改进与外推压力')
    dump_csv(pd.DataFrame([{'panel':t,'condition':k,'spearman':v} for t,rs in panels for k,v in rs]),'A04_model_branches')
    nf=js('evidence/audit_20260924/server_results/new_features.json')
    fw=js('evidence/audit_20260924/server_results/feature_weighting.json')
    fwmean=np.mean([r['weighted_fusion'] for r in fw['results']])
    cands=js('evidence/audit_20260924/ml_squeeze/candidate_demo_summary.json')
    defect=js('evidence/audit_20260924/eval_defect_features_results.json')
    PARTS['DEFECT_RESULTS']=table(['旧特征配置（公式存在问题）','Spearman均值','Δ vs base','95% CI','胜/30'],[
        [k,fmt(v['mean'],6),signed(v['paired_delta_vs_base'],6),interval(v['paired_ci95'],6),v['wins']]
        for k,v in defect['summary'].items()])+'\n\n'+cite('evidence/audit_20260924/eval_defect_features_results.json')+'；'+cite('evidence/audit_20260924/ensemble_defect_features.csv','旧特征表（需重算）')
    PARTS['OTHER_ATTEMPTS']=table(['尝试','实际结果/产物','目前能说什么'],[
        ['近似 Turner 堆叠能 52 维',f"base {fmt(nf['base'])}；加入 {fmt(nf['base_plus_turner'])}；Δ {signed(nf['delta'])}",'历史特征配置无增益，不是精准热力学重新计算'],
        ['历史303维序列基线',f"30种子；均值 {fmt(np.mean([r['baseline'] for r in fw['results']]))}",'同一feature_weighting探索包；不是100种子发布基准'],
        ['历史7子块硬选择的8模型秩融合',f"30种子；均值 {fmt(np.mean([r['hard_useful7'] for r in fw['results']]))}",'基线+七个“序列/单子块”模型；不是单模型303+192维拼接；子块全数据来源限制'],
        ['18 子块+基线加权融合',f"保存 {len(fw['results'])} 个种子；均值 {fmt(fwmean)}；权重接近均匀",'历史预测级混合；不是新发布的有效子块发现'],
        ['三维模板距离特征','历史 3K1V 模板与本任务序列/长度不匹配','没有有效正式成绩，不补零或正收益'],
        ['ensemble defect（系综缺陷）',f"旧30种子 base {fmt(defect['summary']['base']['mean'])}；+defect {fmt(defect['summary']['base+defect']['mean'])}",'本次补回漏写数值，但发现未配对概率公式错误；修正后必须重算再评估'],
        ['31,000 条候选的三分量排序示例','序列 GBSA 预测+dock 预测+MFE 排名；top-200；曾有 1 条训练重叠','历史演示，有目标混合，当前不沿用；没有新真标签'],
        ['锁定 seed=42 发布包','验证 0.316 / 测试 0.451','单次锁定产物，不替代多种子发布基准']])+'\n\n'+cite('evidence/audit_20260924/server_results/new_features.json')+'；'+cite('evidence/audit_20260924/server_results/feature_weighting.json')+'；'+cite('evidence/audit_20260924/ml_squeeze/candidate_demo_summary.json')

    dock=csv('evidence/audit_20260924/formal_dock_fusion_100seed/metrics.csv')
    dockopt=js('evidence/audit_20260924/server_results/dock_optimize.json')
    PARTS['DOCK_TARGET_TABLES']='**历史 30 种子特征实验：**\n\n'+table(['特征','docking Spearman','SD'],[[k,fmt(v['mean']),fmt(v['std'])] for k,v in dockopt.items()])+'\n\n**100 种子对接目标确认：**\n\n'+table(['固定模型','docking Spearman 均值','SD'],[[c,fmt(dock[c].mean()),fmt(dock[c].std(ddof=1))] for c in ['RF','ET','GBR','XGB','rank_fusion','fusion_smooth']])+'\n\n四模型 rank_fusion 正式 95% CI **[0.5456, 0.5590]**。'+cite('evidence/audit_20260924/formal_dock_fusion_100seed/metrics.csv')+'；'+cite('evidence/audit_20260924/server_results/dock_optimize.json')
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    trows=[('独立 GBSA：v8五模型秩融合',float(fusion.mean()),[.3669,.3823]),
           ('docking 分数：四模型秩融合',float(dock.rank_fusion.mean()),[.5456,.5590])]
    for ax,(label,m,ci) in zip(axs,trows):
        ax.bar([0],[m],color='#0072B2',width=.4)
        ax.errorbar([0],[m],yerr=[[m-ci[0]],[ci[1]-m]],fmt='none',color='black',capsize=6)
        ax.set(ylim=(0,.65),xlim=(-.6,.6),ylabel='对应目标的 Spearman 排序相关',title=label)
        ax.set_xticks([0],['100个随机种子；不是100次新测量'])
        ax.text(0,m+.025,f'{m:.4f}\n[{ci[0]:.4f}, {ci[1]:.4f}]',ha='center',fontsize=11)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('两个不同目标分别报告；更易预测不等于更准确的物理指标')
    save(fig,'A12_两种目标的正式预测参照')
    dump_csv(pd.DataFrame([{'target':r[0],'mean':r[1],'formal_ci_low':r[2][0],'formal_ci_high':r[2][1]} for r in trows]),'A12_separate_target_references')
    dn=js('evidence/audit_20260924/audit_dock_denoise_fixed_results.json')
    dr=js('evidence/audit_20260924/audit_dock_rank_corr_fixed_results.json')
    PARTS['DOCK_OLD_CORRECTION']=table(['修复后的 GBSA 比较','Spearman 均值','协议'],[[k,fmt(v['mean']),'30 种子折内变换，独立配置'] for k,v in dn.items()]+[[k,fmt(v),'30 种子排序修正版'] for k,v in dr.items()])+'\n\n'+cite('evidence/audit_20260924/audit_dock_denoise_fixed_results.json')+'；'+cite('evidence/audit_20260924/audit_dock_rank_corr_fixed_results.json')
    dockconf=js('evidence/search_suite_20261005/dock_confirmation/summary.json')
    dockraw=csv('runs/local_dock_residual_confirmation_100seed_20261005/metrics.csv')
    for k,col in [('predicted_dock','predicted_dock_delta_sp'),('observed_dock','observed_dock_delta_sp')]:
        assert np.isclose(dockraw[col].mean(),dockconf[k]['mean_delta_sp'])
    PARTS['DOCK_RESIDUAL_TABLE']=table(['臂','30 种子开发 Δρ','100 种子确认 Δρ','确认 95% CI','胜/100','部署前提'],[
        ['预测 docking 残差','−0.00015；CI [−0.00061, +0.00030]',signed(dockconf['predicted_dock']['mean_delta_sp'],6),interval(dockconf['predicted_dock']['paired_95ci'],6),dockconf['predicted_dock']['wins'],'只靠训练拟合得到 docking 预测'],
        ['已观测 docking 残差','约 +0.05847',signed(dockconf['observed_dock']['mean_delta_sp'],6),interval(dockconf['observed_dock']['paired_95ci'],6),dockconf['observed_dock']['wins'],'测试/新候选真实 docking 可取得']])+'\n\n'+cite('evidence/search_suite_20261005/dock_confirmation/summary.json')+'；'+cite('runs/local_dock_residual_confirmation_100seed_20261005/metrics.csv')
    PARTS['DOCK_RESIDUAL_ABSOLUTE']='**同一100种子上的绝对指标（跨种子均值）：**\n\n'+table(['臂','GBSA Spearman','真实top-10%召回','选20条中最低真实GBSA'],[
        [name,fmt(dockraw[spcol].mean()),fmt(dockraw[recol].mean()),fmt(dockraw[bestcol].mean())]
        for name,spcol,recol,bestcol in [
            ('只用序列基础','baseline_sp','baseline_top10_recall','baseline_best20'),
            ('加预测dock残差','predicted_dock_resid_sp','predicted_dock_top10_recall','predicted_dock_best20'),
            ('加已观测dock残差','observed_dock_resid_sp','observed_dock_top10_recall','observed_dock_best20')]])+'\n\n真实top-10%为400条测试中的40条；召回按预测前40条计算，最低标签另按预测前20条计算。两者不是同一个候选配额。'
    register('scripts/reproduction/dock_residual_screen.py')
    register('configs/experiments/dock_residual_confirmation_100seed_20261005.json')
    PARTS['DOCK_PREFILTER']=table(['保留候选比例','保留真 top-10% 比例','最佳 20 中最低 GBSA 差','95% CI','胜/平/负','需要的 docking'],[
        [f'{k}%',f"{v['mean_true_top10_retained']*100:.2f}%",signed(v['mean_best20_delta']),interval(v['paired_95ci']),f"{v['wins']}/{v['ties']}/{v['losses']}",'全部 400 测试候选'] for k,v in dockconf['prefilter'].items()])+'\n\n'+cite('evidence/search_suite_20261005/dock_confirmation/summary.json')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for i,k in enumerate(['predicted_dock','observed_dock']):
        m=dockconf[k]['mean_delta_sp'];lo,hi=dockconf[k]['paired_95ci']
        axs[0].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=5,color=['#D55E00','#0072B2'][i])
        axs[0].annotate(f'{m:+.6f} [{lo:+.6f}, {hi:+.6f}]',(m,i),xytext=(0,15),textcoords='offset points',ha='center',fontsize=9)
    axs[0].set_yticks([0,1],['预测 docking 残差','真实 docking 残差']);axs[0].axvline(0,color='gray',ls='--');axs[0].set(xlim=(-.025,.09),ylim=(-.45,1.5),xlabel='相对 GBSA-only 的 ΔSpearman',title='100 种子：不同信息可用性')
    vals=[v['mean_true_top10_retained']*100 for v in dockconf['prefilter'].values()]
    axs[1].plot([25,50,75],vals,'o-',color='#0072B2')
    for x,y in zip([25,50,75],vals):axs[1].annotate(f'{y:.2f}%',(x,y),xytext=(0,9),textcoords='offset points',ha='center')
    axs[1].set(xlabel='按已观测 docking 保留候选比例（%）',ylabel='保留真实 GBSA 最佳 10% 的比例（%）',ylim=(40,100),title='预筛会遗漏部分真好候选');axs[1].grid(alpha=.2)
    save(fig,'A05_对接修正与预筛');dump_csv(dockraw,'A05_dock_confirmation_rows')

    static=js('evidence/loop_20260930/strategy_compare_100seed/summary.json')
    sr=csv('evidence/loop_20260930/strategy_compare_100seed/runs.csv')
    last=sr[sr['round'].eq(5)].pivot(index='seed',columns='strategy',values='regret')
    sd=last.greedy-last.random
    sg=static['paired_vs_random_final_round']['greedy']
    assert np.isclose(sd.mean(),sg['delta_regret_mean'])
    assert (sd.lt(-1e-8).sum(),sd.abs().le(1e-8).sum(),sd.gt(1e-8).sum())==(50,37,13)
    per=pd.DataFrame(static['per_round'])
    PARTS['STATIC_ROUNDS']=table(['策略','第1轮平均 regret','第2轮','第3轮','第4轮','第5轮'],[[p]+[fmt(per[(per.strategy.eq(p))&per['round'].eq(i)].mean_regret.iloc[0]) for i in range(1,6)] for p in static['strategies']])+'\n\n'+cite('evidence/loop_20260930/strategy_compare_100seed/summary.json')
    PARTS['STATIC_FINAL']=table(['第5轮 greedy−random 指标','配对均值/数值','95% CI / 分布'],[
        ['regret',signed(sg['delta_regret_mean']),interval(sg['delta_regret_ci95'])],
        ['胜/平/负',f"{sg['regret_wins']}/{sg['regret_ties']}/{sg['regret_losses']}",f"中位差 {fmt(sg['delta_regret_median'])}"],
        ['改善最大10个均值',signed(sg['best10pct_delta_regret_mean']),f"占总体均值差 {sg['best10pct_share_of_mean']*100:.2f}%"],
        ['表现最差10个均值',signed(sg['worst10pct_delta_regret_mean']),'最正的十个配对差'],
        ['真实 top-10% 召回',signed(sg['delta_top10_recall_mean']),interval(sg['delta_top10_recall_ci95'])+f"；{sg['recall_wins']}/100 胜"],
        ['holdout Spearman',signed(sg['delta_holdout_spearman_mean']),interval(sg['delta_holdout_spearman_ci95'])],
        ['累计新标签 top-5 中位数',signed(sg['delta_top5_median_new_mean']),interval(sg['delta_top5_median_new_ci95'])]])
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for p,c in zip(static['strategies'],['#777777','#0072B2','#009E73','#D55E00']):
        sub=per[per.strategy.eq(p)].sort_values('round');axs[0].plot(sub['round'],sub.mean_regret,'o-',label=p,color=c)
    axs[0].set(xlabel='读取历史标签的轮次（每轮50条）',ylabel='平均 regret（历史 GBSA 标度）',title='固定600条池；不是多代变异');axs[0].legend(fontsize=9);axs[0].grid(alpha=.2)
    ss=sd.sort_values().to_numpy();axs[1].bar(range(100),ss,color=['#0072B2' if v< -1e-8 else '#D55E00' if v>1e-8 else '#999999' for v in ss])
    axs[1].axhline(0,color='black',lw=1);axs[1].set(xlabel='100 个种子配对差排序',ylabel='Δregret：greedy−random',title='50 胜 / 37 平 / 13 负；均值 −1.8781')
    save(fig,'A06_固定池选样确认');dump_csv(per,'A06_round_means');dump_csv(pd.DataFrame({'seed':sd.index,'delta_regret':sd.values}),'A06_paired_final')

    sigma=js('evidence/sigma_calibration_30seed/summary.json')
    keys=['spearman_mu','spearman_sigma_error','spearman_sigma_error_shuffled','mean_abs_error',
          'delta_recall_lcb0.5_k0.05','delta_recall_lcb1.0_k0.05','delta_recall_lcb2.0_k0.05',
          'delta_recall_lcb0.5_k0.1','delta_recall_lcb1.0_k0.1','delta_recall_lcb2.0_k0.1']
    PARTS['SIGMA_TABLE']=table(['指标','400训练：均值 [95% CI]','1600训练：均值 [95% CI]'],[
        [k]+[fmt(sigma[f'n_train_{n}'][k]['mean'])+' '+interval(sigma[f'n_train_{n}'][k]['ci95']) for n in [400,1600]] for k in keys])+'\n\n`k0.05/k0.1` 是前 5%/10% 召回终点，`lcb0.5/1.0/2.0` 是 β 值，β 为 sigma 的权重。'+cite('evidence/sigma_calibration_30seed/summary.json')
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    sigma_rows=[]
    for i,n in enumerate([400,1600]):
        s=sigma[f'n_train_{n}']
        for shift,k,col in [(-.12,'spearman_sigma_error','#0072B2'),(.12,'spearman_sigma_error_shuffled','#D55E00')]:
            v=s[k];m=v['mean'];lo,hi=v['ci95'];axs[0].errorbar(m,i+shift,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color=col,label=('真实标签' if shift<0 else '打乱标签') if i==0 else None)
        dec=pd.DataFrame(s['decile_table']);axs[1].plot(dec.decile,dec.mean_abs_error,'o-',label=f'{n} 条训练')
        for k in keys:sigma_rows.append({'n_train':n,'metric':k,**s[k]})
    axs[0].set_yticks([0,1],['400 条训练','1600 条训练']);axs[0].set(xlabel='σ 与绝对误差的 Spearman',title='成员分歧并不是可信误差信号');axs[0].axvline(0,color='gray',ls='--');axs[0].legend(fontsize=9)
    axs[1].set(xlabel='按 sigma 从小到大的十分位',ylabel='平均绝对误差（kcal/mol）',title='误差未形成明显单调分层');axs[1].legend();axs[1].grid(alpha=.2)
    save(fig,'A07_不确定性校准门槛');dump_csv(pd.DataFrame(sigma_rows),'A07_sigma_saved_summary')

    fourdev=js('evidence/search_suite_20261005/development/summary.json')
    fourconf=js('evidence/search_suite_20261005/confirmation/summary.json')
    sevdev=js('evidence/search_seven_20261005/development/summary.json')
    sevconf=js('evidence/search_seven_20261005/confirmation/summary.json')
    aligned=csv('evidence/search_seven_20261005/confirmation/aligned.csv')
    seven_runs=csv('evidence/search_seven_20261005/confirmation/runs.csv')
    pwraw=csv('evidence/search_seven_20261005/confirmation/pairwise.csv')
    stabraw=csv('evidence/search_seven_20261005/development/stability.csv')
    for cell in sevconf['cells']:
        for m in cell['means']:
            rr=aligned[aligned.proxy.eq(cell['proxy'])&aligned.policy.eq(m['policy'])]
            assert len(rr)==100 and np.isclose(rr.best_pred_gbsa.mean(),m['best_pred_gbsa'])
        assert cell['common_n']==[int(rr.common_n.min()),float(rr.common_n.median()),int(rr.common_n.max())]
    for r in sevconf['pairwise_prefix']:
        sub=pwraw[pwraw.proxy.eq(r['proxy'])&pwraw.policy.eq(r['policy'])&pwraw.baseline.eq(r['baseline'])]
        assert len(sub)==100 and np.isclose(sub.delta.mean(),r['mean_delta'])
    PARTS['FIT_RF']=fmt(next(x['holdout_spearman'] for x in sevdev['fits'] if x['proxy']=='rf'))
    PARTS['FIT_GBM']=fmt(next(x['holdout_spearman'] for x in sevdev['fits'] if x['proxy']=='xgb'))
    cfg=sevconf['config']['policy_config']
    PARTS['FROZEN_PARAMETERS']=table(['当前参数','冻结值','用途'],[
        ['共享已测/随机起点',f"{cfg['n_measured_seeds']} / {cfg['n_random_seeds']}",'初始化16条并计分'],
        ['种群 / 每代子代',f"{cfg['population_size']} / {cfg['offspring_size']}",'GA群体及提出量，合法/重复过滤后实际可更少'],
        ['锦标赛 / 精英',f"{cfg['tournament_size']} / {cfg['elite_count']}",'单父本选择和历史较优保留'],
        ['单跳比例 / 重启比例',f"{cfg['one_hop_fraction']} / {cfg['restart_fraction']}",'原四法相关提出设置；不能强加到新法'],
        ['束宽 / 每父本提出量',f"{cfg['beam_width']} / {cfg['offspring_per_parent']}",'束和相关局部候选参数'],
        ['多样性距离 / 池扩展倍率',f"{cfg['diversity_distance']} / {cfg['diversity_pool_factor']}",'原种群/束筛选'],
        ['GA连续停滞代数','4','GA生存群体停滞护栏'],
        ['最新七法最大代数',sevconf['config']['max_generations'],'原四法64；七法全部256'],
        ['ILS扰动位点数',cfg['ils_perturb_sites'],'2位点扰动'],
        ['SA温度初尺度 / 终比例',f"{cfg['sa_temperature_scale']} / {cfg['sa_final_temperature_ratio']}",'训练IQR尺度，按实际次数降温'],
        ['AdaLead容差 / 父本上限 / rollout',f"{cfg['adalead_tolerance_scale']} / {cfg['adalead_parent_cap']} / {cfg['adalead_rollouts']}",'加性容差，固定模型无交叉']])+'\n\n'+cite('evidence/search_seven_20261005/confirmation/suite_config.json')
    def means_table(summary):
        rows=[]
        for cell in summary['cells']:
            mm={m['policy']:m for m in cell['means']}
            for p in summary['policies']:
                m=mm[p];rows.append([PROXIES[cell['proxy']],NAMES[p],fmt(m['best_pred_gbsa']),
                    '/'.join(fmt(x,1) if isinstance(x,float) else str(x) for x in cell['common_n']),fmt(m['actual_predictions'],2)])
        return table(['评分器','算法','共同次数内最低预测均值','共同实际N：最小/中位/最大','整次运行实际预测均值'],rows)
    def paired_table(summary):
        return table(['评分器','算法−对照','平均差','中位差','95% CI','胜/平/负'],[
            [PROXIES[cell['proxy']],NAMES[p['policy']]+'−'+NAMES[p['baseline']],signed(p['mean_delta']),signed(p['median_delta']),interval(p['ci95']),f"{p['wins']}/{p['ties']}/{p['losses']}"] for cell in summary['cells'] for p in cell['paired']])
    PARTS['FOUR_CONFIRM']=means_table(fourconf)+'\n\n'+paired_table(fourconf)+'\n\n'+cite('evidence/search_suite_20261005/confirmation/summary.json')
    PARTS['SEVEN_CONFIRM']=means_table(sevconf)+'\n\n'+cite('evidence/search_seven_20261005/confirmation/summary.json')
    PARTS['SEVEN_PAIRED']=paired_table(sevconf)+'\n\n表内最低分是预测值；负差=试验法预测更低，平局按冻结容差处理。'
    PARTS['SEVEN_PAIRWISE']=table(['评分器','算法−对照','两两N：最小/中位/最大','平均差','95% CI','胜/平/负'],[
        [PROXIES[p['proxy']],NAMES[p['policy']]+'−'+NAMES[p['baseline']],'/'.join(str(x) for x in p['common_n']),signed(p['mean_delta']),interval(p['ci95']),f"{p['wins']}/{p['ties']}/{p['losses']}"] for p in sevconf['pairwise_prefix']])+'\n\n'+cite('evidence/search_seven_20261005/confirmation/pairwise.csv')
    resrows=[]
    for proxy in ['rf','xgb']:
        for p in POLICIES:
            sub=seven_runs[seven_runs.proxy.eq(proxy)&seven_runs.policy.eq(p)]
            assert len(sub)==100
            stops='；'.join(f'{k}: {v}' for k,v in sub.stop_reason.value_counts().items())
            resrows.append([PROXIES[proxy],NAMES[p],f"{sub.actual_predictions.min()}/{fmt(sub.actual_predictions.median(),1)}/{sub.actual_predictions.max()}",
                            fmt(sub.seconds.mean(),3),fmt(sub.prediction_seconds.mean(),3),stops])
    PARTS['SEVEN_RESOURCE']=table(['评分器','算法','实际次数：最小/中位/最大','平均搜索秒数','其中平均预测秒数','100个运行的停止原因'],resrows)+'\n\n'+cite('evidence/search_seven_20261005/confirmation/runs.csv')
    def budgettable(summary):
        rs=[]
        for cell in summary['cells']:
            ms={m['policy']:m for m in cell['means']}
            rs.append([PROXIES[cell['proxy']],cell['budget'],'/'.join(str(n) for n in cell['common_n'])]+[fmt(ms[p]['best_pred_gbsa']) for p in summary['policies']])
        return table(['评分器','上限','共同N：最小/中位/最大']+[NAMES[p] for p in summary['policies']],rs)
    PARTS['BUDGET_TABLES']='**原四法开发（401–430）：**\n\n'+budgettable(fourdev)+'\n\n**最新七法开发（6001–6030）：**\n\n'+budgettable(sevdev)+'\n\n'+cite('evidence/search_suite_20261005/development/summary.json')+'；'+cite('evidence/search_seven_20261005/development/summary.json')
    PARTS['GA_ABLATIONS']=table(['评分器','消融−默认GA','平均差','95% CI','胜/平/负'],[
        [PROXIES[arm['proxy']],p['policy'],signed(p['mean_delta']),interval(p['ci95']),f"{p['wins']}/{p['ties']}/{p['losses']}"] for arm in fourdev['ablations'] for p in arm['paired']])+'\n\n30 种子 401–430，1024 上限，原四法64代条件。'+cite('evidence/search_suite_20261005/development/ablations.csv')
    PARTS['STABILITY_TABLE']=table(['评分器','算法','同类重训排序ρ','换另一代理排序ρ','重训top-50 Jaccard'],[
        [PROXIES[r['proxy']],NAMES[r['policy']],fmt(r['refit_rank_spearman']),fmt(r['cross_proxy_rank_spearman']),fmt(r['refit_top50_jaccard'])] for r in sevdev['stability']])+'\n\n'+cite('evidence/search_seven_20261005/development/stability.csv')
    for r in sevdev['stability']:
        sub=stabraw[stabraw.proxy.eq(r['proxy'])&stabraw.policy.eq(r['policy'])]
        for k in ['refit_rank_spearman','cross_proxy_rank_spearman','refit_top50_jaccard']:assert np.isclose(sub[k].mean(),r[k])
    PARTS['SEVEN_DIAGNOSTICS']=table(['评分器','算法','top-50平均两两距离','top-50 OOD比例','top-50最近训练距离'],[
        [PROXIES[cell['proxy']],NAMES[m['policy']],fmt(m['diversity_top50']),f"{m['ood_fraction_top50']*100:.2f}%",fmt(m['min_train_hamming_top50'])] for cell in sevconf['cells'] for m in cell['means']])
    PARTS['DEMO_TABLE']=table(['产物/示例','唯一预测','可导出预测候选','分组','最低预测/状态','证据性质'],[
        ['原GA独立示例',916,908,2,'6.150741；8代','单次示例'],
        ['ILS独立示例',1024,1016,2,'8.373855；预算耗尽','单次示例'],
        ['SA独立示例',1024,1016,2,'6.150741；预算耗尽','单次示例'],
        ['AdaLead独立示例',367,359,2,'6.150741；邻居耗尽','单次示例'],
        ['两组交接冒烟','见搜索档案','池100/100；批次13/12',2,'25条 PARKED，无真实回填','模拟工程，不是计算订单']])+'\n\n'+cite('evidence/search_seven_20261005/demo_audit.json')+'；'+cite('evidence/search_suite_20261005/demo/metrics.csv')+'；'+cite('runs/loop_search_group_handoff_final_20261005/metrics.csv')
    audits={}
    for family in ['search_suite_20261005','search_seven_20261005']:
        for arm in ['development','confirmation']:
            audits[family,arm]=js(f'evidence/{family}/{arm}/artifact_audit.json')
    PARTS['RUN_COUNTS']=table(['正式套件','子运行组成','子运行数','已审计实际预测记录数'],[
        ['原四法开发','720 主 +240 重训 +180 GA消融',audits['search_suite_20261005','development']['subruns'],f"{audits['search_suite_20261005','development']['evaluated_rows']:,}"],
        ['原四法确认','100×2评分器×4法',audits['search_suite_20261005','confirmation']['subruns'],f"{audits['search_suite_20261005','confirmation']['evaluated_rows']:,}"],
        ['最新七法开发','840 主 +420 重训',audits['search_seven_20261005','development']['subruns'],f"{audits['search_seven_20261005','development']['evaluated_rows']:,}"],
        ['最新七法确认','100×2评分器×7法',audits['search_seven_20261005','confirmation']['subruns'],f"{audits['search_seven_20261005','confirmation']['evaluated_rows']:,}"],
        ['最新七法开发+确认','独立保留本轮条件',2660,'3,554,757'],['新增真GBSA/新增docking/新增MD','当前全部为0','0','不产生新标签']])
    val=js('evidence/search_seven_20261005/validation.json')
    assert val['tests']['count']==139 and val['tests']['passed']
    for key in ['new_real_GBSA','new_docking','new_MD']:assert val[key]==0
    for key,au in audits.items():assert au['ok']

    PARTS['DIAGRAM_ARCHITECTURE']=diagram('A09_当前搜索与未来验证','模型可替换的当前搜索架构', '''flowchart LR
    subgraph INPUT["当前输入和固定模型"]
    direction TB
    D["2000条既有Sequence和GBSA<br/>哈希核验；划分训练与留出"] --> M["代理拟合或读取固定快照<br/>当前RF / 本地五成员GBM<br/>模型身份、特征版本、目标与方向"]
    M --> I["共有16起点<br/>8已测低值 + 8随机<br/>起点由模型重打分并计预算"]
    end
    subgraph ENGINE["当前：固定模型搜索"]
    direction TB
    P["七种策略共享接口<br/>随机游走 / 爬山 / 束搜索 / 无交叉GA<br/>ILS / SA / 无交叉AdaLead"]
    P --> V["单父本提出、约束检查与排重<br/>141nt / 13位点 / 4–13突变<br/>禁止导出标准表已有序列"]
    V --> S["统一评分器<br/>按序列顺序返回pred_gbsa<br/>合法性、缓存、唯一预测预算"]
    S --> A["逐条预测档案<br/>父本ID、序列、跳数、代际、模型与次数"]
    A --> R["按策略接受和保留下一代<br/>固定模型；不读取新真标签"]
    R --> Q{"实际预算/停滞/邻居/代数<br/>达到停止条件？"}
    Q -- 否 --> P
    Q -- 是 --> X["停止内循环<br/>保存最佳候选和完整档案"]
    end
    subgraph OUTPUT["当前输出和未来独立决策"]
    direction TB
    G["未测候选按预测排序<br/>单/双原型分组；单父本谱系<br/>输出可审查的序列建议"]
    G -. 团队未来独立决策 .-> C["统一真实GBSA协议与预算<br/>配体、QC、重复、失败、对照预注册"]
    C -. 获批后才启动 .-> L["真实GBSA计算及受控回传<br/>不是代理预测；目前未开展"]
    L -. 条件通过后 .-> U["判断真实算法效果<br/>另决定重训、主动学习和湿实验"]
    end
    INPUT --> ENGINE
    ENGINE -- 满足停止条件 --> OUTPUT
    classDef future fill:#fff3df,stroke:#c77919,stroke-dasharray:5 5;
    class C,L,U future;
    ''')
    PARTS['DIAGRAM_EVIDENCE']=diagram('A08_证据层级','当前结果与最终生物验证的证据关系','''flowchart LR
    A["已测表模型评估<br/>真标签对预测<br/>v8 100种子ρ=0.3748"] --> B["固定池回顾式选样<br/>读取隐藏历史真标签<br/>greedy Δregret=-1.8781"]
    A --> C["未测序列迭代搜索<br/>固定模型指导并给终点评分<br/>七法已完成；只是预测"]
    C -. 需新真实计算 .-> D["新增候选真实GBSA<br/>统一协议、同成本、独立批次<br/>目前新增标签为0"]
    D -. 需相应湿实验 .-> E["亲和力/功能实验<br/>按真实生物学目标设计<br/>目前没有优胜证据"]
    classDef future fill:#fff3df,stroke:#c77919,stroke-dasharray:5 5;
    class D,E future;
    ''')
    PARTS['DIAGRAM_AUDIT']=diagram('A10_结果到汇报核验','实验产物怎样支持汇报','''flowchart LR
    D["标准数据<br/>registry SHA256"] --> C["冻结配置与源代码<br/>种子、模型、预算、参数"]
    C --> E["实际环境与模型身份<br/>识别本地GBM而非误称XGBoost"]
    E --> R["逐条search_evaluated档案<br/>序列、分数、实际顺序、父本"]
    R --> V["独立验收<br/>计数/缓存/预算/骨架/谱系/排重<br/>相同起点、相同模型、完整分组"]
    V --> S["共同前缀aligned<br/>两两pairwise补充<br/>stability独立口径"]
    S --> T["逐种子汇总与区间<br/>不拼接不同条件<br/>不把预测写成真值"]
    T --> F["详细表格和数据图<br/>来源、摘要、范围与结论边界"]
    ''')
    PARTS['DIAGRAM_ROADMAP']=diagram('A11_阶段进度与触发条件','各阶段进度与后续依赖','''flowchart TB
    A["阶段A：已测表代理与论文迁移<br/>0.506重放及30/100确认已完成<br/>其余十月补实验状态见正文"] --> B1["阶段B1/B2：接口和七种搜索<br/>合同、单父本、预算、恢复已完成"]
    B1 --> B2["阶段B3/B4：公平比较与交接<br/>30开发/100确认、稳定性、谱系、分组<br/>当前计算任务可收尾"]
    A --> D["独立D支线：对接残差与预筛<br/>历史表30/100已完成<br/>候选时真实dock和成本尚待验证"]
    B2 --> G{"下一次研究触发条件"}
    G -- 有明确新代理证据 --> M["新模型身份与合同验收<br/>重新同起点、同实际次数比较"]
    M --> B2
    G -- 团队批准真实验证 --> C0["阶段C0：统一协议与预算<br/>冻结目标、QC、重复、失败、主终点"]
    D -. 候选时dock确实可得 .-> C0
    C0 -. 尚未启动 .-> C1["阶段C1：新增真实GBSA比较<br/>同成本、独立批次、多算法对照"]
    C1 -. 标签通过QC后 .-> C2["阶段C2：真实判定后另决策<br/>重训/是否主动学习/湿实验"]
    classDef future fill:#fff3df,stroke:#c77919,stroke-dasharray:5 5;
    class C0,C1,C2 future;
    ''')
    PARTS['DIAGRAM_NARRATIVE']=diagram('A13_论文到当前搜索的决策链','从论文迁移到当前算法接口的研究顺序','''flowchart TB
    P["1 论文组件迁移<br/>历史ρ：0.1228 / 0.2610<br/>有任务适配，非正式确认"] --> L
    L["2 审查主动学习<br/>旧优劣结论撤回<br/>固定池发现好，模型增益未确认"] --> S
    S["3 保留代理优化成果<br/>历史特征/模型优化至0.506<br/>v8多种子0.3748，泛化有限"] --> D
    D["4 探索docking支线<br/>对接排序ρ=0.5523<br/>真实dock修正有条件收益"] --> J
    J["5 本阶段暂缓代理扩展<br/>正式主动学习暂不启用<br/>独立GBSA目标，模型可替换"] --> I
    I["6 先完成算法接口与搜索<br/>七法、单父本变异、谱系分组<br/>同起点/实际次数确认已完成"] --> F
    F["7 未来真实算法判定<br/>统一GBSA协议与资源待定<br/>新增标签=0；以后决定重训"]
    classDef pending fill:#fff3df,stroke:#c77919,stroke-dasharray:5 5;
    class F pending;
    ''')

    PARTS['DIAGRAM_PAPER_CONTROL']=diagram('A14_预训练受控比较','预训练受控比较的训练与测试边界','''flowchart TB
    D["登记的2000条序列与独立GBSA<br/>外层30个随机种子11001至11030"] --> T["每种子1600训练 / 400测试<br/>GBSA测试标签不进入预训练或回归拟合"]
    I["三臂同一初始编码器权重<br/>随机种子13000 / 6层Transformer"] --> N["无预训练<br/>直接冻结初始编码器"]
    I --> F["外域语料2000条<br/>排除全部已知FAD序列<br/>5轮掩码语言建模与三元组学习"]
    I --> M["域内语料每种子2000条<br/>只从1600训练父本合法变异<br/>排除全部2000条已知序列 / 5轮预训练"]
    T --> M
    F --> FF["冻结外域编码器<br/>独立于划分，跨种子复用"]
    M --> MM["冻结该种子域内编码器"]
    N --> H["每臂相同头初始化及批次顺序<br/>10个独立9层256宽回归头 / 50轮<br/>只用1600条训练GBSA，训练内标准化目标"]
    FF --> H
    MM --> H
    T --> H
    T --> C["同划分经典对照<br/>303维序列特征 / 500树梯度提升<br/>不预训练，不使用NUPACK或dock"]
    H --> E["对共同400条测试序列独立预测<br/>保存真值、均值、成员分歧和分区"]
    C --> E
    E --> A["逐预测重算排序相关与误差<br/>30种子配对均值差、区间和胜平负"]
    A --> G{"域内臂对无预训练、外域和经典对照<br/>三项排序增量区间均严格为正？"}
    G -- 是 --> Y["冻结后追加100种子确认<br/>仍不等于原功能任务完整复现"]
    G -- 否 --> Z["保留开发结果并结束这组配方<br/>不据此否定所有预训练模型"]
    classDef held fill:#fff3df,stroke:#c77919;
    class E held;
    ''')
    old=text('docs/汇报/9月/GBSA优化项目总汇报_生工团队版_20261005.md')
    gloss=old.split('## 九、术语表：本文出现的英文与缩写',1)[1].split('## 十、论文、分区与项目内参考资料',1)[0].strip()
    PARTS['GLOSSARY']=gloss+'\n\n'+table(['新增/补充缩写','完整含义','本报告用途'],[
        ['A / C / G / U','adenine，腺嘌呤 / cytosine，胞嘧啶 / guanine，鸟嘌呤 / uracil，尿嘧啶','RNA四种碱基字母'],
        ['MLM / MLP','masked language modelling，掩码语言建模 / multilayer perceptron，多层感知机','历史预训练/预测头'],
        ['VAE / HMM / SELEX','variational autoencoder，变分自编码器 / hidden Markov model，隐马尔可夫模型 / systematic evolution of ligands by exponential enrichment，指数富集配体系统进化','RaptGen 原任务组件'],
        ['MSE / NLS / Lasso','mean squared error，均方误差 / negative label smoothing，负标签平滑 / least absolute shrinkage and selection operator，最小绝对收缩与选择算子','损失、平滑和交互支线'],
        ['NAND / DBTL','not AND，与非逻辑 / design–build–test–learn，设计—构建—测试—学习','原论文功能与迭代流程，不是当前标签'],
        ['PCA / GELU / FFN / post-LN','principal component analysis，主成分分析 / Gaussian error linear unit，高斯误差线性单元 / feed-forward network，前馈网络 / post-layer normalization，后置层归一化','历史结构表示或编码器实现词语'],
        ['kcal/mol / nt / OOF','千卡每摩尔 / nucleotide，核苷酸 / out-of-fold，折外','单位、长度与训练内残差'],
        ['rollout / root / token','向前展开链 / 根父本 / 序列符号单元','AdaLead 与历史编码器'],
        ['PNG / SVG / PPT','Portable Network Graphics，便携式网络图形 / Scalable Vector Graphics，可缩放矢量图形 / PowerPoint presentation，演示文稿','插图与幻灯片素材']])
    PARTS['GLOSSARY']+='\n\n'+table(['结果文件中的原始方法/配置标识','学术解释与当前含义'],[
        ['raw / LS_k5_a0.5 / NLS_a0.25 / NLS_a0.5','未平滑；5近邻、0.5混合权重的标签平滑；0.25/0.5权重的负平滑'],
        ['xgb_mse / xgb_quantile50','XGBoost均方误差 / 50%分位（中位数）损失'],
        ['lgbm_lambdarank','LightGBM（light gradient boosting machine，轻量梯度提升框架）中的LambdaRank排序机制，本轮整折一组、31级相关度'],
        ['xgbranker_subsample','XGBoost Ranker成对排序回归器；本轮500条子样本配置'],
        ['high_out / mid_out / low_out / random_control','测试域为12–13、10–11、≤9突变的留出 / 随机80/20对照；是划分协议代号'],
        ['rf_52d / xgb_495d / lasso_epistasis','52维突变独热RF / 历史303+192维XGB / Lasso成对上位性表示，1300维'],
        ['mutual_info / spearman / tree_importance','互信息筛选 / 单变量等级相关筛选 / 树重要度筛选；均只在训练折拟合'],
        ['gbsa_Sp / rank_fusion / equal_weight_rank / fusion_smooth','GBSA Spearman列 / 预测等级融合 / 等权等级融合 / 平滑标签后融合'],
        ['matched_seed_random_walk / multi_start_hill_climb / multi_start_beam','同起点随机游走 / 多起点爬山 / 多路径束搜索策略注册名称'],
        ['mutation_only_ga / iterated_local_search / simulated_annealing / mutation_only_adalead','无交叉GA / 迭代局部搜索 / 模拟退火 / 无交叉AdaLead策略注册名称'],
        ['ga_tournament1 / ga_tournament4 / ga_single_hop','锦标赛抽1个（随机父本）/抽4个 / 单跳比例改为1的独立GA消融'],
        ['budget_exhausted / stagnation / neighbors_exhausted / local_optima / max_generations','预算用尽 / 停滞 / 合法未访问邻居耗尽 / 当前局部路径停止 / 达到代数护栏'],
        ['μ / σ / α / β / k','点预测 / 未校准成员分歧 / 平滑权重或岭惩罚（按上下文区分）/LCB分歧权重 / 近邻数量或比例标识（按上下文区分）']])
    PARTS['REFERENCES']=table(['编号：原论文/资料','载体、2024 JCR 指定类别/等级','怎样支撑本方案'],[
        ['R1：[gkag145](https://doi.org/10.1093/nar/gkag145)，NAND 杂交核糖开关深度批量贝叶斯优化','Nucleic Acids Research；Biochemistry & Molecular Biology（生物化学与分子生物学）/Q1','主要计算组件迁移；原生物任务与当前GBSA不同'],
        ['R2：Iwano 等（2022），[Generative aptamer discovery using RaptGen](https://doi.org/10.1038/s43588-022-00249-6)','Nature Computational Science；Computer Science, Interdisciplinary Applications（计算机科学跨学科应用）/Q1','生成表示与潜空间启发，当前未复现'],
        ['R3：Poole 与 Mackworth，[Artificial Intelligence: Foundations of Computational Agents, 3e §4.6](https://artint.info/3e/html/ArtInt3e.Ch4.S6.html)','作者开放教材/专著，不适用JCR','局部改进、随机步骤和重启原理'],
        ['R4：Holland，[Adaptation in Natural and Artificial Systems](https://mitpress.mit.edu/9780262581110/adaptation-in-natural-and-artificial-systems/)（原版1975，1992再版）','MIT Press专著，不适用JCR','种群、选择、变异的理论来源'],
        ['R5：Espah Borujeni 等（2016），[RNA 适配体来源核糖开关自动化设计](https://doi.org/10.1093/nar/gkv1289)','Nucleic Acids Research；Biochemistry & Molecular Biology/Q1','RNA功能设计中GA的应用；原文有重组，当前禁用'],
        ['R6：Freitag 与 Al-Onaizan（2017），[Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/)','ACL神经机器翻译研讨会论文，不适用JCR','束搜索宽度/路径概念，不是本RNA证据'],
        ['R7：Lourenço、Martin、Stützle，[Iterated Local Search](https://arxiv.org/abs/math/0102188)，[2003章节](https://doi.org/10.1007/0-306-48056-5_11)','作者预印本/Handbook of Metaheuristics章节，不适用JCR','局部下降、扰动、接受；指导ILS最小实现'],
        ['R8：Kirkpatrick、Gelatt、Vecchi（1983），[Optimization by Simulated Annealing](https://doi.org/10.1126/science.220.4598.671)','Science；Multidisciplinary Sciences（综合性科学）/Q1','概率接受与退火依据，不保证本项目真GBSA'],
        ['R9：Sinai 等（2020），[AdaLead: A simple and robust adaptive greedy search algorithm for sequence design](https://arxiv.org/abs/2010.02141)','arXiv作者预印本，不适用JCR','相对根的展开与领先者选择；本轮明确改编'],
        ['R10：Bashir 等（2021），[机器学习引导适配体改良](https://doi.org/10.1038/s41467-021-22555-9)','Nature Communications；Multidisciplinary Sciences/Q1','预测、突变、实验迭代；原DNA亲和力不是RNA GBSA'],
        ['R11：Genheden 与 Ryde（2015），[MM/PBSA and MM/GBSA methods](https://doi.org/10.1517/17460441.2015.1032936)','Expert Opinion on Drug Discovery；Pharmacology & Pharmacy（药理学与药学）/Q1','计算标签受近似和协议影响'],
        ['R12：Sindt 等（2025），[原论文](https://doi.org/10.1021/acs.jcim.5c00730)','Journal of Chemical Information and Modeling；Computer Science, Interdisciplinary Applications/Q1','小分子追加打分风险类比，不是本RNA证明'],
        ['R13：Kennedy 与 O’Hagan（2000），[Predicting the output of a complex computer code when fast approximations are available](https://doi.org/10.1093/biomet/87.1.1)','Biometrika；Statistics & Probability（统计学与概率论）/Q1；Biology类别Q2','多保真差异建模依据，不直接证明dock→GBSA映射'],
        ['R14：Angermueller 等（2020），[Population-Based Black-Box Optimization for Biological Sequence Design](https://proceedings.mlr.press/v119/angermueller20a.html)','ICML会议论文/PMLR，不适用JCR','算法接口与组合资源分配，未复现P3BO'],
        ['R15：Stanton 等（2022），[LaMBO](https://proceedings.mlr.press/v162/stanton22a.html)','ICML会议论文/PMLR，不适用JCR','表示、概率代理和采集需求'],
        ['R16：Brookes、Park、Listgarten（2019），[Conditioning by adaptive sampling for robust design](https://proceedings.mlr.press/v97/brookes19a.html)','ICML会议论文/PMLR，不适用JCR','生成先验与预测失真控制，未实现'],
        ['R17：Angermueller 等（2020），[Model-Based Reinforcement Learning for Biological Sequence Design](https://research.google/pubs/model-based-reinforcement-learning-for-biological-sequence-design/)','ICLR会议论文，不适用JCR','后期RL路线；此前正文访问受限，不声称完整复现']])
    evrows=[
        ('起点汇报','docs/汇报/9月/论文复现与改进_dock_gbsa进展.md','全部主题保留，按后续审计勘误'),
        ('全项目审计','docs/项目记录/全项目审计_20260924.md','泄漏、错维度、评估口径和修复记录'),
        ('v8基准','evidence/server_sync_20260902/v8_fusion_100seed/all_results.csv','100种子逐方法值'),
        ('RF基线','evidence/audit_20260924/audit_baseline_rf_100seed/metrics.csv','100种子源行'),
        ('安全特征筛选','evidence/server_sync_20260903/feature_selection_all_safe_30seed/metrics_paired.csv','9组折内配对'),
        ('平滑/排序/负对照','docs/论文改进/A组实验汇总_20260926.md','正文结论按本次边界修正'),
        ('docking目标','evidence/audit_20260924/formal_dock_fusion_100seed/metrics.csv','100种子，不是GBSA'),
        ('对接残差/预筛','evidence/search_suite_20261005/dock_confirmation/summary.json','100种子信息可得条件'),
        ('固定池选择','evidence/loop_20260930/strategy_compare_100seed/runs.csv','400子运行、2000轮次行'),
        ('本次方法复算与历史路径','evidence/report_review_20261006/method_verification.json',f"{audit['statistical_assertions']}项保存统计核验；seed42迁移文档、结构缺陷公式审查"),
        ('σ校准','evidence/sigma_calibration_30seed/summary.json','30种子×2训练量'),
        ('同起点三法','evidence/search_compare_matched_30seed/summary.json','30种子、90子运行'),
        ('原四法','evidence/search_suite_20261005/README.md','64代历史条件和消融'),
        ('最新七法','evidence/search_seven_20261005/README.md','冻结参数、30开发/100确认、来源审计'),
        ('七法详细结果/运行步骤','docs/汇报/9月/七算法同起点比较与实施结果_20261005.md','实践原文'),
        ('GBSA接口总体方案','docs/项目记录/GBSA代理接口与候选闭环实施方案_20260930.md','未来接口与独立GBSA约束'),
        ('突变搜索总体方案','docs/项目记录/突变序列迭代搜索算法与代理接口实施方案_20261002.md','当前阶段B合同'),
        ('七法冻结方案','docs/项目记录/七算法同起点比较冻结方案_20261005.md','确认前固定种子/参数/对照')]
    PARTS['EVIDENCE_INDEX']=table(['阶段','入口','用途'],[[k,f'[{Path(p).name}](../../{p})',v] for k,p,v in evrows])
    for _,p,_ in evrows:
        if (ROOT/p).is_file():register(p)
        else:assert (ROOT/p).is_dir()
    # Preserve the versions actually reviewed, including the repaired formula.
    for p in ['scripts/tools/feature_selection_pipeline.py',
              'scripts/reproduction/label_smoothing.py',
              'scripts/reproduction/eval_smoothing_variants.py',
              'scripts/reproduction/eval_rank_objectives2.py',
              'scripts/reproduction/eval_negative_controls.py',
              'scripts/reproduction/eval_sparse_epistasis.py',
              'scripts/reproduction/eval_extrapolation.py',
              'scripts/reproduction/feature_weighting.py',
              'scripts/reproduction/feature_gating.py',
              'scripts/reproduction/new_features.py',
              'scripts/reproduction/ensemble_defect.py',
              'scripts/reproduction/structure_metrics.py',
              'scripts/reproduction/dock_denoise.py',
              'scripts/reproduction/dock_rank_correction.py',
              'scripts/reproduction/active.py',
              'scripts/loop/selection.py',
              'scripts/loop/strategy_compare.py',
              'tests/test_structure_metrics.py',
              'tools/audit_report_methods_20261006.py']:
        register(p)
    PARTS['METHOD_VERIFICATION']=table(['实验/声明','本次实际核验','仍需区分的限制'],[
        ['论文单次0.1228','读取历史结果与组件代码；原包参数可追溯','只有汇总，无逐候选预测供独立重算；未重训'],
        ['论文融合5种子0.2610','从5个保存种子数值复算均值与SD一致','不同协议，非成对语料消融；转导/全数据筛选限制未消除'],
        ['历史0.506优化路径','追回历史文档；新固定划分重放0.50602865，30开发+100确认；655份预测逐项复算','完整历史每一步仍无原预测档案；新配方跨划分增量见§4.3，不继承固定测试集独立性'],
        ['v8及三种参照','700行来源、100种子配对均值/胜率；审计10种子与原表一致','本次复算指标，未重新训练100模型；随机划分泛化不等于新候选真值'],
        ['三种特征筛选','360行；9组均值/差/胜率/自举CI；120套特征清单与52维强制保留核对','30种子开发；只证明相对各自全1495维对照有改进'],
        ['平滑30/100与融合30','保存逐种子数组及100行CSV；均值/配对/区间复算一致','495维历史useful来源限制；未证明物理真标签去噪'],
        ['负对照10/稀疏交互30','逐种子数组的均值、SD复算；核查实际置换/Lasso方案','10种子负对照是诊断；系数不等于生物学因果'],
        ['外推三域','标准表重算三域样本量，与保存结果匹配；随机30均值重算','三域仅单次域留出；没有逐候选预测独立重算该ρ；不是等样本配对消融'],
        ['旧对接混合/替换','核查实现变换、折内训练/测试边界和保存均值','保存文件只含汇总，不能独立重算逐seed区间/胜率；不作正收益声明'],
        ['对接新100种子支线','依据冻结配置与逐seed包；沿用完整性验收并核对报告表','真实dock仅历史表可用；新候选成本/失败率仍未实测'],
        ['结构缺陷旧30种子','保存数组的均值/差/胜率/CI重算一致；2000唯一行、0缺失','公式错误，1996行WT结构概率为0且异常被置0；旧负结果不能否定正确方法'],
        ['锁定val/test单次包','从每份200条真实值/预测重新计算ρ，舍入0.316/0.451','不同于0.506历史模型；不能证明新方案达到多种子0.451'],
        ['静态池100种子','2000行键完整；regret/CI/胜平负/两尾部由原CSV重算','仅历史有限池的标签揭示，不是新GBSA或多代变异搜索'],
        ['七法搜索100种子','44组主/两两均值、胜平负验证；完整性沿用原验收','目标为固定模型预测；无新增GBSA，不证明真实算法优胜']])+'\n\n'+cite('evidence/report_review_20261006/method_verification.json',f"本次{audit['statistical_assertions']}项统计断言与来源")
    PARTS['REPORT_AUDIT']='旧结果核对完成 **'+str(audit['statistical_assertions'])+' 项统计断言**；另核对七法44组配对统计。新补实验及其逐预测核验分别留痕，不把旧结果复算记作新训练，也不把程序测试当真实GBSA验证。'
    supplement_parts, supplemental_figures = supplement_sections(register,table,fmt,interval,signed,OUT)
    PARTS.update(supplement_parts)
    supplemental_runs = js('evidence/supplement_20261006/index.json')['runs']
    if supplemental_runs:
        PARTS['REPORT_AUDIT']='本次先完成 **'+str(audit['statistical_assertions'])+' 项既有保存结果统计断言**，另核对七法44组配对统计及图片/流程来源；再按冻结协议执行新的补实验并从测试预测重新核对Spearman和MAE。旧结果复算与新模型训练分别留痕，真实候选docking/MD/GBSA新增数仍为0。结构矩阵公式有针对性测试；真实NUPACK官方对照是否完成以完整运行包为准，不能仅凭单元测试宣布物理特征正确。'
    for p in ['scripts/reproduction/supplement_cpu_20261006.py','scripts/reproduction/supplement_active_20261006.py','scripts/reproduction/supplement_paper_20261006.py','docs/项目记录/补实验冻结协议_20261006.md','tools/summarize_supplement_20261006.py','tools/report_supplement_sections_20261006.py','tests/test_supplement_20261006.py']:
        register(p)
    for p in ['scripts/reproduction/supplement_active_bounded_20261006.py','tests/test_defect_error_contract_20261006.py','tests/test_paper_supplement_contract_20261006.py']:
        register(p)
    for p in ['scripts/reproduction/supplement_defect_eval_20261006.py','scripts/reproduction/supplement_defect_force_20261006.py','scripts/reproduction/aggregate_paper_20261006.py','configs/experiments/supplement_defect_forced_20261006.json','configs/experiments/supplement_paper_aggregate_20261006.json','tools/prepare_paper_cohorts_20261006.py','tools/validate_remote_supplement_20261006.py']:
        register(p)
    PARTS['REPORT_AUDIT'] += '\n\n本轮另执行了下列补实验，区别于上述保存结果复算：\n\n'+PARTS['SUPPLEMENT_STATUS']
    template=text('tools/report_templates/阶段总汇报_PPT素材版_20261006.md')
    for k,v in PARTS.items():template=template.replace('{{'+k+'}}',str(v))
    missing=re.findall(r'\{\{([^}]+)\}\}',template)
    assert not missing,missing
    # Local glossary inherited section references point to a different report.
    template=template.replace('见 §4.13','见本报告 §9').replace('见 §4.15/§6.6','见本报告 §9')
    template=template.replace('新增图 A01–A07 从保存的数据','新增图 A01–A07、A12 从保存的数据')
    template=template.replace('在 §4.15/§6.6 的已实现机制中定义','在本报告 §9 的机制中定义')
    DOC.write_text(template.rstrip()+'\n',encoding='utf-8')
    register('tools/build_phase_report_20261006.py')
    register('tools/render_phase_report_20261006.cjs')
    (OUT/'report_sources.json').write_text(json.dumps({'report':DOC.relative_to(ROOT).as_posix(),
        'generated_on':'2026-10-06','no_model_fit':not bool(supplemental_runs),'new_real_GBSA':0,
        'new_model_experiments':len(supplemental_runs),'supplemental_figures':supplemental_figures,
        'source_files':SOURCES,'figure_A03_bootstrap':{'resamples':20000,'seed':20260924,'purpose':'descriptive CI from saved seed rows; formal published CI retained separately'}},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'report':str(DOC),'characters':len(template),'lines':template.count('\n')+1,'source_files':len(SOURCES),'new_data_figures':8+supplemental_figures,'new_diagrams':6,'new_completed_experiment_packages':len(supplemental_runs)},ensure_ascii=False))

if __name__=='__main__':
    main()
