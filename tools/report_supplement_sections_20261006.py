"""Evidence-driven supplemental sections and exportable plots for the main report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NAMES={'mut52':'52维突变','seq303':'303维序列','seq_mt_epi486':'486维序列+突变类型/同碱基指示','seq_nup904':'904维序列+NUPACK','historical1087':'1087维历史完整配方','base303':'303维基线','plus_defect':'加修正缺陷','plus_probability':'加WT结构概率','plus_both':'同时加两项','no_pretrain':'无预训练','foreign_pretrain':'外域语料预训练','in_domain_pretrain':'训练父本生成的域内语料预训练','sequence303_GBR_control':'303维经典GBR对照','random':'随机揭示','greedy':'预测GBSA最低优先','uncertainty':'集成分歧最高优先','mixed':'45/25/20/10混合采集'}
NAMES.update({'forced_defect':'强制保留缺陷','forced_probability':'强制保留WT概率','forced_both':'强制保留两项'})

def build(register,table,fmt,interval,signed,out):
    path=ROOT/'evidence/supplement_20261006/index.json'
    if not path.exists():return {},0
    index=json.loads(path.read_text('utf-8'));register(path)
    import matplotlib.pyplot as plt
    runs=index['runs'];parts={};count=0
    for r in runs:register(ROOT/'evidence/supplement_20261006'/r['run_id']/'summary.json')
    def group(token):return [r for r in runs if token in r['run_id']]
    def panel(rs,title,fig):
        rows=[]
        for r in rs:
            phase='100种子确认' if 'confirm' in r['run_id'] else '30种子开发'
            for name,d in r['summary'].items():
                rows.append([phase,NAMES[name],fmt(d['spearman']['mean']),interval(d['spearman']['ci95']),fmt(d['mae']['mean']),fmt(d['rmse']['mean'])])
        text=table(['阶段','配方','平均Spearman','种子均值95%区间','平均MAE','平均RMSE'],rows)
        figu,ax=plt.subplots(figsize=(11,5.6))
        for k,r in enumerate(rs):
            names=[n for n in ['mut52','seq303','seq_mt_epi486','seq_nup904','historical1087','random','greedy','uncertainty','mixed','no_pretrain','foreign_pretrain','in_domain_pretrain','sequence303_GBR_control','base303','plus_defect','plus_probability','plus_both','forced_defect','forced_probability','forced_both'] if n in r['summary']];means=np.array([r['summary'][n]['spearman']['mean'] for n in names]);ci=np.array([r['summary'][n]['spearman']['ci95'] for n in names])
            pos=np.arange(len(names))+(k-(len(rs)-1)/2)*.22
            ax.errorbar(pos,means,yerr=np.vstack([means-ci[:,0],ci[:,1]-means]),fmt='o',capsize=4,label='100种子确认' if 'confirm' in r['run_id'] else '30种子开发')
        ax.set_xticks(np.arange(len(names)),[NAMES[n] for n in names],rotation=13,ha='right');ax.set_ylabel('Spearman 排序相关系数');ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2);figu.tight_layout()
        for ext in ['png','svg']:figu.savefig(out/f'{fig}.{ext}',dpi=180)
        plt.close(figu)
        pd.DataFrame(rows,columns=['phase','variant','spearman_mean','spearman_ci95','mae_mean','rmse_mean']).to_csv(out/'图表数据'/f'{fig}.csv',index=False,encoding='utf-8-sig')
        return text+f'\n\n![{title}](figures/阶段总汇报_20261006/{fig}.png)\n\n[矢量版](figures/阶段总汇报_20261006/{fig}.svg)。区间为固定已测数据上随机划分种子均值的自助法区间，不是新独立生物样本总体区间。'
    hist=group('history')
    if hist:
        text='### 4.3 十月补实验：历史0.506准确重放与冻结配方的跨划分比较\n\n'
        fixed=next(r for r in hist if 'fixed_split' in r)['fixed_split']
        text+='原1800/200序列分区、601个直接NUPACK特征、训练内筛前100、GBR500已重新训练。实际重放结果如下；没有用历史加权label。\n\n'+table(['固定42配方','Spearman','MAE','RMSE'],[[NAMES[d['variant']],fmt(d['spearman'],8),fmt(d['mae']),fmt(d['rmse'])] for d in fixed])
        text+='\n\n'+panel(hist,'B01：冻结历史配方，开发与确认随机划分','B01_历史配方补实验')
        text+='\n\n'+table(['阶段','试验 / 对照','配对Spearman差','95%区间','胜/平/负'],[['100确认' if 'confirm' in r['run_id'] else '30开发',NAMES[d['variant']]+' / '+NAMES[d['reference']],signed(d['delta']),interval(d['ci95']),f"{d['wins']}/{d['ties']}/{d['losses']}"] for r in hist for d in r['paired'] if d['metric']=='spearman'])
        text+='\n\n**如何解释你的优化成果：**固定42的0.506已经重现，原划分相对303维确实提高约0.0595；不能说这个优化没有发生。新划分比较检验的是冻结配方的稳定性：若复杂配方均值较差，意味着当时固定测试划分的优势不稳定，不能据此把全部特征工程称为无用，也不能把0.506当未来序列的预期相关。52维到303维的整体效应与303维到1087维的增量必须分开。历史固定测试集曾用于多次方法选择，即使现在精确重放也不能恢复独立测试资格。\n\n本实验每次1800/200；旧v8发布参照每次1600/400，模型也不同，不能直接以均值高低宣称替换v8。新种子确认依然来自同一2000条数据，不是100批新GBSA。'
        parts['SUPPLEMENT_HISTORY']=text;count+=1
    al=group('active')
    if al:
        text='### 2.7 十月补实验：修正后主动学习的独立模型与发现终点\n\n共享400条holdout（固定留出测试）、400条初始标签、1200条固定候选池。每轮揭示100条，5轮后900条训练；四臂每轮都重训5成员bootstrap直方图梯度提升集成。标准化仅拟合当轮已揭示标签集合，预测方向为GBSA最小化，测试标签不参与采集。不确定性臂使用未校准的成员标准差，不能解释为真实置信区间。混合策略45条高分歧、25条覆盖最远、20条低预测、10条随机，排重后总数固定100。\n\n'
        text+='bootstrap（自助抽样）是从当轮已标注训练条目有放回抽取相同数量的条目，五个成员分别拟合，因此成员训练集不同，不是新增五批测量。HistGradientBoostingRegressor（直方图梯度提升回归器）把数值特征分箱以加速树的阈值搜索，再逐轮拟合残差；本轮固定200轮、最大深度3、叶最少20、学习率0.03、L2=1。L2是权重平方惩罚的正则化系数；这里使用程序对叶值的正则化，不是新生物标签。使用训练集拟合的同一标准化坐标进行覆盖距离计算，避免旧脚本分别标准化候选/训练/测试造成坐标不一致。\n\n'
        text+='**范围：本轮是树集成评分器下的受控主动学习比较，未重跑原论文深度集成和Kriging-Believer（克里金信徒，用临时预测标签逐项更新批量采集）整条路径。**旧深度主动学习曲线的优劣结论仍撤回；它的重评应在受控预训练结果及有效不确定性评分器明确后独立编号进行，不能用本轮树模型结果替代。\n\n'
        text+=panel(al,'B02：等标签预算主动学习最终holdout表现','B02_主动学习补实验')
        text+='\n\n'+table(['阶段','策略','最终查询regret','top10%召回','相对随机ΔSpearman（95%区间）','相对随机Δregret（95%区间）'],[['100确认' if 'confirm' in r['run_id'] else '30开发',NAMES[name],fmt(d['regret']['mean']),fmt(d['top10_recall']['mean']),next((signed(x['delta'])+' '+interval(x['ci95']) for x in r['paired'] if x['metric']=='spearman' and x['variant']==name),'基线'),next((signed(x['delta'])+' '+interval(x['ci95']) for x in r['paired'] if x['metric']=='regret' and x['variant']==name),'基线')] for r in al for name,d in r['summary'].items()])
        text+='\n\n这里regret（遗憾值）=五轮查询集合中最低真实GBSA−原1200条池中最低真实GBSA，排除共享初始400条；top10%召回分母是固定池真实最低的120条。ΔSpearman正值更好，Δregret负值更好。以配对区间为判断依据，不用单一均值推断全部策略无效。所有标签仍为已有表内标签，没有新序列真实测量。本实验与九月300初始/600池/200holdout、五轮50条的实验条件不同，两表分别保留。\n\n阶段选择仍需结合真标签成本：若某策略只改善发现而没有稳定提高holdout排序，保留它作为有限池发现策略；若改进模型，也仅在本次代理和池条件下成立，不能自动推及深度代理或真实新序列。'
        newest=next((r for r in al if 'confirm' in r['run_id']),al[0]);order=['random','greedy','uncertainty','mixed']
        figu,axes=plt.subplots(1,3,figsize=(14,5.5));plot_rows=[]
        for ax,metric,title in zip(axes,['regret','top10_recall','spearman'],['查询遗憾值（越低越好）','前10%召回（越高越好）','相对随机的排序增量']):
            for j,name in enumerate(order):
                if metric=='spearman':
                    d=next((d for d in newest['paired'] if d['metric']==metric and d['variant']==name),None)
                    mean=0 if d is None else d['delta'];ci=[0,0] if d is None else d['ci95']
                else:d=newest['summary'][name][metric];mean=d['mean'];ci=d['ci95']
                ax.errorbar(j,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],fmt='o',capsize=4,color='#2874a6')
                plot_rows.append([name,metric,mean,ci[0],ci[1]])
            ax.set_xticks(range(4),['随机','低预测','高分歧','混合'],rotation=20);ax.set_title(title);ax.grid(axis='y',alpha=.2)
            if metric=='spearman':ax.axhline(0,color='gray',linestyle='--')
        figu.suptitle('B05：100种子确认——模型提升与候选发现分别判断' if 'confirm' in newest['run_id'] else 'B05：30种子开发——模型提升与候选发现分别判断');figu.tight_layout()
        for ext in ['png','svg']:figu.savefig(out/f'B05_模型与发现终点.{ext}',dpi=180)
        plt.close(figu);count+=1
        pd.DataFrame(plot_rows,columns=['policy','metric','mean_or_delta','ci95_low','ci95_high']).to_csv(out/'图表数据/B05_模型与发现终点.csv',index=False,encoding='utf-8-sig')
        text+='\n\n![模型与候选发现分别判断](figures/阶段总汇报_20261006/B05_模型与发现终点.png)\n\n[矢量版](figures/阶段总汇报_20261006/B05_模型与发现终点.svg)。左、中图是策略本身的种子均值，右图是同种子相对随机的差值；不能只凭左、中图的区间重叠判断配对差。'
        text+='\n\n**本轮评分器自身的分歧校准诊断：**最终holdout上计算成员标准差与绝对预测误差的Spearman，再对种子均值给区间。它衡量错误排序关系，不等同覆盖率或概率校准。\n\n'+table(['阶段','策略','sigma与绝对误差相关','95%区间'],[['100确认' if 'confirm' in r['run_id'] else '30开发',NAMES[name],fmt(d['mean']),interval(d['ci95'])] for r in al for name,d in r.get('sigma_calibration',{}).items()])
        latest=next((r for r in al if 'confirm' in r['run_id']),al[0])
        positives=[d for d in latest['paired'] if d['metric']=='spearman' and d['ci95'][0]>0]
        gains=[d for d in latest['paired'] if d['metric']=='regret' and d['ci95'][1]<0]
        phase='100种子确认' if 'confirm' in latest['run_id'] else '30种子开发'
        text+='\n\n'+table(['最终配对终点','策略 / 随机','均值差','95%区间','胜/平/负（按该终点方向）'],[[d['metric'],NAMES[d['variant']]+' / 随机',signed(d['delta']),interval(d['ci95']),f"{d['wins']}/{d['ties']}/{d['losses']}"] for d in latest['paired'] if d['metric'] in ['spearman','regret','top10_recall']])
        text+='\n\n**本轮'+phase+'结论：**'
        if positives:text+='相对随机，'+ '、'.join(NAMES[d['variant']] for d in positives)+'的holdout排序增量区间为正；这只支持该固定池和评分器条件下的模型增益，不能宣布真实新序列闭环有效。'
        else:text+='没有采集臂的holdout排序增量区间稳定大于零；当前未确认比等标签量随机揭示更能改善这个模型，不能据此否定所有主动学习方法。'
        if gains:text+=' '+ '、'.join(NAMES[d['variant']] for d in gains)+'的平均查询遗憾值降低，支持有限池发现用途；不把这一发现终点的收益写成模型变准。'
        text+=' 本轮同时报告多个策略和终点，所列95%区间没有作多重比较校正；纯分歧选样的微小负向排序差属于本配置下的诊断，不能升级成普遍伤害结论。'
        text+=' 随机揭示500/1200条时，前10%集合的理论期望召回为41.67%；该基线由抽样比例决定，不能把约42%的随机召回称为算法学到了规律。'
        parts['SUPPLEMENT_ACTIVE']=text;count+=1
    paper=group('paper')
    if paper:
        text='### 1.6 十月补实验：相同划分与回归头下的预训练语料控制\n\n30个外层种子11001–11030，1600/400。相同初始编码器、相同冻结编码器和10个9层×256回归头；监督GBSA均方误差训练50轮。三臂为不预训练、历史服务器保留的外域语料、仅由训练父本产生的域内变体；预训练各2000条、5轮。域内语料与全部2000条已知序列均不重叠；外域编码器独立于FAD划分预训练后复用，域内每个划分单独训练。经典303维GBR作为额外同划分对照。这个实验控制语料效应，计算预算低于原论文和历史运行，不能宣称完整原任务复现。外域使用data/ngs_sequences.txt并保存来源文件及实际使用子集哈希；本轮没有从原文测序库重新下载并核对全部原始条目，所以不把文件名及历史NGS标注当作原论文完整语料来源已经重新认证的证据。\n\n'
        text+=panel(paper,'B03：论文迁移预训练受控比较','B03_论文迁移补实验')
        budget=paper[0].get('paper_budget',{})
        if budget:
            text+='\n\n'+table(['实际执行与预算','记录'],[['成员/监督训练轮数/预训练轮数',str(budget['n_members'])+' / '+str(budget['head_epochs'])+' / '+str(budget['pretrain_epochs'])],['外域有效序列数及长度统计',str(budget['foreign_n'])+' / '+str(budget['foreign_length_stats'])],['域内每种子预训练及全表嵌入平均秒数',fmt(budget['mean_domain_pretrain_and_embedding_seconds'])],['外域四组预训练及嵌入总秒数',fmt(budget['shared_foreign_pretrain_and_embedding_seconds'])],['每种子各臂头训练/预测平均秒数',str({NAMES[k]:round(v,2) for k,v in budget['mean_arm_training_prediction_seconds'].items()})],['数值环境',budget['device']+' / PyTorch '+budget['torch']]])
            text+='\n\n时间是程序记录的服务器分项墙钟时间，四组同时运行的分项秒数不能相加当总墙钟耗时；GPU异步执行可能把部分嵌入时间计入后续调用，不当作单候选推理成本基准。外域按文件顺序取前2000条有效、唯一且不属于已知FAD的序列；域内序列都是141核苷酸，二者长度和总词元预算没有强行匹配。外域来源和使用内容的SHA256（安全散列算法256位）保存在paper_manifest.json，域内每种子生成序列及哈希均回收并从训练父本独立重建核对。'
        text+='\n\n'+table(['臂','相对无预训练ΔSpearman','95%区间','胜/平/负'],[[NAMES[d['variant']],signed(d['delta']),interval(d['ci95']),f"{d['wins']}/{d['ties']}/{d['losses']}"] for r in paper for d in r['paired'] if d['metric']=='spearman'])
        text+='\n\n'+table(['误差指标（差值负更好）','臂 / 无预训练','均值差','95%区间','胜/平/负'],[[d['metric'],NAMES[d['variant']],signed(d['delta']),interval(d['ci95']),f"{d['wins']}/{d['ties']}/{d['losses']}"] for r in paper for d in r['paired'] if d['metric'] in ['mae','rmse']])
        text+='\n\n晋级100种子的条件是域内臂对另外两预训练臂以及经典对照的Spearman配对差区间均为正；未达条件则保存开发结果，不追加以挑选好结果为目的的参数或种子。'
        gate=paper[0].get('confirmation_gate',{})
        if gate:
            text+='\n\n'+table(['域内臂 / 对照','排序均值差','95%区间','胜/平/负','预设条件'],[[NAMES[n],signed(d['delta']),interval(d['ci95']),f"{d['wins']}/{d['ties']}/{d['losses']}",'达到' if d['criterion_met'] else '未达到'] for n,d in gate['comparisons'].items()])
            text+='\n\n**本轮配方决定：**'+('三项条件均达到，可以按冻结规则另开100种子确认。' if gate['advance_to_100_seeds'] else '未达到全部晋级条件，不追加本配方的100种子确认；保留三臂/经典对照的完整开发证据。')
        text+='\n\n执行调度分四个互不重叠的种子组，四组分别DONE后再核对并聚合；外域编码器在组内复用，四组使用同一初始权重、预训练随机种子及语料哈希。额外的四次外域预训练是调度开销，不是某一臂额外训练了四倍。没有改变每个模型的预算或挑选种子；首个串行终止包保留为FAILED并不纳入这些数值。'
        parts['SUPPLEMENT_PAPER']=text;count+=1
    defect=group('defect_dev')
    if defect:
        text='### 5.5 十月补实验：修正结构缺陷的真正配对评估\n\n完整2000条NUPACK重新计算；错误的未配对公式已修为矩阵对角线。首个正式包在官方defect核验不兼容WT目标时失败，未用于模型比较；修订后另编号完整重算。官方核对4个兼容WT目标条目，以及前20条的全未配对/自身MFE两个诊断目标，共44个对照。诊断目标不进入特征，全部特征目标仍为WT。非允许配对的目标结构概率明确标0，其余计算异常终止，不做异常到0的替换。303维无历史useful子块基线，训练内筛100，GBR500，1600/400，30个配对种子。\n\n'
        computation=index.get('feature_computations',[])
        if computation:
            c=computation[0]
            text+=table(['重算验收项','实际结果'],[['唯一序列/无失败',str(c['rows'])+' / 0'],['官方对照/最大绝对误差',str(c['official_checks'])+' / '+format(c['official_max_error'],'.8g')],['不兼容WT目标/直接计算概率',str(c['status_counts']['invalid_target_pair'])+' / '+str(c['status_counts']['computed'])],['正确全局缺陷范围',str(c['correct_defect_range'])],['与旧缺陷的平均绝对变化',fmt(c['mean_absolute_change_from_old'])],['重算及核验时间（秒）',fmt(c['elapsed_seconds'])]])+'\n\n'
        text+=panel(defect,'B04：修正结构缺陷特征的30种子增量','B04_结构缺陷补实验')
        text+='\n\n'+table(['特征臂','相对基线ΔSpearman','95%区间','胜/平/负'],[[NAMES[d['variant']],signed(d['delta']),interval(d['ci95']),f"{d['wins']}/{d['ties']}/{d['losses']}"] for r in defect for d in r['paired'] if d['metric']=='spearman'])
        selected=defect[0].get('added_feature_selected_counts',{})
        if selected:
            text+='\n\n'+table(['加特征臂','新特征进入训练内前100的种子数 / 30'],[[NAMES[k],str(v)] for k,v in selected.items()])
            text+='\n\n这记录当前筛选器实际使用新特征的频率。新增列也可能改变树筛选中的随机候选列组合及其他列排序；尤其只有4个非零的WT目标概率，微小均值差不能独立解释为生物学信息增益。未被筛入的特征没有被最终GBR直接使用。'
        text+='\n\n这些正确公式结果才可用于评价本次WT目标的辅助价值。它们与旧495维XGBoost比较协议不同，不能称旧数字的原位纠正，也不能证明野生型MFE结构就是结合构象。'
        positive=[d for r in defect for d in r['paired'] if d['metric']=='spearman' and d['ci95'][0]>0]
        text+='\n\n**本轮结构特征结论：**'+('部分配方的配对排序增量区间为正，但只支持本目标和模型条件。' if positive else '三个加特征臂的配对排序增量区间均跨零，本次没有确认稳定的增益；不晋级主线输入。这个结论来自正确重算后的新实验，旧错误公式负结果仍不恢复有效性。')
        forced=group('defect_force')
        if forced:
            text+='\n\n### 5.6 强制保留诊断：让结构特征实际进入每个最终模型\n\n这是看到前轮筛选频率后提出的解释性消融，不是预注册确认。相同30种子、相同GBSA、相同模型预算；每个输入总数仍100，缺陷/概率分别强制保留1列，联合保留2列，训练重要度排名保留剩余99/98列。因此它包含替换原最低入选列的影响，不能当作无代价纯添加特征的效应。30个基线测试预测与首轮逐条一致，强制特征在30/30进入模型；未根据测试标签决定保留。\n\n'
            text+=panel(forced,'B06：结构特征强制保留的解释性消融','B06_结构强制保留诊断')
            text+='\n\n'+table(['强制臂 / 基线','配对排序差','95%区间','胜/平/负'],[[NAMES[d['variant']],signed(d['delta']),interval(d['ci95']),f"{d['wins']}/{d['ties']}/{d['losses']}"] for r in forced for d in r['paired'] if d['metric']=='spearman'])
            gains=[d for r in forced for d in r['paired'] if d['metric']=='spearman' and d['ci95'][0]>0]
            text+='\n\n**解释性诊断结论：**'+('部分条件出现正向差值，仍需独立冻结确认，不能凭事后消融发布主线优势。' if gains else '强制保留仍没有排序增量区间稳定为正，当前这组目标/表示/模型不支持继续投入结构特征扩展；不推广成所有结构目标或模型均无效。')
            count+=1
        parts['SUPPLEMENT_DEFECT']=text;count+=1
    for key,label,heading in [('SUPPLEMENT_PAPER','预训练受控比较','1.6'),('SUPPLEMENT_ACTIVE','主动学习配对比较','2.7'),('SUPPLEMENT_DEFECT','NUPACK修正重算及配对比较','5.5')]:
        if key not in parts:parts[key]=f'### {heading} 十月补实验：{label}的当前状态\n\n本轮{label}尚无完整DONE运行包，不能写成已完成。用户已明确授权服务器运行，冻结配置的正式运行正在执行或回收；当前进度见补实验证据目录的progress.json。'
    rows=[[f"[{r['run_id']}](../../evidence/supplement_20261006/{r['run_id']}/)",r['n_rows'],r['predictions_verified'],'DONE；逐预测重算通过'] for r in runs]
    parts['SUPPLEMENT_STATUS']=table(['本轮补实验运行','指标行数','独立核对预测文件数','证据状态'],rows)+'\n\n[冻结补实验协议](../项目记录/补实验冻结协议_20261006.md)；[完整汇总与文件哈希](../../evidence/supplement_20261006/index.json)；[服务器输入与终态审计](../../evidence/supplement_20261006/supplement_remote_validation_20261006.json)。另2000条结构重算与44个官方对照完成，不计作2000份回归预测；七个模型比较包合计4135份预测文件、2600个发现轮次核验。新真实docking/MD/GBSA均为0；真实候选试算仍缺统一协议、QC（quality control，质量控制）和预算。'
    return parts,count
