"""Seven-policy figures from saved common-prefix and stability evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/汇报/9月/figures/GBSA总汇报_20261005'
POLICIES=['matched_seed_random_walk','multi_start_hill_climb','multi_start_beam',
          'mutation_only_ga','iterated_local_search','simulated_annealing','mutation_only_adalead']
LABELS=['随机游走','多起点爬山','束搜索','无交叉遗传算法','迭代局部搜索','模拟退火','无交叉AdaLead改编版']
COLORS=['#777777','#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
PROXIES={'rf':'随机森林诊断代理','xgb':'五成员梯度提升诊断代理（本机后端）'}
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):
        fig.savefig(OUT/f'{name}.{suffix}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(name)

def forest(summary,baseline,policies,name,title):
    fig,axes=plt.subplots(1,2,figsize=(14,6.8 if len(policies)>3 else 4.7),constrained_layout=True)
    for ax,(proxy,label) in zip(axes,PROXIES.items()):
        cell=next(c for c in summary['cells'] if c['proxy']==proxy and c['budget']==1024)
        for i,p in enumerate(policies):
            r=next(c for c in cell['paired'] if c['policy']==p and c['baseline']==baseline)
            mu=r['mean_delta'];lo,hi=r['ci95'];color=COLORS[POLICIES.index(p)]
            ax.errorbar(mu,i,xerr=np.array([[mu-lo],[hi-mu]]),fmt='o',capsize=4,color=color)
            ax.annotate(f'{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]',(mu,i),xytext=(0,12),
                        textcoords='offset points',ha='center',fontsize=8)
        ax.axvline(0,color='#333333',linestyle='--',linewidth=1)
        ax.set_yticks(range(len(policies)),[LABELS[POLICIES.index(p)] for p in policies])
        ax.set_ylim(len(policies)-.4,-.6);ax.set_title(label);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('配对最低预测GBSA差（前者减对照；负值更低）')
        ax.margins(x=.3)
    fig.suptitle(title+'\n100个独立确认种子；七者共同实际预测次数；逐项95%自举区间；尚无新候选真实GBSA')
    save(fig,name)

def main():
    root=ROOT/'runs/search_seven_confirmation_20261005'
    summary=json.loads((root/'summary.json').read_text('utf-8'))
    forest(summary,POLICIES[0],POLICIES[1:],'14_七算法确认对随机',
           '七算法统一比较：六种策略相对同起点随机游走')
    forest(summary,POLICIES[1],POLICIES[4:],'15_三新算法确认对爬山',
           '新增机制是否改善现有爬山：三种新策略相对爬山')
    dev=pd.read_csv(ROOT/'runs/search_seven_development_20261005/aligned.csv')
    stab=pd.read_csv(ROOT/'runs/search_seven_development_20261005/stability.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5.3),constrained_layout=True)
    for ax,(proxy,label) in zip(axes,PROXIES.items()):
        for p,text,color in zip(POLICIES,LABELS,COLORS):
            sub=dev[(dev.proxy==proxy)&(dev.policy==p)]
            mean=sub.groupby('budget').best_pred_gbsa.mean()
            ax.plot([0,1],mean.values,'o-',color=color,label=text)
        ax.set_xticks([0,1],['1024上限','4096上限']);ax.set_title(label)
        ax.set_ylabel('共同次数内最低预测GBSA的均值');ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('30个开发种子的两档预算诊断\n每档独立运行；各档共同次数不同，不能视为同一运行的学习曲线')
    save(fig,'16_七算法预算诊断')
    columns=['refit_rank_spearman','cross_proxy_rank_spearman','refit_top50_jaccard']
    fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    for ax,(proxy,label) in zip(axes,PROXIES.items()):
        values=stab[stab.proxy==proxy].groupby('policy')[columns].mean().reindex(POLICIES).to_numpy()
        im=ax.imshow(values,vmin=-1,vmax=1,cmap='RdBu',aspect='auto')
        ax.set_yticks(range(7),LABELS);ax.set_title(label)
        ax.set_xticks(range(3),['同类重训排序相关','跨代理排序相关','重跑前50集合交并比'],rotation=12)
        for i in range(7):
            for j in range(3):
                ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',
                        color='white' if abs(values[i,j])>.6 else 'black')
    fig.colorbar(im,ax=axes,shrink=.8,label='相关系数：−1至1；集合交并比：0至1')
    fig.suptitle('七算法对模型变化的敏感性：30开发种子、1024上限\n原前50条重评分相关与独立重跑集合重叠分列；按各自完整停止档案诊断')
    save(fig,'17_七算法模型敏感性')

if __name__=='__main__':main()
