"""Report plots from audited, saved experiment tables. No model fitting."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.loop.search_compare import bootstrap_ci

OUT=ROOT/'docs/汇报/9月/figures/GBSA总汇报_20261005'
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'font.size':11,'svg.fonttype':'none'})
POLICIES=['matched_seed_random_walk','multi_start_hill_climb','multi_start_beam','mutation_only_ga']
LABELS=['同起点随机游走','多起点爬山','多路径束搜索','无交叉遗传算法']
COLORS=['#666666','#0072B2','#D55E00','#009E73']
PROXIES={'xgb':'五成员梯度提升代理（本地后端）','rf':'随机森林代理（固定诊断模型）'}

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f'{name}.png',dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(name)

def main():
    conf=pd.read_csv(ROOT/'runs/search_suite_confirmation_20261005/aligned.csv')
    dev=pd.read_csv(ROOT/'runs/search_suite_development_20261005/aligned.csv')
    stab=pd.read_csv(ROOT/'runs/search_suite_development_20261005/stability.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for ax,(proxy,title) in zip(axes,PROXIES.items()):
        sub=conf[conf.proxy==proxy].pivot(index='seed',columns='policy',values='best_pred_gbsa')
        lower_bounds=[]
        for i,p in enumerate(POLICIES[1:]):
            delta=(sub[p]-sub[POLICIES[0]]).to_numpy()
            mean=float(delta.mean());lo,hi=bootstrap_ci(delta)
            lower_bounds.append(lo)
            ax.errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',
                        capsize=5,color=COLORS[i+1],markersize=8)
            ax.annotate(f'{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]',(mean,i),
                        xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
        ax.axvline(0,color='black',linewidth=1,linestyle='--')
        ax.set_yticks(range(3),LABELS[1:]);ax.invert_yaxis()
        ax.set_title(title);ax.set_xlabel('最低预测值差：算法减随机游走（越负越低）')
        ax.set_ylim(2.5,-.6);ax.grid(axis='x',alpha=.2)
        # Leave enough room for the full numeric annotation at the left edge.
        span=abs(min(lower_bounds))
        ax.set_xlim(min(lower_bounds)-.28*span,.1*span)
    fig.suptitle('100 个独立确认种子：同起点、共同实际预测次数的配对比较\n点为配对均值；线为逐项 95% 自举区间；没有新序列真实 GBSA')
    save(fig,'09_四算法确认配对结果')

    fig,axes=plt.subplots(1,2,figsize=(12,5.1),constrained_layout=True)
    for ax,(proxy,title) in zip(axes,PROXIES.items()):
        for p,label,color in zip(POLICIES,LABELS,COLORS):
            sub=dev[(dev.proxy==proxy)&(dev.policy==p)]
            means=sub.groupby('budget').best_pred_gbsa.mean()
            ax.plot(means.index,means.values,'o-',label=label,color=color)
        ax.set_xscale('log',base=2);ax.set_xticks([1024,4096,8192],['1024','4096','8192'])
        ax.set_xlabel('唯一预测次数上限（早停后仍按共同实际次数比较）')
        ax.set_ylabel('最低预测 GBSA 的跨种子均值');ax.set_title(title);ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle('30 个开发种子：预算敏感性\n各预算独立运行；实际共同次数另见表；跨代理的绝对值不直接比较')
    save(fig,'10_搜索预算敏感性')

    metrics=['refit_rank_spearman','cross_proxy_rank_spearman','refit_top50_jaccard']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for ax,(proxy,title) in zip(axes,PROXIES.items()):
        values=stab[stab.proxy==proxy].groupby('policy')[metrics].mean().reindex(POLICIES).to_numpy()
        im=ax.imshow(values,vmin=-1,vmax=1,cmap='RdBu',aspect='auto')
        ax.set_xticks(range(3),['同类重训\n排序相关','换另一代理\n排序相关','重训后前50条\n交集/并集'])
        ax.set_yticks(range(4),LABELS);ax.set_title(title)
        for i in range(4):
            for j in range(3):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',
                color='white' if abs(values[i,j])>.6 else 'black')
    fig.colorbar(im,ax=axes,shrink=.8,label='排序相关：−1 到 1；集合重叠：0 到 1')
    fig.suptitle('30 个开发种子、1024 预测上限：模型变化敏感性\n同训练集；重新拟合改变模型随机种子；排序相关与集合重叠分开解释')
    save(fig,'11_跨代理与重训稳定性')

    dock=pd.read_csv(ROOT/'runs/local_dock_residual_confirmation_100seed_20261005/metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    names=['baseline_sp','predicted_dock_resid_sp','observed_dock_resid_sp']
    for i,(name,label) in enumerate(zip(names,['仅序列预测GBSA','预测对接残差辅助','实际对接残差辅助'])):
        arr=dock[name].to_numpy();mean=arr.mean();lo,hi=bootstrap_ci(arr)
        axes[0].bar(i,mean,color=COLORS[i],alpha=.85)
        axes[0].errorbar(i,mean,yerr=np.array([[mean-lo],[hi-mean]]),color='black',capsize=4)
    axes[0].set_xticks(range(3),['仅序列','预测对接','实际对接'])
    axes[0].set_ylabel('预测 GBSA 与已知 GBSA 的斯皮尔曼相关')
    axes[0].set_title('100 个新随机种子的表内残差确认')
    for i,pct in enumerate((25,50,75)):
        retention=dock[f'prefilter_{pct}_true_top10_retained'].mean()
        axes[1].bar(i,retention,color=COLORS[i+1],alpha=.85)
        axes[1].text(i,retention+.025,f'{retention:.1%}',ha='center')
    axes[1].set_xticks(range(3),['保留对接最好25%','保留50%','保留75%'],rotation=10)
    axes[1].set_ylim(0,1.08);axes[1].set_ylabel('真实 GBSA 最好10%候选的预筛保留率')
    axes[1].set_title('预筛会漏掉多少已有表内优胜序列？')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('独立对接支线：实际对接辅助依赖候选阶段能取得额外信息\n当前没有新对接成本、失败率或新序列 GBSA；不能由此宣称节省总成本')
    save(fig,'12_对接支线与预筛损失')

if __name__=='__main__':main()
