# 《面向 GBSA 昂贵标签的主动学习序列优化》逐页详细讲解

> 目的：严格按照当前 PPT 的汇报顺序，把整个项目从第一页到最后一页讲清楚。本文不是简单复述幻灯片，而是为每一个概念补充：**定义、为什么需要、具体怎么做、输入、输出、与下一步如何衔接、汇报时应该怎么说**。
>
> 重要说明：本文以当前 PPT 的术语、顺序和数字为主。PPT 没有给出某些旧特征（例如 BLBFE、cm_score）的精确定义或代码公式时，本文不会凭空补写项目内部定义，而会明确标注“需以代码/实验记录为准”。

---

## 0. 整个项目先用一句话理解

这个项目的核心不是“再换一个模型预测 GBSA”，而是把原来的一次性静态机器学习，升级为一个可以循环的 **模型—昂贵评价—再训练闭环**：

```text
已有 RNA–GBSA 数据
        ↓
训练概率代理模型（surrogate）
        ↓
对大量未测 RNA 同时给出预测值和不确定性
        ↓
主动选择最值得获得真实 GBSA 的一批序列
        ↓
昂贵实验 / MD / MMGBSA / 真实评价
        ↓
获得新 GBSA 标签
        ↓
加入训练集并重新训练
        ↓
下一轮候选选择
        ↓
直到模型学习收益饱和，或转入低 GBSA 优化阶段
```

因此，整份 PPT 的逻辑链条是：

```text
问题定义
→ 为什么现有模型进入瓶颈
→ 特征筛选
→ Sequence + Physics 表示与融合
→ 概率代理模型
→ 参考 riboswitch Bayesian Optimization 框架
→ 批量候选选择
→ 新标签回流
→ 停止标准
```

---

# 第 1 页：封面——核糖开关工作进展报告

## 1.1 这一页在整个汇报中的作用

封面本身没有算法内容，但你在进入下一页之前要让听众知道，这不是一个完全新开的题目，而是**现有核糖开关 GBSA 预测工作进入下一阶段**。

所以开场不要直接说“今天我要讲 Active Learning”。更自然的逻辑是：

> 前期我们已经围绕核糖开关序列到 GBSA 的预测进行了大量特征工程和模型比较；这次汇报主要讨论，在现有预测性能进入平台以后，如何把模型从“静态预测器”升级成能够指导下一批昂贵标签获取的主动学习系统。

这样听众会立刻知道：

- 前半段工作已经存在；
- 这次不是推翻过去；
- 新阶段的关键词是“高价值标签获取”和“闭环”。

---

# 第 2 页：当前问题——Active Learning 和 Bayesian Optimization 交叉的昂贵评价函数优化问题

这一页是**整份 PPT 的问题定义页**。

---

## 2.1 什么是“昂贵评价函数”？

我们最终关心的是一条 RNA 序列对应的 GBSA：

\[
y=f(x)=GBSA(x)
\]

其中：

- \(x\)：RNA sequence；
- \(f\)：真实的昂贵评价过程；
- \(y\)：得到的 GBSA 标签。

这里的“昂贵”不是数学术语，而是工程事实：获取一个新的真实 GBSA 往往需要远高于一次机器学习前向预测的计算或实验成本。

如果真实 GBSA 极其便宜，那么根本不需要主动学习——直接把候选空间全部算完即可。正因为真实标签昂贵，所以才需要用模型帮助决定“下一次最值得算谁”。

---

## 2.2 什么是 Oracle（真实/昂贵评价器）？

在 Active Learning 和 Bayesian Optimization 里，**Oracle** 指的是能返回真实标签的过程。

对于本项目，可以抽象成：

```text
RNA sequence x
      ↓
Expensive Oracle
（实验 / MD / MMGBSA / 统一GBSA流程）
      ↓
真实标签 y = GBSA(x)
```

机器学习模型给的是预测：

\[
\hat y
\]

Oracle 返回的才是用于新增训练数据的真实观测：

\[
y
\]

整个主动学习系统的关键，就是在 Oracle 很贵的情况下，减少无效查询。

---

## 2.3 传统方法是什么？

PPT 中传统路线是：

```text
已有数据
↓
训练模型：RF / XGB / SVR / Ensemble
↓
测试集评估：Spearman / MAE / RMSE
↓
多 seed 验证
↓
结束
```

这叫**静态监督学习（static supervised learning）**。

用数据集表示：

\[
D_0=\{(x_i,y_i)\}_{i=1}^N
\]

训练模型：

\[
D_0 \rightarrow \hat f
\]

然后在测试集评价模型，得到一个最终性能数字，任务结束。

它的核心特点是：

> 模型只消费已有数据，不决定未来新增什么数据。

---

## 2.4 什么是概率代理模型（Probabilistic Surrogate）？

**Surrogate model，代理模型**，指的是用一个便宜的模型去近似昂贵真实函数：

\[
\hat f(x)\approx f(x)
\]

普通回归模型通常只输出：

\[
x\rightarrow \hat y
\]

而概率代理希望输出：

\[
x\rightarrow \mu(x),\sigma(x)
\]

其中：

- \(\mu(x)\)：模型对 GBSA 的预测均值；
- \(\sigma(x)\)：模型对这个预测有多不确定。

为什么要多一个 \(\sigma\)？

因为主动学习不能只问“模型觉得谁最好”，还要问：

> “模型最不了解谁？”

这就是后面候选选择的依据。

---

## 2.5 什么是 Active Learning？

**Active Learning（主动学习）**的主要目标是：

> 在标签昂贵时，用尽量少的新标签，最大程度提高模型学习效率。

假设有 100 万条未标注候选 RNA，但下一轮只能获取 100 条真实 GBSA。

随机方法是：

\[
1,000,000 \xrightarrow{random}100
\]

主动学习则利用当前模型：

\[
1,000,000\xrightarrow{surrogate+acquisition}100
\]

重点优先选择：

- 模型不确定的区域；
- 当前训练集覆盖不足的区域；
- 能增加 batch 多样性的区域；
- 兼顾一定实际优化价值的区域。

因此 Active Learning 首先回答：

> **“测谁最能让模型学得更好？”**

---

## 2.6 什么是 Bayesian Optimization？

**Bayesian Optimization（贝叶斯优化，BO）**面对的也是昂贵函数，但目标更偏向：

> 用尽量少的昂贵评价，尽快找到函数最优点。

如果最终目标是寻找低 GBSA RNA，则可写成：

\[
\min_x GBSA(x)
\]

BO 主要回答：

> **“测谁最有机会帮助我找到更优的 RNA？”**

---

## 2.7 为什么说我们的项目处在 Active Learning 和 BO 的交叉区域？

因为我们有两个阶段性目标。

当前阶段，模型泛化能力仍然有限，所以主要任务是：

\[
\boxed{提高\ GBSA\ surrogate\ 的预测能力}
\]

这更接近 Active Learning。

等 surrogate 足够可靠以后，目标会逐渐转成：

\[
\boxed{寻找更优的低\ GBSA\ sequence}
\]

这更接近 Bayesian Optimization。

因此不是二选一，而是一个连续过渡：

```text
前期：Active Learning
目标：提高模型、补足信息盲区
          ↓
中后期：AL + BO
目标：在学习和优化之间平衡
          ↓
后期：Bayesian Optimization
目标：重点寻找低 GBSA 候选
```

---

## 2.8 这一页与下一页如何衔接？

这一页回答的是：

> “为什么我们要做闭环？”

下一页必须回答：

> “为什么现在就是应该转向闭环，而不是继续换模型？”

这就进入当前瓶颈。

---

# 第 3 页：当前瓶颈——从模型优化逐步转向高价值数据获取

这一页承担的是**研究路线转向的证据**。

PPT 当前给出的核心信息包括：

- seed42：0.506（历史固定划分离群值；旧版此处写 0.511 系笔误，正式口径统一为 0.506）；
- Spearman 正式冻结结果 \(\rho=0.375\)（100-seed 秩融合，CI [0.367, 0.382]）；30-seed 开发口径约 0.38–0.40；
- 采用多随机种子稳定评价；
- 不用单个最好 seed 作为主要结论；
- 更多模型不等于更多有效信息；
- 更多特征不等于更高泛化性能；
- 当前需要改变的是有效标签覆盖和标签质量。

这里需要逐个理解。

---

## 3.1 什么是 random seed？

机器学习实验中存在很多随机过程，例如：

- train/test split；
- bootstrap sampling；
- Random Forest 中的样本和特征随机抽样；
- neural network initialization；
- mini-batch 顺序。

Random seed 是控制这些随机过程的一组固定初始状态。

同一个模型在不同 seed 下可能得到：

```text
seed 1  → ρ = 0.36
seed 2  → ρ = 0.39
seed 3  → ρ = 0.41
seed 42 → ρ = 0.506
```

所以单个 seed 特别好，并不意味着模型真的有 0.506 的稳定能力。

---

## 3.2 什么是 Multi-seed Evaluation？

就是重复多个不同随机种子：

\[
\rho_1,\rho_2,\ldots,\rho_M
\]

然后关注：

- mean / median；
- standard deviation；
- confidence interval；
- paired improvement；
- win rate。

它回答的是：

> 这个方法是普遍有效，还是只在某个幸运划分上有效？

因此 PPT 中“不采用单个最好 seed 作为主要结论”是非常重要的研究规范。

---

## 3.3 Spearman \(\rho\) 是什么？

Spearman Rank Correlation 衡量的是**排名的一致性**。

范围：

\[
-1\le \rho_s\le1
\]

- \(\rho=1\)：真实排序与预测排序完全一致；
- \(\rho=0\)：没有明显单调排序关系；
- \(\rho=-1\)：排序完全相反。

它与 Pearson 的区别是，Spearman 更关注 rank，而不是要求线性关系。

为什么本项目很关注 Spearman？

因为后续往往需要对大量 candidate 按 GBSA 潜力进行排序。即使绝对值存在一定偏差，如果排序稳定，模型仍然有候选筛选价值。

---

## 3.4 MAE 是什么？

MAE = Mean Absolute Error，平均绝对误差：

\[
MAE=\frac1N\sum_{i=1}^{N}|y_i-\hat y_i|
\]

它回答：

> 平均每条样本的 GBSA 数值预测差多少。

MAE 越低越好。

---

## 3.5 RMSE 是什么？

RMSE = Root Mean Squared Error：

\[
RMSE=\sqrt{\frac1N\sum_{i=1}^{N}(y_i-\hat y_i)^2}
\]

因为误差被平方，所以 RMSE 对大错误更敏感。

因此：

- Spearman：看排序；
- MAE：看平均绝对误差；
- RMSE：更惩罚极端预测错误。

三者关注角度不同。

---

## 3.6 为什么“更多模型 ≠ 更多有效信息”？

这是这一页最重要的逻辑。

即使我们训练：

- RF；
- XGB；
- SVR；
- CNN；
- GNN；
- Transformer；
- GP；
- Ensemble；

它们仍然是在同一个已有标签集合上学习：

\[
D_0\rightarrow Model_1,Model_2,\ldots
\]

如果某个 sequence space 区域根本没有标签，模型无法通过“换架构”凭空获得那个区域的真实物理信息。

因此，当多种模型都进入类似平台以后，研究瓶颈很可能从：

\[
Model\ Capacity
\]

逐渐转移到：

\[
Data\ Information\ /\ Label\ Coverage\ /\ Label\ Quality
\]

这就是主动学习阶段的合理性。

---

## 3.7 为什么“更多特征 ≠ 更高泛化性能”？

假设样本只有约 2000 条，但物理/结构特征有数百甚至上千维。

大量特征中可能存在：

- 与 GBSA 无关的噪声；
- 高度重复的变量；
- 偶然相关；
- 高维带来的过拟合。

因此维度增加不等于“信息增加”。

真正需要的是：

> 有效、稳定、非冗余、能在未见样本上泛化的信息。

这就自然引出下一页的 Feature Selection。

---

## 3.8 PPT 中“已经探索过什么”怎么理解？

PPT 列出了四类历史探索：

### A. 特征工程

包括 mutation_only、energy_norm、cm_score、BLBFE、PseKNC、motif、local_RNAfold、structure_guided、pairwise、target_encoding、NUPACK_partition、Turner 等。

这些方法共同的目的，是把 RNA sequence 或结构/能量状态转换成机器学习可用的描述量。

其中若 PPT 没有给出某些缩写的项目内公式，例如 `BLBFE`、`cm_score`，正式汇报时不要凭缩写自行定义，应以代码/实验记录为准。

### B. 模型架构

- ExtraTrees：大量高度随机化决策树的集成；
- SVR：支持向量回归；
- Ridge：带 L2 正则的线性回归；
- CNN：利用卷积学习局部 motif；
- GNN：把 RNA 作为图结构处理；
- Transformer：用 self-attention 学习位置间依赖；
- XGBoost：梯度提升树；
- Gaussian Process：天然输出预测分布；
- Ensemble：多个模型组合；
- RBM：Restricted Boltzmann Machine，受限玻尔兹曼机。

### C. 预训练模型

如 RNA-FM、RNAErnie、RNA-BERTa。

它们通过大规模无标签 RNA 预训练得到 sequence representation，再用于下游预测。

这里 PPT 的论点不是“预训练模型没用”，而是：

> 即使增加 representation complexity，也没有自动解决当前数据/标签信息瓶颈。

### D. 优化策略

- Grid Search：枚举超参数组合；
- Optuna：自动超参数优化；
- MI 特征选择；
- Target Encoding；
- 零样本方法。

这些探索说明模型侧已经投入大量精力，因此下一阶段应更加关注标签信息本身。

---

## 3.9 这一页怎样衔接下一页？

一句话即可：

> 既然继续叠加模型和特征没有稳定突破，那么下一阶段要先把已有信息变得更干净、更紧凑，再构建可以量化不确定性的代理模型，最后让它主动决定下一批值得获取 GBSA 的序列。

这就进入技术路线。

---

# 第 4 页：技术路线——特征筛选 → 多模态表示 → 概率代理 → 批量主动学习

这一页是整份 PPT 的**总方法框架**。

当前 PPT 的主线是：

```text
37 类物理/结构特征
          │
Spearman + MI + 去冗余
          │
筛选后的物理特征表示
          │
          ├──────────────┐
          │              │
RNA Sequence        Physics Representation
     │                    │
Sequence Encoder          │
     │                    │
Sequence Embedding        │
          └──────┬────────┘
                 ↓
          Feature Fusion
                 ↓
      Probabilistic Surrogate
                 ↓
         μ(GBSA) + σ(GBSA)
                 ↓
        Batch Acquisition
                 ↓
           新序列实验
```

这一页一定要把“表示—模型—采集”三个层级区分开。

---

## 4.1 什么是 37 类物理/结构特征？

“37 类”首先表示 **feature category / feature block**，并不等于只有 37 个数值。

例如一个 structure block 内部可能包含几十个位置、窗口或统计量，所以最终原始维度可能远大于 37。

因此要区分：

\[
Feature\ Categories\neq Feature\ Dimensions
\]

这些物理/结构特征希望描述 RNA 的：

- 热力学稳定性；
- 二级结构；
- pairing 情况；
- 局部结构模式；
- ensemble 状态；
- 其他能量/结构统计。

---

## 4.2 为什么先做 Spearman + MI + 去冗余？

因为原始 physics features 可能存在三个问题：

1. **无关**：与 GBSA 没有稳定信息关系；
2. **不稳定**：某个 split 看起来相关，换 split 就消失；
3. **冗余**：多个特征实际上描述同一物理现象。

所以目标不是“维度越低越好”，而是得到：

\[
\boxed{GBSA-relevant+Stable+Nonredundant}
\]

的 physics representation。

---

## 4.3 为什么这里叫“筛选后的物理特征表示”，而不一定叫 Low-dimensional Embedding？

Spearman + MI + redundancy removal 本质上属于 **Feature Selection**：

\[
[f_1,f_2,\ldots,f_P]\rightarrow[f_{i_1},f_{i_2},\ldots,f_{i_K}]
\]

特征还是原来的真实物理变量，只是选得更少。

而严格意义上的 embedding 通常意味着产生新的 latent vector，例如：

- PCA；
- Autoencoder；
- MLP encoder。

例如：

\[
40D\rightarrow PCA\rightarrow10D
\]

才更接近 Low-dimensional Physics Embedding。

所以 PPT 当前使用“筛选后的物理特征表示”更严谨。

---

## 4.4 什么是 Sequence Encoder？

Sequence Encoder 的目标是：

\[
RNA\ Sequence\rightarrow Numeric\ Representation
\]

最简单的方法是 one-hot。

每个碱基：

\[
A=[1,0,0,0]
\]
\[
C=[0,1,0,0]
\]
\[
G=[0,0,1,0]
\]
\[
U=[0,0,0,1]
\]

如果使用 13 个 mutation sites，则可得到 52 维 one-hot 表示。

更复杂的 encoder 可以是：

- k-mer；
- CNN；
- Transformer；
- pretrained RNA encoder。

输出统一记作：

\[
h_{seq}
\]

也就是 Sequence Embedding / Sequence Representation。

---

## 4.5 为什么 Sequence 不能和 physics 一起被普通 Spearman 直接筛掉？

因为 sequence 是我们的**设计变量本身**。

最终我们希望回答：

> 下一条应该设计/测量哪个 RNA sequence？

physics features 是 sequence 的派生描述，sequence 不能被当成普通辅助 feature 随意删掉。

因此更合理的结构是：

```text
Sequence Branch：始终保留
Physics Branch：进行统计筛选和去冗余
最后再融合
```

---

## 4.6 什么是 Feature Fusion？

Feature Fusion 是把两种来源的信息放进同一个 surrogate：

\[
h_{seq}+h_{phy}\rightarrow h_{fusion}
\]

它要回答的科学问题是：

> 已经知道 sequence 以后，physics 是否还能提供额外的、互补的 GBSA 信息？

这比单纯“多加一些 feature”更重要。

---

## 4.7 什么是 Probabilistic Surrogate？

融合表示进入代理模型后，我们不只需要：

\[
\hat y=GBSA
\]

还需要：

\[
\mu(x),\sigma(x)
\]

其中：

- \(\mu\)：预测值；
- \(\sigma\)：预测不确定性。

它把“预测问题”变成“预测 + 置信度问题”。

---

## 4.8 什么是 Batch Acquisition？

Acquisition = 采集函数/候选选择策略。

它对 candidate pool 中每个 RNA 计算“被下一轮测量的价值”。

Batch Acquisition 表示：

> 不是每次只选 1 条，而是一轮并行选择一批，例如 100 条。

因此它既要考虑单条 candidate 的价值，也要考虑 batch 内是否重复。

---

# 第 5 页：特征选择与融合

这一页是技术路线中第一个需要讲细的方法页。

PPT 当前链条是：

```text
原始特征
↓
数据清洗 Missing / Constant
↓
Spearman：单调相关性筛选
↓
MI：非线性信息筛选
↓
去除冗余：Feature-Feature Correlation
↓
模型验证：Nested CV
↓
最终特征集合
↓
Sequence + Selected Features
↓
Probabilistic Surrogate
```

同时文字描述还强调：

```text
Spearman + MI → Stability → Redundancy Removal
```

严格实现中，Stability 和 Nested CV 应当贯穿整个 feature-selection pipeline，而不是只理解为最后才做一次验证。

---

## 5.1 第一步：原始特征（Raw Features）

假设有 \(N\) 条 RNA 和 \(P\) 个 physics features：

\[
X_{phy}\in\mathbb R^{N\times P}
\]

目标标签：

\[
y\in\mathbb R^{N}
\]

每一行是一条 RNA，每一列是一个特征。

目标是从 \(P\) 个候选特征中得到一个紧凑可靠的集合。

---

## 5.2 第二步：Missing——缺失特征处理

第 \(j\) 个 feature 的缺失率：

\[
r_j=\frac{N_{missing,j}}{N}
\]

缺失严重时：删除。

缺失很少时：可在训练集内部使用 median 等方式插补。

关键原则：

> 任何 imputation 参数只能在训练 fold 上拟合，再应用于 validation/test。

否则会把测试信息带入训练过程。

---

## 5.3 第三步：Constant / Near-constant Feature

如果：

\[
Var(X_j)=0
\]

说明所有 RNA 在这个 feature 上完全相同，没有区分能力，应删除。

Near-constant 指绝大多数样本取同一个值、变化极少，也往往没有足够信息。

这一步属于**数据质量控制**，还没有使用 GBSA 标签。

---

## 5.4 第四步：Spearman——单调相关性筛选

对每一个 feature \(X_j\) 计算：

\[
\rho_j=Spearman(X_j,Y)
\]

Spearman 看的是 rank relationship。

为什么通常看：

\[
|\rho_j|
\]

因为正相关和负相关都可能有预测价值。

例如：

- \(\rho=+0.4\)：feature 越大，GBSA 总体越大；
- \(\rho=-0.4\)：feature 越大，GBSA 总体越小。

两者都说明 feature 与目标有明显单调关系。

Spearman 的作用：

> 快速找到与 GBSA 存在稳定单调信息的候选变量。

---

## 5.5 Spearman 为什么不够？

如果真实关系是 U 型：

\[
Y=(X-c)^2
\]

那么 feature 明明包含 GBSA 信息，但因为关系“先下降再上升”，Spearman 可能接近 0。

因此需要 MI 补充非单调的非线性依赖。

---

## 5.6 第五步：MI——Mutual Information（互信息）

Mutual Information：

\[
I(X;Y)
\]

可以直观理解为：

> 知道 feature \(X\) 之后，我们对 GBSA \(Y\) 的不确定性减少了多少。

如果 X 和 Y 独立：

\[
I(X;Y)\approx0
\]

如果存在较强依赖：

\[
I(X;Y)>0
\]

MI 不提供“正相关/负相关方向”，但能捕捉比 Spearman 更一般的非线性依赖。

因此二者角色是：

| 方法 | 主要回答的问题 |
|---|---|
| Spearman | 是否有稳定单调关系？ |
| MI | 是否存在一般非线性信息依赖？ |

---

## 5.7 Spearman 和 MI 怎么组合？

不建议直接把原始数值相加，因为二者尺度不同。

更稳妥的方法是 rank aggregation。

例如：

\[
R_j=\frac{R_{Spearman,j}+R_{MI,j}}{2}
\]

或者分别取 Top-K，再求 union/intersection，并由内层验证确定规则。

重要的是：

> Spearman / MI 是候选生成器，不是最终真理。

最终是否保留要看模型泛化。

---

## 5.8 第六步：Stability——稳定性

某个 feature 如果只在一个 split 排名很高，但换一个 split 就消失，说明它可能依赖偶然抽样。

定义选择频率：

\[
SelectionFrequency_j=
\frac{\text{feature j 被选中的次数}}{\text{总重复次数}}
\]

例如：

- 29/30 次被选中：高度稳定；
- 5/30 次被选中：不稳定。

Stability 的作用：

> 排除只在某一数据划分中偶然显著的特征。

---

## 5.9 第七步：Redundancy Removal——去冗余

前面的 Spearman/MI 在问：

> “这个 feature 和 GBSA 有没有信息关系？”

去冗余问的是：

> “这些 feature 之间是不是重复表达同一信息？”

计算 feature–feature correlation：

\[
C_{ij}=|Spearman(F_i,F_j)|
\]

例如：

\[
C_{12}=0.96
\]

说明 F1、F2 高度相关。

可以设一个候选阈值，例如 0.85，但阈值应通过训练侧验证，不应把 0.85 当成自然定律。

高度冗余组内优先保留：

- GBSA relevance 更强的；
- MI 更高的；
- stability 更高的；
- 物理意义更明确的；
- 缺失更少、数值更稳定的。

---

## 5.10 什么是 Feature Block？

Feature block 是一组来源和物理意义相近的特征，例如：

```text
Pairing block
├─ pairing probability
├─ paired fraction
├─ entropy
└─ local pairing statistics
```

因此“5–10 个核心 feature blocks”不等于只有 5–10 维。

一个 block 内部仍然可能有多个实际数值维度。

按 block 管理的优点：

- 防止高维 block 因维度多而“碰巧”出现更多高相关变量；
- 更容易做消融；
- 生物学解释更清楚。

---

## 5.11 什么是 Biological Rescue Feature？

PPT 计划额外保留少量“单变量相关不高，但有明确结构意义”的 feature。

原因是单变量筛选可能漏掉 interaction：

\[
Y=f(X_1,X_2)
\]

但：

\[
corr(X_1,Y)\approx0,\quad corr(X_2,Y)\approx0
\]

只有二者共同出现时才有效。

因此可以保留少量具有明确生物/物理理由的 rescue features，再通过模型验证决定它们是否真的有增益。

注意：

> Biological rescue 不是“专家觉得重要就永远保留”，仍要接受 held-out evaluation。

---

## 5.12 第八步：Nested CV——嵌套交叉验证

Nested Cross-Validation 有两层：

- Outer CV：负责最终泛化评估；
- Inner CV：负责特征选择规则和超参数选择。

一句话记忆：

> **Outer = 最终考试；Inner = 选方案。**

为什么需要 Nested CV？

因为 Spearman 和 MI 都使用了 GBSA 标签。如果先在全部数据上选好 feature，再做普通 CV，那么所谓 test fold 的标签实际上已经间接参与过 feature selection，会造成 leakage。

正确结构：

```text
全部数据
  │
  ├──────── Outer Train
  │             │
  │             └─ Inner CV
  │                   │
  │                   ├─ Missing/Constant
  │                   ├─ Spearman
  │                   ├─ MI
  │                   ├─ Stability
  │                   ├─ Redundancy Removal
  │                   ├─ 选择 Top-K / threshold
  │                   └─ 调模型参数
  │
  └──────── Outer Test（完全冻结）
```

Inner CV 决定：

- Spearman Top-K；
- MI Top-K；
- 综合规则；
- redundancy threshold；
- 最终 feature 数；
- model hyperparameters。

Outer Test 只做一件事：

> 评价整套“筛选 + 训练”流程面对完全新 RNA 时能有多好。

因此严格来说，Nested CV 不是“去冗余以后再做的一步”，而是包住整个 feature-selection pipeline 的评估框架。

---

## 5.13 最终特征集合怎么定义？

最终集合不是“Spearman 最大的 10 个”，而应该定义为：

> 在训练集内部经过 relevance、stability 和 redundancy 控制，并在 Nested CV 中证明具有泛化价值的物理/结构特征集合。

四个关键词：

\[
\boxed{Relevant+Stable+Nonredundant+Generalizable}
\]

---

# 第 5 页后半：特征融合

---

## 5.14 为什么融合 Sequence + Selected Physics？

Sequence 和 Physics 描述的不是同一层信息。

Sequence 回答：

> 这条 RNA 是什么？

Physics 回答：

> 这条 RNA 可能表现出什么结构/能量状态？

理想情况下，我们希望 physics 提供 sequence 之外的 complementary information。

验证这个问题最重要的三个对照是：

\[
Sequence\ Only
\]

\[
Physics\ Only
\]

\[
Sequence+Physics
\]

如果融合模型稳定优于 Sequence Only，才说明 physics 确实带来额外信息。

---

## 5.15 Early Fusion 是什么？

Early Fusion = 直接拼接。

若：

\[
h_{seq}\in\mathbb R^{d_s}
\]

\[
h_{phy}\in\mathbb R^{d_p}
\]

则：

\[
h=[h_{seq};h_{phy}]
\]

例如：

- sequence 52 维；
- physics 25 维；

则融合后 77 维。

优点：

- 简单；
- 参数少；
- 小样本稳定；
- 最适合作为第一版 fusion baseline。

---

## 5.16 Gated Fusion 是什么？

Gated Fusion 让模型自己学习：

> 对这条 RNA，physics branch 应该被信任多少？

一个简单形式：

\[
g=\sigma(W[h_{seq};h_{phy}]+b)
\]

再让 gate 控制 physics 信息：

\[
h=h_{seq}+g\odot h_{phy}
\]

当 \(g\) 大时，更依赖 physics；当 \(g\) 小时，更依赖 sequence。

优点：更灵活。

风险：参数更多，小样本下更容易过拟合。

因此合理顺序是：

```text
Early Fusion baseline
↓
只有在稳定证明需要更复杂融合时
↓
Gated Fusion
```

---

## 5.17 特征融合最终输出什么？

不是最终 GBSA 标签本身，而是一个供 surrogate 使用的统一表示：

\[
h_{fusion}
\]

然后：

\[
h_{fusion}\rightarrow Probabilistic\ Surrogate
\]

这就自然进入下一页“代理模型选择”。

---

# 第 6 页：代理模型选择 + 参考文献框架

PPT 参考：

> *Iterative design of a NAND hybrid riboswitch by deep batch Bayesian optimization*（2026，Nucleic Acids Research）

这一页最重要的作用不是告诉大家“我们要复制论文”，而是：

> **证明在 riboswitch 问题中，用不确定性驱动的 batch Bayesian Optimization 做多轮真实实验闭环已经有直接先例。**

然后我们再说明：

> 我们借它的 closed-loop framework，但将输入、目标和 acquisition 改造成 GBSA 预测导向。

---

## 6.1 参考论文的整体框架

PPT 当前流程：

```text
大量 riboswitch sequences
↓
Sequence Encoder Pre-training
↓
MLM + Triplet Loss
↓
少量真实实验功能标签
↓
Deep Ensemble
↓
预测功能 + 预测不确定性
↓
UCB
↓
Kriging Believer
↓
每轮 16 条 RNA
↓
in vivo validation
↓
真实标签返回
↓
模型重训
```

这和我们的闭环结构高度相似。

---

## 6.2 Sequence Encoder Pre-training 是什么？

当真实功能标签很少时，直接训练一个大 encoder 容易过拟合。

因此先使用大量无标签 RNA：

\[
\{x_1,x_2,\ldots,x_M\}
\]

做 self-supervised pretraining，让模型先学一般 sequence regularity，再用少量真实标签训练下游任务。

---

## 6.3 MLM 是什么？

MLM = Masked Language Modeling。

例如：

```text
原序列：A C G U A C G
掩码后：A C [MASK] U A C G
```

模型需要预测被遮住的 G。

通过大量这种任务，encoder 学习：

- nucleotide context；
- motif；
- 局部依赖；
- 序列规律。

MLM 本身不是在预测 GBSA，而是在学 representation。

---

## 6.4 Triplet Loss 是什么？

Triplet Loss 使用：

- Anchor \(A\)；
- Positive \(P\)；
- Negative \(N\)。

希望：

\[
d(A,P)<d(A,N)
\]

一个常见形式：

\[
L=\max(d(A,P)-d(A,N)+margin,0)
\]

作用：让相似 sequence 的 embedding 更接近，不相似 sequence 更远。

它依然属于 representation learning，而不是直接 GBSA regression。

---

## 6.5 Deep Ensemble 在参考论文里为什么重要？

因为 BO 需要 uncertainty。

一个单独神经网络通常只给一个预测值，而多个独立模型可以通过 prediction disagreement 估计 epistemic uncertainty。

这使得：

\[
Prediction+Uncertainty
\]

可以直接用于 acquisition。

---

## 6.6 UCB 是什么？

UCB = Upper Confidence Bound：

\[
UCB(x)=\mu(x)+\kappa\sigma(x)
\]

其中：

- \(\mu(x)\)：模型预测性能；
- \(\sigma(x)\)：预测不确定性；
- \(\kappa\)：控制 exploration 的强度。

它同时考虑：

- Exploitation：预测值高；
- Exploration：不确定性高。

为什么参考论文用 UCB？

因为它是在做 maximization 类设计目标。

如果我们后期目标是更低 GBSA，则可能使用：

\[
LCB(x)=\mu(x)-\kappa\sigma(x)
\]

即 Lower Confidence Bound。

但当前提高 surrogate 的阶段，不应该机械照搬纯 LCB/UCB。

---

## 6.7 Kriging Believer 是什么？

Batch BO 面临一个问题：

> 如果一次选 16 条，直接拿 UCB Top-16，可能选出 16 条彼此非常相似的 RNA。

Kriging Believer 的思想：

1. 先选第 1 条 \(x_1\)；
2. 真实标签还没回来，暂时“相信”其标签等于模型预测 \(\mu(x_1)\)；
3. 将这个伪标签临时加入模型；
4. 再选第 2 条；
5. 依次重复直到 batch 填满。

这样能够让同一个 batch 更分散，减少重复采样。

---

## 6.8 为什么参考论文每轮 16 条，我们可以每轮 100 条？

Batch size 不是算法上的固定常数。

它取决于：

- 实验/计算 throughput；
- 单轮预算；
- 周期；
- 并行能力；
- batch 内 redundancy 风险。

参考论文的 16 条是其工作流选择，不代表本项目必须 16。

但我们若每轮 100 条，更需要做好 batch diversity。

---

## 6.9 参考模型和我们的模型到底有什么关联？

### 参考论文

```text
Sequence
↓
Pretrained Encoder
↓
Deep Ensemble
↓
UCB
↓
Kriging Believer
↓
Experiment
```

### 我们

```text
Sequence + Selected Physics
↓
Feature Fusion
↓
Probabilistic Surrogate
↓
前期：Uncertainty + Coverage + Diversity
后期：逐渐转向 LCB / Optimization
↓
Expensive GBSA Oracle
↓
Retrain
```

所以我们的关系是：

\[
\boxed{借鉴闭环范式，不照搬输入、目标和采集函数}
\]

---

# 第 7 页：概率代理模型

这一页回答：

> 我们具体用什么 surrogate，怎么得到 \(\mu\) 和 \(\sigma\)？

PPT 当前分三层：

1. Baseline 1：现有强传统模型；
2. Baseline 2：Gaussian Process；
3. Main Candidate：Deep Ensemble。

---

## 7.1 为什么必须保留传统模型 baseline？

现有强传统模型包括：

- Random Forest；
- Extra Trees；
- Gradient Boosting Regression；
- XGBoost；
- SVR。

它们的作用不是继续无止境优化，而是建立一个必须被新 surrogate 对比的预测基线。

如果新概率模型 uncertainty 很漂亮，但 Spearman 大幅下降，就不能自动认为它更好。

因此新模型至少要比较：

- prediction performance；
- stability；
- uncertainty usefulness。

---

## 7.2 Random Forest（RF）是什么？

RF 是多棵决策树的集成。

回归输出可写作：

\[
\hat y=\frac1M\sum_{m=1}^{M}T_m(x)
\]

不同树使用 bootstrap sample 和随机 feature subsets。

优点：

- 对表格数据强；
- 能拟合非线性；
- 对小中等数据较稳健。

---

## 7.3 Extra Trees（ET）是什么？

Extra Trees = Extremely Randomized Trees。

与 RF 类似，但分裂阈值更随机，进一步增加树之间差异。

作用：通过更多随机性降低 ensemble correlation，有时可以降低方差。

---

## 7.4 GBR 是什么？

Gradient Boosting Regression 采用顺序加法：

\[
F_M(x)=F_{M-1}(x)+\eta h_M(x)
\]

每一棵新树都在纠正前面模型的残差。

它与 RF 的“并行随机树平均”思想不同。

---

## 7.5 XGBoost 是什么？

XGBoost 是正则化的梯度提升树框架，擅长：

- 非线性；
- feature interaction；
- tabular data；
- 正则化控制。

在中小规模结构化数据上通常是很强的 baseline。

---

## 7.6 SVR 是什么？

Support Vector Regression 通过一个函数：

\[
f(x)=w^T\phi(x)+b
\]

拟合目标。

借助 kernel 可学习非线性关系。

SVR 在样本不大的连续回归问题中经常有竞争力。

---

# 7.7 Baseline 2：Gaussian Process（GP）

Gaussian Process 把未知函数 \(f(x)\) 看作一个随机函数：

\[
f(x)\sim GP(m(x),k(x,x'))
\]

其中：

- \(m(x)\)：mean function；
- \(k(x,x')\)：kernel / covariance function。

GP 最重要的优势：

> 对任意未知输入天然给出预测均值和预测方差。

\[
p(y_*|x_*,D)=\mathcal N(\mu(x_*),\sigma^2(x_*))
\]

所以特别适合小数据 Bayesian Optimization。

---

## 7.8 Matérn Kernel 是什么？

Kernel 用来描述两个输入之间的相似性，以及函数随输入变化的平滑程度。

Matérn kernel 是 BO 中常用的一类核，比非常平滑的 RBF 更允许函数呈现不那么光滑的变化，适合复杂物理/生物映射作为 baseline。

---

## 7.9 ARD 是什么？

ARD = Automatic Relevance Determination。

普通 kernel 可能所有输入维度共享同一个 length scale \(\ell\)。

ARD 为每一维学习不同：

\[
\ell_1,\ell_2,\ldots,\ell_d
\]

直觉上：

- length scale 很大：函数对这个维度不太敏感；
- length scale 较小：这个维度的变化会显著影响函数。

因此 ARD 可以一定程度反映 feature relevance。

---

## 7.10 GP 风险是什么？

PPT 已经指出两个风险：

### 计算成本

经典 GP 训练常见复杂度约为：

\[
O(N^3)
\]

数据继续增长时成本增加。

### 高维表达

若输入是非常高维的 sequence embedding 或大量 physics features，kernel distance 可能变得难以建模。

因此 GP 更适合：

> 作为低中维、概率定义清楚的强 baseline。

---

# 7.11 Main Candidate：Deep Ensemble

Deep Ensemble = 训练多个独立模型：

\[
M=20\sim50
\]

参考论文使用更大的 ensemble，但本项目可以先测试 20–50 个以平衡成本和 uncertainty stability。

对于一个 candidate \(x\)，第 \(m\) 个模型预测：

\[
\hat y_m(x)
\]

预测均值：

\[
\mu(x)=\frac1M\sum_{m=1}^{M}\hat y_m(x)
\]

模型分歧：

\[
\sigma_{epi}(x)=Std(\hat y_1(x),\ldots,\hat y_M(x))
\]

后者作为 epistemic uncertainty 的近似。

---

## 7.12 Epistemic Uncertainty 是什么？

Epistemic uncertainty = 认知不确定性。

来源是：

> 模型缺少知识或训练数据覆盖不足。

如果同一条 RNA：

```text
Model 1 → -20
Model 2 → -10
Model 3 → -27
Model 4 → -15
```

说明模型意见分歧很大，通常意味着这条 RNA 在当前知识下不确定。

这类不确定性可以通过获取新的高价值标签来降低，所以它是 Active Learning 重点利用的 uncertainty。

---

## 7.13 Aleatoric Uncertainty 是什么？

Aleatoric uncertainty = 数据/过程本身的随机性。

例如同一 RNA 的多个 replicate：

```text
Replica 1 → -20
Replica 2 → -15
Replica 3 → -24
```

如果测量/模拟过程本身波动很大，那么新增训练样本不能完全消除这种 uncertainty。

因此后面 stopping criteria 需要检查 oracle noise。

---

## 7.14 Deep Ensemble 和当前 Ensemble 有什么区别？

当前传统 ensemble 的重点通常是：

```text
RF + ET + GBR + XGB + SVR
↓
融合预测
↓
提高预测稳定性
```

Deep Ensemble 在主动学习场景中除了提高 prediction，还承担：

```text
多个独立模型
↓
平均预测 μ
+
模型 disagreement σ
```

因此：

> “用于预测的 ensemble”和“用于概率采集的 deep ensemble”在功能上不完全相同。

---

## 7.15 Probabilistic Surrogate 的评价不能只看 Spearman

一个概率代理模型至少要同时回答两个问题：

### 预测是否准确？

看：

- Spearman；
- MAE；
- RMSE；
- multi-seed stability。

### uncertainty 是否有意义？

例如考察：

\[
corr(\sigma(x),|\hat y-y|)
\]

如果 uncertainty 大的样本确实更容易预测错，说明 \(\sigma\) 对 Active Learning 有价值。

也可以进一步做：

- prediction interval coverage；
- NLL；
- calibration analysis。

---

# 第 8 页：回流序列选择 + 停止标准

这一页是整个闭环真正开始“行动”的地方。

PPT 当前计划每轮 100 条：

| 类型 | 参考数量 | 目的 |
|---|---:|---|
| High epistemic uncertainty | 45 | 解决模型最不会的区域 |
| Coverage / novelty | 25 | 补足当前数据空间盲区 |
| Physics-aware diversity | 联合约束 | 避免物理上重复 |
| Predicted promising | 20 | 保留实际优化价值 |
| Random control | 10 | 判断 AL 是否优于随机 |

总计：

\[
45+25+20+10=100
\]

其中 Physics-aware diversity 不是额外再加一组，而是对整个 batch 施加的多样性约束。

---

## 8.1 High Epistemic Uncertainty——高认知不确定性

通过 probabilistic surrogate 得到：

\[
\sigma_{epi}(x)
\]

选择高 \(\sigma\) 的 candidate，代表优先测：

> 当前模型最不知道的区域。

目的不是直接找到最优 GBSA，而是降低模型知识盲区。

这属于 Exploration。

---

## 8.2 Coverage / Novelty——覆盖度与新颖性

即使 uncertainty 很高，也可能有一大群候选集中在同一个局部区域。

如果全选它们，batch 会浪费。

Coverage 关注：

> 当前训练数据覆盖了哪些区域，还有哪些区域明显缺失？

Novelty 关注：

> 新 candidate 与已有训练样本相比有多新？

对于 13 个离散突变位点，一个简单指标是 Hamming Distance。

若两个序列在 13 个位置中有 \(k\) 个位置不同：

\[
d_H(x_i,x_j)=k
\]

可以用 candidate 到训练集最近邻的最小 Hamming distance 作为 novelty 的一个简单指标。

---

## 8.3 Physics-aware Diversity——物理感知多样性

两条 sequence 可能 nucleotide 不同，但预测的结构/物理状态非常相似。

因此除了 sequence distance，还可以在 selected physics representation 或 fused representation 上定义距离：

\[
d_{phy}(x_i,x_j)
\]

目的：

> 防止同一个 batch 在物理状态上高度重复。

可实现方法包括：

- clustering；
- farthest-point / max-min selection；
- representation-space distance；
- DPP 等。

当前 PPT 不需要一次性承诺最复杂方法，核心是明确“多样性作为联合约束”。

---

## 8.4 Predicted Promising——预测有潜力的候选

如果只追求 uncertainty，模型可能越来越会学习，但实际并没有测到潜在优秀 RNA。

因此保留一部分 exploitation：

> 选择模型预测 GBSA 有潜力的候选。

后期如果目标明确是低 GBSA，可以逐渐提高这部分权重，甚至使用 LCB 等 BO acquisition。

---

## 8.5 Random Control——随机对照

Random control 非常重要，因为它回答：

> 模型指导选样真的优于“随便再增加 100 条”吗？

如果没有 random control：

```text
增加100条 → 模型提升
```

我们无法判断提升来自：

- 主动学习策略；还是
- 单纯因为数据更多。

真正应该比较的是 learning curve：

\[
Performance_{AL}(Budget)
\]

vs.

\[
Performance_{Random}(Budget)
\]

在同样标签预算下，若 AL 更快提升，才证明 acquisition 有额外价值。

---

## 8.6 一轮 Active Learning 的完整数学过程

当前数据：

\[
D_t
\]

训练 surrogate：

\[
D_t\rightarrow p(y|x,D_t)
\]

对 candidate pool \(\mathcal C\) 计算：

\[
\mu(x),\sigma(x),Novelty(x),Diversity(x)
\]

通过 acquisition 选 batch：

\[
B_t\subset\mathcal C,\quad |B_t|=100
\]

调用 expensive oracle：

\[
Y_t=Oracle(B_t)
\]

更新数据：

\[
D_{t+1}=D_t\cup(B_t,Y_t)
\]

然后 retrain。

如果最初 2000 条：

```text
Round 0: 2000
↓ +100
Round 1: 2100
↓ +100
Round 2: 2200
↓ +100
Round 3: 2300
```

关键点：

> 每轮新标签返回后必须重训练，再决定下一轮。

因此“100 → retrain → 100”通常比“一开始用旧模型一次选 200”更符合主动学习思想。

---

# 第 8 页后半：停止标准（Stopping Criteria）

PPT 当前有五类停止条件：

1. Prediction Plateau；
2. Uncertainty Saturation；
3. Active Learning ≈ Random；
4. Experimental Cost；
5. 转向 Optimization。

这不是“五个任一触发就机械终止”的死规则，而是判断**下一轮昂贵标签的边际信息收益是否还值得**。

---

## 8.7 Prediction Plateau——预测平台期

PPT 当前建议：

\[
\Delta Spearman<0.01
\]

连续两轮，而且增益不稳定。

例如：

```text
0.38 → 0.43 → 0.48 → 0.52
```

明显还在涨，应继续。

如果：

```text
0.480 → 0.486 → 0.489 → 0.490
```

则进入平台。

注意：0.01 是工程阈值，不是自然定律，应结合实验成本决定。

---

## 8.8 “增益不稳定”是什么意思？

不能只看平均提升一点点。

应看多个 seeds / splits 下的 paired difference：

\[
\Delta_i=\rho_{new,i}-\rho_{old,i}
\]

并关注：

- bootstrap confidence interval；
- win rate；
- 是否多数 split 均提升。

如果均值略正但 CI 跨 0，说明提升证据不够稳定。

---

## 8.9 Uncertainty Saturation——不确定性饱和

随着高价值标签不断加入，candidate pool 上的 epistemic uncertainty 应逐渐降低。

如果大部分候选：

\[
\sigma_{epi}(x)\downarrow
\]

并且已经很低，说明模型认为主要空间已被较充分覆盖。

但是“模型说自己很确定”不等于真的可靠，因此必须结合 uncertainty calibration。

---

## 8.10 什么是 Uncertainty Calibration？

理想情况下：

\[
\sigma(x)\uparrow\Rightarrow Error(x)\uparrow
\]

也就是说模型越不确定的样本，实际越容易预测错。

可以检查：

\[
corr(\sigma(x),|\hat y-y|)
\]

如果完全不相关，则当前 uncertainty 不能很好指导 acquisition。

这时不是简单继续采样，而应先重新诊断 surrogate uncertainty。

---

## 8.11 Active Learning ≈ Random

如果在相同标签预算下：

\[
LearningCurve_{AL}\approx LearningCurve_{Random}
\]

说明当前 acquisition strategy 没有明显提高 label efficiency。

可能原因：

- 数据已经接近饱和；
- uncertainty 估计不可靠；
- diversity 定义不合适；
- candidate pool 本身信息有限。

所以这个标准意味着“停止或调整策略”，不一定意味着整个项目结束。

---

## 8.12 Experimental Cost——实验成本

最终要看边际收益：

\[
\frac{Performance\ Gain}{Cost}
\]

例如每增加 100 条昂贵标签，Spearman 只提高 0.003，而成本和周期都很高，则即使数学上还在涨，工程上也可能不值得继续。

因此可以讨论：

> 每提升 0.01 Spearman 的成本是否可接受？

---

## 8.13 转向 Optimization

这不是“项目终止”，而是**阶段切换**。

前期 acquisition 主要是：

\[
Uncertainty+Coverage+Diversity
\]

目标：提高 surrogate。

当 surrogate 已满足预测需求后，重点逐渐变成：

\[
Low\ GBSA\ Candidates
\]

可以转向：

\[
LCB(x)=\mu(x)-\kappa\sigma(x)
\]

或者其他 BO acquisition。

因此整个研究路线可以理解为：

```text
Active Learning
（先学准）
↓
AL + BO 混合
↓
Bayesian Optimization
（再重点找最优）
```

---

## 8.14 Oracle Noise——为什么停止判断还要检查真实标签噪声？

观测标签可以写成：

\[
Y_{obs}=Y_{true}+\epsilon_{oracle}
\]

如果同一条 RNA 多个 replica 差异很大，说明 \(\epsilon_{oracle}\) 很大。

这可能来自：

- trajectory convergence；
- initial state；
- sampling window；
- multiple replicas；
- protocol variance；
- 实验/计算噪声。

如果 oracle 自身噪声已经成为主要误差源，继续增加模型复杂度或标签数量未必能显著提高 Spearman。

所以 performance plateau 后要检查：

```text
Uncertainty
+
Coverage
+
Oracle Noise
```

从而判断究竟是：

- 还应该继续采样；
- 应该调整 acquisition；还是
- 应该先优化 GBSA label protocol。

---

# 第 8 页最终决策流程如何理解

PPT 最后一张 decision flow 可以用下面的逻辑讲：

```text
完成一轮新标签获取
        ↓
重训练代理模型
        ↓
独立测试集评估
        ↓
性能仍在提升？
   ┌────┴────┐
  Yes       No / Plateau
   │           │
   ↓           ↓
进入下一轮    检查：
             uncertainty
             coverage
             oracle noise
                 │
          ┌──────┴──────┐
         尚有信息       已饱和
          │              │
          ↓              ↓
       调整策略          停止
                      或转向Optimization
```

这里最关键的一句话是：

> **停止不是由“达到某个固定 Spearman”机械决定，而是由新增昂贵标签的边际信息价值决定。**

---

# 9. 把整个项目统一成一个数学闭环

当前 labelled dataset：

\[
D_t=\{(x_i,y_i)\}_{i=1}^{N_t}
\]

其中：

- \(x_i\)：RNA sequence + derived physics information；
- \(y_i\)：真实 GBSA。

### Step 1：Feature / Representation

\[
x\rightarrow h_{seq},h_{phy}
\]

### Step 2：Fusion

\[
h=[h_{seq};h_{phy}]
\]

或 gated fusion。

### Step 3：Probabilistic Surrogate

\[
p(y|x,D_t)
\]

得到：

\[
\mu(x),\sigma(x)
\]

### Step 4：Acquisition

概念上：

\[
A(x)=f(\mu(x),\sigma(x),Novelty(x),Diversity(x))
\]

### Step 5：Batch Selection

\[
B_t=SelectBatch(\mathcal C,A),\quad |B_t|\approx100
\]

### Step 6：Expensive Oracle

\[
Y_t=Oracle(B_t)
\]

### Step 7：Update

\[
D_{t+1}=D_t\cup(B_t,Y_t)
\]

### Step 8：Retrain + Stopping

若 learning curve 仍有显著收益：继续。

若进入 plateau：检查 uncertainty、coverage、oracle noise、cost。

若预测能力已满足需求：转向 optimization。

---

# 10. 汇报时整份 PPT 最推荐的主讲逻辑

如果你需要把整份 PPT 连贯地讲出来，可以按下面这段主线组织语言。

## 第一段：问题转向

> 前期我们主要在现有 RNA–GBSA 数据上做静态监督学习，即固定数据集以后比较不同特征和模型，再用 Spearman、MAE、RMSE 和多随机种子评价泛化能力。经过大量模型和特征探索以后，单个 seed 可能出现很高结果，但多 seed 稳定评价显示整体提升有限。因此当前瓶颈逐渐从“模型是否足够复杂”转向“训练数据是否覆盖了真正有信息的区域，以及 GBSA 标签本身是否足够稳定”。

## 第二段：为什么先做特征筛选

> 原始结构和物理特征来源多、维度高，而且内部存在无效信息和高度冗余。我们首先做 Missing、Constant 和低方差清洗，再利用 Spearman 捕捉与 GBSA 的单调信息，用 Mutual Information 补充非单调的非线性依赖，通过多 split stability 排除偶然关系，再通过 feature–feature correlation 去除重复信息。需要强调的是，所有依赖 GBSA 标签的筛选都必须放在 Nested Cross-Validation 的训练侧完成，外层测试集完全不可见，从而避免 feature-selection leakage。

## 第三段：为什么做 Sequence + Physics 融合

> Sequence 是最终真正要设计的变量，因此它始终保留；physics features 是由序列派生的结构和能量状态描述。我们希望验证 physics 是否提供了 sequence 之外的 complementary information，所以分别构建 sequence representation 和 selected physics representation，再从简单稳定的 Early Fusion 开始，如果确有稳定收益再尝试 Gated Fusion。最终通过 Sequence Only、Physics Only、Sequence + Physics 三组对照确认融合是否真正有用。

## 第四段：为什么要概率代理模型

> 普通模型只给出一个 GBSA 点预测，不能告诉我们下一条数据究竟哪里最值得测。Active Learning 和 Bayesian Optimization 需要知道模型对预测有多确定，因此下一阶段的 surrogate 需要输出 \(\mu(x)\) 和 \(\sigma(x)\)。我们保留现有 RF、ET、GBR、XGB、SVR 作为 prediction baseline；用 Gaussian Process 作为天然概率模型 baseline；以 Deep Ensemble 作为主要候选，通过多个独立模型的平均预测得到 \(\mu\)，通过模型间 disagreement 近似 epistemic uncertainty。

## 第五段：参考论文和我们的关系

> 我们参考 2026 年 NAR 的 NAND hybrid riboswitch deep batch Bayesian optimization 工作，因为它已经在 riboswitch 场景中证明了“sequence encoder—deep ensemble—uncertainty-driven acquisition—batch selection—真实实验—模型重训”的闭环可行性。但我们不会原样复制它：参考工作主要面向功能最优化并使用 UCB，而我们的当前首要目标是提高 GBSA surrogate 的泛化能力，所以前期 acquisition 更强调 uncertainty、coverage 和 diversity；只有当预测能力成熟后，才逐步转向低 GBSA 的 LCB/BO 优化。

## 第六段：一轮到底选什么

> 每轮暂定回流 100 条，其中约 45 条解决高 epistemic uncertainty 区域，25 条补足 coverage/novelty，20 条保留 predicted promising candidates，10 条作为 random control；physics-aware diversity 作为整个 batch 的联合约束。这里 random control 很关键，因为只有与随机新增标签在相同预算下比较 learning curve，才能证明主动学习是否真正提高了 label efficiency。

## 第七段：什么时候停

> 每一轮新标签返回以后重新训练 surrogate，再在冻结独立测试集上评价。如果性能仍稳定提高就继续；如果连续多轮进入 plateau，则进一步检查 uncertainty 是否饱和、candidate space 是否仍有未覆盖区域、Active Learning 是否仍明显优于 Random，以及 oracle 本身是否存在较大的 GBSA 噪声。停止条件不是机械达到某个 Spearman，而是继续增加昂贵标签已经没有足够的边际信息收益。如果模型已经足够可靠，则不是结束项目，而是把 acquisition 的重心从“学习模型”转向“寻找低 GBSA 序列”。

---

# 11. 最重要的术语速查表

| 术语 | 中文 | 在本项目中的作用 |
|---|---|---|
| GBSA | 广义 Born / 表面积相关能量评价 | 需要预测/获取的昂贵标签 |
| Oracle | 昂贵真实评价器 | 返回真实 GBSA |
| Surrogate | 代理模型 | 近似昂贵 GBSA 函数 |
| Probabilistic Surrogate | 概率代理模型 | 同时输出预测均值和不确定性 |
| Active Learning | 主动学习 | 优先获取最能提升模型的新标签 |
| Bayesian Optimization | 贝叶斯优化 | 用少量昂贵评价寻找最优 sequence |
| Spearman | 秩相关系数 | 评价排名；也用于单变量特征相关筛选 |
| Mutual Information | 互信息 | 捕捉一般非线性依赖 |
| Stability | 稳定性 | 判断特征是否跨 split 稳定出现 |
| Redundancy Removal | 去冗余 | 删除重复表达同一信息的特征 |
| Nested CV | 嵌套交叉验证 | 内层选方案，外层评估真正泛化 |
| Sequence Encoder | 序列编码器 | RNA sequence → 数值 representation |
| Feature Fusion | 特征融合 | 合并 sequence 与 physics 信息 |
| Early Fusion | 早期融合 | 直接 concatenate 两类表示 |
| Gated Fusion | 门控融合 | 学习每条样本对不同信息源的权重 |
| Gaussian Process | 高斯过程 | 天然输出预测均值和方差的概率模型 |
| Matérn Kernel | Matérn 核 | GP 中衡量样本相似性的协方差函数 |
| ARD | 自动相关性确定 | 为不同输入维度学习不同 length scale |
| Deep Ensemble | 深度集成 | 多独立模型产生预测均值和模型分歧 |
| Epistemic Uncertainty | 认知不确定性 | 模型知识不足，可通过新数据降低 |
| Aleatoric Uncertainty | 偶然/数据噪声 | Oracle 或系统本身的随机波动 |
| Acquisition Function | 采集函数 | 决定下一条/下一批测谁 |
| UCB | 置信上界 | maximization 时平衡均值和不确定性 |
| LCB | 置信下界 | minimization 时兼顾低预测值和不确定性 |
| Kriging Believer | 克里金信念批量策略 | 减少一个 batch 内候选的重复 |
| Coverage | 覆盖度 | 补足训练数据空间盲区 |
| Novelty | 新颖性 | 选择与已有样本不同的 candidate |
| Hamming Distance | 汉明距离 | 衡量离散 mutation pattern 的差异 |
| Random Control | 随机对照 | 验证主动学习是否优于随机加数据 |
| Prediction Plateau | 预测平台期 | 新标签不再显著提升性能 |
| Uncertainty Saturation | 不确定性饱和 | candidate pool 的认知不确定性已普遍降低 |
| Oracle Noise | 真实评价噪声 | 判断 GBSA label 本身是否成为性能上限 |

---

# 12. 汇报时最容易被追问的 10 个问题

## Q1：为什么不继续换 Transformer / GNN？

因为前期大量模型探索说明模型复杂度并没有自动带来稳定泛化增益。现阶段更值得检验的是：高价值新标签能否比继续堆模型更有效。

## Q2：为什么 Spearman 和 MI 两个都要？

Spearman 擅长稳定单调关系，MI 能补充一般非线性依赖，两者互补。

## Q3：为什么还要去冗余？

因为多个 feature 可能表达同一物理信息；高维冗余在 2000 左右样本下容易增加方差和过拟合。

## Q4：为什么必须 Nested CV？

因为 Spearman/MI 使用 GBSA 标签，若先在全数据选 feature 再 CV，会让 test label 间接参与筛选，造成 leakage。

## Q5：为什么 Sequence 不跟 physics 一起筛掉？

Sequence 是最终设计变量本身；physics 是由 sequence 派生的辅助状态描述。两者角色不同。

## Q6：为什么要概率代理而不是普通 XGB？

Active Learning 不只需要预测值，还需要判断“模型哪里最不确定”，因此需要可靠 uncertainty。

## Q7：为什么 GP 不直接作为最终模型？

GP 概率定义很清楚，适合作为 baseline；但高维复杂表示和数据规模增长会带来计算和表达限制，所以同时测试 Deep Ensemble。

## Q8：为什么参考论文用 UCB，我们不用完全照抄？

参考论文直接优化功能值；我们当前首要目标是提高 GBSA surrogate 泛化，因此前期更强调 uncertainty + coverage + diversity，后期才转向 LCB/BO。

## Q9：为什么是 100 条？

100 是预算/throughput 方案，不是数学固定值。真正应该通过 retrospective simulation 和实验能力比较 50/100/200 等 batch size 的 label efficiency。

## Q10：为什么不能直接把 0.6 设为停止标准？

因为不知道 oracle noise ceiling 和可达到上限。科学的停止规则应该看新增标签的边际信息收益、uncertainty、coverage、random baseline 和成本。

---

# 13. 最终一张图：整个项目从输入到闭环

```text
                        Existing RNA–GBSA Data
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
            RNA Sequence                Physics / Structure
                    │                    Raw Features
                    │                           │
                    │              Missing / Constant QC
                    │                           │
                    │                Spearman + MI
                    │                           │
                    │                    Stability
                    │                           │
                    │               Redundancy Removal
                    │                           │
                    │            Nested-CV Validation
                    │                           │
                    ▼                           ▼
            Sequence Representation    Selected Physics Representation
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                         Feature Fusion
                    Early Fusion / Gated Fusion
                                  │
                                  ▼
                      Probabilistic Surrogate
                       p(GBSA | sequence)
                                  │
                          ┌───────┴───────┐
                          ▼               ▼
                        μ(x)             σ(x)
                  Predicted GBSA    Uncertainty
                          └───────┬───────┘
                                  ▼
                         Batch Acquisition
              Uncertainty + Coverage + Diversity
                   + Promising + Random Control
                                  │
                                  ▼
                          ~100 Candidates
                                  │
                                  ▼
                     Expensive GBSA Oracle
                                  │
                                  ▼
                         New True Labels
                                  │
                                  ▼
                          Update Dataset
                                  │
                                  ▼
                         Retrain Surrogate
                                  │
                                  ▼
                        Independent Evaluation
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
              Meaningful Gain?             Plateau?
                     │                         │
                    Yes                        ▼
                     │              Check uncertainty / coverage /
                     │                   random / oracle noise / cost
                     ▼                         │
                 Next Round          ┌─────────┴─────────┐
                                     ▼                   ▼
                                Still useful          Saturated
                                     │                   │
                                     ▼                   ▼
                              Adjust strategy     Stop / Shift to BO
```

---

# 14. 最后应该记住的四句话

1. **特征选择不是为了把维度压得越低越好，而是为了留下与 GBSA 有信息、跨 split 稳定、彼此不重复且真正能泛化的 physics information。**

2. **Sequence 是设计变量，Selected Physics 是辅助状态描述；融合的核心科学问题是 physics 是否提供了 sequence 之外的 complementary information。**

3. **概率代理模型与普通回归模型最大的区别，是不仅输出 \(\mu(GBSA)\)，还要输出能够指导主动学习的 \(\sigma(GBSA)\)。**

4. **整个系统的价值不在于“选出 100 条”，而在于建立一个可重复的 model → acquisition → expensive oracle → retrain 闭环，并证明在相同标签预算下比随机新增标签更有效。**

