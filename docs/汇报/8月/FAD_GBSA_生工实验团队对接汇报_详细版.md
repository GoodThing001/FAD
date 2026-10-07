# FAD 核糖开关 GBSA 代理模型与昂贵实验闭环优化——生工实验团队对接汇报稿

> **用途**：与生工实验团队对接，讨论如何利用有限、昂贵的新 GBSA（项目中的目标自由能指标）标签，迭代提升 RNA 核糖开关 GBSA 预测模型，并逐步建立可执行的模型—实验闭环。  
> **建议汇报时长**：25–35 分钟；讨论 20–40 分钟。  
> **汇报定位**：这不是“已经确定全部算法细节”的项目结题汇报，而是“提出一个有文献依据、可验证、可逐轮调整的实验—算法联合方案”，希望在会上确定实验容量、标签质量、迭代周期与第一轮候选数。  
> **当前核心目标**：第一阶段优先提高 GBSA 预测模型的泛化精度；第二阶段在代理模型足够可靠后，再逐步转向寻找更优 GBSA 序列。

---

## 0. 术语约定

为避免汇报中英文术语造成理解障碍，本文首次出现的重要英文术语均给出中文含义。

- **GBSA**：本项目的目标自由能指标；下文直接简称 GBSA。
- **Feature Selection（特征选择）**：从已有候选特征中筛出与 GBSA 有效相关且信息不冗余的特征。
- **Feature Extraction / Representation（特征提取 / 表示学习）**：把 RNA 序列和筛选后的物理/结构特征表示成适合模型学习的向量。
- **Surrogate Model（代理模型）**：用较低成本预测昂贵目标函数（这里为 GBSA）的模型。
- **Bayesian Optimization, BO（贝叶斯优化）**：利用代理模型的预测和不确定性，迭代选择最值得真实评估的候选。
- **Active Learning, AL（主动学习）**：优先选择“最能提升模型认知”的样本进行标注。
- **Deep Ensemble（深度集成）**：训练多个独立神经网络，用多个模型的预测形成经验预测分布。
- **Uncertainty Quantification, UQ（不确定性量化）**：估计模型对某个预测有多大把握。
- **Upper Confidence Bound, UCB（置信上界）**：贝叶斯优化中兼顾预测值和不确定性的候选选择函数。
- **Lower Confidence Bound, LCB（置信下界）**：当目标是最小化某个值时使用的置信界选择函数。
- **Kriging Believer, KB（克里金信念批量策略）**：构造一批候选时，把已经选中的候选暂时用模型预测值作为“伪标签”加入模型，再继续选择下一个候选，从而降低批内重复。
- **Design–Build–Test–Learn, DBTL（设计—构建—测试—学习闭环）**：模型设计候选、实验构建与测试、结果返回、模型再学习的迭代流程。
- **Mutual Information, MI（互信息）**：衡量变量之间一般统计依赖关系，可捕获非线性、非单调关系。
- **Spearman Correlation（斯皮尔曼秩相关）**：衡量两个变量之间单调关系的非参数相关系数。
- **Principal Component Analysis, PCA（主成分分析）**：无监督线性降维方法。
- **Gaussian Process, GP（高斯过程）**：经典概率代理模型，可直接给出预测均值和不确定性。
- **Masked Language Modeling, MLM（掩码语言建模）**：随机遮盖部分序列片段，并让模型恢复被遮盖内容的自监督预训练任务。
- **Triplet Loss（三元组损失）**：使相似样本的表示更接近、不同样本的表示更远的一类度量学习损失。
- **Epistemic Uncertainty（认知不确定性）**：源于模型“没有见过或没学懂”的不确定性，可通过增加数据降低。
- **Aleatoric Uncertainty（数据噪声不确定性）**：源于实验或模拟本身噪声的不可约不确定性。
- **Retrospective Active-Learning Simulation（回顾性主动学习模拟）**：把已有标签暂时隐藏，用已知历史数据模拟未来主动学习过程，以验证候选选择方法和批量大小。

---

# 一、建议的汇报总体结构

建议把汇报主线固定为：

```text
当前 GBSA 预测能力与瓶颈
        ↓
特征选择：哪些物理/结构信息真的有用？
        ↓
特征表示：RNA 序列 + 物理/结构信息如何融合？
        ↓
概率代理模型：既要预测准，也要知道“哪里不确定”
        ↓
直接对标：NAR 2026 核糖开关 Deep Batch Bayesian Optimization
        ↓
针对 FAD 的四项关键改造
        ↓
先用已有 2000 条数据做离线主动学习模拟
        ↓
确定 50 / 100 / 200 中哪种批量更合理
        ↓
模型选择第一批高信息量 RNA
        ↓
昂贵 GBSA / 实验返回
        ↓
重训模型
        ↓
下一轮
```

汇报的核心句建议统一为：

> **我们的目标不是一次性让模型“生成一批看起来最好的 RNA”，而是建立一个能够反复学习的昂贵标签闭环：每一轮优先选择最有信息价值的 RNA 获得真实 GBSA，使有限实验资源最大程度转化为模型泛化能力。**

---

# 第 1 页：研究目标——从“预测模型”升级为“模型—实验闭环”

## 页面标题

**面向昂贵 GBSA 标签的核糖开关迭代学习框架**

英文副标题可写：

**Expensive-Oracle Learning for Riboswitch GBSA（面向昂贵评价函数的核糖开关 GBSA 学习）**

## 页面建议内容

目前已有约 2000 条带 GBSA 标签的 RNA 序列，核心预测任务为：

\[
f_{\theta}(x) \rightarrow \widehat{\mathrm{GBSA}}
\]

其中：

- \(x\)：RNA 序列以及与其相关的结构/物理特征；
- \(\widehat{\mathrm{GBSA}}\)：代理模型预测值；
- 真实 GBSA 获取成本较高，因此无法对整个候选空间逐条真实评估。

本项目下一阶段的真正问题不是：

> “Random Forest（随机森林）、XGBoost（极端梯度提升）还是 Transformer（变换器模型）哪个更好？”

而是：

> **“如果下一轮只能新获得 50、100 或 200 个昂贵 GBSA 标签，应该选择哪些 RNA，才能让后续模型提升最大？”**

## 图怎么画

画一个左右对比图：

```text
传统方式
已有数据 → 训练模型 → 报一个 Spearman → 结束

我们希望的方式
已有数据 → 概率代理模型 → 选高价值序列 → 昂贵 GBSA
   ↑                                      ↓
   └──────────── 新标签 ←─────────────────┘
```

在右图上标注：

**DBTL（设计—构建—测试—学习）闭环**

## 现场讲稿

> 我们当前已经不是完全没有模型，而是模型已经能够学习到一定的 GBSA 排序信号。下一阶段最重要的问题，是如何把新增实验数据变成真正能够提高模型能力的数据。  
> 因此我们希望实验团队不仅承担“最终验证”，而是和算法一起进入迭代闭环。每一轮模型根据现有认知选择最值得验证的序列；实验返回 GBSA 后模型重新训练，再决定下一轮。  
> 这类问题在机器学习里属于 Active Learning（主动学习）和 Bayesian Optimization（贝叶斯优化）交叉的昂贵评价函数优化问题。

## 本页需要避免的说法

不要说：

> “我们通过三轮实验一定可以做到 0.6。”

应该说：

> “我们希望通过学习曲线验证模型能否逐轮突破当前瓶颈，0.6 是阶段性的高目标，而不是目前可以保证的结果。”

---

# 第 2 页：当前状态——为什么下一步不能只继续堆模型

## 页面标题

**当前瓶颈：模型已经学到信号，但新增信息比继续堆模型更重要**

## 页面建议内容

当前项目可以采用以下汇报口径：

- 已有约 **2000 条带 GBSA 标签的 RNA 序列**；
- RNA 长度约 **141 nt**；
- 设计空间主要由 **13 个可变位点**决定；
- 当前稳定评价应以多随机划分结果为主；
- 当前 GBSA 排序预测的稳定基准约为 **Spearman \(\rho \approx 0.37\)** 左右；
- 历史上某些单次划分可出现更高结果，但不能作为稳定泛化能力的证明；
- 已经测试过多种树模型、提升模型、支持向量模型和集成策略；
- 高维物理/结构特征中存在有效信号，但“把所有特征直接加入模型”并没有稳定提高最终表现。

> **注**：如果会前最新正式实验结果有更新，建议把本页数字替换为最新冻结版本；原则是不使用单个最优 seed（随机种子）作为主要结果。

## 图怎么画

建议画一个“边际收益递减”示意图：

```text
预测性能
  ↑
  |                ______
  |             __/
  |          __/
  |       __/
  |______/________________→ 模型复杂度
       当前区域
```

旁边写：

- 更多模型 ≠ 必然更多有效信息；
- 更多特征 ≠ 必然更高泛化性能；
- 当前需要改变的是 **有效标签覆盖** 和 **标签质量**。

## 现场讲稿

> 从目前结果看，我们面临的不只是“模型还不够复杂”。  
> 如果一个问题的主要瓶颈是模型类型，那么 RF 换 XGB、XGB 换深度网络应该出现持续的大幅提升；但我们的实验更像是进入了平台期。  
> 因此下一阶段的核心假设是：当前 2000 条 GBSA 对 13 位点组合空间的覆盖不足，尤其是某些互作区域和模型高不确定区域没有足够标签。  
> 如果这个假设成立，那么有策略地新增一小批标签，应当比随机新增同样数量标签更有效。

---

# 第 3 页：总体方法——四个模块，而不是简单“做贝叶斯优化”

## 页面标题

**技术路线：特征筛选 → 多模态表示 → 概率代理 → 批量主动学习**

## 页面建议内容

建议正式把路线分成四层：

### ① Feature Selection（特征选择）

目标：

> 从现有物理、结构、序列衍生特征中找出与 GBSA 真实相关、稳定且低冗余的特征块。

方法：

\[
\text{Spearman} + \text{Mutual Information（互信息）}
\rightarrow
\text{Stability（稳定性）}
\rightarrow
\text{Redundancy Removal（去冗余）}
\]

### ② Representation / Fusion（表示学习 / 融合）

两条输入支路：RNA Sequence（RNA 序列）和 Selected Physics / Structure Features（筛选后的物理/结构特征），分别编码，再融合。

### ③ Probabilistic Surrogate（概率代理模型）

输出不只是一条预测，而是：

\[
\mu(x) \quad \text{和} \quad \sigma(x)
\]

即：

- \(\mu(x)\)：预测 GBSA；
- \(\sigma(x)\)：模型对该预测的不确定程度。

### ④ Batch Active Learning / Bayesian Optimization（批量主动学习 / 贝叶斯优化）

根据模型当前知识选择下一批最有价值序列。

## 图怎么画

```text
37类物理/结构特征              RNA序列
        │                        │
Spearman + MI                Sequence Encoder
   + 去冗余                  （序列编码器）
        │                        │
低维 Physics Embedding       Sequence Embedding
（物理特征表示）             （序列表示）
        └──────────┬─────────────┘
                   ↓
              Feature Fusion
              （特征融合）
                   ↓
          Probabilistic Surrogate
            （概率代理模型）
                   ↓
          μ(GBSA) + σ(GBSA)
                   ↓
         Batch Acquisition
          （批量候选选择）
                   ↓
             新序列实验
```

## 现场讲稿

> 我们希望把“贝叶斯”放在正确的位置。贝叶斯优化并不是简单换一个叫 Bayesian 的回归器，而是要求模型对未测序列同时给出预测和不确定性，然后利用不确定性决定下一批实验。  
> 所以在进入昂贵优化之前，前面还有两个基础问题必须解决：哪些物理特征值得保留，以及 RNA 序列本身和物理特征应该怎么共同表达。

---

# 第 4 页：Feature Selection（特征选择）——Spearman + MI + 去冗余

## 页面标题

**第一步：以“高相关、低冗余、跨划分稳定”为原则筛选物理/结构特征**

## 4.1 为什么采用 Filter Method（过滤式特征选择）

导师提出的思路：

\[
\text{Spearman} + \text{MI} \rightarrow \text{去除冗余}
\]

是合理的，尤其适合当前目标：

- 候选特征种类有限；
- 很多特征来自不同物理/结构模块；
- 不希望为了筛一个特征就反复训练复杂模型；
- 希望筛选过程可解释、计算成本低、方便汇报。

上传的特征选择方案同样建议使用 Spearman 和 Mutual Information（互信息）评估与目标的关系，并对高相关特征去冗余。

## 4.2 为什么同时用 Spearman 和 MI

### Spearman Correlation（斯皮尔曼秩相关）

衡量单调关系：

\[
\rho_s(X,Y)
\]

优点：

- 不要求线性；
- 对异常值通常比 Pearson（皮尔逊相关）更稳健；
- 与我们最终重视 Spearman 排序指标的方向一致。

局限：U 型、倒 U 型等非单调关系可能得到低相关。

### Mutual Information（互信息）

衡量一般统计依赖：

\[
I(X;Y)
\]

优点：

- 可发现非线性关系；
- 可补 Spearman 对非单调关系的不足。

因此两者互补：

\[
\text{Spearman：单调依赖}
+
\text{MI：一般非线性依赖}
\]

## 4.3 关键修改：以 Feature Block（特征块）为筛选单位

我们的某些“特征”并不是单个数，而是一个高维特征块，例如：

- `wd`：可能有数百维；
- `motif_positional`：数十维；
- `pos_pair_prob`：多维；
- `contact_pca`；
- `energy_land`；
- `struct`；
- `mut`；
- 其他 NUPACK（核酸热力学分析）相关特征。

不能把所有维度直接混在一起后简单取 Top-K（前 K 个），因为一个 208 维特征块比一个 5 维特征块更容易仅由于维度数量大而“碰巧”出现一个很高的相关维度。

因此建议：

> **37 类特征作为第一层筛选单位，而不是把全部几百/上千维直接当作同等级单变量。**

## 4.4 推荐的 Block Score（特征块评分）

对于第 \(g\) 个特征块：

\[
X_g \in \mathbb{R}^{N\times d_g}
\]

在训练数据内部，对其中每一维 \(j\) 计算：

\[
r_{g,j}=|\rho_s(X_{g,j},Y)|
\]

以及：

\[
m_{g,j}=MI(X_{g,j};Y)
\]

为了避免“最大值偏好高维块”，不建议直接使用最大值，更推荐使用：

\[
R_g=\operatorname{mean}(\text{Top-}k\%\ r_{g,j})
\]

\[
M_g=\operatorname{mean}(\text{Top-}k\%\ m_{g,j})
\]

例如取每个特征块内部前 10%–20% 的维度做平均。

由于 Spearman 和 MI 数值尺度不同，不建议直接相加，而建议先在 37 个特征块间转成排名：

\[
Score_g =
0.5\cdot Rank(R_g)
+
0.5\cdot Rank(M_g)
\]

这里的 0.5/0.5 是初始方案，可在回顾性实验中测试，但不要在实验数据上反复调到过拟合。

## 4.5 加入 Stability（稳定性）

只在一次完整数据上算特征排名不够。建议在多个训练划分中重复特征筛选，例如 20–30 个随机种子或 bootstrap（自助采样）。

对每个特征块统计：

\[
Stability_g=
\frac{\text{进入 Top-K 的次数}}{\text{总重复次数}}
\]

例如：

- 特征 A：30 次中 27 次进入 Top-10，稳定性 0.90；
- 特征 B：30 次中 8 次进入 Top-10，稳定性 0.27。

优先保留 A。

**核心原则：不是只选“相关最高”的特征，而是选“在不同数据划分下持续有用”的特征。**

## 4.6 Redundancy Removal（去冗余）

导师提出 \(|\rho(X_i,X_j)|>0.85\) 则认为高度冗余，这个规则对单维特征很好用。

但对于高维特征块，需要先把每个特征块压缩成低维摘要。第一版推荐：

1. 对每个候选特征块单独做 PCA（主成分分析）；
2. 保留前 2–5 个主成分，或解释 80%–90% 方差；
3. 根据这些低维摘要判断特征块之间是否高度重复；
4. 如果两个块高度冗余，则优先保留与 GBSA 关联更强、跨 seed 更稳定、生物物理解释更明确、维度更低的一个。

第二版可作为方法学补充测试 Distance Correlation（距离相关）、CKA（中心化核对齐）或 HSIC（希尔伯特—施密特独立性准则），但明天汇报不需要把这些复杂方法作为主线。

## 4.7 防止 Feature Leakage（特征泄漏）

这一步必须强调：

> **Spearman、MI、去冗余阈值、PCA 都只能在训练 fold（训练折）内部拟合。**

正确流程：

```text
数据划分
 ↓
只用训练集做 Spearman / MI / 去冗余
 ↓
只用训练集拟合 PCA / scaler
 ↓
应用到验证集
 ↓
训练代理模型
 ↓
评估
```

不能先用全部 2000 条筛特征，再做交叉验证，否则会高估模型能力。

## 4.8 本阶段建议产出

建议形成三级结果：

- Top 12–15：初筛集合；
- 经过冗余过滤后约 5–10 个核心 feature blocks（特征块）；
- 额外保留 1–3 个具有明确结构意义但单变量相关不高的 biological rescue features（生物学保留特征），用于防止纯过滤法遗漏互作信息。

## 现场讲稿

> 特征选择阶段我们不打算堆很多复杂算法。核心原则就是与 GBSA 有关系、彼此不要高度重复，而且这种关系在不同数据划分下要稳定。  
> 我们会特别按“特征块”来筛，而不是让维度大的特征天然占优势。  
> 同时我们会保留少数结构上合理、可能通过交互作用起效的特征，因为单变量 Spearman 和 MI 仍可能漏掉 epistasis（上位性 / 位点互作）。

---

# 第 5 页：Feature Representation（特征表示）——RNA 序列和物理特征怎么融合

## 页面标题

**第二步：建立 Sequence–Physics Representation（序列—物理联合表示）**

这一页是整个项目方法创新的核心之一。

## 5.1 RNA 序列的两层信息

设计空间主要由 13 个可变位点决定：

\[
x=(x_1,x_2,\ldots,x_{13}),\quad x_i\in\{A,C,G,U\}
\]

最简单表示是 One-Hot Encoding（独热编码）：

\[
13\times4=52
\]

它必须保留为可靠基线。

但 13 个位点嵌入在约 141 nt 的 RNA scaffold（骨架）中，位点效应可能依赖相邻序列、配对位置、motif（基序）、长程结构背景以及 epistasis（上位性），所以需要测试学习型序列表示。

## 5.2 Main Sequence Encoder（主序列编码器）：参考 Kelvin et al. 2026

Kelvin 等人在 2026 年 Nucleic Acids Research（《核酸研究》）的核糖开关工作中采用：

- 3-mer tokenizer（3-mer 分词器）；
- BERT-like Transformer Encoder（类 BERT 变换器编码器）；
- 6 个 Transformer blocks（变换器模块）；
- embedding dimension（嵌入维度）128；
- MLM（掩码语言建模）；
- Triplet Margin Loss（三元组间隔损失）；
- 使用 36,925 条 riboswitch（核糖开关）序列预训练；
- `[CLS]` token（分类标记）的 128 维输出作为 sequence embedding（序列嵌入）。

## 5.3 我们不能直接照搬，需要针对 FAD 修改

原论文 RNA 最长约 70 nt，而我们约 141 nt。同时我们最大的特殊性是 141 nt 中真正变化的主要只有 13 个设计位点。

如果完全照搬随机 MLM，大量 mask（掩码）会落在几乎永远不变化的 scaffold（骨架）位置，模型可能把能力浪费在“恢复固定序列”。因此建议比较两个版本。

### Version A（版本 A）：Full-Length Encoder（全长编码器）

输入完整 141 nt：

\[
141nt \rightarrow 3\text{-mer} \rightarrow Transformer
\]

把论文中的 `max_sequence_length = 70` 改为适应约 141 token 的长度。

### Version B（版本 B）：Mutation-Aware Encoder（突变感知编码器）

仍使用全长 RNA，但预训练时让 mask（掩码）更集中于：

- 13 个可变位点；
- 可变位点附近局部窗口；
- 与可变位点可能形成配对的结构位置。

初始可尝试：70% masking（掩码）来自突变相关区域，30% 保留普通随机 masking。

## 5.4 无标签候选序列也有价值

Transformer（变换器）预训练不需要 GBSA 标签。因此如果我们已有数万到十万条候选序列，可以用于：

\[
MLM + Triplet\ Learning（三元组学习）
\]

得到 FAD-specific encoder（FAD 专用序列编码器）。

> **候选池不仅是未来要搜索的空间，同时也是序列表征的自监督训练语料。**

## 5.5 Physics Branch（物理特征支路）

经过第 4 页筛选后，假设留下 5–10 个 feature blocks（特征块）。不要再次把所有原始维度直接拼回去。

建议对每一个块单独压缩：

```text
contact / structure / energy / motif / pair-probability ...
                ↓
          Standardization（标准化）
                ↓
               PCA
                ↓
          2–5维 block embedding
```

最终把所有物理块组合成：

\[
z_{phys}\approx20\sim40d
\]

第一版优先使用 PCA 而不是复杂 autoencoder（自编码器），因为 2000 个有标签样本仍属小数据，PCA 不使用 GBSA 标签，泄漏风险和过拟合风险更低，同时易解释、易复现。

---

# 第 6 页：Feature Fusion（特征融合）——先简单，再逐步复杂化

## 页面标题

**RNA Sequence（序列） + Physics（物理）如何融合：从可验证基线开始**

## 6.1 必须做的三个 Ablation（消融实验）

### Model A：Sequence Only（仅序列）

输入 \(z_{seq}\)，确认序列表示本身能学到多少 GBSA 信息。

### Model B：Physics Only（仅物理特征）

输入 \(z_{phys}\)，确认筛选后的结构/物理信息单独能达到什么水平。

### Model C：Sequence + Physics（序列 + 物理）

\[
z=[z_{seq};z_{phys}]
\]

检验两种信息是否真正互补。

## 6.2 第一版建议：Early Fusion（早期融合 / 直接拼接）

例如：

\[
z_{seq}=128d,\quad z_{phys}=32d
\]

则：

\[
z_{fusion}=160d
\]

然后进入小型回归网络：

```text
160 → 128 → 64 → 1 个 GBSA
```

或者：

```text
160 → 256 → 128 → 64 → 1 个 GBSA
```

不建议直接照搬 Kelvin 论文的 9 层、每层 256 的回归网络，因为我们的监督标签只有约 2000 个，过深网络更容易过拟合。

## 6.3 第二阶段可尝试 Gated Fusion（门控融合）

如果直接拼接证明有效，再尝试让模型学习不同 RNA 上序列信息和物理信息的权重，例如：

\[
g=\sigma(W[z_{seq};z_{phys}])
\]

但这不是第一阶段必须完成的工作。

## 现场讲稿

> 我们不希望把所有东西混成一个 500 或 1500 维向量，然后完全依赖模型自己处理。更合理的是让 RNA 序列和结构物理信息成为两个信息源：序列支路负责学习突变组合和上下文，物理支路负责提供 NUPACK、能量、配对概率等显式物理信息。两支路都压到小维度后再融合，这样更适合我们 2000 条标签的规模，也更方便证明每一类信息到底有没有贡献。

---

# 第 7 页：直接对标论文——NAR 2026 的核糖开关 Deep Batch Bayesian Optimization

## 页面标题

**直接相关工作：核糖开关已实现真实实验闭环的 Deep Batch Bayesian Optimization（深度批量贝叶斯优化）**

核心文献：

Daniel Kelvin, Erik Kubaczka, Marianna Karava, Heinz Koeppl, Beatrix Suess.  
**Iterative design of a NAND hybrid riboswitch by deep batch Bayesian optimization（利用深度批量贝叶斯优化迭代设计 NAND 混合核糖开关）**  
*Nucleic Acids Research（核酸研究）*, 2026, 54(5), gkag145.  
DOI: `10.1093/nar/gkag145`

开源：

- GitHub：`Self-Organizing-Systems-TU-Darmstadt/NANDRiboswitchDesignByBayesOpt`
- Zenodo：`10.5281/zenodo.15185168`

## 7.1 论文流程

```text
大量 riboswitch 序列
        ↓
Sequence Encoder Pre-training（序列编码器预训练）
MLM + Triplet Loss
        ↓
少量真实实验功能标签
        ↓
Deep Ensemble（深度集成）
        ↓
预测功能 + 预测不确定性
        ↓
UCB（置信上界）
        ↓
Kriging Believer（克里金信念批量策略）
        ↓
每轮 16 条 RNA
        ↓
in vivo validation（体内实验验证）
        ↓
真实标签返回
        ↓
模型重训
```

## 7.2 关键数字

论文中：

- 预训练 RNA：**36,925 条 riboswitch sequences（核糖开关序列）**；
- candidate pool（候选池）：**3,256 条**；
- 初始 individually characterized（逐条实验表征）序列：**34 条**；
- ensemble size（集成模型数量）：**100**；
- batch size（每轮批量）：**16**；
- UCB coverage（置信覆盖率）：**95%**；
- 共做 3 轮 Bayesian optimization（贝叶斯优化）；
- 新增实验序列：\(16\times3=48\)；
- 总共约 82 条序列进入精细体内表征；
- 最优 NAND score（与非评分）从约 1.70 提高到 3.42。

## 7.3 论文真正值得借鉴的部分

不是简单“它用了 Transformer”，更重要的是四个模块：

1. **Task-specific Sequence Pretraining（任务相关序列预训练）**；
2. **Deep Ensemble Uncertainty（深度集成不确定性）**；
3. **Kriging Believer Batch Selection（克里金信念批量选择）**；
4. **DBTL Cycle（设计—构建—测试—学习闭环）**。

## 7.4 论文对我们的直接意义

这篇工作提供了强先验证据：

> **在 riboswitch（核糖开关）问题中，使用不确定性驱动的 batch Bayesian optimization（批量贝叶斯优化）进行多轮真实实验闭环是可行的。**

我们真正需要验证的是：

> **把该框架改造成 GBSA 预测导向以后，是否能够比随机新增标签更快提高 FAD 的泛化性能。**

---

# 第 8 页：Kelvin 2026 → FAD：四项关键改造

## 页面标题

**从 NAND 核糖开关设计到 GBSA 学习：四项关键改造**

| Kelvin et al. 2026 | FAD 拟采用方案 |
|---|---|
| Hybrid riboswitch（混合核糖开关） | PreQ1 / RF00522 核糖开关 |
| 优化 P1 stem（P1 茎区） | 主要优化 13 个可变位点组合 |
| Sequence only（主要基于序列） | Sequence + Physics（序列 + 物理/结构） |
| 4 个表达状态 | 1 个 GBSA 连续目标 |
| NAND score 最大化 | 第一阶段提高 GBSA 预测精度 |
| UCB（置信上界） | 第一阶段 uncertainty-driven AL（不确定性主动学习） |
| UCB 找高性能序列 | 第二阶段可改为 LCB 找低 GBSA |
| 100 个神经网络 ensemble（集成） | 先从 20–50 个开始验证 |
| 16 条 / round（轮） | 50 / 100 / 200 由离线模拟 + 实验容量决定 |
| KB batch（克里金信念批量选择） | 保留，并加入 physics-aware diversity（物理感知多样性） |
| wet-lab expression（实验表达量） | 昂贵 GBSA / 实验真实标签 |

## 改造 1：Sequence-only → Sequence + Physics

Kelvin 论文主要用序列作为输入，而论文 Discussion（讨论）明确提出未来可以把 structural information（结构信息）整合进模型。

我们的优势是已有：

- NUPACK（核酸热力学计算）；
- structure（结构）；
- base-pair probability（碱基配对概率）；
- energy landscape（能量景观）；
- contact（接触）；
- motif（基序）；
- positional features（位置特征）等。

因此可构建：

\[
\boxed{Sequence\ Encoder + Selected\ Physics}
\]

## 改造 2：NAND Score → GBSA

原论文：

\[
Sequence \rightarrow (\phi_{w/o},\phi_{Tc},\phi_{Neo},\phi_{both}) \rightarrow NAND\ Score
\]

我们：

\[
Sequence + Physics \rightarrow GBSA
\]

原论文的 Bernoulli KL divergence（伯努利 KL 散度）不适合我们的连续自由能目标。

建议第一版使用 Huber Loss（Huber 稳健回归损失），后续可测试：

\[
L=L_{Huber}+\lambda L_{Rank}
\]

## 改造 3：UCB Optimization（UCB 优化） → Active Learning First（先主动学习）

原论文目标是 \(\max S(x)\)，所以 UCB 适合找“可能非常好的序列”。

我们的当前首要目标是提高整体预测精度，所以第一阶段更应该找“模型最缺信息、测完后最可能提升泛化能力的序列”。

因此第一阶段 acquisition function（采集函数）应以：

\[
\sigma_{epistemic}(x)
\]

即认知不确定性为核心。

## 改造 4：Sequence Diversity → Physics-aware Diversity

Kelvin 论文后续分析发现，普通 Levenshtein distance（编辑距离）与功能差异关系有限，而 base-pair distance（碱基配对距离）与功能差异关系更强。

因此我们的批内多样性不应只看 Hamming distance（汉明距离），建议：

\[
d(i,j)=\lambda_s d_{seq}(i,j)+\lambda_p d_{physics}(i,j)+\lambda_{str}d_{structure}(i,j)
\]

这可以形成 **Physics-aware Batch Selection（物理感知批量选择）**。

---

# 第 9 页：Probabilistic Surrogate（概率代理模型）——“贝叶斯”到底怎么做

## 页面标题

**代理模型选择：不仅比较预测精度，还必须比较不确定性是否可信**

## 9.1 关键概念

Bayesian Optimization（贝叶斯优化）真正需要的是：

\[
p(y\mid x,D)
\]

即对未知序列给出预测分布，而不是单一数值。

我们希望得到：

\[
\mu(x)=E[y\mid x,D]
\]

和：

\[
\sigma(x)=Uncertainty[y\mid x,D]
\]

## 9.2 建议比较三类代理模型

### Baseline 1：现有强传统模型

- Random Forest, RF（随机森林）；
- Extra Trees, ET（极端随机树）；
- Gradient Boosting, GBR（梯度提升回归）；
- XGBoost（极端梯度提升）；
- Support Vector Regression, SVR（支持向量回归）。

### Baseline 2：Gaussian Process（高斯过程）

推荐测试 Matérn Kernel + ARD（Matérn 核 + 自动相关性确定）。

优点：概率定义清楚，天然输出均值与方差，适合小数据 Bayesian Optimization（贝叶斯优化）。

风险：特征维度和样本规模上升时计算成本增加；对复杂离散序列空间的表达可能不如学习型 encoder（编码器）。

### Main Candidate：Deep Ensemble（深度集成）

训练：

\[
M=20\sim50
\]

个独立模型，每个模型：

\[
f_{\theta_m}(x)\rightarrow\widehat{GBSA}_m
\]

得到：

\[
\mu(x)=\frac{1}{M}\sum_{m=1}^{M}\widehat{GBSA}_m(x)
\]

\[
\sigma_{ensemble}(x)=Std[\widehat{GBSA}_1(x),...,\widehat{GBSA}_M(x)]
\]

## 9.3 为什么不直接照论文用 100 个模型

Kelvin 论文采用 100 个模型用于构建稳定 empirical predictive distribution（经验预测分布）。但其计算复杂度与 ensemble size × batch size（集成规模 × 批量大小）相关。

我们的候选池更大、预期 batch 可能达到 50–100，因此建议：

1. 从 20 个集成成员开始；
2. 比较 20 / 50 / 100 的 uncertainty stability（不确定性稳定性）；
3. 如果 20 和 50 给出的候选排序已高度一致，就没有必要为形式上“复现论文”强行使用 100。

## 9.4 代理模型选择标准

不能只看 Spearman，至少同时看：

### A. Prediction Accuracy（预测精度）

- Spearman correlation（斯皮尔曼相关）；
- MAE（平均绝对误差）；
- RMSE（均方根误差）。

### B. Uncertainty Calibration（不确定性校准）

例如：

\[
corr(\sigma(x),|\widehat{y}-y|)
\]

### C. Coverage（覆盖率）

检查预测区间的真实覆盖率。

### D. Active-Learning Efficiency（主动学习效率）

固定新增 100 个标签时，哪个代理最终让 frozen test（冻结测试集）的 Spearman 提升最多。

---

# 第 10 页：Kriging Believer（KB）——为什么不能一次直接拿“不确定性最高的前 100”

## 页面标题

**批量实验的核心问题：100 条必须“信息互补”，而不是 100 个相似的高不确定序列**

## 10.1 普通 Top-K 的问题

高不确定序列可能集中在同一个局部区域。一次实验测 100 个高度相似序列，相当于花 100 次成本重复回答一个问题。

## 10.2 Kriging Believer（克里金信念批量策略）

第一条：

\[
x_1=\arg\max A(x)
\]

然后暂时假设：

\[
y_1^{pseudo}=\mu(x_1)
\]

把它加入训练集，更新代理模型，再选择第二条。如此重复直到形成：

\[
B=\{x_1,x_2,...,x_m\}
\]

实验返回后，用真实 GBSA 替换所有 pseudo labels（伪标签）。

## 10.3 对 FAD 的工程改造：KB-lite（轻量克里金信念）

如果 \(M=50\) 个 ensemble 模型且 batch=100，每选一个样本都完整重新训练 Transformer 会很重。

建议：

### 实验轮之间

真实 100 条 GBSA 回来后完整训练 encoder（编码器）+ physics branch（物理支路）+ regression head（回归头）。

### 同一实验 batch 内

进行 KB 时：

- 冻结 sequence encoder（序列编码器）；
- 冻结 physics representation（物理表示）；
- 只快速更新 regression head / uncertainty head（回归头 / 不确定性头）。

这样保留 KB 的核心作用，同时降低计算量。

---

# 第 11 页：Acquisition Function（采集函数）——先把模型学准，再找最优 GBSA

## 页面标题

**候选选择分两阶段：Learning（学习）→ Optimization（优化）**

## Stage I：Bayesian Active Learning（贝叶斯主动学习）

当前目标：

\[
\boxed{\text{提高全局 GBSA 预测精度}}
\]

关注：

1. Epistemic Uncertainty（认知不确定性）；
2. Distance to Labeled Set（与已标注数据的距离）；
3. Batch Diversity（批内多样性）；
4. 少量 Exploitation（利用），保留当前预测有潜力的候选。

第一版混合采集函数不建议直接把不同量纲原值相加，而先转成 percentile rank（百分位排名）：

\[
A_{learn}(x)=w_U R_U(x)+w_D R_D(x)+w_Q R_Q(x)
\]

初始可测试：

\[
w_U=0.6,\quad w_D=0.3,\quad w_Q=0.1
\]

这只是待验证初始值，应由回顾性模拟决定，而不是人为固定。

## Stage II：Bayesian Optimization（贝叶斯优化）

当模型足够可靠后，如果目标是寻找更低 GBSA，可使用：

\[
LCB(x)=\mu(x)-\beta\sigma(x)
\]

即 Lower Confidence Bound（置信下界）。

整个项目可以自然从“提高模型精度”转向“用模型寻找最优序列”。

---

# 第 12 页：正式实验前，先用已有 2000 条做 Retrospective AL Simulation（回顾性主动学习模拟）

## 页面标题

**先用历史真实标签回答三个问题：选什么？每轮多少？能提升多少？**

## 12.1 为什么必须先做

目前我们仍不知道：

1. Uncertainty（不确定性）选择是否真的比 Random（随机选择）更有效？
2. KB（克里金信念策略）是否能带来额外收益？
3. batch=50、100、200 哪一个在本项目上最划算？
4. 新增 300 个标签有没有可能显著推动 Spearman？
5. 0.6 是否有任何数据驱动依据？

Kelvin 论文证明的是“贝叶斯优化可以找到性能更好的核糖开关”，它没有证明“能让我们的 GBSA held-out Spearman 从约 0.37 提高到 0.6”。所以必须用我们的历史数据做模拟。

## 12.2 推荐模拟设计

例如将 \(N=2000\) 划成：

- Initial Train（初始训练集）：1200；
- Simulated Candidate Pool（模拟候选池）：500；
- Frozen Test（冻结测试集）：300。

500 条候选真实 GBSA 暂时隐藏；300 条冻结测试集从头到尾不参与候选选择、特征筛选或参数选择。

## 12.3 模拟过程

```text
1200 个已知标签
     ↓
训练 surrogate
     ↓
从 500 个“假装无标签”的候选里选 50/100
     ↓
揭示这些样本真实 GBSA
     ↓
加入训练集
     ↓
重新训练
     ↓
再选下一批
     ↓
每一轮只在 frozen test 上评估
```

## 12.4 必须比较的采集策略

- Random Sampling（随机抽样）；
- Greedy Predicted Best（贪心预测最优）；
- Uncertainty Sampling（不确定性采样）；
- Uncertainty + Diversity（不确定性 + 多样性）；
- UCB / LCB + KB（置信界 + 克里金信念）；
- Physics-aware Uncertainty + KB（物理感知不确定性 + KB）。

## 12.5 必须比较的 batch size（批量大小）

建议至少模拟：

\[
25,50,100,200
\]

每种设置用 20–30 个随机种子，并保持总新增标签预算一致，例如总共 +400：

- batch 50：50×8；
- batch 100：100×4；
- batch 200：200×2。

## 12.6 最终关键图

横轴：Number of Acquired GBSA Labels（新增 GBSA 标签数）。  
纵轴：Frozen-test Spearman（冻结测试集 Spearman）。

画 Random、Uncertainty、KB、Physics-aware KB 四条学习曲线。

如果同样新增 300 条后主动学习明显优于随机，才真正有依据投入新实验。

---

# 第 13 页：第一轮到底 50、96、100 还是 200？

## 页面标题

**Batch Size（批量大小）不是固定常数：由实验吞吐量 + 离线学习效率共同决定**

## 13.1 96 没有算法上的特殊意义

96 的常见来源是 96-well plate（96 孔板）实验格式，不是贝叶斯优化的数学要求。

Kelvin 论文实际 batch=16。其他主动学习实验也可以每轮几十到几百条。

所以：

> **如果实验平台适合 100 条，100 完全可以，而且从项目管理上比 96 更直观。**

## 13.2 为什么不默认第一轮 200

如果总预算 200：

- 一次做 200：\(2000\rightarrow2200\)，后 100 条仍由旧模型选择；
- 分两轮做 100：\(2000\rightarrow2100\rightarrow2200\)，第二个 100 利用了第一轮真实反馈。

理论上分轮更有机会提高信息效率，但分轮增加实验周期间隔和项目协调成本，所以最终要共同优化。

## 13.3 明天建议讨论三个候选

### 方案 A：50 / round（每轮 50）

反馈频繁，但实验轮数多。

### 方案 B：100 / round（每轮 100）

迭代和实验效率折中，目前最适合作为默认讨论值。

### 方案 C：200 / round（每轮 200）

适合一次启动成本高、不能频繁实验的场景。

最终通过：

\[
\boxed{Retrospective\ Simulation + Wet-lab\ Constraints}
\]

共同决定。

## 13.4 如果使用 96 孔板

“96 个孔”不等于“96 个新序列”，必须预留 positive control（阳性对照）、negative control（阴性对照）、reference sequence（参考序列）、technical replicate（技术重复）、biological replicate（生物学重复）。因此每板真正的新序列可能只有 70–90，需要实验团队给出实际约束。

---

# 第 14 页：推荐的第一阶段闭环设计

## 页面标题

**初步建议：先规划 3 轮，每轮约 100 条，但以离线模拟结果为准**

如果实验容量允许，可以把项目预算先按 \(3\times100\approx300\) 个新 GBSA 标签进行规划，但不是一次性承诺 300 条，而是设置 checkpoint（检查点）。

### Round 0（第 0 轮）

已有 \(N\approx2000\)，完成：

- 特征筛选；
- Sequence–Physics representation（序列—物理表示）；
- surrogate benchmark（代理模型比较）；
- uncertainty calibration（不确定性校准）；
- retrospective AL simulation（回顾性主动学习模拟）。

### Round 1（第 1 轮）

模型选择约 100 条高信息量序列，实验返回后：

\[
2000\rightarrow2100
\]

重新完整训练全部模型，并检查 frozen benchmark、uncertainty、GBSA 分布覆盖和 mutation cluster（突变簇）表现。

### Round 2 / 3

每轮重新计算 acquisition（采集值），不能重复使用上一轮排名，因为：

\[
p(y|x,D_{2100})\neq p(y|x,D_{2000})
\]

第三轮后得到约 \(N=2300\)，再判断是否继续到 400–500 新标签，或开始从 Active Learning（主动学习）切换到 Bayesian Optimization（贝叶斯优化）。

---

# 第 15 页：第一轮 100 条应该包含什么样的序列

## 页面标题

**第一轮目标：最大化信息增益，而不是把 100 个名额全部给“预测最优”**

如果回顾性模拟支持 hybrid strategy（混合策略），可采用如下初始构成讨论：

| 类型 | 参考数量 | 目的 |
|---|---:|---|
| High epistemic uncertainty（高认知不确定性） | 45 | 解决模型最不会的区域 |
| Coverage / novelty（覆盖 / 新颖性） | 25 | 补足当前数据空间盲区 |
| Physics-aware diversity（物理感知多样性） | 与上述联合约束 | 避免选到物理上重复的 RNA |
| Predicted promising（预测有潜力） | 20 | 保留实际优化价值 |
| Random control（随机对照） | 10 | 证明主动学习是否优于随机 |

注意：Diversity（多样性）更适合作为整个 batch 的约束，而不是机械单独划 20 条。

更严格的流程是：

```text
先根据 acquisition 产生候选
     ↓
KB 逐条构造 batch
     ↓
对整个 batch 施加 sequence + physics diversity
     ↓
保留少量 random controls
```

---

# 第 16 页：GBSA Label Quality（标签质量）——0.6 能不能达到，取决于“可学习信号”有多少

## 页面标题

**在追求更高 Spearman 前，必须估计 GBSA 标签本身的噪声上限**

模型看到的可能是：

\[
GBSA_{observed}=GBSA_{true}+\epsilon
\]

其中 \(\epsilon\) 可能来自实验测量噪声、MD（分子动力学）采样不足、初始构象差异、trajectory（轨迹）长度、replicate（重复）差异、protocol（实验/模拟协议）差异等。

## 建议实验团队共同建立 QC（质量控制）信息

对于一部分序列，尽量获取：

- replicate mean（重复均值）；
- replicate standard deviation（重复标准差）；
- protocol version（协议版本）；
- batch ID（实验批次编号）；
- failed / censored samples（失败 / 截断样本）；
- 质量评分。

如果可能，优先选择少量历史代表序列做 repeated measurement（重复测量），用于估计 \(Var(\epsilon)\)。

如果重复实验一致性很差，模型存在天然上限；反之，如果同一序列跨重复的排序一致性很高，则说明 0.6 至少不是被标签噪声直接否定的。

---

# 第 17 页：0.6 到底应该怎么表达

## 页面标题

**Spearman 0.6：作为 Stretch Target（挑战目标），不作为实验承诺**

当前稳定模型若约为 \(\rho\approx0.37\)，那么 \(0.37\rightarrow0.60\) 是非常大的提升。

不能因为某篇论文“目标性能翻倍”，就推导我们 Spearman 也能达到 0.6。Kelvin 论文优化的是 best NAND functionality（最佳 NAND 功能），不是 global held-out Spearman（全局留出集 Spearman）。

## 更严谨的目标定义

### Minimum Success（最低成功）

主动学习显著优于随机新增同样标签：

\[
\Delta\rho_{AL}>\Delta\rho_{Random}
\]

### Stage Goal（阶段目标）

\[
0.37\rightarrow0.45+\rightarrow0.50+
\]

### Stretch Goal（挑战目标）

如果 learning curve（学习曲线）持续上升，并且 label noise（标签噪声）允许：

\[
\rho\sim0.6
\]

## 现场建议说法

> “0.6 是我们的挑战目标，但不是今天用理论直接保证的数字。我们会在正式消耗大量实验资源前，利用已有 2000 条标签做回顾性主动学习实验，建立 learning curve。第一、二轮新实验完成后再根据真实增益判断 0.6 是否可达以及需要多少新增样本。”

---

# 第 18 页：实验团队需要提供什么——数据接口必须一次定义清楚

## 页面标题

**实验输出不仅要有 GBSA 数字，还要能用于机器学习质量控制**

| 字段 | 说明 |
|---|---|
| `sequence_id` | 唯一编号 |
| `sequence` | 完整 RNA 序列 |
| `mutation_pattern` | 13 位点突变模式 |
| `gbsa` | 最终目标值 |
| `replicate_id` | 重复编号 |
| `gbsa_raw` | 每次重复原始值 |
| `gbsa_mean` | 重复均值 |
| `gbsa_std` | 重复标准差 |
| `protocol_version` | 实验 / 模拟协议版本 |
| `batch_id` | 实验批次 |
| `qc_status` | 质量控制状态 |
| `failure_reason` | 如失败，记录原因 |
| `date` | 完成日期 |

如果 GBSA 是由多个中间量计算得到，建议中间量也全部保留，不要只返回最后一列。

---

# 第 19 页：明天必须和实验团队确定的 10 个问题

## 页面标题

**会后需要形成明确决策，而不是只讨论算法**

1. **单轮最大可处理多少“新序列”？** 50 / 100 / 200 的实际工作量分别是多少？
2. **实验是否受 96-well plate（96 孔板）限制？** 如果受限，一块板能容纳多少真正的新序列？
3. **每条序列需要多少 biological replicates（生物学重复）和 technical replicates（技术重复）？**
4. **单轮 turnaround time（周转时间）是多少？**
5. **一次做 200 与分两次做 100，成本差多少？**
6. **是否可以重复测量少量历史样本？** 用于估计标签噪声和批次效应。
7. **序列是否存在不可实验约束？** 如 GC 含量、长度、motif（基序）、二级结构、合成难度等。
8. **GBSA 数据失败率大约多少？**
9. **是否存在 batch effect（批次效应）？** 不同轮实验是否完全可比？
10. **第一阶段更优先“模型变准”还是“同时找最好序列”？** 建议明确第一阶段 KPI（关键绩效指标）。

---

# 第 20 页：项目实施路线图

## 页面标题

**从现在到第一轮实验的建议实施顺序**

### Phase 1：Feature Selection（特征选择）

\[
37\ blocks\rightarrow Spearman+MI\rightarrow stability\rightarrow redundancy\rightarrow5\sim10\ blocks
\]

输出：特征块排名、稳定性、冗余图、最终保留列表。

### Phase 2：Representation Benchmark（表示方法比较）

比较：

1. 13-site one-hot（13 位点独热编码）；
2. full-sequence encoder（全序列编码器）；
3. mutation-aware encoder（突变感知编码器）；
4. physics-only representation（仅物理表示）；
5. sequence + physics fusion（序列 + 物理融合）。

### Phase 3：Probabilistic Surrogate Benchmark（概率代理比较）

比较 GP（高斯过程）、Deep Ensemble（深度集成）、RF / XGB ensemble（树模型集成）。

指标：Spearman、MAE / RMSE、不确定性校准。

### Phase 4：Retrospective AL Benchmark（回顾性主动学习评测）

比较 Random、Greedy、Uncertainty、UCB / LCB、KB、Physics-aware KB，同时比较 batch=25/50/100/200。

### Phase 5：Round 1（第一轮真实昂贵标签）

根据离线模拟和实验容量，选择 50–100 条作为第一轮优先候选。

### Phase 6：Iterative Learning（迭代学习）

真实数据返回后重新运行特征流水线、代理模型、不确定性评估、采集函数和下一轮候选选择。

---

# 第 21 页：本项目拟形成的方法学创新

## 页面标题

**我们不是简单复现 Kelvin 论文，而是针对 GBSA 做方法扩展**

### Innovation 1（创新点 1）

**Block-wise Stable Feature Selection（特征块级稳定筛选）**：

\[
Spearman+MI+Stability+Redundancy
\]

用于高维异构 RNA 物理特征。

### Innovation 2（创新点 2）

**Sequence–Physics Fusion（序列—物理融合）**：参考深度 RNA sequence encoder，同时引入 NUPACK、secondary structure（二级结构）、energy landscape（能量景观）、pairing probability（配对概率）、motif（基序）等。

### Innovation 3（创新点 3）

**Prediction-oriented Bayesian Active Learning（面向预测提升的贝叶斯主动学习）**：Kelvin 的目标是最大化 NAND score，我们第一阶段改成最大化 information gain（信息增益），先提高 GBSA surrogate（代理模型）准确性。

### Innovation 4（创新点 4）

**Physics-aware Kriging Believer（物理感知的克里金信念批量选择）**：批次多样性同时考虑 sequence + structure + physics（序列 + 结构 + 物理），避免“序列不同、物理状态重复”的实验浪费。

---

# 第 22 页：建议的结论页

## 页面标题

**希望今天和实验团队共同确认的方案**

> **1. 当前模型瓶颈需要从“继续堆模型”转向“高价值新增 GBSA 标签”。**

> **2. 特征侧采用 Spearman + Mutual Information（互信息）+ 稳定性 + 去冗余，形成紧凑物理特征集合。**

> **3. 模型侧参考 NAR 2026 核糖开关 deep batch Bayesian optimization（深度批量贝叶斯优化），构建 Sequence + Physics（序列 + 物理）的概率代理模型。**

> **4. 实验前先用已有约 2000 条 GBSA 做 retrospective active-learning simulation（回顾性主动学习模拟），确定 acquisition（采集函数）和 50 / 100 / 200 的合理 batch size（批量大小）。**

> **5. 正式建立“模型选样 → 昂贵 GBSA → 数据回流 → 重训 → 再选样”的 DBTL（设计—构建—测试—学习）闭环。**

底部：

\[
\boxed{2000\rightarrow+100\rightarrow2100\rightarrow+100\rightarrow2200\rightarrow\cdots}
\]

> **Primary KPI（主要指标）**：稳定 held-out Spearman（留出集斯皮尔曼相关）的逐轮提升。  
> **Stretch Target（挑战目标）**：约 0.6，但由实际 learning curve（学习曲线）决定。

---

# 二、建议的完整技术方案（备份页 / 会后技术文档）

## 23. Feature Selection（特征选择）的推荐算法规范

设共有 \(G=37\) 类特征块：

\[
\mathcal{F}=\{F_1,F_2,\ldots,F_G\}
\]

每个特征块：

\[
F_g\in\mathbb{R}^{N\times d_g}
\]

目标 \(y\in\mathbb{R}^N\) 为 GBSA。

### 外层验证

每个 seed 先划分：

\[
D=D_{train}\cup D_{test}
\]

所有 Feature Selection（特征选择）只在 \(D_{train}\) 内完成。

### 单块关联度

\[
r_{g,j}=|\rho_s(F_{g,j},y)|
\]

\[
m_{g,j}=MI(F_{g,j};y)
\]

定义：

\[
R_g=\operatorname{mean}(\text{Top }20\%\ r_{g,j})
\]

\[
M_g=\operatorname{mean}(\text{Top }20\%\ m_{g,j})
\]

也可比较 median（中位数）、Top-3 mean（前三平均）、Top-10% mean（前 10% 平均），通过 nested CV（嵌套交叉验证）确定最稳定方案。

### Rank Fusion（排名融合）

\[
S_g=0.5\cdot rank_{\rho}(g)+0.5\cdot rank_{MI}(g)
\]

### Stability Selection（稳定性选择）

重复 \(K=20\sim30\) 次：

\[
Stability_g=\frac{1}{K}\sum_{k=1}^{K}I(g\in TopM_k)
\]

### Redundancy（冗余）

\[
F_g\rightarrow PCA(F_g)\rightarrow Z_g
\]

如果两个 block 的低维表示高度相似，则根据 target relevance（目标相关性）、stability（稳定性）、biological interpretability（生物学可解释性）和 cost（成本）保留一个。

### 防止纯 Filter（过滤法）漏掉 Interaction（交互）

可能存在：

\[
F_a\not\rightarrow y,\quad F_b\not\rightarrow y,\quad F_a\times F_b\rightarrow y
\]

所以保留少量具有明确物理意义的 interaction-prone blocks（可能存在交互的特征块），后续通过 ablation（消融）确认。

---

# 三、Sequence Encoder（序列编码器）改造建议

## 24. Baseline（基线）

13 位点 One-hot（独热编码）：

\[
13\times4=52d
\]

## 25. Kelvin-style Encoder（Kelvin 风格编码器）

原论文架构：3-mer tokenizer（3-mer 分词）、6-layer Transformer encoder（6 层变换器编码器）、\(d_{model}=128\)、4-head attention（4 头注意力）、MLM（掩码语言模型）、triplet loss（三元组损失）。

我们的修改：

- `max_sequence_length` 增加到适应 141 nt；
- 使用 FAD candidate pool（FAD 候选池）做自监督预训练；
- 对 mutation positions（突变位点）提高 masking probability（掩码概率）；
- 比较 encoder frozen（冻结编码器）和 fine-tuned（微调编码器）。

## 26. 为什么不是直接使用原论文 checkpoint（检查点）

原论文预训练语料来自其四环素—新霉素混合核糖开关体系，与 PreQ1 / RF00522 的 scaffold（骨架）、sequence distribution（序列分布）和 function（功能）不同。

因此原 checkpoint 可作为 transfer-learning baseline（迁移学习基线），但不应默认就是最佳 FAD encoder。更合理的主方案是使用我们自己的候选序列做 domain-specific self-supervised pretraining（领域专用自监督预训练）。

---

# 四、Regression Head（回归头）与 Loss（损失函数）

## 27. 第一版网络

\[
z_{seq}\in\mathbb{R}^{128},\quad z_{phys}\in\mathbb{R}^{20\sim40}
\]

拼接后约 148–168 维，MLP（多层感知机）可采用：

\[
160\rightarrow128\rightarrow64\rightarrow1
\]

## 28. Loss（损失）

第一版：

\[
L=L_{Huber}
\]

第二版：

\[
L=L_{Huber}+\lambda L_{rank}
\]

其中 \(L_{rank}\) 为 pairwise ranking loss（成对排序损失），用于增强排序一致性。

---

# 五、Deep Ensemble（深度集成）设计

## 29. Ensemble Diversity（集成多样性）

建议每个成员至少在 random initialization（随机初始化）、mini-batch order（小批次顺序）方面不同；bootstrap resampling（自助重采样）作为可选实验。

## 30. 是否训练 Encoder（编码器）

### 初始阶段

encoder 固定，只训练 regression head（回归头），防止 2000 标签破坏预训练表示，也降低 ensemble 训练成本。

### 后续阶段

比较 frozen encoder（冻结编码器）、last 1–2 Transformer blocks fine-tuning（最后 1–2 层微调）、full fine-tuning（全量微调），由验证结果决定。

---

# 六、Uncertainty Quantification（不确定性量化）必须单独验证

## 31. 不能把 ensemble std（集成标准差）自动当作“正确不确定性”

建议将测试样本按 \(\sigma(x)\) 分成五组，比较各组 MAE。如果最不确定组误差显著更高，说明 uncertainty 有效。

## 32. Calibration（校准）

检查 50%、80%、90% prediction interval（预测区间）的真实覆盖率。必要时使用 conformal calibration（保形校准）等方法后处理。

---

# 七、Physics-aware KB（物理感知 KB）的建议算法

设 \(\mathcal{X}_{pool}\) 为尚未获得昂贵 GBSA 的候选池。

对每条候选预先计算 sequence embedding（序列嵌入）、physics embedding（物理嵌入）、current mean prediction（当前均值预测）和 current uncertainty（当前不确定性）。

第一个候选：

\[
x_1=\arg\max_x A(x)
\]

加入 pseudo observation（伪观测）：

\[
(x_1,\mu(x_1))
\]

对于后续候选，与已选集合 \(S\) 保持足够距离：

\[
d(x,S)=\min_{s\in S}[\lambda_s d_{seq}(x,s)+\lambda_p d_{phys}(x,s)]
\]

如果 \(d(x,S)<\tau\)，降低候选优先级。重复直到 \(|S|=B\)，随后整批交给实验。真实 GBSA 返回后删除伪标签并完整重训。

---

# 八、Retrospective Simulation（回顾性模拟）的统计设计

## 33. 多 seed（随机种子）

建议 20–30 组不同划分，汇报 mean（均值）、median（中位数）和 95% bootstrap CI（95% 自助置信区间）。

## 34. Random Split（随机划分）之外增加 Cluster Split（簇划分）

普通随机划分可能让训练集和测试集包含几乎相同的 mutation pattern（突变模式）。

因此建议同时做：

- Random Split（随机划分）：一般泛化；
- Cluster Split（簇划分）：按 mutation pattern / sequence distance（突变模式 / 序列距离）划分，评估新区域泛化。

主动学习的真实价值可能在 cluster split 上更明显。

---

# 九、Stopping Criteria（停止标准）

建议预先定义：

1. **Prediction Plateau（预测平台期）**：连续两轮 \(\Delta Spearman<0.01\) 且增益不稳定；
2. **Uncertainty Saturation（不确定性饱和）**：候选池大部分样本不确定性已明显下降；
3. **Active Learning ≈ Random（主动学习与随机接近）**；
4. **Experimental Cost（实验成本）**：每提升 0.01 Spearman 的成本过高；
5. **转向 Optimization（优化）**：预测精度已满足需求，开始把采集重点转向低 GBSA 候选。

---

# 十、明天可能被追问的问题及答辩话术

## Q1：为什么是 96？100 不可以吗？

> 96 没有算法上的特殊意义，主要来自 96 孔板实验格式。贝叶斯优化本身不要求 96。Kelvin 的核糖开关论文每轮实际只选择 16 条。对我们来说 100 完全可以。我们希望先用已有 2000 条做回顾性主动学习模拟，比较 50、100、200 的信息效率，再结合实验团队一次实验的真实吞吐量决定。如果使用 96 孔板，还要给阳性、阴性对照和重复实验留位置，因此 96 个孔也通常不等于 96 个新序列。

## Q2：为什么这些新序列能够提高模型？

> 关键不在于“多 100 条”本身，而在于这 100 条是不是补在模型真正缺数据的地方。随机加数据可能重复已有分布；主动学习利用模型不确定性找到模型最不会的区域，再利用序列和物理多样性避免重复。我们不会只凭理论假设这一点成立，而会先把已有 2000 条标签隐藏一部分做回顾性模拟，直接比较主动学习和随机抽样在同样新增样本数量下谁的 held-out Spearman 提升更多。

## Q3：0.6 凭什么？

> 目前没有充分证据保证 0.6。0.6 应定义为 Stretch Target（挑战目标）。我们首先要求主动学习显著优于随机新增数据，然后看 learning curve（学习曲线）。如果从 2000 增加到 2100、2200、2300 的过程中 Spearman 仍持续稳定上升，同时 GBSA 重复测量表明标签噪声允许达到更高相关，那么 0.6 才有数据依据。如果学习曲线在 0.45 或 0.5 附近饱和，也应接受真实数据上限。

## Q4：为什么不直接照 Kelvin 论文用 UCB？

> 因为优化目标不同。Kelvin 的目标是尽快找到 NAND score 最高的序列，所以 UCB 合适。我们当前第一目标是提高 GBSA 代理模型整体预测准确性，因此采集更应该优先选择模型最不确定、最缺覆盖的区域。等模型足够准确以后，再逐渐转成 LCB寻找低 GBSA 候选。

## Q5：为什么还要保留随机样本？

> 少量随机样本是重要对照。如果所有序列都由算法选择，最后即使模型变好了，也难以证明是主动学习有效还是单纯“多了 100 个标签”就会提高。保留约 5%–10% random control（随机对照）可以估计主动学习相对随机采样的真实增益。

## Q6：为什么不用一个最准的 XGBoost，非要做 Bayesian（贝叶斯）？

> 如果只做一次预测，可以只选 Spearman 最高的模型。但主动学习还需要回答模型在哪里最没有把握。普通单个 XGBoost 主要输出点预测，不天然提供可靠的认知不确定性。因此下一阶段模型选择要同时看“预测准不准”和“不确定性可信不可信”。如果最后 XGB ensemble（XGB 集成）在预测和不确定性上都更好，我们也不会为了形式上做贝叶斯优化而强行使用神经网络。

## Q7：为什么不直接使用 Gaussian Process（高斯过程）？

> GP 是很好的概率基线，而且我们会保留。但我们的 RNA 是离散序列，同时还有结构/物理高维信息；Kelvin 论文已经证明 deep ensemble 在核糖开关闭环中可以有效工作。所以我们计划让 GP、deep ensemble 和现有树模型集成都参加统一 benchmark，最终根据预测精度、不确定性校准和主动学习效率选择。

## Q8：为什么要加入 NUPACK / 结构特征？论文只用序列也成功了。

> 一方面，我们自己的已有实验表明物理/结构特征中存在与 GBSA 有关的信息，只是大量高维特征直接堆叠并不稳定。另一方面，Kelvin 论文在讨论中明确提出未来可整合结构信息，而且其分析显示更结构化的 base-pair distance 与功能变化关系更强。对 GBSA 这种物理自由能目标，结构和热力学信息具有更直接的机制关联，所以我们计划先筛掉冗余，再以低维形式与序列融合。

## Q9：100 条为什么不全部选高 uncertainty？

> 因为高不确定序列可能高度集中在同一区域。模型可能对一组只差一个碱基的序列都很不确定，但测完其中几条以后，剩下的边际信息快速下降。所以我们计划使用 Kriging Believer 加 sequence/physics diversity，让整批 100 条互相补充，而不是简单做 uncertainty Top-100。

## Q10：实验团队返回什么数据最有利于模型？

> 除最终 GBSA 外，我们最希望保留每次重复原始值、实验批次、协议版本和 QC。模型最终能达到多高不仅取决于样本数量，也取决于标签噪声。如果可以估计同一序列重复测量方差，我们就能判断误差中多少来自模型没学好，多少来自实验本身不可约噪声。

---

# 十一、建议在会上展示的关键参考论文

## [1] Kelvin et al., 2026 ——最直接的 riboswitch（核糖开关）对标

Kelvin D, Kubaczka E, Karava M, Koeppl H, Suess B.  
**Iterative design of a NAND hybrid riboswitch by deep batch Bayesian optimization（利用深度批量贝叶斯优化迭代设计 NAND 混合核糖开关）**.  
*Nucleic Acids Research（核酸研究）*. 2026;54(5):gkag145.  
DOI: `10.1093/nar/gkag145`.

关键借鉴：RNA sequence encoder（RNA 序列编码器）、deep ensemble（深度集成）、uncertainty quantification（不确定性量化）、UCB（置信上界）、Kriging Believer（克里金信念批量策略）、DBTL（设计—构建—测试—学习）闭环以及真实多轮核糖开关 wet-lab validation（湿实验验证）。

开源仓库：`https://github.com/Self-Organizing-Systems-TU-Darmstadt/NANDRiboswitchDesignByBayesOpt`  
Zenodo：`10.5281/zenodo.15185168`

## [2] Groher et al., 2019 ——riboswitch + ML + biophysical features（核糖开关 + 机器学习 + 生物物理特征）

Groher A-C, Jager S, Schneider C, et al.  
**Tuning the Performance of Synthetic Riboswitches using Machine Learning（利用机器学习调控合成核糖开关性能）**.  
*ACS Synthetic Biology（ACS 合成生物学）*. 2019;8(1):34–44.  
DOI: `10.1021/acssynbio.8b00207`.

关键借鉴：机器学习预测 riboswitch switching behavior（核糖开关切换行为），并结合 biophysical parameters（生物物理参数）和序列模式。

## [3] Angenent-Mari et al., 2020 ——大规模 RNA sequence-to-function（序列到功能）

Angenent-Mari NM, Garruss AS, Soenksen LR, et al.  
**A deep learning approach to programmable RNA switches（可编程 RNA 开关的深度学习方法）**.  
*Nature Communications（自然·通讯）*. 2020;11:5057.  
DOI: `10.1038/s41467-020-18677-1`.

关键借鉴：91,534 条 toehold switches（趾端开关）数据；深度神经网络直接学习 RNA sequence-to-function（RNA 序列到功能）。

注意：其有标签数据远多于我们的约 2000 条，所以不能用它证明“2000 条一定适合训练大型 Transformer”。

## [4] Valeri et al., 2020 ——RNA sequence-to-function frameworks（RNA 序列到功能框架）

Valeri JA, Collins KM, Ramesh P, et al.  
**Sequence-to-function deep learning frameworks for engineered riboregulators（工程化 RNA 调控元件的序列到功能深度学习框架）**.  
*Nature Communications（自然·通讯）*. 2020;11:5058.  
DOI: `10.1038/s41467-020-18676-2`.

## [5] Yang et al., 2025 ——Active Learning（主动学习）真实实验闭环

Yang J, Lal RG, Bowden JC, et al.  
**Active learning-assisted directed evolution（主动学习辅助定向进化）**.  
*Nature Communications（自然·通讯）*. 2025;16:714.  
DOI: `10.1038/s41467-025-55987-8`.

关键借鉴：low-N batch setting（小样本批量场景）、uncertainty-guided selection（不确定性驱动选样）、多轮实验—模型迭代。

注意：对象是蛋白质，主要用于支持主动学习的通用实验设计原则。

## [6] Andersen et al., 2026 ——昂贵 MMGBSA + Bayesian Optimization（贝叶斯优化）

Andersen L, Rausch-Dupont M, Martínez Leon A, et al.  
**Accelerating ligand discovery by combining Bayesian optimization with MMGBSA-based binding affinity calculations（通过结合贝叶斯优化与基于 MMGBSA 的结合亲和力计算加速配体发现）**.  
*Digital Discovery（数字发现）*. 2026;5:3082–3095.  
DOI: `10.1039/D5DD00522A`.

关键借鉴：昂贵 MD/MMGBSA（分子动力学 / MMGBSA）作为 oracle（真实评价函数），代理模型选择少量候选进行昂贵计算，新标签返回后更新模型。

注意：对象是 protein–ligand（蛋白质—配体），不是核糖开关；主要用于支持“昂贵自由能计算 + 主动学习”范式。

## [7] Stanton et al., 2022 ——Biological Sequence Bayesian Optimization（生物序列贝叶斯优化）

Stanton S, Maddox W, Gruver N, et al.  
**Accelerating Bayesian Optimization for Biological Sequence Design with Denoising Autoencoders（利用去噪自编码器加速生物序列设计的贝叶斯优化）**.  
*ICML（国际机器学习大会）*. 2022;162:20459–20478.

关键借鉴：离散生物序列空间中的 Bayesian Optimization（贝叶斯优化）、latent representation（潜在表示）以及 explore–exploit tradeoff（探索—利用平衡）。

---

# 十二、参考文献在汇报中的使用建议

如果时间有限，正文只重点讲三篇：

1. **Kelvin et al., NAR 2026**：同对象——riboswitch；同方法——batch Bayesian optimization；有真实 wet-lab loop；有完整开源代码。
2. **Groher et al., ACS Synthetic Biology 2019**：直接说明 riboswitch 功能与 biophysical features（生物物理特征）结合机器学习的可行性。
3. **Andersen et al., Digital Discovery 2026**：直接说明昂贵 MMGBSA 可以进入 active-learning / BO（主动学习 / 贝叶斯优化）闭环。

其余放备份页或参考文献页。

---

# 十三、明天汇报可以直接使用的开场稿

> 我们当前希望解决的问题，不只是继续寻找一个更强的 GBSA 回归模型，而是如何最大化每一个新增昂贵标签的价值。  
> 目前我们已有约 2000 条带 GBSA 的核糖开关序列，并且已经进行了多类序列、结构和物理特征构建。下一步我们首先会通过 Spearman、互信息和冗余分析把特征空间收紧；然后分别建立 RNA 序列表示和物理特征表示，构建能够同时预测 GBSA 和输出不确定性的代理模型。  
> 我们最近找到了一项 2026 年发表在 Nucleic Acids Research 的直接相关工作。它已经在真实核糖开关体系中采用深度集成、Kriging Believer 和批量贝叶斯优化完成了三轮实验闭环。因此我们希望以它作为核心参考框架，但针对我们的 GBSA 任务做关键改造：第一，我们加入 NUPACK 等结构物理信息；第二，我们输出 GBSA 而不是 NAND 功能评分；第三，我们第一阶段以提升模型预测精度为目标，所以候选选择将更偏向主动学习和信息增益，而不是直接追求当前预测最优值；第四，我们希望把序列多样性升级为结构和物理多样性。  
> 在真正让实验团队投入大量资源之前，我们还会先用已有 2000 条真实 GBSA 做回顾性主动学习模拟，比较随机、UCB、不确定性、Kriging Believer 和我们拟采用的物理感知策略，并直接比较每轮 50、100 和 200 条的效率。这样我们希望让第一轮实验的规模和候选选择都有数据依据。

---

# 十四、建议的结束稿

> 今天希望和实验团队确定的不是一个最终固定算法，而是闭环接口。  
> 算法侧我们负责从现有数据中建立可靠的概率代理模型，并在每一轮给出一批具有明确选样理由的 RNA；实验侧最关键的是确定单轮容量、重复数、实验周期、质量控制和数据返回格式。  
> 如果这个闭环跑通，我们后续评估的不只是“有没有找到几个好序列”，还可以量化每新增 100 个 GBSA 标签到底带来了多少模型提升，以及主动学习是否显著优于随机采样。  
> 最终目标是把当前一次性的 GBSA 建模任务升级成一个可以持续自我改进的 RNA 设计平台。

---

# 十五、最终推荐方案摘要

## 计算主线

\[
\boxed{
\text{Feature Selection（特征选择）}
\rightarrow
\text{Sequence–Physics Representation（序列—物理表示）}
\rightarrow
\text{Probabilistic Surrogate（概率代理）}
\rightarrow
\text{KB Batch Active Learning（KB 批量主动学习）}
\rightarrow
\text{Expensive GBSA Oracle（昂贵 GBSA 评价）}
}
\]

## 特征选择

\[
\boxed{Spearman+MI+Stability+Redundancy\ Removal}
\]

## 序列表示

\[
\boxed{13-site\ One-hot\quad vs\quad FAD-specific\ Transformer\ Encoder}
\]

## 物理表示

\[
\boxed{Selected\ Blocks\rightarrow PCA\rightarrow20\sim40d}
\]

## 概率代理

主候选：Deep Ensemble（深度集成），\(M=20\sim50\)。  
同时保留 GP（高斯过程）和 RF/XGB Ensemble（树模型集成）作为 baseline（基线）。

## 第一阶段 Acquisition（采集）

\[
\boxed{Epistemic\ Uncertainty+Coverage+Physics-aware\ Diversity+Kriging\ Believer}
\]

## 第二阶段 Acquisition（采集）

代理模型足够准确后：

\[
\boxed{LCB+KB}
\]

用于寻找更优 GBSA 序列。

## 实验批次

不预设 96，优先比较：

\[
\boxed{50,100,200}
\]

通过：

\[
\boxed{Retrospective\ AL\ Simulation+Experimental\ Constraints}
\]

确定。

## 0.6 的表述

\[
\boxed{0.6=Stretch\ Target（挑战目标）}
\]

真正决策依据是：

\[
\boxed{Learning\ Curve+Label\ Noise\ Ceiling}
\]

---

# 十六、资料来源与开源实现

1. Kelvin D, Kubaczka E, Karava M, Koeppl H, Suess B. **Iterative design of a NAND hybrid riboswitch by deep batch Bayesian optimization（利用深度批量贝叶斯优化迭代设计 NAND 混合核糖开关）**. *Nucleic Acids Research（核酸研究）*. 2026;54(5):gkag145. DOI: `10.1093/nar/gkag145`.
2. Kelvin et al. 开源代码：`https://github.com/Self-Organizing-Systems-TU-Darmstadt/NANDRiboswitchDesignByBayesOpt`
3. Kelvin et al. Zenodo 归档：`https://doi.org/10.5281/zenodo.15185168`
4. Groher A-C, Jager S, Schneider C, et al. **Tuning the Performance of Synthetic Riboswitches using Machine Learning（利用机器学习调控合成核糖开关性能）**. *ACS Synthetic Biology（ACS 合成生物学）*. 2019;8(1):34–44. DOI: `10.1021/acssynbio.8b00207`.
5. Angenent-Mari NM, Garruss AS, Soenksen LR, et al. **A deep learning approach to programmable RNA switches（可编程 RNA 开关的深度学习方法）**. *Nature Communications（自然·通讯）*. 2020;11:5057. DOI: `10.1038/s41467-020-18677-1`.
6. Valeri JA, Collins KM, Ramesh P, et al. **Sequence-to-function deep learning frameworks for engineered riboregulators（工程化 RNA 调控元件的序列到功能深度学习框架）**. *Nature Communications（自然·通讯）*. 2020;11:5058. DOI: `10.1038/s41467-020-18676-2`.
7. Yang J, Lal RG, Bowden JC, et al. **Active learning-assisted directed evolution（主动学习辅助定向进化）**. *Nature Communications（自然·通讯）*. 2025;16:714. DOI: `10.1038/s41467-025-55987-8`.
8. Andersen L, Rausch-Dupont M, Martínez Leon A, et al. **Accelerating ligand discovery by combining Bayesian optimization with MMGBSA-based binding affinity calculations（通过结合贝叶斯优化与 MMGBSA 结合亲和力计算加速配体发现）**. *Digital Discovery（数字发现）*. 2026;5:3082–3095. DOI: `10.1039/D5DD00522A`.
9. Stanton S, Maddox W, Gruver N, et al. **Accelerating Bayesian Optimization for Biological Sequence Design with Denoising Autoencoders（利用去噪自编码器加速生物序列设计的贝叶斯优化）**. *ICML（国际机器学习大会）*. 2022;162:20459–20478.

---

# 十七、汇报前最后检查清单

在正式汇报前，建议把下列数字更新成当前最新版本：

- [ ] 当前 canonical GBSA 标签样本数；
- [ ] 当前正式 30-seed / 100-seed Spearman；
- [ ] 37 个 feature blocks（特征块）的最新完整名单；
- [ ] 每个 feature block 的维度；
- [ ] 当前候选池真实可用数量；
- [ ] 是否所有候选都有 NUPACK 特征；
- [ ] 实验 GBSA 的具体获得流程；
- [ ] 是否存在 replicate（重复）；
- [ ] 当前一次实验的最大吞吐量；
- [ ] 50 / 100 / 200 的实验成本和周期；
- [ ] 哪些序列约束必须在模型候选生成前过滤；
- [ ] 实验团队希望第一阶段优先提高模型精度，还是同时强调发现低 GBSA 序列。

---

> **文档版本说明**：本方案把 Kelvin et al. 2026 的真实核糖开关 deep batch Bayesian optimization（深度批量贝叶斯优化）作为核心直接对标，但没有把其方法机械照搬到 FAD。文中明确区分了“原论文已经证明的部分”和“FAD 需要通过回顾性模拟 / 新实验验证的改造部分”。特别是 Spearman≈0.6，目前只应视为挑战目标；最终可达性能必须由新增标签 learning curve（学习曲线）与 GBSA label noise（标签噪声）共同判断。
