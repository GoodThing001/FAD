# 昂贵且噪声标签下的主动学习与贝叶斯优化：文献调研报告

**项目背景**：RNA 适配体 GBSA 代理建模；2000 条序列、标签为单轨迹 MM-GBSA（噪声大，可预测性 Spearman 上限 ~0.41）；已复现 gkag145（NAR 2026，NAND 核糖开关深度批量 BO：预训练 transformer + 深度集成 + UCB + Kriging Believer），结论为三种代理下主动学习 ≈ 随机采样；项目判断根因为标签噪声主导、特征已充分覆盖。本报告围绕该负面结论寻找机制解释与可能出路。

**检索方法说明**：共 10 组 web_search（约 30 个查询）；关键条目用 arXiv / ACL Anthology / ICML / NeurIPS 官方页面以及 Crossref、EuropePMC 元数据 API 做核验（nature.com 与 PubMed 直连被拦截、bioRxiv 触发限流，故改用 API）。标记 ⚠ 的条目为「检索命中但未核读全文」。

---

## A. 分主题论文清单

### A1. 分子/生物序列设计中的批量贝叶斯优化（2019–2025）

1. **Kriging is well-suited to parallelize optimization** — Ginsbourger, Le Riche, Carraro；2010；Springer《Computational Intelligence in Expensive Optimization Problems》pp. 131–162；[arXiv:1003.1119](https://arxiv.org/abs/1003.1119)。
   核心发现：提出 q-EI 与 **Kriging Believer（常数骗子）** 伪观测策略，使批量查询可并行评估；KB 用「乐观幻觉」近似 EI，是 gkag145 所采用批量策略的源头，也是后续文献批评/随机化修正的对象。

2. **Batch Bayesian Optimization via Local Penalization** — González, Dai, Hennig, Lawrence；2016；AISTATS，PMLR 51:648–657；<https://proceedings.mlr.press/v51/gonzalez16a.html>。
   核心发现：用局部惩罚函数压制已选点邻域、将批量采集化为顺序 greedy，无需伪观测；对光滑 GP 有收敛保证。

3. **Parallelised Bayesian Optimisation via Thompson Sampling** — Kandasamy, Krishnamurthy, Schneider, Póczos；2018；AISTATS，PMLR 84:133–142；<https://proceedings.mlr.press/v84/kandasamy18a.html>。
   核心发现：并行 TS 批量（从后验采样多个函数、每个最大化一个采样函数）天然带批内多样性，有遗憾界、可异步扩展。

4. **GIBBON: General-purpose Information-Based Bayesian OptimisatioN** — Moss, Leslie, González, Rayson；2021；JMLR 22(235):1–49；[arXiv:2102.03324](https://arxiv.org/abs/2102.03324)。
   核心发现：批量互信息采集显式奖励批内多样性，普遍优于批量 UCB/q-EI 类方法。

5. **Population-Based Black-Box Optimization for Biological Sequence Design (P3BO)** — Angermueller, Belanger, Gane, Mariet, Dohan, Murphy, Colwell, Sculley；2020；ICML，PMLR 119；[arXiv:2006.03227](https://arxiv.org/abs/2006.03227)（已核验摘要）。
   核心发现：**单优化方法在任务间表现波动极大**（"performance of existing methods varies drastically across optimization tasks"）；P3BO 用「方法集成 + 在线超参进化」对冲脆弱性，比任何单方法更稳、批内更多样。该文也是 Design-Bench 基准的出处。

6. **Accelerating Bayesian Optimization for Biological Sequence Design with Denoising Autoencoders (LaMBO)** — Stanton, Maddox, Gruver, Maffettone, Delaney, Greenside, Wilson；2022；ICML，PMLR 162（stanton22a）；<https://proceedings.mlr.press/v162/stanton22a.html>（已核验）。
   核心发现：去噪自编码器 + 多任务 GP 头，在隐空间做多目标采集；无需大预训练语料，是生物序列批量 BO 的代表框架。

7. **Randomized Kriging Believer for Parallel Bayesian Optimization with Regret Bounds** — Sugiura, Takeuchi；arXiv 预印本（期刊未确认 ⚠）；[Semantic Scholar 记录](https://www.semanticscholar.org/paper/a85673b5ac394dea292679dfc1e1a28805fdc78a)。
   核心发现：给 KB 的伪观测注入随机化以获得遗憾界——说明确定性 KB 的「乐观幻觉」本身是需要修正的对象（噪声标签下该问题更严重）。

8. **Protein Design with Guided Discrete Diffusion** — Gruver et al.；2023；NeurIPS；<https://proceedings.neurips.cc/paper_files/paper/2023/file/29591f355702c3f4436991335784b503-Paper-Conference.pdf>。
   核心发现：代理引导的离散扩散作为批量序列设计器，与批量 BO 形成互补的生成式路线。

### A2. 深度集成/BNN 不确定性用于分子科学主动学习

1. **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles** — Lakshminarayanan, Pritzel, Blundell；2017；NeurIPS；[arXiv:1612.01474](https://arxiv.org/abs/1612.01474)。
   核心发现：深度集成 + 对抗训练给出校准良好的不确定性；是小数据下比单 BNN 更实用的 UQ 基线（gkag145 即采用深度集成）。

2. **Evaluating Scalable Uncertainty Estimation Methods for Deep Learning-Based Molecular Property Prediction** — Scalia, Grambow, Pernici, Li, Green；2020；JCIM 60(6):2697–2717；DOI [10.1021/acs.jcim.9b00975](https://doi.org/10.1021/acs.jcim.9b00975)。
   核心发现：系统比较分子性质预测的 UQ 方法（MC-dropout、集成、深度核学习等），校准质量参差不齐——UQ 误标定会直接污染 AL 采集。

3. **Evidential Deep Learning for Guided Molecular Property Prediction and Discovery** — Soleimany et al.；2021；ACS Cent. Sci. 7(8):1356–1367；DOI [10.1021/acscentsci.1c00546](https://doi.org/10.1021/acscentsci.1c00546)。
   核心发现：证据深度学习显式分离数据（aleatoric）与模型（epistemic）不确定性，在噪声数据上校准优于贝叶斯方法——天然适合噪声标签代理。

4. **Leveraging Uncertainty in Machine Learning Accelerates Biological Discovery and Design** — Hie, Bryson, Berger；2020；Cell Systems 11(5):461–477；DOI [10.1016/j.cels.2020.09.007](https://doi.org/10.1016/j.cels.2020.09.007)（DOI 经 crossmark 检索确认）。
   核心发现：明确区分可约（epistemic）与不可约（aleatoric）不确定性；**模型引导设计只有在可约不确定性显著时才有效**——为「噪声主导 ⇒ AL 无增益」提供概念框架。

5. **Deep Batch Active Learning for Drug Discovery** — Bailey, Moayedpour, Li, Corrochano-Navarro, Kötter, Kogler-Anele, Riahi, Grebner, Hessler, Matter, Bianciotto, Mas, Bar-Joseph, Jager；2023；bioRxiv DOI [10.1101/2023.07.26.550653](https://doi.org/10.1101/2023.07.26.550653)（作者与摘要经 EuropePMC 核验；eLife reviewed preprint 文章号 89679 ⚠）。
   核心发现：深度集成 + Greedy k-center 批量 AL，在多个公共数据集与自建时间序实验数据上优于现有批量选择方法。注意：其 oracle 多为低噪声模拟或高通量测定，与单轨迹 MM-GBSA 的高噪声环境不同，正面结论不可直接外推。

### A3. 噪声标签/实验噪声下的主动学习与 noisy BO

1. **Get Another Label? Improving Data Quality and Data Mining Using Multiple, Noisy Labelers** — Sheng, Provost, Ipeirotis；2008；KDD，pp. 614–622；DOI [10.1145/1401890.1401965](https://dl.acm.org/doi/10.1145/1401890.1401965)。
   核心发现：标签噪声显著时，**重复标注的投资回报常高于获取新样本**，且存在最优复测分配——为「把预算投给 replica 而非新序列」提供直接依据。

2. **Active Learning with Neural Networks: Insights from Nonparametric Statistics** — Zhu, Nowak；2022；NeurIPS；<https://proceedings.neurips.cc/paper_files/paper/2022/hash/01025a4e79355bb37a10ba39605944b5-Abstract-Conference.html>（已核验）。
   核心发现：深度 AL 首个近最优标注复杂度理论，但**依赖 low-noise（Tsybakov）条件**；无该条件时需引入 abstention 才能维持优势——即高标签噪声下不确定性 AL 失去理论保障。

3. **Activized Learning: Transforming Passive to Active with Improved Label Complexity** — Hanneke；2012；JMLR 13:587–625；<https://jmlr.csail.mit.edu/papers/volume13/hanneke12a/hanneke12a.pdf>。
   核心发现：AL 理论奠基工作之一，标签复杂度保障依赖「标签噪声随距决策边界距离减小」等噪声条件——噪声与 AL 收益的关系在理论层面早已被形式化。

4. **Mind Your Outliers! Investigating the Negative Impact of Outliers on Active Learning for Visual Question Answering** — Karamcheti, Krishna, Fei-Fei, Manning；2021；ACL-IJCNLP；[arXiv:2107.02331](https://arxiv.org/abs/2107.02331)。
   核心发现：不确定性 AL **系统性偏爱离群/错标样本**，在噪声数据上损害性能；建议平衡「不确定性 + 代表性」的混合采集。

5. **Quality-controlled active learning via Gaussian processes for robust structure–property learning in autonomous microscopy** — Chowdhury, Narasimha, Yang, Funakubo, Ehara, Liu, Vasudevan；2026；npj Computational Materials，文章号 2248；DOI [10.1038/s41524-026-02248-x](https://doi.org/10.1038/s41524-026-02248-x)（经 Crossref 核验，开放获取）。
   核心发现：**标准 AL 在噪声数据上会「主动优先低质量测量」**；用物理信息 QC 门控（SHO 拟合质量）在采集阶段排除低质量数据后，优于随机采样、标准 AL 与多任务学习，并在真实显微镜实验中闭环验证。

6. **Batched Bayesian Optimization for Drug Design in Noisy Environments** — Bellamy, Rehim, Orhobor, King；2022；JCIM 62(17):3970–3981；DOI [10.1021/acs.jcim.2c00602](https://doi.org/10.1021/acs.jcim.2c00602)，开放获取 [PMC9472273](https://europepmc.org/articles/PMC9472273)（经 EuropePMC 核验）。
   核心发现：在噪声 assay 下系统比较批量 BO 技术，并**提出 retest（复测）策略缓解噪声**；批量 BO 在大量噪声下仍有效，但需要复测机制才能以相同实验数找到更多活性化合物。

7. **A Tutorial on Bayesian Optimization** — Frazier；2018；[arXiv:1807.02811](https://arxiv.org/abs/1807.02811)。
   核心发现：noisy BO 的标准处理（GP 噪声核与噪声方差建模、noise-aware EI 变体）——测量噪声建模的基线方法。

### A4. 重复测量预算分配 + 主动学习

1. **Batch Bayesian Optimization for Replicable Experimental Design (BTS-RED)** — Dai, Nguyen, Tay, Urano, Leong, Low, Jaillet；2023；NeurIPS；<https://mlanthology.org/neurips/2023/dai2023neurips-batch/>，[OpenReview](https://openreview.net/forum?id=Xj4LJiXvlX)（已核验）。
   核心发现：固定总预算下「更多不同条件 vs 每条件更多复测」的权衡；**噪声方差大的输入自动获得更多复测**，在异方差观测噪声下有理论保证且渐进 no-regret；另含风险厌恶变体（Mean-Var-BTS-RED）。验证场景为精准农业与 AutoML——分子/湿实验应用仍是空白，可直接迁移。

2. **Re-Active Learning: Active Learning with Relabeling** — Lin, Mausam, Weld；2016；AAAI；<https://ojs.aaai.org/index.php/AAAI/article/view/10315>。
   核心发现：把「重标注」作为显式动作并入 AL 决策空间，与「获取新样本」共同优化——对噪声标签 AL 的直接改进路线。

3. **An autonomous laboratory for the accelerated synthesis of novel materials (ARES)** — Szymanski et al.；2023；Nature 624:86–91；DOI [10.1038/s41586-023-06734-w](https://doi.org/10.1038/s41586-023-06734-w)。
   核心发现：自主实验室闭环实验规划中显式处理复测与实验噪声——湿实验噪声下闭环实践的代表。

4. （交叉引用）Bellamy 2022 的 retest policy 同样属于本主题：以复测作为采集动作的一部分。

### A5. 小数据分子性质预测中「AL vs 随机采样」的负面结果/失效条件

1. **Practical Obstacles to Deploying Active Learning** — Lowell, Lipton, Wallace；2019；EMNLP-IJCNLP，pp. 21–30；DOI [10.18653/v1/D19-1003](https://aclanthology.org/D19-1003/)（已核验）。
   核心发现：**AL 的收益在不同模型/任务间不可靠**；小标注预算下收益更差；主动获取的数据集与引导模型耦合，训练「后继模型」时并不持续优于 i.i.d. 采样。与「三种代理下 AL≈随机」同构，是最接近的机制性负面证据（NLP 域）。

2. **Angermueller et al. 2020（见 A1.5）**：单方法跨任务高度脆弱 → 单一采集策略无稳健优势，本身就是 AL 不可靠的证据。

3. **Zhu & Nowak 2022（见 A3.2）**：理论失效条件——标签噪声高（low-noise 假设不成立）时不确定性 AL 无保障。**这是「噪声主导 ⇒ AL≈随机」最直接的理论背书。**

4. **Karamcheti et al. 2021（见 A3.4）**：机制——AL 追逐离群点/错标样本，噪声下反而有害。

5. **Chowdhury et al. 2026（见 A3.5）**：机制——AL 主动优先劣质测量，标准采集在噪声数据上不如随机采样（该文明确报告了这一对比）。

6. **Finding Drug Candidate Hits With a Hundred Samples: Ultra-low Data Screening With Active Learning** — ChemRxiv 2025 预印本 ⚠；<https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-jlml5>。百样本级 AL 筛选（正面结果，未核读；其 oracle 设定需自行核对）。

7. **gkag145（NAR 2026, 54(5)）** — 项目自身复现的负面结论（三种代理下 AL≈随机）；<https://academic.oup.com/nar/article/54/5/gkag145>。本次检索未发现 RNA 领域的其他同类负面报道——该负结果本身具有发表价值。

### A6. 核糖开关/RNA 设计的实验闭环（DBTL）

1. **Tuning the Performance of Synthetic Riboswitches using Machine Learning** — Groher, Suess；2019；ACS Synth. Biol. 8(1):34–44；DOI [10.1021/acssynbio.8b00207](https://doi.org/10.1021/acssynbio.8b00207)。
   核心发现：用 ML 模型指导核糖开关性能调优的早期工作（ML+核糖开关，非批量 BO 闭环）。

2. **A deep learning approach to programmable RNA switches** — Angenent-Mari, Garruss, Soenksen, Church, Collins；2020；Nat. Commun. 11:5057；DOI [10.1038/s41467-020-18677-1](https://doi.org/10.1038/s41467-020-18677-1)。
   核心发现：深度学习大规模设计 toehold 开关并实验验证，序列→功能学习指导 RNA 元件设计。

3. **Sequence-to-function deep learning frameworks for engineered riboregulators** — Valeri et al.；2020；Nat. Commun. 11:5058；DOI [10.1038/s41467-020-18676-2](https://doi.org/10.1038/s41467-020-18676-2)。
   核心发现：伴生工作，序列→功能深度模型指导核糖调节器设计。

4. **Machine learning guided aptamer refinement and discovery** — Bashir, Yang, Wang, Hoyer, Chou, McLean, Davis, Gong, Armstrong, Jang, Kang, Pawlosky, Scott, Dahl, Berndl, Dimon, Ferguson；2021；Nat. Commun. 12:2366；DOI [10.1038/s41467-021-22555-9](https://doi.org/10.1038/s41467-021-22555-9)，开放获取 [PMC8062585](https://europepmc.org/articles/PMC8062585)（经 EuropePMC 核验）。
   核心发现：粒子展示 + ML 亲和力预测的闭环；模型预测高亲和 DNA 适配体的命中率比随机扰动高 11 倍，并生成截短 70% 的更高亲和适配体。DNA 适配体 ML 闭环代表作（其标签来自分区/高通量测量，同样含实验噪声）。

5. **InstructNA（标题以检索命中为准）** — 2026；Nature Computational Science ⚠；DOI [10.1038/s43588-026-00965-3](https://doi.org/10.1038/s43588-026-00965-3)（据 PDF 链接推断，未核读）。LLM 驱动的蛋白靶适配体序列生成，属于 ML 指导适配体设计的最新路线。

6. **gkag145（NAR 2026）** — 项目基线本身（不再展开）。

### A7. 采集函数鲁棒性：代理误标定（miscalibrated UQ）的修正

1. **Bayesian Optimization with Conformal Prediction Sets** — Stanton, Maddox, Wilson；2023；AISTATS，PMLR 206:959–986；[arXiv:2210.12496](https://arxiv.org/abs/2210.12496)（已核验）。
   核心发现：模型误标定与协变量漂移下，用共形预测给采集函数加**覆盖保证**；在多数任务上不牺牲样本效率即可显著提升查询有效性——代理误标定时直接可用的修正。

2. **Accurate Uncertainties for Deep Learning Using Calibrated Regression** — Kuleshov, Fenner, Ermon；2018；ICML，PMLR 80:2796–2804；[arXiv:1807.00263](https://arxiv.org/abs/1807.00263)。
   核心发现：回归的 recalibration 方法，修正系统性高估/低估的方差——深度代理方差误标定的直接修法。

3. **On Calibration of Modern Neural Networks** — Guo, Pleiss, Sun, Weinberger；2017；ICML；[arXiv:1706.04599](https://arxiv.org/abs/1706.04599)。
   核心发现：现代 DNN 普遍误标定，温度缩放等后校准有效（分类设定；与回归校准同源的思想）。

4. **Combining Bayesian and Evidential Uncertainty Quantification for Improved Bioactivity Modeling** — 2025；JCIM 65(24):13057 ⚠（检索命中，链接为镜像：[检索命中链接](https://pubs.acs.org/doi/10.1021/acs.jcim.5c05855) 未经核验，谨慎引用）。贝叶斯 + 证据 UQ 组合改善生物活性建模决策——A2.3 与 A7.2 路线的近期结合。

---

## B. 可借鉴方法：噪声标签下 AL 是否可行、如何改进

### B1. 文献如何解释「AL ≈ 随机」（为项目结论提供机制背书）

| 机制 | 出处 | 与本项目的对应 |
|---|---|---|
| AL 理论保障依赖 low-noise（Tsybakov）条件；噪声高时不确定性采集失去优势，需 abstention | Zhu & Nowak 2022（NeurIPS）；Hanneke 2012（JMLR） | 单轨迹 MM-GBSA 噪声主导 → 理论条件不成立 |
| 只有可约（epistemic）不确定性才可被 AL 削减；不可约（aleatoric）噪声主导时模型引导设计无效 | Hie et al. 2020（Cell Systems） | Spearman 上限 0.41 ≈ 不可约噪声上限 |
| AL 系统性偏爱离群点/错标样本，噪声下反害 | Karamcheti et al. 2021（ACL） | 采集函数在噪声排序里选点 |
| 标准 AL 会「主动优先劣质测量」，不如随机采样 | Chowdhury et al. 2026（npj Comput. Mater.） | 该文是明确的 AL<随机 的实证 |
| AL 收益跨模型/任务不一致、小预算更差、数据集与引导模型耦合 | Lowell et al. 2019（EMNLP） | 三种代理换着试都≈随机 |
| 单优化方法跨任务高度脆弱 | Angermueller et al. 2020（ICML） | 方法选择本身不可靠 |

**综合判断（检索结果支持）：** 文献中「噪声标签主导 + 特征充分覆盖 ⇒ 不确定性 AL 无法优于随机」是一条被多次独立验证的失效模式（理论：Zhu & Nowak / Hanneke；实证：Lowell / Karamcheti / Chowdhury）。项目的负面结论不是实现缺陷，而是该失效模式的一个 RNA 领域实例。

### B2. 文献给出的出路（按可操作性排序）

1. **复测优先于新样本（最直接的出路）。** 当标签噪声主导时，预算应投向 replica 而非新序列：
   - Sheng et al. 2008（KDD）：重复标注的投资回报高于新样本，且有最优复测分配；
   - Bellamy et al. 2022（JCIM）：retest 策略在噪声 assay 下以相同实验数找到更多活性化合物；
   - Dai et al. 2023（NeurIPS BTS-RED）：自适应复测（噪声方差大者多复测）有 no-regret 保证。
   - 对项目：复测直接压缩标签噪声 → 提高有效 Spearman 上限（0.41 之上）；AL 只能削减 epistemic 部分，而复测削减 aleatoric 部分（Hie 2020 的框架）。

2. **把「复测/重标注」并入采集动作空间。** Lin et al. 2016（Re-Active Learning）、Bellamy 2022（retest policy）：采集决策 = {测新序列，复测旧序列}，由代理的逐样本噪声估计驱动。

3. **先量化噪声再决定 AL 是否值得做。** 用 ~10–20 条序列 × 3–5 replicas 的小型试点估计逐样本/全局噪声方差（BTS-RED 的 known/unknown noise 设定；Bellamy 2022 的 retest 框架）。若 aleatoric 占主导 → 报告 AL 不可行（文献支持）；若试点显示可约不确定性显著 → 再启用 AL。

4. **QC 门控采集。** Chowdhury et al. 2026（npj Comput. Mater.）：在采集阶段用物理/统计质量过滤（如 SHM/SHO 拟合质量、复测一致性）排除劣质测量，再让 AL 选点。对项目可类比为：用轨迹稳定性指标（如 MM-GBSA 收敛性、重复轨迹偏差）过滤后再采集。

5. **修正误标定的不确定性。** Kuleshov et al. 2018（recalibration，回归）；Stanton et al. 2023（conformal BO，覆盖保证）；Guo et al. 2017（校准思想）。若坚持 UCB/KB 类采集，先校准代理方差；共形 BO 可在代理误标定时不牺牲样本效率地修正查询方向。

6. **批量策略避开乐观伪观测。** KB 的伪观测在噪声标签下会放大乐观幻觉（Sugiura & Takeuchi 的随机化 KB 即为此提出；带遗憾界）。噪声场景优先：并行 TS（Kandasamy 2018）、local penalization（González 2016）、GIBBON（Moss 2021，批内多样性），或至少随机化 KB。

7. **集成对冲 + 混合采集。** P3BO（Angermueller 2020）：方法集成按近期表现加权采样，对冲单一采集策略的脆弱性；深度集成 UQ（Lakshminarayanan 2017；Bailey 2023 的 ensemble + Greedy k-center）提供批内多样性。Karamcheti 2021 建议「不确定性 + 代表性」混合采集。

### B3. 对本项目的落地建议（综合，非文献原文）

- **论文叙事**：把「AL≈随机」写成有理论支撑的负面结论（B1 表），引用 Zhu & Nowak / Hie / Lowell / Karamcheti / Chowdhury 作为机制证据；这比「我们调参失败」强得多。
- **实验一（推荐优先做）**：复测分配实验——固定总 GBSA 预算，比较 {全部新序列} vs {部分复测}（BTS-RED / Bellamy / Sheng 路线），预期复测能提升排序质量上限。
- **实验二**：QC 门控 AL（Chowdhury 路线）+ 校准/共形采集（Kuleshov / Stanton 路线），作为「修正后 AL」的对照。
- **实验三**：若仍要批量 BO，替换 KB 为并行 TS 或 GIBBON（噪声友好），并报告随机化 KB（Sugiura & Takeuchi）作为基线。

---

## C. 检索局限与证据不足（如实说明）

1. **分子领域「AL 与随机无差异」的直接负面研究稀缺。** 找到的负面证据主要来自 NLP（Lowell 2019）、理论（Zhu & Nowak 2022、Hanneke 2012）与显微镜闭环（Chowdhury 2026）；分子药物设计文献多为正面结果（Bailey 2023、Bellamy 2022），且其 oracle 噪声通常远低于单轨迹 MM-GBSA。未见「Spearman 上界 ~0.4 量级噪声下 AL 失效」的直接定量研究。
2. **单轨迹 MM-GBSA 的特定噪声规模的文献不存在**（预期之内；未检索到任何 MM-GBSA/GBSA 标签噪声 + AL 的研究）。
3. **RNA 适配体/核糖开关的噪声感知闭环工作除 gkag145 外基本空白。** Groher 2019、Angenent-Mari 2020、Valeri 2020 是 ML 指导设计而非批量 BO 闭环，且无噪声下 AL 失效分析；Bashir 2021 为 DNA 适配体。InstructNA（2026）未核读全文 ⚠。
4. **Replica-aware BO 的分子/湿实验应用仍是空白。** BTS-RED（Dai 2023）的实证场景为精准农业与 AutoML；Szymanski 2023（ARES）实践了复测但未给出可复用的算法框架。项目若做「GBSA 复测预算分配」有填补空间。
5. **个别条目未完全核验**：Sugiura & Takeuchi「Randomized Kriging Believer」仅见预印本记录，期刊/年份未确认 ⚠；Bailey 2023 的 eLife reviewed preprint 文章号 89679 来自检索命中路径，仅核验了 bioRxiv DOI；Hie 2020 的 DOI（10.1016/j.cels.2020.09.007）经 crossmark 页面间接确认；JCIM 2025「Combining Bayesian and Evidential UQ」及 ChemRxiv 2025 超低数据 AL 为预印本/镜像链接，未核读全文 ⚠。
6. **若干经典条目未逐篇 fetch 摘要**（Ginsbourger 2010、González 2016、Kandasamy 2018、GIBBON、deep ensembles、Scalia 2020、Soleimany 2021、Frazier 2018、Szymanski 2023、Angenent-Mari 2020、Valeri 2020、Guo 2017），均以 DOI/arXiv/官方 proceedings 链接给出，属领域标准引用。
7. 访问限制说明：nature.com 文章页重定向被拦截、PubMed 触发 reCAPTCHA、bioRxiv 限流 429；相应条目的元数据改用 Crossref / EuropePMC API 核验（Chowdhury 2026、Bashir 2021、Bellamy 2022、Bailey 2023 已通过此途径完整核验）。
