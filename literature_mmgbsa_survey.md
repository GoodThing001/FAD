# 文献调研报告：MM-GBSA/MM-PBSA 精度、收敛性与去噪方法
（RNA 适配体-配体结合自由能 ML 代理建模项目）

**检索方法说明**：共执行 10 组 web_search 检索（中英文关键词，覆盖综述/精度、收敛性、多轨迹、igb/协议参数、RNA 体系、ML 代理、熵/3D-RISM/sGCMC 等主题）。ACS、Wiley、bioRxiv、ChemRxiv 的正文页均被 Cloudflare 拦截（HTTP 403），因此所有 DOI/作者/卷期页均改经 **Crossref REST API** 逐条核实；全文内容确认限于 PMC 开放论文（3 篇）。下文每条文献的 DOI 均已机器核实，个别条目注明核实方式。

---

## A. 论文清单（按主题分组）

### A1. 综述与精度评估

**1. Genheden S, Ryde U. "The MM/PBSA and MM/GBSA methods to estimate ligand-binding affinities." *Expert Opin Drug Discov*. 2015;10(5):449-461.**
DOI: [10.1517/17460441.2015.1032936](https://doi.org/10.1517/17460441.2015.1032936)；开放全文 [PMC4487606](https://pmc.ncbi.nlm.nih.gov/articles/PMC4487606/)（已读全文）。
核心发现（对噪声问题直接相关）：
- 仅用 ~20 帧时，单个配体的 ΔG_bind 帧间标准差高达 47–62 kJ/mol，均值标准误 SE = 11–14 kJ/mol——「使该方法在比较亲和力相近的配体时基本无用」（与项目 gbsa std=10.4 kcal/mol ≈ 44 kJ/mol 同一量级，见第 3 节 Table 1 原文）。
- 三平均（3A，独立模拟复合物/受体/配体）法的标准误比一平均（1A，仅模拟复合物）大 **4–5 倍**；实践中 1A 通常更准。
- 「多条短独立模拟比单条长模拟更有效」（并引 [38–41] 支持）。
- 熵项（normal mode）是不确定性最大的分量（SD1/SD7 列中 −TΔS 的 SD 达 46/57 kJ/mol），且大量已发表研究直接省略熵项而不损失排序。
- 溶质介电常数 ε 最优值随体系和溶剂模型在 1–25 之间变化，但**大样本研究中最优值通常为 ε=2–4**；高电荷结合位点需要更大 ε。
- Hou 等发现 MM/GBSA 结果随 MD 时长变化，但 **>4 ns 之后无增益**。
- 多种「改进」（QM 电荷、极化力场、3D-RISM 替代溶剂化）大多没有提升、甚至变差；3D-RISM 介于中间（r²=0.80–0.90，MAD 16–17 kJ/mol）。
- 最小化结构偶发不真实构象，需过滤（离群值来源之一）。

**2. Wang E, Sun H, Wang J, Wang Z, Liu H, Zhang JZH, Hou T. "End-Point Binding Free Energy Calculation with MM/PBSA and MM/GBSA: Strategies and Applications in Drug Design." *Chem Rev*. 2019;119(16):9478-9508.**
DOI: [10.1021/acs.chemrev.9b00055](https://doi.org/10.1021/acs.chemrev.9b00055)。
端点法策略与应用的标准综述（Hou 组）：力场/GB 模型/介电常数/熵近似/采样协议各环节的推荐与取舍。

**3. Sun H, Li Y, Tian S, Xu L, Hou T. "Assessing the performance of MM/PBSA and MM/GBSA methods. 4. Accuracies of MM/PBSA and MM/GBSA methodologies evaluated by various simulation protocols using PDBbind data set." *Phys Chem Chem Phys*. 2014;16(31):16719-16729.**
DOI: [10.1039/c4cp01388c](https://doi.org/10.1039/c4cp01388c)。
1800+ 个 PDBbind 晶体复合物上系统评估不同模拟协议（模拟时长/介电常数/GB 模型等）对 MM/GBSA 与 MM/PBSA 精度的影响——协议敏感性最直接的证据来源。

**4. Hou T, Wang J, Li Y, Wang W. "Assessing the performance of the MM/PBSA and MM/GBSA methods. 1. The accuracy of binding free energy calculations based on molecular dynamics simulations." *J Chem Inf Model*. 2011;51(1):69-82.**
DOI: [10.1021/ci100275a](https://doi.org/10.1021/ci100275a)。
（元数据经多篇论文引用列表交叉确认；出版页未直接访问。）Hou 系列第 1 篇，MD 采样时长与 GB 模型选择的基准。

**5. Wang C, Nguyen PH, Pham K, Huynh D, Le T-BN, Wang H, Ren P, Luo R. "Calculating Protein-Ligand Binding Affinities with MMPBSA: Method and Error Analysis." *J Comput Chem*. 2016;37(27):2436-2446.**
DOI: [10.1002/jcc.24467](https://doi.org/10.1002/jcc.24467)。
（元数据经引用列表交叉确认。）MMPBSA 协议的分项误差分析（方法与误差传播）。

**6. Aldeghi M, Bodkin MJ, Knapp S, Biggin PC. "Statistical Analysis on the Performance of Molecular Mechanics Poisson-Boltzmann Surface Area versus Absolute Binding Free Energy Calculations: Bromodomains as a Case Study." *J Chem Inf Model*. 2017;57(9):2203-2221.**
DOI: [10.1021/acs.jcim.7b00347](https://doi.org/10.1021/acs.jcim.7b00347)。
用 bootstrap 等统计手段对 MMPBSA 与绝对结合自由能（ABFE）做严谨比较：相关性、系统偏差与不确定性传播的模板（参考文献含 Efron bootstrap）。

### A2. 收敛性、模拟时长/帧数

**7. Genheden S, Ryde U. "How to obtain statistically converged MM/GBSA results." *J Comput Chem*. 2010;31(4):837-846.**
DOI: [10.1002/jcc.21366](https://doi.org/10.1002/jcc.21366)；PMID 19598265。摘要已核实（Crossref）。
核心发现（收敛判据的直接依据）：
- 单条 10 ns 长模拟**不够**——其标准误会低估真实不确定性（用抗生物素蛋白四聚体 4 个等价结合位点做内部检验，单条模拟得到的 4 个位点结果彼此不一致）。
- MM/GBSA 能量的自相关时间 **~5 ps**；需 ~100 ps 平衡。
- 建议每条独立模拟采样 **20–200 ps**；配合 **5–50 条独立模拟**才能达到 SE=1 kJ/mol（每配体共 800–8000 次能量计算、总时长 1.5–15 ns）——比常规做法高一个数量级。
- 结论：应跑多条短模拟并取平均，任意精度可通过增加独立模拟条数达到（SE ∝ 1/√n）。

**8. Xu X, Zhou F, Zheng L, Wang S, Peng X, Li D. "Sampling Challenges of MM/PBSA Binding Energy Calculations." *J Phys Chem B*. 2025;129(45):11666-11678.**
DOI: [10.1021/acs.jpcb.5c04908](https://doi.org/10.1021/acs.jpcb.5c04908)。
（2025 年新作，标题即主题；出版页被反爬拦截，仅核实元数据，未确认具体结论。检索片段显示其讨论长时间模拟与采样问题。）

**9. Wright DW, Hall BA, Kenway OA, Jha S, Coveney PV. "Computing Clinically Relevant Binding Free Energies of HIV-1 Protease Inhibitors." *J Chem Theory Comput*. 2014;10(3):1228-1241.**
DOI: [10.1021/ct4007037](https://doi.org/10.1021/ct4007037)。
（元数据经多条引用列表交叉确认。）多重复本 ensemble 端点评分（ESMACS 类协议）的代表作，验证多副本对排序可靠性的作用。

### A3. 多 replica 平均降低噪声、提升排序

**10. Adler M, Beroza P. "Improved Ligand Binding Energies Derived from Molecular Dynamics: Replicate Sampling Enhances the Search of Conformational Space." *J Chem Inf Model*. 2013;53(8):2065-2072.**
DOI: [10.1021/ci400285z](https://doi.org/10.1021/ci400285z)。
多条独立 replicate 采样（而非单条延长）提升 MM-GBSA 结合能排序质量的早期直接证据。

**11. Crean RM, Pudney CR, Cole DK, van der Kamp MW. "Reliable In Silico Ranking of Engineered Therapeutic TCR Binding Affinities with MMPB/GBSA." *J Chem Inf Model*. 2022;62(3):577-590.**
DOI: [10.1021/acs.jcim.1c00765](https://doi.org/10.1021/acs.jcim.1c00765)；预印本 [bioRxiv 10.1101/2021.06.21.449221](https://doi.org/10.1101/2021.06.21.449221)（摘要已核实）。
**与项目最接近的变体排序 benchmark**（TCR 变体 vs pHLA）：
- **5–10 条短（4 ns）MD replica 即可得到可重复且准确的变体排序**；
- 变体集按突变数区分协议：≤4 个突变 vs 多突变，**多突变变体集必须加熵修正**（项目 13 个突变位点属于多突变情形）；
- 可**提前识别潜在离群值（outlier）**；
- 用 bootstrap 重采样计算相关系数的置信区间。
（注：Bristol 大学机构库中同工作版本以 "How Many Replicas Are Required for Reproducible MMPB/GBSA Calculations" 为题存放；Crossref 精确标题查询未检索到独立于此的正式论文，故以 JCIM 2022 正式版为准。）

**12. Wan S, Knapp B, Wright DW, Deane CM, Coveney PV. "Rapid, Precise, and Reproducible Prediction of Peptide-MHC Binding Affinities from Molecular Dynamics That Correlate Well with Experiment." *J Chem Theory Comput*. 2015;11(7):3346-3356.**
DOI: [10.1021/acs.jctc.5b00179](https://doi.org/10.1021/acs.jctc.5b00179)（摘要已核实）。
pMHC 12 肽排序：1-轨迹 vs 3-轨迹协议（构型熵）；文中结论（检索片段+摘要）：「使用 ensemble 模拟时，**3-轨迹法优于 1-轨迹法**（平均 Pearson 相关更高）」，且结果精确、可重复。

### A4. igb 模型选择、介电常数、协议优化

**13. Nguyen H, Roe DR, Simmerling C. "Improved Generalized Born Solvent Model Parameters for Protein Simulations." *J Chem Theory Comput*. 2013;9(4):2020-2034.**
DOI: [10.1021/ct3010485](https://doi.org/10.1021/ct3010485)。
igb8 参数化原始文献（注意：针对蛋白参数化，对 RNA/核酸体系外推需谨慎）。

**14. Wang E, Fu W, Jiang D, Sun H, Wang J, Zhang X, Weng G, Liu H, Tao P, Hou T. "VAD-MM/GBSA: A Variable Atomic Dielectric MM/GBSA Model for Improved Accuracy in Protein-Ligand Binding Free Energy Calculations." *J Chem Inf Model*. 2021;61(6):2844-2856.**
DOI: [10.1021/acs.jcim.1c00091](https://doi.org/10.1021/acs.jcim.1c00091)。
变原子介电常数 MM/GBSA——介电常数协议优化的近期代表（残基特异性介电常数思想见 Genheden & Ryde 2015 综述 §6 引用 [68]）。

**15. Chen F, Sun H, Wang J, Zhu F, Liu H, Wang Z, Lei T, Li Y, Hou T. "Assessing the performance of MM/PBSA and MM/GBSA methods. 8. Predicting binding free energies and poses of protein-RNA complexes." *RNA*. 2018;24(9):1183-1194.**
DOI: [10.1261/rna.065896.118](https://doi.org/10.1261/rna.065896.118)；开放全文 [PMC6097651](https://pmc.ncbi.nlm.nih.gov/articles/PMC6097651/)（已读全文）。
**RNA 体系协议选择的直接证据**：
- 55 个蛋白-RNA 复合物亲和力预测：MM/GBSA **GBn1 模型 + ε_in=2 + 显式水最小化**最优（r_p=−0.557，r_s=−0.519）；优于所有 docking 打分函数。
- **MM/PBSA 对蛋白-RNA 体系明显更差（r_p≈−0.04~−0.50 依协议），且对构象采样更敏感，不推荐用于 RNA**。
- 蛋白-RNA 体系对 ε_in（1/2/4/8）**不敏感**（r_p 仅 −0.508~−0.524 波动，GBn2 除外），与蛋白体系行为不同。
- 1 ns MD 轨迹后 GBn1 + ε_in=8 最优（r_p=−0.578，r_s=−0.568），但收益相对最小化结构有限；重打分识别 near-native pose 用 GBn1 + ε_in=1（top-10 成功率 79.1%，148 个体系）。

**16. Sun H, et al. "Assessing the performance of MM/PBSA and MM/GBSA methods. 5. Improved Docking Performance Using High Solute Dielectric Constant MM/GBSA and MM/PBSA Rescoring." *Phys Chem Chem Phys*. 2014;16(41):22035-22045.**
DOI: [10.1039/C4CP03179B](https://doi.org/10.1039/C4CP03179B)。
（元数据经引用列表交叉确认。）高溶质介电常数重打分改善 docking 结果——介电常数协议的又一直接证据。

**盐浓度**：专门研究 MM-GBSA 盐浓度影响的系统性论文**未检索到**（见 C 节）。Genheden & Ryde 2015 指出：MM/GBSA 结果对设置中的任意选择（溶剂化、构象、质子化状态等）不敏感，除非受影响残基紧邻配体（[PMC4487606](https://pmc.ncbi.nlm.nih.gov/articles/PMC4487606/) §3）。

### A5. RNA-小分子应用案例

**17. Chen J, Wang X, Pang L, Zhang JZH, Zhu T. "Effect of mutations on binding of ligands to guanine riboswitch probed by free energy perturbation and molecular dynamics simulations." *Nucleic Acids Res*. 2019;47(13):6618-6631.**
DOI: [10.1093/nar/gkz499](https://doi.org/10.1093/nar/gkz499)；开放全文 [PMC6649850](https://pmc.ncbi.nlm.nih.gov/articles/PMC6649850/)（已读全文）。
**与项目最接近的 RNA 突变-配体亲和力案例**（鸟嘌呤核糖开关 WT + 5 种突变 × 2 配体）：
- FEP 与 MM-GBSA 的突变相对亲和力相关 **R²=0.83**——说明突变集内 MM-GBSA 相对排序有意义；
- 协议：**1 μs MD**，取最后 800 ns 共 **200 帧（间隔 4 ns）** 做 MM-GBSA（GB 模型为 Onufriev GB），nmode 熵仅用 50 帧（因代价高）；
- 由其 Table 2 反推：单帧 ΔG 分量标准差约 3–15 kcal/mol（SEM 0.2–1.0 × √200）——即使 1 μs 采样，帧级噪声仍然很大，依赖大量帧平均；
- 突变影响集中于关键核苷酸 U22/U51/C74 的相互作用。

**18. （蛋白-RNA 大基准）** Chen F et al. 2018 *RNA*（见 A4 #15）。

### A6. ML 代理 / MM-PBSA 噪声处理

**19. Al Masri C, Yu J. "Combining Physics-Based Protein-DNA Energetics with Machine Learning to Predict Interpretable Transcription Factor-DNA Binding." *J Chem Inf Model*. 2025.**
DOI: [10.1021/acs.jcim.5c01143](https://doi.org/10.1021/acs.jcim.5c01143)；PubMed 41131660。
用 gmx_MMPBSA 计算的结合自由能分量与 ML 结合预测蛋白-DNA 结合——「**MM-PBSA 能量作为 ML 标签/特征的物理先验**」范式的直接先例（作者与卷期待核对，期刊与 DOI 已确认）。

**20. Wei X-X, Chen Y, Yang Y, Xu M, Dral PO, Chen H. "Integrating Machine Learning Interatomic Potentials with MMPBSA for Accurate Protein-Ligand Binding Free Energy Calculations." *J Phys Chem B*. 2026;130(19):4982-4993.**
DOI: [10.1021/acs.jpcb.6c00306](https://doi.org/10.1021/acs.jpcb.6c00306)。
ML 原子间势 + MMPBSA 端点评分：加速后处理同时改善能量函数——「MMPBSA 管道 + ML」的最新组合。

**21. Wingbermühle S, Biswas AD, Bonanni D, et al. (含 Lindahl E). "Molecular Dynamics Workflows to Compute Large-Scale Sets of Absolute Binding Free Energies Aiding Drug Candidate and Binding Pose Selection." *J Chem Theory Comput*. 2026;22(10):5122-5142.**
DOI: [10.1021/acs.jctc.5c02127](https://doi.org/10.1021/acs.jctc.5c02127)；ChemRxiv 预印本 [10.26434/chemrxiv.15000165](https://doi.org/10.26434/chemrxiv.15000165)。
大规模（数千级）ABFE 工作流；文中（检索片段来源）指出「**MM/PBSA 的性能对体系（the investigated system）的敏感性高于 MM/GBSA**」——支持"GB 更稳、PB 更敏感"的判断。

### A7. 熵近似与替代/改进方法

**22. Duan L, Liu X, Zhang JZH. "Interaction Entropy: A New Paradigm for Highly Efficient and Reliable Computation of Protein-Ligand Binding Free Energy." *J Am Chem Soc*. 2016;138(17):5722-5728.**
DOI: [10.1021/jacs.6b02682](https://doi.org/10.1021/jacs.6b02682)。
Interaction Entropy（IE）法：从轨迹直接算熵、免 nmode，是 MM-GBSA 熵项的常用替代。

**23. Ekberg V, Ryde U. "On the Use of Interaction Entropy and Related Methods to Estimate Binding Entropies." *J Chem Theory Comput*. 2021;17(8):5379-5391.**
DOI: [10.1021/acs.jctc.1c00374](https://doi.org/10.1021/acs.jctc.1c00374)。
对 IE 类方法的批判性检验（Ryde 组）——采用 IE 前必须读的对照文献。

**24. Genheden S, Luchko T, Gusarov S, Kovalenko A, Ryde U. "An MM/3D-RISM Approach for Ligand Binding Affinities." *J Phys Chem B*. 2010;114(25):8505-8516.**
DOI: [10.1021/jp101461s](https://doi.org/10.1021/jp101461s)。
MM/3D-RISM 结合自由能经典文献；综述评价其精度介于 PB/GB 与更昂贵方法之间（r²=0.80–0.90，MAD 16–17 kJ/mol，见 A1 #1）。

**25. Bian H, Shao X, Chipot C, Cai W, Fu H. "A Formally Exact Method for High-Throughput Absolute Binding Free-Energy Calculations." *Nat Comput Sci*. 2025;5(8):621-626.**
DOI: [10.1038/s43588-025-00821-w](https://doi.org/10.1038/s43588-025-00821-w)；HAL 存档 [hal-05412328](https://hal.science/hal-05412328v1)（摘要已核实）。
高通量 ABFE：热力学循环设计减小扰动 + double-wide 采样 + 氢质量重分配，较传统 double-decoupling 提速 4–8 倍；45 个复合物中 34 个验证体系平均无符号误差 <1 kcal/mol、滞后 <0.5 kcal/mol。**注意：这是 alchemical ABFE 加速方法，并非 sGCMC**（sGCMC 专项文献未检索到，见 C 节）。

---

## B. 可借鉴方法（面向本项目的 gbsa 去噪与协议优化）

1. **多 replica 协议（最直接、证据最强）**
   - 5–10 条 4 ns 短 replica + 平均，即可获得可重复的变体排序；短多副本优于单条长模拟（[Crean 2022](https://doi.org/10.1021/acs.jcim.1c00765)；[Adler & Beroza 2013](https://doi.org/10.1021/ci400285z)）。
   - 每条独立模拟 20–200 ps 生产 + ≥100 ps 平衡；能量自相关时间 ~5 ps，帧间隔应 ≥ 数倍自相关时间（[Genheden & Ryde 2010](https://doi.org/10.1002/jcc.21366)）。
   - 文献明确支持：**单条长模拟会低估不确定性、产生看似收敛实则未收敛的结果**（[Genheden & Ryde 2010](https://doi.org/10.1002/jcc.21366)）——若项目 2000 条变体每条只有单轨迹，则「单轨迹 MM-GBSA 未收敛」的怀疑有直接文献依据。
   - 达到 SE=1 kJ/mol 需 5–50 条独立模拟/配体（[Genheden & Ryde 2010](https://doi.org/10.1002/jcc.21366)）；SE 随 √n 下降，可用此预算所需 replica 数。
   - 3-轨迹协议在 ensemble 模拟时排序优于 1-轨迹（[Wan 2015](https://doi.org/10.1021/acs.jctc.5b00179)）；但 3A 法 SE 比 1A 大 4–5 倍（[Genheden & Ryde 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4487606/)），需权衡。

2. **分块收敛判据与不确定性量化**
   - Block averaging + 自相关时间估计作为收敛判据（[Genheden & Ryde 2010](https://doi.org/10.1002/jcc.21366)）。
   - Bootstrap 重采样计算 Spearman/Pearson 相关及其置信区间，而不是只看点估计（[Crean 2022](https://doi.org/10.1021/acs.jcim.1c00765)；[Aldeghi 2017](https://doi.org/10.1021/acs.jcim.7b00347)）。
   - 内部一致性检验：等价位点/等价构象应给出相同结果（抗生物素蛋白四聚体检验，[Genheden & Ryde 2010](https://doi.org/10.1002/jcc.21366)）。

3. **离群值处理**
   - 文献层面：Crean 2022 证明可「提前识别潜在离群值」；Genheden & Ryde 2015 指出最小化会偶发产生不真实构象、需过滤（[PMC4487606](https://pmc.ncbi.nlm.nih.gov/articles/PMC4487606/) §2）。项目 25 个 max=+109 的离群样本可先按帧级诊断（最小化是否崩溃/GB 是否出现奇点）分类，而非盲目全局截断。
   - **注意：未检索到任何「对 MM-GBSA 分数做截断均值/Winsorize/中位数去噪」的专门文献**（见 C 节）。若采用此类稳健统计，应作为本项目自创的标签后处理，并以保留集的排序相关性提升作为唯一判据，引用上只能标注"无直接文献支持"。

4. **熵项协议**
   - 排序任务可省略 nmode 熵（大样本测试中省略不降序、文献广泛省略）（[Genheden & Ryde 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4487606/) §7）。
   - 但多变体集（本项目 13 突变位点）**熵修正重要**（[Crean 2022](https://doi.org/10.1021/acs.jcim.1c00765)）；如需廉价熵可用 Interaction Entropy（[Duan 2016](https://doi.org/10.1021/jacs.6b02682)），但须先了解其局限（[Ekberg & Ryde 2021](https://doi.org/10.1021/acs.jctc.1c00374)）。
   - nmode 只算子集帧（如 50 帧）即可（[Chen 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6649850/)）。

5. **igb 模型与介电常数（RNA 体系）**
   - RNA 体系亲和力：**GBn1 + ε_in=2 + 显式水最小化**是文献最优组合；**不推荐 MM/PBSA 用于 RNA**（对构象采样更敏感、结果更差）（[Chen 2018 RNA](https://pmc.ncbi.nlm.nih.gov/articles/PMC6097651/)）。
   - RNA 体系对 ε_in 在 1–8 内相对不敏感（[Chen 2018 RNA](https://pmc.ncbi.nlm.nih.gov/articles/PMC6097651/)）；蛋白体系大样本 ε=2–4 最优（[Genheden & Ryde 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4487606/)）。
   - igb8 为蛋白参数化（[Nguyen 2013](https://doi.org/10.1021/ct3010485)），用于核酸需谨慎；变介电模型 VAD-MM/GBSA 可作替代尝试（[Wang 2021](https://doi.org/10.1021/acs.jcim.1c00091)）。
   - **重要限定**：项目标签是 Rosetta 的 `r_psp_MMGBSA_dG_Bind`，与 AMBER 系 MM-GBSA 实现不同；上述 igb/ε 结论迁移前需在 Rosetta 打分框架内重验。

6. **标签层级的替代/补充方案**
   - 子集 ABFE 校准：高通量 ABFE 已可做到平均误差 <1 kcal/mol、4–8 倍加速（[Bian 2025](https://doi.org/10.1038/s43588-025-00821-w)）——可对 2000 条中的子集（如 50–100 条）算 ABFE 作为低噪声参考标签，用于验证 gbsa 去噪效果。
   - ML 势加速 MM-PBSA 后处理（[Wei 2026](https://doi.org/10.1021/acs.jpcb.6c00306)）；MM-PBSA 能量分量作为 ML 特征的范式（[Al Masri & Yu 2025](https://doi.org/10.1021/acs.jcim.5c01143)）。
   - 若重建标签：RNA 突变集内相对排序用 1 μs MD + 200 帧（4 ns 间隔）+ 熵子集帧的做法有先例（[Chen 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6649850/)）；模拟时长超过 ~4 ns 的排序增益有限（[Genheden & Ryde 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4487606/) 引 Hou 等）。

---

## C. 检索无结果 / 证据不足（如实说明）

1. **sGCMC**：未检索到以 semi-grand canonical Monte Carlo 为核心、可引用的结合自由能文献。找到的最接近"高通量严格 ABFE"是 Bian et al. 2025（Nat Comput Sci，alchemical 方法）。sGCMC 方向目前无可用引用。
2. **MM-GBSA 专用去噪/截断均值/稳健统计文献**：未找到。最接近的是 Aldeghi 2017（bootstrap 统计比较）与 Crean 2022（离群值提前识别），但均非"对标签做截断均值"的方法论文。
3. **「用 ML 直接回归 MM-GBSA 分数并建模其噪声」的代理工作**：未找到完全同款工作。最接近：Al Masri & Yu 2025（MM-PBSA 能量+ML 预测结合）、Wei 2026（MLIP+MMPBSA）、Wingbermühle 2026（大规模 ABFE 工作流）。本项目的"ML 代理复现 gbsa 标签"这一设定在文献中暂无直接先例。
4. **141 nt 规模适配体的 MM-GBSA 应用**：未找到。RNA 案例最大规模为核糖开关（~70–80 nt，Chen 2019）与蛋白-RNA 复合物基准（Chen 2018 RNA）。大 RNA（>100 nt）的 MM-GBSA 收敛行为无直接文献证据。
5. **盐浓度专门研究**：未找到系统性论文；仅有 Genheden & Ryde 2015 关于设置不敏感性的一般性结论。
6. **Rosetta MM-GBSA（r_psp_MMGBSA_dG_Bind）的精度验证文献**：未检索到直接评估；相关检索只返回 Rosetta RNA 结构预测与药物发现案例。
7. **访问受限**：ACS/Wiley/bioRxiv/ChemRxiv 正文均被 Cloudflare 拦截（403）。ChemRxiv 7327019（"influence of different analysis choices on ensemble MMPBSA"）因此未能确认，**不列入引用**。所有 DOI 元数据经 Crossref API 核实；全文级确认仅限 3 篇 PMC 开放论文（Genheden 2015、Chen 2019 NAR、Chen 2018 RNA）。条目 #4、#5、#9、#16 的元数据来自多篇论文引用列表的交叉确认，未直接打开出版页。
8. **中文检索**（MM-GBSA 收敛性/噪声/核糖开关等关键词）：未发现独立于英文文献的中文新来源。
9. **Bristol 机构库标题疑义**：题为 "How Many Replicas Are Required for Reproducible MMPB/GBSA Calculations" 的 PDF（第一作者 Crean）经 Crossref 精确标题查询，未找到独立正式论文，判定为 Crean 2022 JCIM 工作的版本之一；引用以 JCIM 2022 正式版为准（见 A3 #11 注）。
