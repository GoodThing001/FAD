# Change log

## 2026-10-06（历史配方与修正主动学习补实验）

- 通过主运行器完成历史固定42重放、30开发与100确认的五配方比较，以及树集成四种采集策略的30开发和100确认，四个正式包均DONE；655份历史测试预测、3120份逐轮holdout预测和2600个发现轮次独立复算通过。
- 0.50602865已准确重现，保留固定划分上的优化成果；1087维相对303维在100确认中的排序差−0.03500，52维到303维的增益+0.03567，不能将全部特征工程一概归为偶然。旧v8发布参照协议单列。
- 100确认中树集成低预测采集未确认模型排序改善，但有限池查询regret差−2.90193、前10%召回差+0.27758；原论文深度集成/Kriging-Believer整链没有重评。没有新增真实GBSA或自动安排旧96条测量。
- 新增详细冻结协议、独立预测审计、可导出补实验数据图、总汇报动态证据章节。默认环境8项与Torch环境2项新增合同测试通过。
- NUPACK异常处理及官方对照准备、受控预训练脚本和配置已完善，但上传服务器被自动审批拒绝，待目的地授权；没有声称这两类实验完成或本轮服务器同步。

### 后续服务器授权与结构重算

- 用户明确授权现有服务器连接及上传后，已完成2000条NUPACK修正重算和30种子四臂特征比较。官方utility借用设计约束求解器，拒绝不兼容WT目标，首个运行FAILED并保留；v2核对全部4条兼容WT目标及前20条的两个诊断目标，共44个对照通过。
- 正确缺陷、WT目标概率及二者联合的排序增量区间均跨零；不恢复旧错误公式结果的有效性，不晋级主线输入。
- 预训练串行任务改为四个不重叠种子组，科学参数不变，串行部分包保留FAILED不并入统计；正式聚合必须四组DONE、哈希一致、30种子120行无缺漏。
- 四组预训练均DONE，30种子聚合与120份预测复算通过；三臂排序均值0.11618/0.13954/0.11927，经典对照0.35967，域内晋级条件未达，不追加本配方100种子。历史外域语料哈希保留，原论文完整语料来源未重新认证。
- 结构筛选频率发现4/30及0/30后，追加明确标为事后诊断的强制保留四臂30种子；新特征30/30实际入模，排序增量区间仍跨零，基线逐条一致。新增120份预测及筛选清单核验。
- 本轮总计七个完整模型比较包、4135份预测文件和2600个发现轮次独立复算；另2000条结构重算及44个官方对照完成。没有新增真实GBSA或恢复历史96条安排。

## 2026-10-05（三个新策略实现及七算法统一比较）

- 已实现ILS（迭代局部搜索）、SA（模拟退火）和无交叉AdaLead改编版，与原四法形成七种同起点策略。新冻结比较：30开发种子6001–6030，两代理、1024/4096上限、1260子运行；100确认种子8001–8100，1024上限、1400子运行。七者共同实际预测次数为主表，预设两两共同次数为补充。139项本地测试及逐条档案审计通过；无新真实GBSA、MD、对接或服务器同步。
- 新增独立操作配置、七算法冻结方案与完整生工结果汇报；总汇报§4.15、阶段图及索引更新。原四算法的旧证据不改写。确认启动前因开发重训种子重叠，审计调整确认编号7001–7100为8001–8100；算法/参数/预算/比较对未改。
- 新方法精确恢复、负分阈值、退火概率、扰动机制与未来暂停分组交接均测试；独立审计重算主/两两共同次数最低预测。结果仍限固定代理景观，真实GBSA算法优胜未判定。

## 2026-10-05（算法文献、评分器、选择流程与阶段进度补充）

- 生工总汇报增加§4.13四法原理与当前版本、§4.14评分器合同/当前代理/替换步骤，以及§6.5–6.10选择关口、未来真实GBSA定案规则、文献扩展、新算法实验流程、分阶段任务与进度。
- 核查AdaLead含重组，拟议无交叉版明确为改写；加入ILS、SA、算法组合、BO/LaMBO、CbAS和RL的条件、优先级与原始参考，期刊/会议/预印本等级区分。OpenReview正文受限、Holland仅核对出版信息均如实标注。
- 新增第九张框架图（阶段步骤与条件分支）的Mermaid图源、PNG/SVG；更新文档索引和实践指南阅读入口。此轮只更新说明与图，不新增算法运行或真实标签，不修改原冻结比较包。

## 2026-10-05（无交叉 GA、两代理四算法冻结比较与阶段 B 收尾）

- 实现仅变异遗传算法，单父本、锦标赛选择、精英保留、局部一/两位点变异、全局重启；没有交叉。注册到独立搜索、比较与未来外循环；人工标签验证钉协议→混合成功/失败→重训→DONE。
- 搜索接口0.1.1：训练标签/骨架/可变位点与源码严格恢复校验，已拟合模型身份含权重摘要；旧静态池忽略不生效搜索参数；拒绝独立续跑时不覆盖原模型权重。爬山正常代数上限与随机尾部邻居截断已补齐。评分器缓存固定训练侧OOD几何，计入实际批次预测调用时间。
- 新固定RF诊断适配器及高效套件，每种子每代理只训练一次；30开发×两代理×三预算，含重训稳定性与3项固定GA消融，共1140子运行；100独立确认×两代理×四法，共800子运行。主指标按同起点/同拟合/共同实际预测次数比较，完整性与独立逐产物审计通过。
- GA相对随机游走的预测最低值优势确认，但未显示稳定优于爬山；代理变化后的名单仍不稳定。保持爬山简洁基线、GA种群备选、束多样性诊断、随机控制；真实效果等待新序列GBSA，不无限扩大代理参数网格。
- 既有对接残差加入独立100种子确认及固定25/50/75%预筛补充诊断；不增加实际对接/MD/GBSA。详细数字与信息/成本边界以本轮总汇报和证据为准。
- 生工总汇报统一术语、数据与科学对象说明、当前四算法、文献/JCR类别与分区、八张框架图及四张数据图、完成清单与操作指南。代理优化本阶段暂停；历史96条不安排计算或回填，主动学习未选择。
- 本地完整128项测试通过，语法编译通过；最终链接与数值产物检查记录见本轮证据。本轮没有服务器同步/双端验证声明。先前10月5日下条记录是本次扩展前的历史复核，不是当前完成状态。

## 2026-10-05（生工总汇报扩写与直接父本证据复核）

- 扩写 `docs/汇报/9月/GBSA优化项目总汇报_生工团队版_20261005.md` §3.3/§4：对接残差的训练/测试边界与 30 种子数据、候选阶段信息及成本门槛；模型无关搜索合同、爬山/束搜索真实控制流程、同起点实际预测数比较、待实现 GA 的双亲合同及未来条件接口。附八张可查看框架图和可编辑 Mermaid 图源，增补原论文及指定类别的 2024 JCR 分区。
- 修正 `search_policies.py` 中爬山及束搜索的 `parent_evidence`：按直接父本是否为已测起点判断，避免第二代以后沿用祖先的 `measured` 标记。新增三种同起点策略的多代测试；该字段不参与预测、选择和预算计数，历史指标无需因此重算。旧运行档案不改写；旧父本证据字段仍须按直接父本核对。源码指纹改变后新搜索应另建档案。
- 本地针对本改动验证：搜索合同 28、搜索闭环集成 8、搜索比较合同 7，共 43 项通过。没有进行服务器复验，也没有启动真实 GBSA、对接或历史 96 条回填。报告同时注明未接入正常代数上限的爬山控制、预算尾部邻居截断、软多样性及批次时间字段等实现限制，未把这些建议写成已完成修复。

## 2026-10-02（σ 校准结果：uncertainty 采集门槛关闭）

- **30-seed 研究完成**（服务器 `srv_sigma_calibration_30seed_20261002_172402`，DONE，60 行；证据与判读 `evidence/sigma_calibration_30seed/`）：
  - n_train=1600：σ 与 |误差| 仅**弱相关**（Spearman 0.028 [0.010, 0.045]，十分位 7.23→7.75）；n_train=400（闭环尺度）：**不显著**（0.014 [−0.014, 0.040]，与打乱标签对照 0.016 同量级）；
  - **LCB(μ−βσ) 无任何 β 增益**：top-5%/10% 召回 Δ 全部 CI 跨 0 或为负；β=2 在 1600 档 top-10% 显著**有害**（−0.046 [−0.064, −0.028]）；
  - **放行决定**：`gate_sigma_tracks_error`（400 档 False）/`gate_lcb_beats_greedy`（全 False）→ uncertainty 采集保持关闭；`sigma` 继续只作记录字段；批量 BO/EI/Thompson 门槛未过。结合阶段 B（跨模型 Spearman 0.18–0.30），当前瓶颈是代理细粒度排序噪声而非采集函数 → 下一步优先代理/标注改善。

## 2026-10-02（σ 校准研究 + 阶段 C 预注册模板）

- **`scripts/reproduction/eval_sigma_calibration.py`**（方案 §5 门槛的放行检验）：30 seeds（301–330）× 两档 n_train（1600/400），XGB 自举集成（5 成员，折内 screen-100）在独立测试折上检验 `sigma` 是否分层 |误差|（Spearman + 十分位表）、LCB（μ−βσ，β=0.5/1/2）是否优于纯 greedy 的 top-5%/10% 召回（配对 Δ+CI）、打乱标签负对照；`summary.json` 输出放行门槛（`gate_sigma_tracks_error` / `gate_lcb_beats_greedy`）。`tests/test_sigma_calibration_contract.py` 3 项（召回/十分位/微型端到端）；本地 2-seed 冒烟 DONE（门槛 false——n=2 不构成结论，等 30-seed）。
- **30-seed 研究已投递**：服务器 `srv_sigma_calibration_30seed_20261002_172402`（nice 19/2 线程）。uncertainty 采集（LCB/EI/Thompson/`uncertainty_mixed`）在门槛通过前保持关闭。
- **阶段 C 预注册模板**：`docs/项目记录/阶段C预注册模板_真实GBSA决策性比较_20261002.md`（冻结项/主终点与判据/盲法/复测失败/停止门槛/签发前检查单；带 `<待定>` 占位，测量前必须填写冻结；96 条既有 PARKED 批次不受影响）。

## 2026-10-02（阶段 B 结果：爬山≈束、排序不稳定被量化）

- **30-seed 同预算比较完成**（服务器 `srv_search_compare_30seed_20261002_024621`，120/120 子运行、DONE；证据与判读 `evidence/search_compare_30seed/`，含 `runs.csv/summary.json/consistency.csv/run_manifest.json`）：
  - 主诊断：爬山/束 best_pred 6.02/6.07 vs 随机 10.85（配对 Δ≈−4.8，CI 上界 <0，**30/30 胜**）；爬山 30/30 local_optima 停机（4143 预测）、束 27/30 stagnation（4769）、随机满预算 8192；引导式 top-50 OOD≈0（随机 0.055）、更靠近训练集（3.83–4.04 vs 4.74）、多样性收窄（3.9–4.2 vs 8.3）；
  - **爬山 vs 束：无差异**（配对 Δbest_pred +0.047，CI [−0.038, +0.156]，中位数 0，20/30 完全平局）→ 按方案 §5 门槛，阶段 C 预注册候选默认建议多起点局部爬山（最简可审计），束为次要配置；GA/BO 暂无依据；
  - **不稳定风险量化**：top-50 跨模型 Spearman 0.18–0.30（随机基线亦仅 0.27–0.30）；换代理种子重跑 top-50 重叠（arm B Jaccard）随机 0.242 / 爬山 0.029 / 束 0.037——引导式搜索的最低预测候选名单对代理重训高度敏感，最优点附近主要由噪声驱动（方案 R7 类风险）；
  - 结论限代理表计算诊断；下一步按门槛优先改善代理与标注，真实 GBSA 阶段 C 须预注册终点与对照份额后再签批次。

## 2026-10-02（阶段 B 驱动：三策略同预算配对比较 + 30-seed 投递）

- **`scripts/loop/search_compare.py`**（方案 §5 阶段 B）：每 (seed, policy) 一个独立搜索子运行，同划分（300 init）/同代理快照/同硬唯一预测预算；汇总 runs.csv + 配对诊断（vs `stratified_random`：best_pred_gbsa、top-50 多样性、top-50 OOD 比例、top-50 训练邻近，各带 bootstrap 95% CI 与胜平负）+ 停机分布 + 跨模型排序稳定性（top-50 由两个重训代理重打分 → Spearman）+ 重训一致性（arm B，`--consistency-seeds` 个 seed 换代理种子重跑 → top-50 Jaccard）+ `completeness.json`（按 (seed, policy, arm) 逐行验收，缺行即非零退出）；子运行已有 metrics.csv 即复用（改代码换新 out-dir）。
- `scripts/loop/search_run.py` 新增 `--model-seed`（与 `--seed` 分离：同一划分、不同代理种子；search_id/metrics 记录 model_seed）。
- `tests/test_search_compare_contract.py` 6 项（配对 Δ/bootstrap CI/胜平负/Jaccard/完整性缺行检测/model-seed 与划分分离）；全套测试 **107/107 通过**；本地 3-seed 冒烟 DONE（9/9 行）：爬山/束 best_pred 相对随机 Δ 均为负、爬山多样性 4.1 < 束 8.2、束更靠近训练集（Δhamming −3.57）、跨模型 Spearman 0.20–0.55、arm-B Jaccard 随机 0.43/爬山 0.0/束 0.19（2 seeds，不构成结论）。
- **30-seed 开发比较已投递**：服务器 `srv_search_compare_30seed_20261002_024621`（seeds 201–230 × 3 策略 × 8192 预算 + 10 seed arm B，约 120 子运行，nice 19 / 2 线程，预计数小时）。**结论边界**：全部为代理表计算诊断，best_pred_gbsa 是代理输出，不构成发现效果证据；阶段 C 需真实 GBSA。

## 2026-10-02（搜索层接入闭环：步骤 5 + 服务器 8192 开发吞吐）

- **loop.py 接入（方案 §6 步骤 5）**：新增 `--candidate-source search`（仅前瞻式；回溯式在参数校验处直接报错——搜索候选在标准表之外，TableOracle 无标签可揭示）。每轮流程：当前已标注集 → 拟合代理 → `_rebuild_search_pool` 用搜索层重建候选池（`round_NNN_search/` 档案，恢复时按已标注集摘要复用已完成档案或确定性重跑；搜索 seed = loop seed + round；排除集 = 完整标准表 ∪ 已标注/已选/待回填/失败 ∪ 历史批次）→ 既有 `select_batch`/Oracle/`PARKED`/回填/重训流程不变。搜索超参全部落盘并进入恢复指纹（CONFIG_KEYS + fingerprint），配置变更续跑被拒；台账 `source=search_pool`，metrics 新增 `search_stop_reason/search_n_unique_predicted/search_best_pred_gbsa/search_pool_size` 列。新增 `configs/experiments/loop_search_prospective_park.json` 与 `tests/test_search_loop_integration.py` 5 项（前瞻式限定/停车流与档案/续跑幂等/配置变更拒续/两轮回填重训与新池）；全套测试 **101/101 通过**，本地 `loop_search_park_review` 冒烟 PARKED(rc 3)。
- **服务器 8192 开发吞吐测试**：`runs/srv_search_dev_beam_8192_20261002_014350` = DONE（nice 19 / 2 线程，约 108 s；`xgb_boot5_xgboost` 后端）：5598 个唯一预测（stagnation 停机，未耗尽预算）、best_pred 5.838、2 组；吞吐 ≈52 预测/秒（该规模全预算约 2.5–3 分钟）。小体积证据回收至 `evidence/loop_20261002_search/`（metrics/run_manifest/DONE/summary/snapshot/generations/groups/files.sha256）。**仅为计算过程观察，不构成发现效果证据**。
- `tools/launch_loop_jobs.sh` 新增 `search_8192` 投递模式；`scripts/README.md` 更新闭环与搜索层条目。

## 2026-10-02（迭代突变搜索层 v0.1.0：第一版实现）

- 依据 [突变序列迭代搜索算法与代理接口实施方案_20261002.md](docs/项目记录/突变序列迭代搜索算法与代理接口实施方案_20261002.md) 实现独立于静态选样的**多代突变搜索层**（内循环只花计算预算，不查询 Oracle、不写真实标签；外循环 `loop.py` 与 96 条 PARKED 批次未改动）：
  - `scripts/loop/search.py`：`SEARCH_INTERFACE_VERSION=0.1.0`——`SearchContext/ScoredCandidate/CandidateGroup/SearchResult` 与 `SearchPolicy/SequenceScorer` 协议；`SurrogateScorer` 适配器（批量顺序/有限值校验、模型快照缓存隔离、硬唯一预测预算、训练侧 OOD/最近距离标记，与 `loop.py` 的 OOD 实现逐值等价并由测试锁定）；
  - `scripts/loop/search_archive.py`：`search_snapshot.json`（数据/模型/约束/策略指纹，恢复强制匹配）、`search_state.json`（每代检查点：父代、RNG 状态、预算计数）、`predictions.jsonl`/`search_evaluated.jsonl`（全部评分含被淘汰子代）、成品四件套 + `files.sha256`；崩溃窗口按已提交行对账（cache→log→state 写序）；
  - `scripts/loop/search_policies.py`：`stratified_random`（分层随机基线，水位填充耗尽层自动再分配）、`multi_start_hill_climb`（共享种子集的一跳爬山，局部最优重启，预算内截断提案）、`multi_start_beam`（默认推荐：束宽/子代数/80-20 一跳两跳混合/全局重启配额/显式可关闭的多样性规则/停滞与预算停机；一跳配额按 RNG 采样避免位点截断偏置）；三种策略同上下文/同代理/同硬预算，边界恢复与不中断运行**逐位一致**（除墙钟字段）；
  - `scripts/loop/search_run.py` 入口 + `configs/experiments/search_{smoke_random,smoke_hillclimb,smoke_beam,dev_beam_8192}.json`（256 冒烟 × 3 策略 + 8192 开发预算；所有默认参数落盘，不依赖隐式默认）；
  - `tests/test_search_contract.py` 26 项：最小化方向、批量顺序、13 位点约束、去重、39 邻居上限、硬预算、同 seed 可复现、模型缓存隔离、束/爬山边界恢复逐位一致、快照不匹配拒续、完成后幂等、NaN/错长/耗尽/无种子清晰失败、1–2 组输出规则；全套测试 **96/96 通过**；
  - 本地 256 级三策略冒烟（真实代理，`sklearn_gbm` 回退后端已记录在 `model_id`）：random `best_pred=13.78`（budget_exhausted）、hill_climb `8.21`（budget_exhausted）、beam `9.22`（stagnation，5 代，2 组，top10 OOD=0）——仅为计算过程观察，**不构成发现效果证据**；8192 开发预算配置留待服务器按负载投递；
  - GA 与批量 BO 不在第一版范围（方案 §1）；loop.py 集成（步骤 5）为下一阶段工作。

## 2026-10-01（复核勘误与回填闭环补强）

- **PARKED 验收不被误伤（用户复核发现）**：`finish_run` 此前对所有运行要求 `status=retrained`，正常停机（`awaiting_labels` + rc 3）被同时记录 `acceptance_ok=false`/`terminal_state_ok=false`。已改为**按标记施加**：仅当准备写 `DONE`（rc 0 且未 parked）时才要求检查点 `retrained`；`PARKED` 运行如实记录 `checkpoint_status=awaiting_labels` 且 `acceptance_ok=true`。新增 `test_parked_acceptance_not_poisoned_by_awaiting_state`。
- **清单一致性验收补全（用户复核发现）**：`verify_loop_run.py` 的 Oracle 副本检查此前只比 `protocol_id`，现同时比 `approved_protocols`（规范化比对）与内嵌指纹的协议/批准列表；钉定测试中直接用 `tools/verify_loop_run.py` 子进程验收钉定后的运行包。
- **框架文档过时结论修正**：`docs/论文改进/候选闭环框架_20260930.md` 的钉定指引改为 `--pin-protocol` 审计流程；v2 重跑/100-seed 状态更新为已完成（含 50 胜/37 平/13 负与主判据）。测试增至 70 项（本地）/59 项（服务器闭环契约）。

- **尾部统计勘误（用户复核发现）**：初版 `summary.json` 把“最差 10% seed”标反——`worst10pct` 取的是配对差值**最负**的 10 个（greedy 改善最大），被当成了“表现最差”。已修 `strategy_compare.py`（`tail_stats`：worst10pct = 最正、best10pct = 最负，另加胜/平/负与 best10 占均值差比例），并用 `tools/recompute_compare_summary.py` 从 `runs.csv` 重算 100-seed 包（`runs.csv` 与服务器运行目录未改动；主判据与其余指标逐位一致）。修正后的 greedy 尾部：**50 胜/37 平/13 负**；最差 10% seed（相对 random 最差）均值 **+1.7627**（seeds 1062,1069,1055,1056,1078,1085,1081,1095,1065,1072）；改善最大 10 个 −8.4125（占平均差 ≈44.8%）。`completeness.json.observed_runs` 语义修正为子运行数（400），不再是逐轮行数。
- **协议钉定审计流程（用户复核发现）**：96 条批次的协议是占位符、批准列表为空——直接回填会接受任意非空协议串，临时改 `--approved-protocols` 又会被恢复指纹拒绝。新增 `run_experiment.py --resume-run … --pin-protocol <协议串>`：写 `checkpoints/protocol_changes.jsonl` 审计行，同步重写 config/指纹/台账/批次清单（含 `protocol_pinned_to`），之后受控回填；**禁止手改旧运行配置文件**。空批准列表回填时打印警告。**二次复核补强（用户复核发现）**：钉定现在同时更新检查点清单与 Oracle 副本 `pending_batches/batches/batch_<id>/batch_manifest.json`，并改写两份清单内嵌指纹的协议字段；拒绝空白协议串；仅允许对仍 `PARKED` 且尚无任何回填（labels/failures 均空）的运行钉定；`verify_loop_run.py` 新增两份清单与内嵌指纹的一致性检查。
- **回填→重训闭环补全（用户合成测试发现）**：全批回填后程序退 0 但状态仍 `labels_ingested`、台账仍 `awaiting_labels`、无重训指标。已修：`_absorb_returned_labels` 把吸收的 pass 行更新为 `labeled` 并填 `measured_gbsa`；恢复运行在批次闭合且轮次用尽时执行 `_final_retrain`（重训 + 记录 `retrained` 行与 holdout 分数）。`verify_loop_run.py` 对 DONE 运行新增核验：labeled 行都有 `measured_gbsa`、`state.status=retrained`。新增端到端测试：全批 pass、混合 pass-fail、协议钉定（错协议拒收/对协议接受）。**二次复核补强（用户复核发现）**：`--no-post-retrain-eval` 只跳过 holdout 打分、不再跳过末次重训；运行器 `finish_run` 对存在 `checkpoints/state.json` 的运行要求 `status=retrained` 才写 `DONE`。
- **分层口径收窄**：`round_NNN_ood_strata.csv` 的 MAE 只覆盖被选中且揭示标签的候选、“层内最优”指预测值最优——文档（汇报/预注册/策略/方案）相应改写，不再暗示整个候选池误差平坦或真实命中。
- 测试增至 **65 项全过**（闭环契约 54 项）；41 个 Markdown 链接 0 断链；`tools/recompute_compare_summary.py` 新增。

## 2026-10-01（100-seed 发布候选确认：结果）

- **预注册的 100-seed 确认完成**：服务器 `srv_strategy_compare_100seed_20260930_201309` = `DONE`，`completeness.json` ok=true（400 子运行 = 100 seeds × 4 策略，2000 行 = × 5 轮；此前的“1200 子运行/6000 行”为笔误，已修正）。seeds 1001–1100 与开发集不重叠，口径与预注册文档 §1–§4 完全一致。
- **主判据达成**：greedy 相对 random 的第 5 轮配对平均 Δregret = **−1.8781，95% CI [−2.5189, −1.2614]**（上界 <0）。**性质必须如实**：配对中位数 −0.0173、Q25/Q75 = −3.45/0.00、分布 **50 胜/37 平/13 负**；平均优势由改善最大的 10 个 seed（−8.41，占均值差 ≈44.8%）驱动，最差 10 个 seed 平均 +1.76（**初版把二者标反，见上方“复核勘误”条目**）——greedy 确认的作用是少数种子上大幅避免差批次，而非多数 seed 严格领先；与 30-seed 模式逐点一致。
- **附加指标（预注册）**：Δtop-10% 池召回 **+0.2153 [0.1983, 0.2323]，99/100 胜**（最强信号）；Δtop-5 中位数 **−0.9424 [−1.2144, −0.6833]，70/100 胜**；Δholdout Spearman −0.0043 [−0.0150, +0.0066]，46/100 —— **30-seed 开发集的 +0.031 未在本 seed 范围复现**，如实记录为未复现。
- **次要策略（探索）**：greedy_diverse Δregret −0.92 [−1.65, −0.19]（45 胜/24 平/31 负）、maxmin_coverage Δregret −0.89 [−1.58, −0.22]（42 胜/26 平/32 负）；两者 top-5 改善不显著。默认策略结论不变：**greedy**。
- **分层压力检验**（`ood_strata_aggregate.csv`，44 行 = 11 层 × 4 策略）：逐层 MAE 平坦（7–9，无强烈误差分层）；OOD 集中在稀疏层（n=0/4 为 0.60–1.00，n≥5 ≈0–0.07）；greedy 选中候选更靠近训练集（最小 Hamming ≈4.3–4.9 vs 覆盖类 ≈5.1–5.4），且在稠密层（7–12）几乎总选中层内预测最优（0.82–1.00）。
- 边界：结论仅限同一张 2000 条代理表的回顾式发现效率；真实前瞻式闭环（F 包）仍待真实 GBSA。

## 2026-09-30（第三阶段：发布候选冻结与 100-seed 预注册投递）

- **冻结发布候选**：`configs/experiments/loop_strategy_compare_100seed.json`（seeds 1001–1100 与开发集不重叠；同 GBSA 目标/300+200+600 划分/5×50 预算/XGB 5 成员/主指标）。
- **预注册（运行前写死）**：`docs/项目记录/发布候选预注册_100seed_20260930.md`——主指标 = 第 5 轮配对平均 regret（greedy vs random），判据 = bootstrap 95% CI 上界 <0，同时必须报告中位数/Q25-Q75/胜率/最差 10% seed/逐轮曲线；附加稳健指标（top-5 中位数、top-10% 池召回、每预算发现数）与分层压力检验的算法先于运行实现并测试。
- **模型卡**：`docs/项目记录/模型卡_XGB自举集成_20260930.md`（`model_id/feature_version`、后端区分、σ 未校准声明、明确不是 v8 秩融合、使用范围与限制）。
- **比较驱动扩展**：`pool_truth()`（复刻闭环划分用于召回计算）、`top10_recall()`、逐轮累计 top-5 中位数与 top-10% 召回入 `runs.csv`；`summary.json` 配对块新增 Δregret 中位数/Q25/Q75、最差 10% seed、Δtop5、Δrecall（各带 bootstrap CI 与胜率）；新增 `ood_strata_aggregate.csv`（按策略×突变层：池 OOD 率、到采集前训练集的平均最小 Hamming 距离、揭示数、MAE、层内最优选中率）；loop 的 `round_NNN_ood_strata.csv` 新增 `mean_min_train_hamming(_selected)`。测试 +2（`DiscoveryMetricsTest`），本地 61 项全过、服务器 50 项闭环契约全过。
- **100-seed 确认已投递**：服务器 `srv_strategy_compare_100seed_20260930_201309`（100 seeds × 4 策略 = 400 子运行，`nice -n 19`/2 线程，约 11–12 小时）；早期子运行已抽查：指纹/台账/清单/OOD 分层齐全、`ood_threshold_hamming=6.0`。结果回填至预注册文档 §5（见 2026-10-01 条目）。

## 2026-09-30（第二阶段：修 §7 P0/P1 并交接真实待测批次）

- **P0-1 台账索引错位修复**：`loop.py` 原先把 `mu_all.mu[k]` 写到第 k 个被选候选（把批次顺序当可选池索引）；现全部预测/OOD/角色字段按 `sel_pos`（可选池内位置）对齐，台账新增 `selection_position`。`LedgerAlignmentTest` 用第 1 轮训练集重拟合代理、逐行比对 `pred_gbsa`（1e-9 容差）并断言选中确实非前缀。**旧批次（含 30-seed v1）的 `pred_gbsa`/`best_pred_in_batch` 字段作废**；发现指标（`best_selected/regret` 出自状态标签）不受影响，v2 重跑已投递。
- **P0-2 回填校验与失败持久化**：`ParkOracle` 新增 `_validate_ingest_row`（`candidate_id` 重算比对、`protocol_id` 非空且过允许列表、状态仅 `pass/fail`、同文件/跨文件冲突一致校验、已确认值不被污染）与 `_atomic_append`（临时文件 + `os.replace`）原子写入 `labels.csv`/`failures.csv`；`loop._absorb_returned_labels` 把 QC fail 吸收进 `state.failed`，批次闭合后清 `open_batch` 并删除陈旧 `PARKED`。新增 6 个回填/重启/闭批测试。
- **P0-3 恢复指纹与安全恢复入口**：`loop.check_fingerprint()` 写 `fingerprint.json`（接口版本、配置、数据 SHA256、候选池/留出/初始标注集哈希、池统计、源码哈希），`--resume` 强制匹配 config/data/pool/holdout（不符即报错），源码变动只告警并保留原始指纹；`run_experiment.py` 新增 `--resume-run <run_id> [--ingest csv]`（复用原 `config.json`、`resume_history.json`、先删旧标记再跑、单一标记落盘）。新增 `FingerprintResumeTest` 与 `test_resume_run_reuses_the_run_and_updates_the_marker`（PARKED→回填→DONE 端到端）。
- **P1-a 全量排重**：生成期排除完整 2000 条标准表 + init + holdout（`pool_stats.excluded_canonical_table=2000` 入指纹），出口期 `_available()` 再排除历史批次全部序列。服务器 96 条交接批次实测排除 2000 条后保留 600 条新候选。测试 `FullLibraryDedupeTest`。
- **P1-b 采集前 OOD 与分层报告**：`ood_threshold()` 在采集前用当轮训练集计算；metrics 新增 `ood_threshold_hamming`、`ood_train_n_from_wt`、`batch_ood_fraction`；每轮写 `round_NNN_ood_strata.csv`（按突变数分层：池/选中/OOD 计数、回顾式逐层 MAE、层内最优是否被选中）。测试 `OodTimePointTest`。
- **P1-c 运行器严格化与比较完整性**：`require_registry_hash()` 缺注册项即失败（`allow_missing_registry` 显式豁免），`registry_match` 不再缺项时为 true；`strategy_compare.check_completeness()` 缺任一 seed/策略/轮次、行数不符、键重复或 regret 为负 → 写 `completeness.json` 并非零退出，配置加 `expected_rows`。测试 `test_unregistered_dataset_fails_for_formal_runs`、`CompletenessTest`。
- **P2**：`surrogates.py` 文件头升到 v0.2.0（含后端区分与“非 v8 秩融合”说明）。
- **真实待测批次交接**：`configs/experiments/loop_handoff_96.json` → 服务器 `runs/srv_handoff_20260930_182400`：96 条候选（1 板，seed 101，greedy，`n_from_wt` 5–13），`PARKED`/rc 3；导出含 `batch_id/candidate_id/sequence/pred_gbsa/ood_flag/protocol_id` 与 `batch_manifest.json`（模型/协议/数据指纹）。`tools/verify_loop_run.py`（已扩展：指纹、清单、OOD 分层、陈旧标记）复核 `[ok]`；证据回收至 `evidence/loop_20260930/handoff_96/`。协议串为占位符 `gbsa_pending_protocol`，回填前须与团队确认并经 `--resume-run … --pin-protocol <协议串>` 审计流程钉定（勿手改旧运行配置）。
- **测试 59 项全过**（闭环契约 48 项）；本地 `verify_smoke_v2_20260930`（DONE）与 `verify_park_v2_20260930`（PARKED）经扩展后的验证器全部 `[ok]`；38 个文档 0 断链。
- **30-seed v2 复核完成**（服务器 `srv_strategy_compare_30seed_v2_20260930_182900` = `DONE`，`completeness.json` ok=true，120 子运行/600 行无缺失）：发现指标与 v1 **逐位一致**（greedy 末轮 regret 0.451 vs random 2.130，配对 Δ −1.679 [−2.637,−0.785]），证实修复只影响台账预测列与 OOD 记录方式；**v2 的 `pred_gbsa/ood_flag` 台账字段可信，引用预测台账或 OOD 分层一律以 v2 为准**（`evidence/loop_20260930/strategy_compare_30seed_v2/`）。
- QC 失败的台账行在吸收时由 `awaiting_labels` 转为 `failed`（`selection_role` 追加 `:qc_fail`），`_persist_state` 同步重写 `ledger.csv`；`--resume-run` 的 `expected_rows` 改为仅在配置显式给 `resume_expected_rows` 时校验（恢复会合法追加历史行）。

- **候选生成与测量库的范围错位（本轮最重要的更正）**：标准 2000 条库的 `n_from_wt` 分布为 `{0:1,4:2,5:9,6:34,7:80,8:247,9:410,10:481,11:444,12:226,13:66}`（均值 9.87），即测量空间是**半径 4–13 的壳**而不是以 WT 为中心的球；原先 `mutate` 默认生成 1–5 突变候选——**1–3 超出已测突变数范围，4–5 极稀疏（各 2/9 条）；约 51% 是该 OOD 规则下的比例，不能称“整池 OOD”**。闭环现默认取已标注集合的 `n_from_wt` 范围（`--min-mutations/--max-mutations` 可覆盖），日志打印池/标注范围对比，同范围生成 OOD=0。证据：`evidence/loop_20260930/candidate_pool_report.json`（2000 生成→1998 保留，2 条与标准表重复，候选 ID 全唯一，与最近标准序列距离中位数 4）；范围 4–13 的总容量 67,100,400。
- 修复 §7.1 OOD 规则的退化缺陷：LOO 最近邻距离用“全体最小距离”计算时被自身距离 0 覆盖，阈值恒为 `inf`、`ood_flag` 失效；新增 `selection.loo_min_distance`（显式排除自身）并接入 `compute_ood`，每轮 `metrics.csv` 增加 `ood_fraction`；`OodRuleTest` 锁定该行为。
- 新增 D 包同预算策略比较驱动 `scripts/loop/strategy_compare.py` + 配置 `loop_strategy_compare_30seed.json`（30 seeds，random/greedy/greedy_diverse/maxmin_coverage，各 5 轮×50）：同划分/同池/同预算/同种子配对，输出逐轮 `runs.csv`、相对 random 的配对 Δregret 与 Δholdout-Spearman（bootstrap 95% CI）与胜率。发现指标使用 `best_selected`（只统计该策略自己查询过的候选）——3-seed 冒烟中曾因把共享初始标签算进去导致 regret 为负，已修正口径。3-seed 冒烟通过（`evidence/loop_20260930/strategy_compare_3seed`），30-seed 已于服务器 `nice -n 19`、2 线程投递。
- 新增 `tools/candidate_pool_report.py`（只读序列的候选池报告：分层容量/填充、与标准表及训练集去重、Hamming 几何、OOD 比例）、`tools/sync_remote.py`（按路径推送/远程执行，避免嵌套引号问题）、`tools/launch_loop_jobs.sh`（服务器礼貌投递脚本）。
- 新增运行包独立复核工具 `tools/verify_loop_run.py`：对任一 `runs/<run_id>` 逐条检查标记与退出码、每轮批次预算与 `candidate_id` 唯一性、跨轮批次不重叠、`n_labeled` 增量、`n_new_unique` 与已揭示台账行一致、指标时间点（`model_n_train==n_labeled_before`）、σ 合法性（要么缺失要么 >0）、台账待回填行数。本轮用它复核重跑的 `loop_smoke`（`verify_smoke_20260930`=DONE）与 `loop_prospective_park`（`verify_park_20260930`=PARKED，1 批 50 条待回填、`n_from_wt` 4–13）均通过；该工具第一次运行即抓出“PARKED 运行台账含未揭示行”的自有断言错误（`sum(n_new_unique)` 只应等于已揭示行数）。同时补 `test_chunked_hamming_is_chunk_invariant`（分块 Hamming 与 `loo_min_distance` 对块大小不变），测试增至 45 项。
- D 包 30-seed 同预算策略比较完成（服务器 `srv_strategy_compare_30seed_20260930_135716` = `DONE`，约 76 分钟）**：`random`/`greedy`/`greedy_diverse`/`maxmin_coverage`，同划分（300 初始 + 200 holdout + 600 池）、同预算（5×50）、同种子配对。末轮发现 regret 均值：greedy **0.451**、greedy_diverse 1.043、maxmin_coverage 1.880、random 2.130；相对 random 的配对 Δregret：greedy **−1.679，CI [−2.637, −0.785]**、greedy_diverse −1.087 [−1.966, −0.303]、maxmin_coverage −0.251 [−1.402, +0.868]（CI 跨 0）；holdout Spearman 配对 Δ：greedy **+0.0312 [+0.0163, +0.0462]，23/30 胜**、greedy_diverse +0.0242、maxmin_coverage +0.0159（CI 跨 0）。**如实记录两点**：(1) greedy 只在 15/30 个 seed 上更好、配对中位差 ≈ −0.08，均值优势来自尾部（避免灾难性批次，最差 −8.06 对 +3.34）；(2) 预注册的判读规则把“CI 是否跨 0”与“胜率≈15/30”并列，本次两者冲突，已把“发布级运行必须预先声明主统计量（均值 vs 中位数）”记为教训。结论仅限代理表，详见 `docs/论文改进/策略对比_20260930.md`。
- 新增 `tools/summarize_strategy_compare.py`（从 `summary.json`/`runs.csv` 生成 markdown 表）与 `tools/check_doc_links.py`（37 个文档、0 断链）。
- 冻结 GBSA 代理↔下游接口（`INTERFACE_VERSION=0.2.0`）并按 A→B→C 完成实施方案的缺陷修复，闭环框架可跑通：
  - **目标口径**：固定独立 `gbsa`（越小越强）；`label=0.961*Gap2+0.039*gbsa` 不得作为训练目标、采集分数、候选排序分数或成功指标（权重未验证）；
  - `loop/interface.py`：`Prediction(mu, sigma 可选, uncertainty_kind, feature_version, model_id)`、非有限值拒绝、`validate_sequences(check_scaffold=True)`、`candidate_id=sha256[:16]`、`TargetSpec(report_column="pred_gbsa")`；
  - `loop/selection.py`（由 `select.py` 改名，避免遮蔽标准库 `select`）：6 策略、`exclude`/`labeled_seqs` 过滤、零配额与负配额语义、耗尽显式报告、确定性 tie-break、分块 Hamming 新颖度；
  - `loop/mutate.py` v1：按 `stratum_capacity(k)=C(13,k)·3^k` 水填充分层生成（k=1 → 39）、父序列/WT 溯源、三重去重、OOD 分层标记；
  - `loop/oracle.py`：`TableOracle` 池/留出边界、`ParkOracle`（持久批次台账 + 严格回填校验 + 幂等）、`CommandOracle`、三者统一 `peek`（不新增揭示）；
  - `loop/loop.py`：每轮按“已标注∪已选∪待回填∪失败”过滤候选池、`_absorb_returned_labels` 幂等吸收、指标时间点（`model_n_train`=采集前）、**存在待回填一律 `PARKED`+rc 3，绝不写 `DONE`**、同 `batch_id` 续跑幂等；修复“轮次已用尽但批次未闭合时退 0”的误判路径；
  - `run_experiment.py`：registry 哈希比对、`source_manifest`、`DONE/PARKED/FAILED` 三态，配置可写 `expect_marker/expect_exit_code` 把“该 PARK 的运行不许 DONE”变成硬断言；
  - 测试：43 项全过（闭环契约 31 项），覆盖实施方案 §11.1 全部 8 类场景（跨轮唯一、选样边界、最小化方向、预测契约、真值隔离、指标时间点、停机恢复、错误回填、来源状态）；
  - 端到端证据（服务器 nice 19，2 线程）：`runs/srv_loop_smoke` DONE（2 轮，holdout 0.173→0.320）、`runs/srv_loop_park` 与 `runs/srv_loop_park_clean` PARKED（rc 3，awaiting 50）；`tools/loop_resume_probe.py` 验证部分回填后 awaiting 50→45→45、labeled 400→405→405、同一 `batch_id`、无新批次目录、重复续跑日志不增长；本地 Windows 侧同样 DONE/PARKED；
  - 新增文档 [候选闭环框架_20260930.md](docs/论文改进/候选闭环框架_20260930.md)（接口、状态机、产物、运行与回填命令、MD 接入方式、已验证/未验证边界）与 `tools/sync_remote.py`、`tools/loop_resume_probe.py`；
  - **明确边界**：冒烟运行读的是表内标签，不构成“选样优于随机”的证据；30-seed 策略对比（D 包）与真实前瞻式闭环（F 包）仍未做。

## 2026-09-26

- 「A. 现在就能做」批次全部尝试（30-seed 配对，服务器 nice 19），汇总见 `docs/论文改进/A组实验汇总_20260926.md`：
  - 负对照通过：打乱标签 0.016≈0、置换特征 −0.012≈0（发布健全性证据补齐）；
  - 平滑变体定界：LS(k5,a0.5) +0.0228（CI [0.0157,0.0301]、26/30）；**NLS 单调有害**（−0.009/−0.023）——标准平滑已是本家族最优；
  - 排序目标全面失败：lambdarank −0.271、xgbranker 子采样 −0.215、quantile −0.032 vs MSE——**MSE 保持最优目标（有证据关闭）**；
  - 强外推协议：突变数三分位留出给出真实 OOD 梯度（高突变区 0.318/中 0.286/低 0.269 vs 随机 0.355）——低突变区外推最差，量化了代理外推边界；
  - 稀疏 epistasis（Lasso 1300d）0.3632 无增益（一阶可加确认）；可解释输出：p31×p127_AA 等跨簇交互 30/30 稳定；
  - 候选闭环演示：31 个已验证批次合规搬入 `data/candidates/`（SHA256 manifest），release 融合+dock 融合+MFE 三秩排序 31,000 候选 → top-200（注意 1 条与训练集重叠、缺 Aft/Gap2 待补）；
  - 锁定测试集+模型卡：`runs/release_package/`（locked val/test 预测、model_card.json）；
  - 服务器清单核查：项目目录 90 文件/49MB，无 MD 逐帧数据 → Phase A 需生工侧对接；诊断子集 100 条清单已生成（runs/phaseA_diagnostic_subset_100.csv）；
  - 平滑×融合组合（30 seeds×2 配置）：配对 Δ=−0.0016、CI [−0.0054,+0.0026]、13/30 胜——平滑增益不叠进发布融合（融合本身已消减噪声），**发布配置保持不变**；A 组全部完成，总评见汇总文档 §11。

## 2026-09-24

- 全项目审计：通读全部文档与脚本，独立重算全部 evidence 检查点（偏差全部 ≤0.0002）；服务器（nice 19 小规模）验证管线跨环境逐位一致。
- 补充缺失验证：新鲜 100-seed mutation-only RF 基线（0.3644±0.0424，复现 0.365 声明）；v8 10-seed 与 dock_fusion 5-seed 新鲜复现（与历史证据逐位一致）；配对融合 vs 基线重算（+0.0104、CI [0.0061,0.0147]、66/100 胜）。
- 修复 NUPACK useful 块维度错配：feature_selection_pipeline 384d（误含 delta_nupack_）、fusion 377d（宽前缀）→ 严格 192d，并新增回归测试（单测 11/11）。
- 确认 phase1 eval_rank_targets 的 rank 目标泄漏与 XGB loss 未接线问题；登记 14 处文档口径矛盾与修复清单，见 docs/项目记录/全项目审计_20260924.md。
- 文档修复（执行方案_v8/0814特征汇报/特征整理_v2/逐页讲解/改进方案/验证结果/工作流决策/README/AGENTS）：混合基线表拆分为「vs 基线 66/100」与「vs 最优单模型 44/100」两口径、0.395 标注 oracle 口径、oracle bound 0.412 标注为经验观察、MAE +0.007 标注混口径、5 vs 7 子块矛盾按 v10b 证据修正、seed42 0.511→0.506、评估次数算术勘误、dock 定位明确（可作目标不可作输入/不可修 gbsa）。
- 脚本修复：dock_denoise/dock_rank_correction 目标变换折内化（消除测试标签泄漏）；run_active 预训练语料与标准化仅限非 holdout；run_paper_fusion 语料默认 30,000 与汇报一致、riboswitch 语料缺失时显式报错；feature_gating 补 seed 并修正熵正则符号；ensemble_defect 修复 Windows spawn（initializer 传参）；active.select_batch 联合标准化；dock_fusion 移除死 SVR 导入并修正 docstring；new_features 26d→52d docstring；surrogate σ 改 ddof=1；run_reproduce 增加 --outer-seeds 多 seed 支持；usage 统一为 python -m。历史快照 eval_rank_targets 添加 KNOWN_ISSUES 注释（不篡改快照）。
- 文献调研（4 路并行 + OpenAlex 核验）：MM-GBSA 去噪（多 replica 5-10×4ns 为最强手段、RNA 用 GBn1+ε_in=2）、RNA 适配体 ML（Ward 2023 概率>系综缺陷、CoBRA/GRASSP RNA-LM 嵌入、压缩感知 epistasis）、噪声标签 AL/BO（「AL≈random」理论背书 + 复测预算分配 BTS-RED 路线为可填补空白）、dock/去噪（Sindt 2025 佐证 dock 修 gbsa 失败、ECR 秩共识、Rank-N-Contrast、ConFrag/NLS 边界）。产出 `docs/论文改进/优化空间调研_20260924.md` 及 4 份子报告。
- 升格验证（服务器 nice 19）：k-NN Hamming 平滑 100-seed **配对 Δ=+0.0191、CI [0.0146,0.0237]、82/100 胜**（正式化门槛通过）；dock_fusion 100-seed **四模型秩融合 0.5523 ± 0.0342、CI [0.5456,0.5590]**（dock 升格为 100-seed 正式目标）。
- 正式结果表更新：dock 100-seed 与 k-NN 平滑 100-seed 写入验证结果.md/README/PROJECT_STATUS/工作流决策/AGENTS；两实验均通过 run_experiment.py 生成正式运行包（config/数据哈希/环境/逐 seed 指标/DONE）；dock_fusion 与 label_smoothing 增加 CSV 输出模式与 plain-script 兼容回退。
- nupack 环境修复：段错误根因为 Python 3.10 的 nupack_env，改用 **nupack_env_py39**（mfe/pairs/pfunc/structure_probability 全正常，单条 0.2s）；ensemble_defect.py 修复 Structure→dot-paren 解析（str() 而非迭代）与非规范配对容错（突变破坏 WT 配对时结构概率置 0）；新增 wt_struct_prob 特征（Ward 2023）。
- 系综缺陷/目标结构概率特征评估（30-seed 配对，303d+192d 基线上）：**无增益**（+defect Δ+0.0001 19/30、+wt_prob Δ−0.0001 3/30、CI 均跨 0）——信息已被 1192d 系综表覆盖，特征路线有证据收尾。
- 复测预算分配实验（BTS-RED 路线，replica_allocation.py，30 seeds × 均匀/异方差噪声 × 低覆盖/饱和覆盖）：低覆盖时 NEW（新序列）全面占优（瓶颈是特征覆盖）；**饱和覆盖（n_init=1400）出现交叉点——顶部真最优区排序可靠性上 MIX（复测+新序列）反超 NEW（0.115 vs 0.110）**；p 扫描定出预注册比例 **p=0.5~0.75**（p=0.25 有害）。执行方案（含 MD 副本去噪的天花板估计公式 ρ(r)=ρ_obs/√(ρ_rr+(1−ρ_rr)/r) 与 Phase A–D 协议）见 `docs/论文改进/执行方案_MIX复测与MD去噪_20260924.md`。

## 2026-09-03

- 完成 30-seed 嵌套特征选择（1495 维含 NUPACK），9/9 配对达标（Δ+0.022~+0.085，CI 下界>0，胜率 24–30/30）。
- cluster split 外推检查 9/9 通过；但突变 pattern 全唯一导致区分度弱，仅作稳健性验证。
- 新增 cluster_split_check.py 脚本与 feature_selection_all_safe_{30seed,smoke} 配置。
- 特征选择流水线接入 canonical nupack_features_full.csv（按 Sequence 对齐、强制校验），并加固数据/参数/结果矩阵校验。
- 回收 30-seed + cluster split 完整运行包至 evidence/server_sync_20260903/，登记 SHA256。
- 主线推送到 GitHub（GoodThing001/FAD）。

## 2026-09-02

- 建立安全的 clean mainline 整理区；旧项目未移动、未删除。
- 固定唯一数据入口、实验运行包和文档职责。
- 将 v8 100-seed 结果登记为正式发布结果。
- 将 2026-09 特征选择登记为探索性实验。
- 登记旧 `eval_rank_targets.py` 的 rank 变换测试标签依赖风险，等待修复与复核。
- 完成30,996个历史文件的定向SHA256清单，识别精确重复和大型文件候选。
- 将1,019个≥5 MiB精确重复文件（约7.02 GiB）移入外部可恢复隔离区；终态校验1,019/1,019通过。
- 增加长路径安全处理、失败项定向重试和隔离终态验证工具。
- 回收并核验phase1、v10–v13共6组完整30-seed历史检查点及运行脚本快照。
- 修正特征选择流程：加入全特征对照、mutation强制保留、折内候选重算、逐seed检查点和配对汇总。
- 增加服务器回收清单工具与项目路线决策文档。
- 回收并核验v8的30-seed与100-seed完整checkpoint、日志和运行元数据；100-seed融合为0.37477，bootstrap 95% CI [0.36691, 0.38231]。
- 回收NUPACK增强数据并生成无目标泄漏的2000×1192安全特征表，登记来源与SHA256。
- 归档旧版30-seed特征选择stdout及脚本快照；因缺失逐seedCSV，仅标记为补充证据。
- 将1,531个可用历史内容文件整理为带SHA256清单的冷归档；将27,326个文件、13.885 GiB生成物与重复副本移入待永久删除隔离区。
