# evidence/sigma_calibration_30seed — XGB 自举集成 σ 校准研究（判读）

来源：服务器 `runs/srv_sigma_calibration_30seed_20261002_172402`（`configs/experiments/eval_sigma_calibration_30seed.json`，DONE，60 行 = 30 seeds × 2 档 n_train）。seeds 301–330；每 seed：`RandomState(seed)` 划分 train/test(400)，折内 screen-100、winsor 2.5、5 成员 XGB；打乱标签负对照同折复跑。

## 结果（均值 + bootstrap 95% CI，n=30）

| 指标 | n_train=1600 | n_train=400（闭环尺度） |
|---|---|---|
| spearman_mu（参考技能） | 0.353 [0.341, 0.365] | 0.250 [0.234, 0.266] |
| spearman_sigma_error（σ vs |误差|） | **0.028 [0.010, 0.045]**（弱正相关） | 0.014 [−0.014, 0.040]（**不显著**） |
| 打乱标签对照 | −0.007 [−0.022, 0.008] ≈0 ✓ | 0.016 [0.002, 0.029]（与真标签同量级） |
| Δrecall LCB−greedy，β=0.5/1/2，top-5% | −0.008 / −0.017 / −0.007（均 CI 跨 0） | +0.003 / +0.008 / −0.002（均跨 0） |
| Δrecall LCB−greedy，β=2，top-10% | **−0.046 [−0.064, −0.028]（显著有害）** | −0.015 [−0.031, +0.001] |
| 十分位表（decile1→10 平均 |误差|） | 7.23→7.75（弱单调，跨度 ≈7%） | 7.49–8.14（无单调） |

## 门槛结论（summary.json 已固化）

1. **`gate_sigma_tracks_error`**：1600 档 True（但效应量极小，Spearman 0.028）、**400 档 False**——闭环尺度下 σ 与误差的分层不超过打乱对照水平。
2. **`gate_lcb_beats_greedy`：全部 False**；β=2 在 1600 档 top-10% 召回上显著**有害**（−0.046）。
3. **放行决定：uncertainty 采集保持关闭**。`sigma=ensemble_spread_uncalibrated` 继续只作记录字段；`mixed_control` 的 uncertainty 桶保持禁用、`uncertainty_mixed` 保持拒绝；批量 BO/EI/Thompson 的门槛未过（方案 §5：先证明 σ 与误差/命中风险的校准关系——本研究的回答是：当前 XGB 成员标准差不是可用的误差/命中风险信号）。
4. 与阶段 B 一致：当前瓶颈是代理本身的细粒度排序噪声（跨模型 Spearman 0.18–0.30），而不是采集函数选择；下一步应优先代理/标注改善（去噪、特征、更多标签），而非加采集函数。
