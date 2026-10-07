# eval_rank_targets.py 快照已知问题（2026-09-24 审计确认）

本目录的 `eval_rank_targets.py` 是 phase1_v8 运行时的**历史快照**，用于证据追溯，
不会被修改。审计确认其存在以下问题，引用其结果时必须遵守对应口径：

1. **rank_uniform / rank_gaussian 目标变换泄漏测试标签**：`transform_target()` 对
   `np.concatenate([y_train, y_test])` 做联合排名（第 87-100 行）。测试标签进入训练目标。
   影响：phase1 CSV 中 `target=rank_uniform/rank_gaussian` 的行不构成独立证据；
   `raw` 与 `winsor_2.5` 行不受影响（winsor 分位数只用训练折）。
2. **XGB 的 loss 参数未接线**：`make_model("XGB", loss)` 忽略 `loss`（第 255-259 行），
   四种损失（mse/mae/huber_1/huber_2）下 XGB 结果逐位相同。影响：XGB 的损失消融结论无效。
3. **GBR 的 huber_1/huber_2 全部静默失败**：`GradientBoostingRegressor(loss="huber",
   alpha=delta)` 的 alpha 越界（合法区间 (0,1)），被主循环 `except Exception: pass`
   （第 493 行）吞掉，GBR huber 行全部缺失。影响：「GBR huber」参与 0.395 候选的归因不成立。
4. **静默异常丢行**：计划 4×4×3×5×4×30 = 28,800 行，实际产出 19,440 行（含 2,160 行
   screen-k 别名重复：mutation_only 的 sk∈{0,75,100} 都写 full_52d）。失败与未运行无法区分。
5. **「0.395 ± 0.039」是 oracle 口径**：等于逐 seed 在 {GBR×mae} ∪ {XGB×mae/huber_1/huber_2}
   @nested_100d 中取最优的均值（重算 0.3944 ± 0.0398）。单一固定配置最高为
   XGB@winsor@nested_100d = 0.3907。

结论：v8 发布路径（phase3_fusion_v8.py，winsor+秩融合）不依赖上述缺陷行；
本快照仅作为 phase1 开发阶段的证据留存。详见 `docs/项目记录/全项目审计_20260924.md`。
