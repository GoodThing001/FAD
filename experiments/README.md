# Experiment policy

实验定义放在 `configs/experiments/`，运行结果统一进入 `runs/<run_id>/`。

状态词：

- `validated`：完成规定的多 seed、无泄漏审计和结果归档。
- `exploratory`：可以运行，但不能写进正式结论。
- `deprecated`：保留历史说明，不再消耗计算资源。
- `invalid_leakage`：结果无效，仅用于审计。

