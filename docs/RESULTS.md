# Validated results

## 正式结果

| 方法 | 协议 | gbsa Spearman | 状态 |
|---|---|---:|---|
| mutation-only + RF | 100 seeds | 0.365，95% CI 约 ±0.008 | baseline |
| v8 五模型秩融合 | 100 seeds | 0.375，95% CI [0.367, 0.382] | release |
| mut+struct+wd + winsor + GBR/XGB | 30 seeds | 约 0.395 ± 0.039 | development only |
| 固定 seed=42 + NUPACK + epistasis | 单一拆分 | 0.506 | historical outlier |

## 解释

最终发布必须使用预先固定的方法在全部 seeds 上的均值，不能对每个 seed 事后选择最佳模型。0.395 是开发阶段配置比较结果；0.506 是固定拆分偶然高值。

当前数据的经验 oracle bound 约为 0.412。继续堆叠同类模型或序列特征预期收益很小；突破需要新增 MD 数据、3D 接触信息或更可靠的 GBSA 标签。

## 已恢复的 NUPACK / RNA 配对结果

| 对比 | 配对ΔSpearman | bootstrap 95% CI | 胜率 | 判断 |
|---|---:|---:|---:|---|
| XGB：+contact_pca vs 序列基线 | +0.0116 | [0.0039, 0.0198] | 21/30 | 单块信号成立，保留线索 |
| XGB：+NUPACK 192d vs 303d | +0.0008 | [-0.0061, 0.0077] | 14/30 | 无稳定组合增益 |
| GBR：+NUPACK 192d vs 303d | -0.0057 | [-0.0115, -0.0003] | 14/30 | 有害 |
| GBR：NUPACK+RNA组合 vs 303d | +0.0090 | [0.0008, 0.0171] | 18/30 | 可做一次固定复核，未达主线门槛 |
| XGB：NUPACK+RNA组合 vs 303d | -0.0089 | [-0.0183, -0.0004] | 14/30 | 有害且模型依赖 |

这些结果说明物理特征中存在局部信号，但高维组合不能稳定迁移到不同模型。不要据此继续扩展特征族。

## 已关闭路线

- 全数据 ED/目标统计特征：标签泄漏。
- energy_terms 与二阶 epistasis：多 seed 下有害。
- RNA-FM、RNAErnie、RNA-BERTa：域偏移，效果弱。
- NN + NUPACK BPP：与 GBSA 的 3D 物理机制不匹配。
- XGBRanker/LGBMRanker：不适配单组连续标签。
- 全量 NUPACK 直接输入 boosting：维度高且分布不稳。
