# Scripts — PreQ1 Riboswitch gbsa Proxy Model (v2.2)

> 交付基线: `eval_gbsa_unified.py --screen top-100` → **gbsa_Sp=0.506**

---

## 目录结构

```
scripts/
├── eval_gbsa_unified.py                ★ 主评估 (核心交付)
├── add_nupack_local_features.py         ★ NUPACK 特征生成编排
├── nupack_pairprob_worker.py            ★ NUPACK 逐序列 worker (P1-P11)
│
├── tools/                               辅助工具
│   ├── feature_shgr.py                  SHGR-lite 特征筛选
│   ├── test_d3_features.py              3D 特征独立测试
│   └── test_advanced_features.py        上位效应+3D 组合测试
│
├── experiments/                         已完成的优化实验 (全部 :x: 未超越基线)
│   ├── train_softrank.py                Phase 1: MLP Soft Spearman (0.472)
│   ├── train_gnn_gbsa.py                Phase 2: GNN + NUPACK (0.15)
│   ├── train_siamese.py                 Phase 3: Siamese RNAErnie (0.04)
│   ├── extract_3d_proxy.py              Phase 4a: DBSCAN 3D 代理 (0 簇)
│   ├── extract_aform_3d.py              Phase 4b: A-form 3D (0.451)
│   ├── train_lgbm_softspear.py          Phase A: LGBM SoftSpear (0.303)
│   ├── run_farfar2_3d.py                Phase B: Rosetta FARFAR2 (API不兼容)
│   ├── generate_trrna2_pdbs.py          Phase B2: trRosettaRNA2 PDB 生成 (100%成功)
│   └── extract_local_3d.py              Phase B2: 局部 3D 特征 (0.447)
│
└── legacy/                              v1/v2 早期脚本 (归档)
    ├── compare_mfe_sources.py
    └── verify_v2_integrity.py
```

---

## 交付基线

### v2.2 最终: gbsa_Sp=0.506 (2026-08-01)

**复现命令**:

```bash
conda run -n nupack_env_py39 python scripts/add_nupack_local_features.py
conda activate /home/hzeng/envs/FAD_env
python scripts/eval_gbsa_unified.py --metric spearman --models tree --scale group --no-rnafm --screen top-100
```

**最佳结果**:

| 指标 | 数值 | 方法 |
|---|---|---|
| **gbsa_Sp** | **0.5060** | GBR + epistasis+NUPACK (top-100) |
| gbsa_RMSE | 8.6436 | 同上 |
| gbsa_Pearson | 0.4693 | 同上 |
| **Hybrid_Sp** | **0.9236** | 同上 |
| RMSE-weighted Ensemble | 0.5061 | GBR+LGBM+Ensemble |

**最佳单模型组合**:

| 特征组合 | 维度→top-100 | 模型 | gbsa_Sp |
|---|---|---|---|
| **mut+struct+wd+mt+epistasis+NUPACK** | 1087d→100d | **GBR(500)** | **0.5060** |
| mut+struct+wd+mt+epi+epi3pca | 586d→100d | GBR(500) | 0.4618 |
| mut+struct+wd+NUPACK | 447d→100d | LGBM(500) | 0.4781 |
| mut+struct+wd+mt+sq+nuss | 413d→100d | LGBM(500) | 0.4756 |
| mut+struct+wd+mt | 317d→100d | XGB(500) | 0.4553 |

### v2.1: gbsa_Sp=0.492 (2026-07-29)

**复现命令**:

```bash
python scripts/eval_gbsa_unified.py --metric spearman --models tree --scale group --no-rnafm --screen top-100
# 备注: v2.1 无 epistasis 特征
```

**最佳结果**: Ensemble top-3 = **0.4922** (LGBM + LGBM + RF, 简单平均)

---

## gbsa_Sp 完整进化路径

```
v1 起点 (mutation_only SVR):                       0.391
+ structure_role:                                   0.402  (+0.011)
+ mut_window_dimer (PyFeat 2019):                   0.432  (+0.030) 🏆 最大单步
+ GBR/XGB (换模型):                                  0.461  (+0.029) 🏆
+ energy_terms (MutPNI 2025):                       0.462  (+0.001)
+ SQUARNA + Nussinov:                               0.463  (+0.001)
+ top-100 screening (LGBM+NUPACK fix):              0.478  (+0.015) 🏆
+ Ensemble top-3:                                   0.492  (+0.014) 🏆 v2.1 交付
+ epistasis 外积 169d (v2.2):                       0.506  (+0.014) 🏆 v2.2 交付
────────────────────────────────────────────────────────────────
目标:                                               0.600  (剩余 0.094)
```

---

## eval_gbsa_unified.py 详解

### 核心协议

- **数据**: train (1600) + val (200) 联合训练 → test (200) 锁定评估
- **特征缩放**: Group-wise StandardScaler (每族特征独立)
- **特征筛选**: `--screen top-100` → ExtraTrees 重要性筛选 (关键参数!)
- **模型**: ET(500), RF(500), GBR(500,dp3,lr=0.03), XGB(500,dp3,lr=0.03), LGBM(500,dp3,lr=0.03)
- **集成**: 按 gbsa_Sp 选 top-3, 简单平均

### 全部特征组合

| 组合 | 维度 | 特征族 | 说明 |
|---|---|---|---|
| mutation_only | 52d | mut | 基线 |
| mut+struct | 95d | +structure_role | |
| mut+struct+wd | 303d | +window_dimer | 🏆 核心 |
| mut+struct+wd+mt | 317d | +mut_type | 基线组合 |
| mut+struct+wd+mt+energy | 387d | +energy_terms | |
| mut+struct+wd+mt+sq | 369d | +SQUARNA stem-max | |
| mut+struct+wd+mt+nuss | 361d | +Nussinov DP | |
| mut+struct+wd+mt+sq+nuss | 413d | +sq+nuss | |
| mut+struct+wd+mt+epistasis | 486d | +2阶上位效应 169d | v2.2 新增 |
| mut+struct+wd+mt+epi+epi3pca | 586d | +3阶上位效应 PCA 100d | v2.2 新增 |
| mut+struct+wd+mt+sq+nuss+epistasis | 582d | +sq+nuss+epi | |
| +NUPACK | +601d | NUPACK 1192d 全部 | 需 NUPACK 数据 |
| +epistasis+NUPACK | +601d+169d | NUPACK+epi | 🏆 最佳: 0.506 |
| +epi+epi3pca+NUPACK | +601d+169d+100d | 全组合 | |
| +epi×NUP_sparse | +601d+169d+100d | 稀疏交叉 | |

### 关键参数

| Flag | 推荐值 | 说明 |
|---|---|---|
| `--metric` | `spearman` | 主评估指标 (rmse/mae/r2/pearson/hybrid) |
| `--models` | `tree` | ET/RF/GBR (+XGB/LGBM 自动检测) |
| `--scale` | `group` | :trophy: 按特征族分组 StandardScaler |
| `--screen` | `top-100` | :trophy: 修复 Boosting+NUPACK 崩溃的关键 |
| `--no-rnafm` | — | 禁用 LLM 嵌入 (已确认无效) |
| `--rank` | — | pairwise ranking (已确认无效) |
| `--3dproxy` | — | 3D 代理特征 (需 `_3dproxy.csv`) |

### 输出的模型类型

| 类型 | 说明 |
|---|---|
| Tree Models | ET/RF/GBR/XGB/LGBM — 各类特征组合 |
| Two-Stage | GBR/RF/ET→Ridge/SVR — 序列→NUPACK 残差 |
| CV Two-Stage | OOF stage1 → stage2 — 修复分布偏移 |
| Interaction | seq × NUPACK top-10/20 交叉 |
| Ensemble | top-3 简单平均 / RMSE加权 / Diverse(E2EPP) |

---

## 所有实验方法汇总

| 脚本 | 方法 | gbsa_Sp | 结论 |
|---|---|---|---|
| `experiments/train_softrank.py` | MLP Pure Spearman 损失 | **0.472** | 0.472<0.506, 小样本过拟合 |
| `experiments/train_gnn_gbsa.py` | GNN + NUPACK 边权重 | 0.15 | 1600 图样本不足 |
| `experiments/train_siamese.py` | Siamese RNAErnie Δgbsa | 0.04 | LLM 领域漂移 |
| `experiments/extract_3d_proxy.py` | DBSCAN 聚类 3D 代理 | — | 0 个簇 |
| `experiments/extract_aform_3d.py` | A-form 理想螺旋 3D | 0.451 | 保守结构无差异 |
| `experiments/train_lgbm_softspear.py` | LGBM Soft Spearman 自定义 | 0.303 | 树分裂/排序梯度不兼容 |
| `experiments/run_farfar2_3d.py` | Rosetta FARFAR2 | — | API 不兼容 |
| `experiments/generate_trrna2_pdbs.py` | trRosettaRNA2 2000 PDB | 100%成功 | :white_check_mark: 技术栈就绪 |
| `experiments/extract_local_3d.py` | 局部 3D 特征 | 0.447 | 树模型不兼容 3D 坐标 |
| `tools/test_advanced_features.py` | epi×NUPACK 稀疏交叉 | 0.449 | :x: 无效 |

### 已确认无效的方向 (勿重试)

| 方向 | 证据 |
|---|---|
| 预训练 RNA LLM (RNA-FM/RNAErnie/Siamese) | gbsa_Sp ≤0.04 |
| Pairwise Ranking (XGBRanker/LGBMRanker) | gbsa_Sp ≤0.31 |
| GNN (NUPACK 边权重) | gbsa_Sp 0.15 |
| 3D 坐标特征 (A-form/trRosettaRNA2/Rosetta) | gbsa_Sp ≤0.451 |
| LGBM Soft Spearman 自定义目标 | gbsa_Sp 0.303 |
| Feature Group-wise Ensemble | gbsa_Sp 0.402 |
| Dual Global+Local Model | gbsa_Sp 0.461 (≈GBR) |
| RBFN Surrogate | gbsa_Sp 0.442 |

---

## 环境依赖

| 环境 | Python | 关键包 | 用途 |
|---|---|---|---|
| `nupack_env_py39` | 3.9 | nupack, numpy | NUPACK 物理计算 |
| `FAD_env` | 3.10 | sklearn, xgboost, lightgbm, scipy | ML 训练/评估 |

## 相关文档

| 文档 | 路径 |
|---|---|
| 交付包 v2.2 | `docs/97_BestModel/0.5060/` |
| 交付包 v2.1 | `docs/97_BestModel/0.4922/` |
| 完整总结 | `docs/08_数据集矫正后记录/v2_本阶段完整总结.md` |
| 实验记录 | `docs/98_论文/调研/v2/实验记录.md` |
| 3D 分析 | `docs/98_论文/调研/v3_3d/3d特征分析.md` |
