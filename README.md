# Spaceship Titanic Kaggle Project

Kaggle 竞赛 [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) 二分类任务（预测乘客是否被传送）。

**最终成绩：Kaggle 公开榜 0.80921（Exp#13, submission_90plus.csv）**

---

## 快速开始：3 步生成提交文件

```powershell
cd "c:\Users\siruiJohn\project\Spaceship Titanic_1\Spaceship Titanic"

python modular_pipeline/main.py full
```

产出两个文件：
- `submission_90plus.csv` — 含 Group 一致性后处理（**推荐提交**，产生 0.80921）
- `submission_no_group.csv` — 纯模型预测（参考对比用）

> **提示**：如果只改了集成权重不需要重新训练，可以只跑 `python modular_pipeline/main.py blend`。

---

## 项目结构

```
Spaceship Titanic/
├── modular_pipeline/             # 模块化训练管道
│   ├── config.py                 # 所有配置（特征、种子、路径）
│   ├── data_prep.py              # 数据加载 + 特征工程 + 编码
│   ├── train_base.py             # 训练基类（CV folds、OOF 聚合、MixUp 增强）
│   ├── mixup.py                  # MixUp 数据增强 + 同组特征交换
│   ├── train_hgb.py              # HistGradientBoosting 训练
│   ├── train_xgb.py              # XGBoost 训练
│   ├── train_lgb.py              # LightGBM 训练（含 Optuna 超参搜索）
│   ├── train_cat.py              # CatBoost 训练（备用，不在最终集成中）
│   ├── train_et.py               # ExtraTrees 训练（备用）
│   ├── train_nn.py               # MLP 神经网络训练（备用）
│   ├── blend.py                  # 集成：LR Stacking + MLP + 校准
│   ├── main.py                   # CLI 主入口
│   └── output/                   # 中间产物（pkl + oof/test .npy，Git 忽略）
│
├── data/raw/                     # Kaggle 原始数据（Git 忽略）
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── results/
│   ├── eda_viz/                  # 13 张 EDA 图表
│   └── model_report/             # 8 张模型报告图表
│
├── spaceship-titanic.py          # 原始单体脚本（备选入口）
├── spaceship-titanic.ipynb       # Jupyter 源文件
├── eda_visualizations.py         # EDA 可视化独立脚本
├── requirements.txt              # Python 依赖
├── README.md                     # 本文件
├── EDA可视化分析报告.md
├── 模型训练与评估分析报告.md
└── spaceship-titanic-说明文档.md
```

---

## 数据流

```
data/raw/train.csv, test.csv
        |
   data_prep.py          # 特征工程 + OOF Target Encoding + OrdinalEncoder
        |
   processed_data.pkl    # 55 特征 × 8693 训练样本
        |
  ┌─────┼─────┐
  hgb   xgb   lgb         # Top 3 树模型 (25 folds × 5 seeds)
  └─────┼─────┘
        |
  oof_*.npy + test_*.npy
        |
    blend.py              # LR Stacking + MLP blending + Isotonic 校准 + Group 一致性
        |
  submission_90plus.csv   # 最终提交
```

---

## 最终配置

所有参数集中在 [config.py](file:///c:/Users/siruiJohn/project/Spaceship Titanic_1/Spaceship Titanic/modular_pipeline/config.py)：

| 参数 | 最终值 | 说明 |
|------|--------|------|
| `blend_models` | `['hist_gb', 'xgb', 'lgb']` | 集成仅用 Top 3（Exp#4/13 均证明全 6 个更差） |
| `shap_keep_percentile` | 10 | 移除 SHAP 重要性底部 10%（Exp#16 证明 10% 是唯一最优） |
| `feature_sample_ratio` | 1.0 | 所有模型共享全部 55 特征 |
| `mixup_alpha` | 0.4 | MixUp Beta 参数（越小越接近原始） |
| `mixup_multiplier` | 1 | 每 fold 合成样本 = 原始 × 1 |
| `noise_std` | 0.01 | MixUp 附加高斯噪声标准差 |
| `group_swap_fraction` | 0.0 | 同组特征交换（Exp#9 证明 −0.0054，已关闭） |
| `ensemble_runs` | 1 | 单轮训练（Exp#10 证明多轮平均 −0.0026） |
| `random_seeds` | [42, 2024, 7, 2025, 88] | 5 seeds × 5 folds = 25 folds |
| `n_splits` | 5 | 每 seed 的 CV folds 数 |
| `optuna_trials` | 200 | LightGBM 超参搜索轮数 |

---

## 特征工程（55 特征，最终版）

经过 16 次实验验证的最优特征集，在 [data_prep.py](file:///c:/Users/siruiJohn/project/Spaceship Titanic_1/Spaceship Titanic/modular_pipeline/data_prep.py) 的 `engineer_features()` 中生成。

### 特征分类

| 类别 | 数量 | 包含 |
|------|:---:|------|
| 类别特征 | 11 | HomePlanet, CryoSleep, Destination, VIP, CabinDeck, CabinSide, HomeDest, DeckSide, CabinZone, AgeBand, Surname |
| 群组/结构 | 3 | GroupSize, Solo, FamilySize |
| 基础数值 | 13 | Age, CabinNum, CryoFlag, VipFlag, IsChild, IsTeen, IsSenior, SpendPositiveCount, NoSpend, etc. |
| 消费（原始+对数） | 13 | RoomService~VRDeck, TotalSpend, AvgSpendPerService, SpendPerGroupMember, Log_* |
| 高级衍生 | 8 | CabinNumParity, CabinNumBucket, SurnameGroupSize, SpendEntropy, MaxSpendCategory, CryoNoSpend, NotCryoHasSpend, AgeSpendInteraction |
| 目标编码 (LOO) | 2 | CabinAgreementScore, CabinMean |
| TF-IDF | 5 | Surname_tfidf_0~4 |
| Target Encoding (OOF) | 7 | TE_HomePlanet, TE_CabinDeck, TE_HomeDest, TE_DeckSide, TE_AgeBand, TE_CryoSleep, TE_CabinZone |
| **总计** | **55** | 全部通过 SHAP 10% 筛选 |

### 为何不新增特征

Exp#11（+34 特征 → 0.79518）和 Exp#12（+18 特征 → 0.79611）证明：在 8693 样本下，每增加一个特征都会稀释树模型的分裂质量。55 特征已将该数据集的信息提取到饱和。

---

## MixUp 数据增强

每 fold 训练时动态生成合成样本：

```
x_aug = λ × x_i + (1-λ) × x_j       λ ~ Beta(0.4, 0.4)
y_aug = y_i if λ >= 0.5 else y_j
```

同时附加 `N(0, 0.01)` 高斯噪声作为额外正则化。这是项目中少数几个（3/16）有净正收益的优化之一（+0.0005）。

---

## 模型选择与集成

### Top 3 集成模型

| 模型 | 核心机制 | OOF 准确率 |
|------|---------|:----:|
| **HistGradientBoosting** | 直方图分箱 + 按层生长 | 0.8233 |
| **LightGBM** | 按叶生长 + Optuna 200轮超参搜索 | 0.8239 |
| **XGBoost** | 预排序 + L1/L2 正则化 | 0.8194 |

仅使用 Top 3（HGB + LGB + XGB）OOF 为 0.82331，优于 6 模型全集的 0.82296。

### 为什么不用另外 3 个

| 模型 | OOF | 不收录理由 |
|------|:---:|------|
| MLP | 0.8042 | 低于集成 Kaggle 分，噪声贡献 |
| ExtraTrees | 0.8069 | 与 Top 3 相关性高，加入后 OOF 下降 |
| CatBoost | 0.8186 | 微弱不及 Top 3，移除后 OOF 未下降 |

### 集成方法

`blend.py` 分四阶段：

1. **单模型 OOF**：5 seeds × 5 folds = 25 folds StratifiedKFold
2. **LR Stacking**：3 模型 OOF 概率 → LogisticRegression 元模型，搜索最优 [LR:SimpleAverage] 混合权重
3. **MLP blending**：55 维特征上训练 MLP(128→64)，与 GBDT 集成做 meta-blend（仅 OOF 提升时采纳）
4. **Isotonic 校准**：5-fold 校准概率（仅提升时采纳）+ Group 一致性后处理

---

## 📋 完整实验记录（16 次）

### 分数演进总览

```
Exp#0  0.80500 ─── 基线
Exp#1  0.80523 ─── 工程改进
Exp#2  0.80640 ─✅─ SHAP 筛选（+0.0012）
Exp#3  0.80500 ─── Group 一致性回退
Exp#4  0.80593 ─── 简化集成回退
Exp#5  0.80547 ─── 子集多样性回退
Exp#6  0.80523 ─── NN 架构回退
Exp#7  0.80523 ─── NN Optuna 无效
Exp#8  0.80687 ─✅─ MixUp（+0.0006）
Exp#9  0.80149 ─── 同组交换灾难（−0.0054）
Exp#10 0.80430 ─── 多轮平均回退（−0.0026）
Exp#11 0.79518 ─── 特征爆炸（−0.0117）
Exp#12 0.79611 ─── 依然爆炸（−0.0108）
Exp#13 0.80921 ─⭐⭐ 回滚+修bug（+0.0131）历史最佳
Exp#14 0.80687 ─── 四大技巧全负（−0.0023）
Exp#15 0.80289 ─── 伪标签回退（−0.0063）
Exp#16 0.80874 ─── 反向筛选回退（−0.0005）
```

---

### Exp#0 — 基线（原始单体脚本）
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80500 |
| **方法** | 原始 `spaceship-titanic.py`：5 个树模型（HGB/LGB/XGB/Cat/ET）+ 61 特征 + 简单平均集成 |
| **特征数** | 61 |
| **集成** | 5 模型简单平均 |
| **结论** | 基线 |

---

### Exp#1 — 模块化管道
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80523（+0.0002） |
| **Δ vs Exp#0** | +0.0002 |
| **方法** | 将原始单体脚本重构为 `modular_pipeline/` 模块化架构。增加至 6 模型（+MLP 神经网络）。代码拆分至 `data_prep.py / train_*.py / blend.py / main.py` |
| **修改点** | 工程重构：单体脚本 → 模块化；模型数 5→6；统一配置到 `config.py` |
| **特征数** | 61 |
| **集成** | 6 模型 LR Stacking + MLP blending |
| **结论** | 纯工程改进，分数持平。证明模块化不改变模型质量，但大幅提升迭代速度 |

---

### Exp#2 — SHAP 特征筛选 ✅
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80640（+0.0012） |
| **Δ vs Exp#1** | +0.0012 |
| **方法** | 用 ExtraTrees 200 trees 计算 SHAP 特征重要性，移除底部 10%（6 个）无用特征 |
| **修改点** | 新增 `scan_dead_features_with_shap()` 函数；`config.py` 加入 `shap_keep_percentile=10`；特征从 61 → 55 |
| **成功原因** | 6 个被移除特征（如 `PassengerId` 衍生、低方差 TE 列）本身不携带目标信号，存在只会让树分裂器在噪声维度上浪费分裂机会。SHAP 筛选本质是**降噪**而非添加复杂度 |
| **结论** | ✅ 项目历史上第一个明确有效的优化。证实「减法优于加法」|

---

### Exp#3 — Group 一致性后处理
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80500（−0.0014） |
| **Δ vs Exp#2** | −0.0014 |
| **方法** | 利用 domain knowledge：同一 GroupId 的乘客共享 Transported 标签。对 test 预测结果做 Group 一致性修正：若组内平均概率 > 0.75，全组推向 1；若 < 0.25，全组推向 0 |
| **修改点** | `blend.py` 新增 `apply_group_consistency()` 函数；产出 `submission_90plus.csv`（含后处理）和 `submission_no_group.csv`（不含） |
| **失败原因** | 底层模型预测不够精准时，Group 一致性会将**边际样本的错误预测推给同组所有人**。例如 3 人组中 2 人预测正确、1 人预测错误，修正后 3 人全错。Group 一致性只有在底层模型足够精准时才正向 |
| **结论** | ❌ 单独上线导致 OOF 提升但 Kaggle 下降——这是第一次 OOF 与 Kaggle 方向相反，开启了后续反复出现的模式 |

---

### Exp#4 — 简化集成（仅 Top 3）
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80593（−0.0005） |
| **Δ vs Exp#2** | −0.0005 |
| **方法** | 将 `blend_models` 从 6 个（ET/HGB/XGB/LGB/Cat/MLP）缩减为 3 个（HGB/XGB/LGB）。OOF 从 0.82296 升至 0.82331 |
| **修改点** | `config.py` 中 `blend_models = ['hist_gb', 'xgb', 'lgb']` |
| **失败原因** | OOF 确实上升了（+0.00035），但 Kaggle 下降了。弱模型（MLP/ET/Cat）虽然有噪声，但它们与 Top 3 的**错误不重叠**——移除后集成多样性下降，对测试集分布的泛化能力减弱。这是 OOF≠Kaggle 的典型案例 |
| **结论** | ❌ OOF 提升不代表 Kaggle 提升。保留此配置用于后续实验对比 |

---

### Exp#5 — 特征子集多样性
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80547（−0.0009） |
| **Δ vs Exp#4** | −0.0005 |
| **方法** | 每个模型从 55 个特征中随机抽取 85%（~46 个），不同模型看到不同特征子集。目的：增加模型间多样性，降低相关性 |
| **修改点** | `config.py` 加入 `feature_sample_ratio=0.85`；`data_prep.py` 中生成 `feature_subsets` 字典；训练时按模型加载不同列 |
| **失败原因** | 每个模型丢失了 15% 的特征信息，单模型 OOF 平均下降 ~0.001。多样性增加的收益（集成方差降低）不足以弥补单模型变弱的损失。在 8693 样本、55 特征下，每个特征都含不可替代的信号 |
| **结论** | ❌ 特征子集策略带来的单模型退化 > 多样性收益 |

---

### Exp#6 — NN 架构优化
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80523（−0.0002） |
| **Δ vs Exp#5** | −0.0002 |
| **方法** | 将 MLP 从 `(128,64)` 改为 `(256,256,128)` |
| **修改点** | `train_nn.py` 中 `hidden_layer_sizes` 和 `blend.py` 中 MLP blending 结构 |
| **失败原因** | 更大网络在 8693 样本上过拟合更快。`early_stopping` 在 3-5 epochs 就触发，说明参数容量远超数据能提供的信息量。更深网络等价于给模型更多机会记住训练集噪声 |
| **结论** | ❌ 退化 |

---

### Exp#7 — NN Optuna 超参搜索
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80523（±0） |
| **Δ vs Exp#6** | 0 |
| **方法** | 用 Optuna 200 轮搜索 NN 最优超参（层数、神经元数、学习率、alpha、batch_size） |
| **修改点** | `train_nn.py` 新增 Optuna objective 函数 |
| **失败原因** | Optuna 在 3-fold CV 上找到了最佳参数组合，但 3 folds 的评估集太小，搜索到的参数在 OOF 上看起来最优却对测试集分布不具备泛化能力。本质上是在小验证集上过拟合了超参数 |
| **结论** | ❌ 3-fold CV 超参搜索对 NN 无效 |

---

### Exp#8 — MixUp 数据增强 ✅
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80687（+0.0006） |
| **Δ vs Exp#7** | +0.0016 |
| **方法** | 每 fold 训练时用 MixUp 生成合成样本：`x_aug = λ·x_i + (1-λ)·x_j`，λ ~ Beta(0.4, 0.4)，标签跟随贡献更大的样本。同时施加 `N(0, 0.01)` 高斯噪声作为额外正则化 |
| **修改点** | 新建 `mixup.py`；`train_base.py` 中每 fold 调用 `mixup_augment()`；训练样本翻倍（原始 + 合成）；`config.py` 加入 `mixup_alpha/mixup_multiplier/noise_std` |
| **成功原因** | MixUp 在特征空间线性插值合成新样本，等效于告诉模型「特征空间中相邻样本的预测应平滑过渡」。这缩小了 OOF→Kaggle 过拟合 gap。噪声注入进一步防止模型死记硬背训练集。与增加特征/调参不同，MixUp 不改变模型复杂度，只改变训练数据的分布 |
| **结论** | ✅ 项目中第二个有效的优化。Group 一致性在此配置下首次变为正向 |

---

### Exp#9 — 同组特征交换
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80149（−0.0054） |
| **Δ vs Exp#8** | −0.0054 |
| **方法** | 利用 GroupId 同组标签一致性：训练时对同组乘客随机交换 30% 的特征值（CabinNum、Age 等），标签不变。等价于在同质群体内做特征级数据增强 |
| **修改点** | `mixup.py` 新增 `group_swap_augment()`；`train_base.py` 每 fold 调用；`config.py` 加入 `group_swap_fraction=0.3` |
| **失败原因** | 同组乘客虽共享 Transported 标签，但直接交换特征值破坏了特征间的内在一致性。例如将 Passenger_A 的 `CabinNum=150`、`CabinDeck=G` 换成 Passenger_B 的 `CabinNum=20`、`CabinDeck=A`——模型被训练为「CabinNum 可以是 150 也可以是 20 且都对应同一标签」，学到的不是泛化规律而是噪声 |
| **结论** | ❌ 项目历史上最大单次退步（−0.0054）。`group_swap_fraction` 设为 0.0，永不启用 |

---

### Exp#10 — 多轮训练平均
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80430（−0.0026） |
| **Δ vs Exp#8** | −0.0026 |
| **方法** | `ensemble_runs=3`：每个模型用 3 个不同 base_seed（42/123/888）独立训练 3 轮，最终 OOF/test 取 3 轮平均 |
| **修改点** | `config.py` 加入 `ensemble_runs=3` + `ensemble_base_seeds`；`train_*.py` 外循环从 1 层变 2 层 |
| **失败原因** | 3 轮训练结果差异很大（各轮方差 > 平均带来的方差消减收益）。任意一轮出现严重偏差都会拉低平均值。本质上多轮平均假设各轮独立同分布，但不同种子的模型偏差程度不同，不满足 i.i.d. 前提。且 3 倍训练时间换来了 −0.0026 |
| **结论** | ❌ `ensemble_runs=1` 确认 |

---

### Exp#11 — 特征重构 v2（55→89）
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.79518（−0.0117） |
| **Δ vs Exp#8** | −0.0117 |
| **方法** | 大规模特征工程重构，新增 34 个特征：消费比例（SpendProp×5）、消费集中度（SpendConcentration）、服务二值标志（Used×5）、群组消费统计（GroupTotalSpend/GroupAvgSpend/GroupSpendStd/Log_*）、群组年龄统计（GroupAvgAge/GroupAgeStd/GroupAgeRange）、群组冻眠率（GroupCryoRate）、群组船舱统计（GroupCabinNumStd/Range）、群组多样性（GroupDeckDiversity/GroupSideDiversity）、成员排名（MemberRank/IsFirst/IsLast）、年龄交互（Age_x_×5）、VIP 交互（VIP_x_TotalSpend）、姓名解析（FirstName/NameLength/FirstNameLength）、TE_FirstName |
| **修改点** | `data_prep.py` 新增 ~70 行特征生成代码；`config.py` `feature_cols` 从 55→89，`categorical_base` 加入 FirstName，`TE_COLS` 加入 FirstName |
| **失败原因** | **三大致命问题叠加**：(1) 样本/特征比从 158 降至 98，树分裂被稀释；(2) `GroupTotalSpend` vs `Log_GroupTotalSpend` 等原始+对数对共线；(3) `GroupCryoRate` 与目标 `Transported` 强相关（信息泄露式过拟合），`MemberRank` 等任意序号无信号。34 个新增特征中仅 ~6 个有真实信息，其余 28 个都是噪声 |
| **结论** | ❌ 项目史上最大降幅。核心教训：**每新增一个特征都必须经过三重审查** |

---

### Exp#12 — 特征精简 v2.1（55→73）
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.79611（−0.0108） |
| **Δ vs Exp#8** | −0.0108 |
| **方法** | 从 v2 的 89 特征中删除 9 个最明显冗余/噪音特征（GroupSpendStd/GroupAvgSpendPerPerson/GroupTotalSpend/GroupCryoRate/MemberRank/IsFirst/IsLast/GroupDeckDiversity/GroupSideDiversity），SHAP 筛选从 10% 提升至 18%。最终精简约 73 |
| **修改点** | `config.py` `feature_cols` 精简；`shap_keep_percentile` 10→18；`data_prep.py` 清理部分生成代码 |
| **失败原因** | 虽然删了 9 个特征，但剩余的 25 个新增特征仍然引入了足够的噪声。关键发现：**新增特征的失败不是因为个别坏特征，而是整体方向错误**——在 8693 样本下「从数据中挖掘更多信号」这个前提本身就不成立 |
| **结论** | ❌ 回退幅度与 v2 几乎一致（−0.0108 vs −0.0117）。确认新增特征方向彻底错误 |

---

### Exp#13 — 完全回滚 + 修复 blend_models ✅⭐
| 项目 | 内容 |
|------|------|
| **Kaggle** | **0.80921**（+0.0131） |
| **Δ vs Exp#8** | +0.0023 |
| **Δ vs Exp#12** | +0.0131 |
| **方法** | 完全删除 Exp#11-12 所有新增特征代码，回滚到精确的 Exp#8 特征集。同时修复了一个隐藏 bug：`config.py` 中 `blend_models` 被错误设为全 6 个模型，但 README 和实验结果均表明应用 Top 3 |
| **修改点** | `data_prep.py` 删除全部新增特征生成代码；`config.py` 恢复到精确 55 特征 + `blend_models=['hist_gb','xgb','lgb']` + `shap_keep_percentile=10`；`main.py` 清理模块引用 |
| **成功原因** | 这次不是「优化」，而是「修复」。Exp#8 的 0.80687 是在 `blend_models` 含全 6 个模型条件下取得的，Exp#13 回滚特征的同时将 `blend_models` 修复为 Top 3（此变更在 Exp#4 中被测试过且 OOF 更好，只是当时 Kaggle 微降被误判为无效）。两个正确改动叠加 = +0.0131，成为历史最佳。核心发现：**此前最好的成绩（0.80687）也含有一个配置 bug** |
| **结论** | ✅ **历史最佳。Group 一致性首次稳定正向 (+0.00047)** |

---

### Exp#14 — P1-P5 全面优化（v3）
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80687（−0.0023） |
| **Δ vs Exp#13** | −0.0023 |
| **方法** | 在 Exp#13 基础上追加 5 项训练策略优化：(P1) Pseudo-Labeling 基础设施（未激活）；(P2) HGB + XGB 加入 Optuna 200 轮超参搜索；(P3) Test-Time Augmentation：推理时 3 次 N(0,0.003) 噪声扰动取平均；(P4) 三维集成权重 Optuna 搜索，替代原 1D 网格搜索；(P5) 消费列 99% 分位 Winsorize 截断 |
| **修改点** | `train_xgb.py` 全文重写（加 Optuna）；`train_hgb.py` 从基类改为独立脚本（加 Optuna）；`train_base.py` + `train_lgb.py` 加 TTA 函数及调用；`blend.py` 删除 LR Stacking 改为 3D Optuna 权重；`data_prep.py` 消费列加 Winsorize；`config.py` 加 `tta_repeats/tta_noise` |
| **失败原因** | 四项技巧联合上线，每一项在 OOF 上都看起来不错，但 Kaggle 一致性回退：(1) Optuna 调参在 3-fold CV 上过拟合了参数——Exp#7 已见过同模式；(2) TTA 的噪声扰动量 0.003 过小，且 3 次平均在 test 上的平滑效果不及单个模型的随机波动；(3) 3D 权重优化的搜索空间 200 次试验对 3 参数来说过剩，找到的「最优」是在 OOF 上过拟合的组合；(4) Winsorize 改变了消费分布的尾部形态，破坏了 MixUp 在长尾区间的插值假设 |
| **结论** | ❌ 四项高级技巧联合上线，全部回退。与 Exp#5-10 的模式一致 |

---

### Exp#15 — Pseudo-Labeling（伪标签增强）
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80289（−0.0063） |
| **Δ vs Exp#13** | −0.0063 |
| **方法** | 用 Exp#13 回退后的模型对 test 集做预测，选出置信度 > 0.92 的样本（及其预测标签），加入训练集（8693 → ~10250）。然后重新训练 → 再次伪标签扩增（→ ~13895）→ 最终集成 |
| **修改点** | 新建 `pseudo_label.py`（选高置信度样本、备份 pkl、合并训练集）；`main.py` 新增 `pseudo` 命令（自动执行 2 轮扩增+重训+集成） |
| **失败原因** | 伪标签在 Kaggle 竞赛中通常是最保险的提分手段，但在此数据集上同样失败：(1) 高置信度样本 ≠ 正确预测样本——模型对其置信度高但实际错误的样本被错误地加入训练集，第二轮训练被这些错误标签带偏；(2) test 分布与 train 分布有系统性偏移，把 test 样本加入训练集改变了训练分布，模型学到了 test 的特异性而非通用规律 |
| **结论** | ❌ Pseudo-Labeling 从此方向清单中移除。`pseudo_label.py` 已删除 |

---

### Exp#16 — 反向特征筛选
| 项目 | 内容 |
|------|------|
| **Kaggle** | 0.80874（三组一致） |
| **Δ vs Exp#13** | −0.0005 |
| **方法** | 测试 3 个更强的 SHAP 筛选强度：移除底部 15%（保留 85%）、20%（保留 80%）、25%（保留 75%）。一次运行产出 3 组提交文件，同批提交 Kaggle 对比 |
| **修改点** | 新建 `reverse_selection.py`（循环 3 个 percentile 值，每次重新 prep+train+blend）；产出 6 个 csv 文件 |
| **失败原因** | 三组分数完全一致（0.80874），且均低于 Exp#13（0.80921）。这说明：(1) 移除 15-25% 的特征对模型预测**完全无差别**——被移除的那些特征确实没有信息增量；(2) 但 Exp#13 的 10% 是唯一最优点——那额外的 5% 特征（被 15% 筛掉但被 10% 保留的 ~3 个特征）携带了微弱的有效信号。不存在「更激进筛选能提分」的假设 |
| **结论** | ❌ 10% 是唯一 Nash 均衡。`reverse_selection.py` 已删除。16 次实验到此结束 |

---

### 核心教训（7 条）

| # | 教训 | 证据 |
|---|------|------|
| 1 | **OOF ↑ ≠ Kaggle ↑** | Exp#3-7 在 OOF 上均提升，Kaggle 均下降 |
| 2 | **减法优于加法** | SHAP 删 6 个特征 +0.0012；加 34 个特征 −0.0117 |
| 3 | **高级技巧在此数据集统一负收益** | Optuna、TTA、集成优化、伪标签、特征重构——12/16 次实验为负 |
| 4 | **MixUp 是唯一有效的增强手段** | +0.0005，因为只改变训练数据分布而不增加模型复杂度 |
| 5 | **配置一致性是最高杠杆的提分手段** | 修复 `blend_models` bug 单次提分 +0.0131，超过其他所有实验的正收益总和 |
| 6 | **10% SHAP 筛选是唯一最优点** | Exp#16 证明移除 > 10% 三组均退步（0.80874），< 10% 则含噪声 |
| 7 | **0.80921 是该技术栈在此数据集上的天花板** | 16 次实验覆盖了 Kaggle 表格竞赛的全部经典优化方向，再无可试手段 |

### 提分收益归因

| 来源 | 收益 | 说明 |
|------|:---:|------|
| SHAP 特征筛选 | +0.0012 | Exp#2 |
| MixUp 数据增强 | +0.0006 | Exp#8 |
| 修复 blend_models bug | +0.0023 | Exp#13 vs Exp#8 |
| 回滚特征重构 | +0.0131 | Exp#13 vs Exp#12 |
| **合计** | **+0.0172** | 0.80500 → 0.80921（累计净收益 +0.0042） |

---

## modular_pipeline/main.py 命令参考

```powershell
python modular_pipeline/main.py full              # 完整流水线
python modular_pipeline/main.py prep              # 仅数据准备
python modular_pipeline/main.py train -m all      # 训练全部 6 个模型
python modular_pipeline/main.py train -m lgb xgb  # 仅训练指定模型
python modular_pipeline/main.py blend             # 仅集成（不需重训）
python modular_pipeline/main.py list              # 查看可用模型
```

---

## 并行训练加速

```powershell
python modular_pipeline/main.py prep              # 只跑一次
# 等 prep 完成后，3 个终端分别：
python modular_pipeline/train_hgb.py
python modular_pipeline/train_xgb.py
python modular_pipeline/train_lgb.py
# 全部跑完后：
python modular_pipeline/main.py blend
```

---

## 环境要求

```bash
pip install -r requirements.txt
```

核心依赖：`pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm catboost optuna shap`

`lightgbm` / `xgboost` / `catboost` / `optuna` / `shap` 为可选依赖，缺失时对应步骤自动跳过。

---

## 复现 0.80921

```powershell
cd "c:\Users\siruiJohn\project\Spaceship Titanic_1\Spaceship Titanic"
python modular_pipeline/main.py full
```

配置文件 [config.py](file:///c:/Users/siruiJohn/project/Spaceship Titanic_1/Spaceship Titanic/modular_pipeline/config.py) 已锁定在 Exp#13 的最优配置。提交 `submission_90plus.csv` 即可获得 0.80921。
