# Kaggle Spaceship Titanic 竞赛优化报告 (最终版)

## 1. 优化历程回顾

我们通过多轮迭代，不断提升模型的预测能力。

### 第一阶段：基线模型
- **Random Forest**: 0.7809
- **XGBoost**: 0.7913

### 第二阶段：特征工程与初步调优
- **改进**: 引入 `GroupSize`, `Cabin_Num`, `LogTotalSpend` 等特征。
- **XGBoost (调优后)**: 0.8022 (+1.09%)
- **LightGBM**: 0.7993
- **Voting Classifier**: 0.7987

### 第三阶段：深度调优与引入新模型 (本阶段)
- **LightGBM (Optuna 调优)**: 使用 Optuna 对 LightGBM 进行了超参数搜索。
- **CatBoost**: 引入了 CatBoost 模型，该模型通常对类别特征处理更优。

## 2. 最终实验结果 (验证集准确率)

| 模型 | 准确率 | 备注 |
| :--- | :--- | :--- |
| Random Forest | 0.7890 | 基线 |
| XGBoost | 0.8022 | 初步调优后 |
| **LightGBM (Tuned)** | **0.8108** | **Optuna 深度调优 (+0.86% over XGBoost)** |
| **CatBoost** | **0.8120** | **新引入模型 (最佳单模型)** |
| Voting Classifier | 0.8056 | Soft Voting (RF+XGB+LGB+Cat) |

**最佳模型**: **CatBoost** (0.8120)，相比最初的基线 (0.7809) 提升了 **3.11%**。

## 3. 提交文件

- **`submission_v3.csv`**: 基于全量数据重新训练的最佳 **CatBoost** 模型。
- **`submission_v2.csv`**: 基于 XGBoost 模型。

## 4. 总结与建议

- **LightGBM 调优有效**: Optuna 成功找到了一组参数，使 LightGBM 的性能从 0.7993 提升到 0.8108，超过了之前的 XGBoost。
- **CatBoost 表现强劲**: 无需大量调参，CatBoost 直接达到了 0.8120 的最高分，证明其在处理此类混合数据（数值+类别）时的优势。
- **融合策略**: 目前简单的 Soft Voting (0.8056) 没有超过最佳单模型。这可能是因为 Random Forest (0.7890) 相对较弱，拉低了整体表现。
    - **建议**: 尝试去掉 Random Forest，仅融合 XGBoost, LightGBM (Tuned) 和 CatBoost。
    - **建议**: 使用 Stacking，以 CatBoost 或逻辑回归为元学习器。

## 5. 关键代码参考
- [tune_lgbm.py](file:///c%3A/Users/siruiJohn/project/Spaceship%20Titanic/tune_lgbm.py): Optuna 调参脚本。
- [train_model.py](file:///c%3A/Users/siruiJohn/project/Spaceship%20Titanic/train_model.py): 最终模型训练与集成脚本。
