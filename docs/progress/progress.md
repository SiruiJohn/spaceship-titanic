# 任务进展与特征可视化清单

## 任务进展
- 环境与依赖已就绪（pandas/numpy/sklearn/seaborn/xgboost/lightgbm/catboost/optuna）
- 数据探索与预处理完成，新增特征：IsAlone、AnySpend、NumSpendCats、Log 各消费项、CryoSleepSpendConflict、CabinBin 等
- 模型基线与调优：
  - Random Forest: 0.7890
  - XGBoost: 0.8022
  - LightGBM（Optuna 调优）: 0.8108
  - CatBoost: 0.8120
  - 加权融合（验证集网格搜索）: 0.8154
  - OOF 融合（5-fold + 多 seed + Logistic 回归）: 0.8138（精简 seed 版本日志已生成）
- 提交文件：
  - submission_v5.csv（网格加权）
  - submission_v6.csv（OOF 融合）
  - submission_v7.csv（OOF 融合 + meta 层嵌套CV）

## 特征可视化
- 目标分布：[target_distribution.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/target_distribution.png)
- 数值分布（示例）：
  - [dist_Age.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/dist_Age.png)
  - [dist_LogTotalSpend.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/dist_LogTotalSpend.png)
  - [dist_NumSpendCats.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/dist_NumSpendCats.png)
  - [dist_CabinBin.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/dist_CabinBin.png)
- 特征与目标关系（示例）：
  - [Age_by_target.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/Age_by_target.png)
  - [RoomService_by_target.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/RoomService_by_target.png)
  - [VIP_by_target.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/VIP_by_target.png)

## 代码参考
- 特征工程：[preprocess.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/preprocess.py)
- 可视化脚本：[plot_features.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/plot_features.py)
- 融合训练脚本：[train_model.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/train_model.py)
