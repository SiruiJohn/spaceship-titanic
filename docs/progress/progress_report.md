# 5. Submission Requirements

## 5.1 Progress Report

### (1) Abstract
本项目针对 Kaggle “Spaceship Titanic” 二分类任务（预测乘客是否被传送 Transported），完成了从数据探索、数据预处理、特征工程、模型调参到多模型集成的端到端流程，并进一步引入深度学习（MLP + Embedding）作为差异化基学习器增强融合效果。当前最佳方案为基于 5-fold OOF 的 Stacking（LightGBM + XGBoost + CatBoost + MLP-DL，meta=LogisticRegression），离线 OOF 指标达到 0.815714（阈值 0.535），并已生成可提交文件 [submission_v8.csv](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/submission_v8.csv) 与结果记录 [results_v8.txt](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/results_v8.txt)。

### (2) Introduction
- 任务目标：利用乘客属性与消费等表格特征，预测 Transported（True/False）。
- 核心挑战：缺失值较多、类别字段较多、特征交互复杂；同时需要避免在单次切分验证集上过拟合。
- 总体思路：以 GBDT 系列模型作为强基线，通过特征工程提升可分性，再用 OOF 集成提高泛化能力；深度学习作为差异化基学习器加入融合以提升上限。

#### 数据与评测说明
- 数据来源：Kaggle 官方提供的 train/test 表格数据（PassengerId 为主键，Transported 为标签）。
- 评测方式：以准确率（Accuracy）为核心指标；为降低随机切分带来的方差，使用 StratifiedKFold 产生 OOF 预测并训练 meta-learner。

### (3) Related Work / Existing Techniques
- GBDT 系列（LightGBM / XGBoost / CatBoost）：Kaggle 表格任务的常用强基线，擅长捕捉非线性关系与特征交互。
- 特征工程：Cabin 拆分、组团规模、消费聚合与对数变换等对本题有效。
- 集成学习：
  - Voting / Weighted blending：对多个模型概率进行简单融合，易实现但对验证集切分敏感。
  - OOF Stacking：用 K-fold 产生 out-of-fold 预测训练二层模型（meta-learner），更稳健、更接近线上表现。
- 深度学习（MLP + Embedding）：对类别特征学习稠密表示，通常单模型不一定超越 GBDT，但作为融合成员可提供互补性。

### (4) Methodology (Work Completed So Far)
#### 数据预处理与特征工程
- Cabin 拆分：将 Cabin 解析为 Deck / Cabin_Num / Side，并构造 CabinBin（Cabin_Num 分位数分桶）。
- 组团特征：从 PassengerId 构造 GroupId，得到 GroupSize 与 IsAlone（是否独自出行）。
- 消费特征：
  - TotalSpend、LogTotalSpend
  - AnySpend（是否有任何消费）、NumSpendCats（消费项数）
  - LogRoomService / LogFoodCourt / LogShoppingMall / LogSpa / LogVRDeck（各消费项对数）
  - CryoSleepSpendConflict（CryoSleep=1 且 TotalSpend>0）
- 编码策略：
  - 二值列：CryoSleep、VIP 转换为 0/1
  - 类别列：HomePlanet/Destination/Deck/Side 做 One-Hot（drop_first=True）
- 缺失值填补：数值列用中位数、类别列用众数（保证 train/test 一致处理）。
- 参考实现：[preprocess.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/preprocess.py)

#### 模型与调参
- LightGBM：使用 Optuna 进行超参数搜索并固化参数（best_lgbm_params.txt）。
  - 参考脚本：[tune_lgbm.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/tune_lgbm.py)
- CatBoost / XGBoost：作为强基学习器加入对比与融合。
- 融合策略：从单次验证集加权融合迭代到 5-fold OOF Stacking（meta=LogisticRegression）。
  - 参考脚本：[train_model.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/train_model.py)

#### 训练与融合细节（当前 v8）
- Base level（一级模型）：LightGBM、XGBoost、CatBoost（多 seed），输出每个模型的 OOF 概率与 test 概率。
- Meta level（二级模型）：LogisticRegression，对一级模型 OOF 概率进行二次学习（stack_method 等价于 “使用 predict_proba 作为特征”）。
- 阈值搜索：在 OOF 上遍历阈值区间 [0.3, 0.7]（步长 0.005）以确定最终分类阈值。
- 结果记录：在 [results_v8.txt](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/results_v8.txt) 中保存 OOF 指标、阈值、特征列表与 meta 系数。

#### 深度学习基学习器（差异化增强）
- 预处理：对类别特征做 Label Encoding，对数值特征做 Standard Scaling。
  - 参考脚本：[preprocess_dl.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/preprocess_dl.py)
- 模型：PyTorch MLP + Embedding（类别特征嵌入），5-fold 训练导出 OOF 与测试预测（oof_preds_dl.npy / test_preds_dl.npy），作为融合额外特征加入。
  - 参考脚本：[train_dl.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/train_dl.py)
- 工件与输出：
  - 预处理工件：dl_preproc_artifacts.pkl
  - OOF/test 预测：oof_preds_dl.npy、test_preds_dl.npy

### (5) Current Progress and Current Kaggle Ranking
#### 当前进展
- 环境与依赖已就绪，数据预处理、特征工程与可视化图表已完成。
- 已完成多模型训练与 OOF Stacking，并引入 MLP-DL 作为融合特征。
- 特征可视化脚本：[plot_features.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/plot_features.py)
- 图表输出目录：figs（例如相关热力图、分布图、按目标分组箱线图、特征重要性 Top20）。

#### 可视化产出（示例）
- 相关热力图：[corr_heatmap.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/corr_heatmap.png)
- LightGBM 特征重要性 Top20：[lgb_feature_importance_top20.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/lgb_feature_importance_top20.png)
- 目标分布：[target_distribution.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/target_distribution.png)

#### 当前离线最优结果（OOF）
- Stacking（lgb/xgb/cat + mlp_dl，5-fold base + 5-fold meta）：
  - OOF Meta Accuracy：0.815714
  - Meta Threshold：0.535
  - 详情记录：[results_v8.txt](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/results_v8.txt)
- 主要提交文件：
  - [submission_v8.csv](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/submission_v8.csv)

#### 基学习器表现（来自 results_v8）
- cat_s202 OOF Accuracy@0.5：0.815829
- cat_s42 OOF Accuracy@0.5：0.814218
- xgb_s202 OOF Accuracy@0.5：0.812033
- xgb_s42 OOF Accuracy@0.5：0.810192
- lgb_s202 OOF Accuracy@0.5：0.807201
- lgb_s42 OOF Accuracy@0.5：0.806051
- mlp_dl OOF Accuracy@0.5：0.806626

#### 融合权重解读（meta 系数，来自 results_v8）
- meta 系数显示 CatBoost 与 MLP-DL 在融合中占比较高（系数更大），说明 DL 为融合提供了有用的互补信号；XGBoost/LGBM 系数较小但仍提供边际增益。

#### Current Kaggle Ranking（待补充）
- Public Leaderboard Rank：____ / ____
- Public Score：____
- Private Leaderboard Rank：____ / ____（如已公布）
- Private Score：____（如已公布）

### (6) Next Steps and Plan for Completion
- 伪标签（Pseudo-labeling）：
  - 基于测试集高置信预测扩充训练集后重训并再做 OOF 融合。
  - 风险控制：仅选择极高置信度样本（例如 p>=0.98 或 p<=0.02）以减少噪声标签引入。
  - 一致性问题：若扩充训练集，需重新生成与新训练集对齐的 DL OOF（或在伪标签阶段暂不引入 DL）。
- 更强正则化的 meta-learner：
  - 用 Ridge / ElasticNet 替代 LogisticRegression，以抑制二层过拟合并提升稳定性。
  - 对 meta 层使用嵌套 CV（OOF-of-OOF）进一步降低信息泄露风险。
- 增强模型多样性：
  - 增加 seeds 数量与差异化参数（subsample/colsample/深度等）。
  - 引入不同类型基学习器（如 ExtraTrees、线性模型）以增加互补性（谨慎选择，避免弱模型拖累）。
- 进一步特征工程：
  - 姓名/家庭特征（LastName 家庭规模、家庭消费统计）。
  - Cabin 组合特征（Deck+Side、CabinBin 与其他字段交叉）。
  - 组团统计特征（组内均值/最大/是否存在消费等）。

#### 可复现性与交付物
- 一键运行流程建议：preprocess → train_dl → train_model → 输出 submission + results。
- 关键产出文件清单：
  - 预处理：train_processed.csv / test_processed.csv
  - DL 预处理：train_processed_dl.csv / test_processed_dl.csv
  - 融合结果：submission_v8.csv / results_v8.txt
