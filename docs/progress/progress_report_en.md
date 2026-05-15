<br />

## 5.1 Progress Report

### (1) Abstract

This project targets Kaggle’s “Spaceship Titanic” binary classification task (predicting whether a passenger was transported). We have completed an end-to-end pipeline including EDA, data preprocessing, feature engineering, hyperparameter tuning for tree-based models, multi-model ensembling, and an additional deep-learning base learner (MLP with embeddings) to increase ensemble diversity. Our current best approach is 5-fold OOF stacking (LightGBM + XGBoost + CatBoost + MLP-DL with a LogisticRegression meta-learner), achieving an offline OOF accuracy of 0.815714 (threshold 0.535). The ready-to-submit file is [submission\_v8.csv](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/submission_v8.csv) and the recorded results are in [results\_v8.txt](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/results_v8.txt).

### (2) Introduction

- Goal: predict Transported (True/False) from passenger attributes and spending-related tabular features.
- Key challenges: many missing values, multiple categorical fields, complex feature interactions, and the risk of overfitting to a single train/validation split.
- Approach: use GBDT family models as strong baselines, improve separability with feature engineering, and adopt OOF ensembling to improve generalization; add a DL model as a complementary learner to raise the ensemble ceiling.

#### Data and Evaluation

- Data source: Kaggle-provided train/test CSV files (PassengerId as the key, Transported as the label).
- Metric: Accuracy. To reduce split variance, we generate OOF predictions with StratifiedKFold and train a meta-learner on OOF outputs.

### (3) Related Work / Existing Techniques

- GBDT models (LightGBM / XGBoost / CatBoost): common strong baselines for Kaggle tabular tasks, effective at non-linearities and feature interactions.
- Feature engineering: Cabin parsing, group-size features, spending aggregation and log transforms are known to be effective for this competition.
- Ensembling:
  - Voting / weighted blending: simple probability aggregation, but sensitive to validation splitting.
  - OOF stacking: train a meta-learner on out-of-fold predictions, usually more robust and closer to leaderboard performance.
- Deep learning (MLP + embeddings): learns dense representations for categorical variables; often not superior as a standalone model but helpful as an ensemble member.

### (4) Methodology (Work Completed So Far)

#### Data preprocessing and feature engineering

- Cabin parsing: split Cabin into Deck / Cabin\_Num / Side; add CabinBin via quantile binning of Cabin\_Num.
- Group features: derive GroupId from PassengerId; compute GroupSize and IsAlone.
- Spending features:
  - TotalSpend and LogTotalSpend
  - AnySpend (any spending indicator) and NumSpendCats (number of spending categories)
  - LogRoomService / LogFoodCourt / LogShoppingMall / LogSpa / LogVRDeck (log1p per spend column)
  - CryoSleepSpendConflict (CryoSleep=1 with TotalSpend>0)
- Encoding:
  - Binary: CryoSleep and VIP mapped to 0/1
  - Categorical: one-hot encode HomePlanet/Destination/Deck/Side (drop\_first=True)
- Missing values: median imputation for numeric and mode imputation for categorical to keep train/test processing consistent.
- Reference implementation: [preprocess.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/preprocess.py)

#### Modeling and tuning

- LightGBM: tuned with Optuna and persisted parameters (best\_lgbm\_params.txt).
  - Reference: [tune\_lgbm.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/tune_lgbm.py)
- CatBoost / XGBoost: included as strong base learners.
- Ensemble strategy: evolved from single-split weighted blending to 5-fold OOF stacking (meta=LogisticRegression).
  - Reference: [train\_model.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/train_model.py)

#### Training and ensembling details (current v8)

- Base level: LightGBM, XGBoost, CatBoost (multi-seed), producing OOF probabilities and test probabilities per model.
- Meta level: LogisticRegression trained on the OOF probability matrix.
- Threshold search: scan thresholds in \[0.3, 0.7] (step 0.005) on OOF predictions.
- Logging: OOF accuracy, threshold, feature list and meta coefficients are stored in [results\_v8.txt](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/results_v8.txt).

#### Deep learning base learner (diversity boost)

- DL preprocessing: label-encode categorical features and standard-scale numeric features.
  - Reference: [preprocess\_dl.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/preprocess_dl.py)
- DL model: PyTorch MLP with embeddings; 5-fold training exports OOF and test predictions (oof\_preds\_dl.npy / test\_preds\_dl.npy), used as an additional stacking feature.
  - Reference: [train\_dl.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/train_dl.py)
- Artifacts:
  - dl\_preproc\_artifacts.pkl
  - oof\_preds\_dl.npy, test\_preds\_dl.npy

### (5) Current Progress and Current Kaggle Ranking

#### Current progress

- Environment and dependencies are ready; preprocessing, feature engineering and visualization have been completed.
- Multi-model OOF stacking is implemented; MLP-DL predictions are integrated as an additional feature.
- Visualization script: [plot\_features.py](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/plot_features.py)
- Figures directory: figs (correlation heatmap, distributions, target-conditioned boxplots, feature importance).

#### Visualization outputs (examples)

- Correlation heatmap: [corr\_heatmap.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/corr_heatmap.png)
- LightGBM top-20 feature importance: [lgb\_feature\_importance\_top20.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/lgb_feature_importance_top20.png)
- Target distribution: [target\_distribution.png](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/figs/target_distribution.png)

#### Current best offline result (OOF)

- Stacking (lgb/xgb/cat + mlp\_dl, 5-fold base + 5-fold meta):
  - OOF Meta Accuracy: 0.815714
  - Meta Threshold: 0.535
  - Details: [results\_v8.txt](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/results_v8.txt)
- Main submission file:
  - [submission\_v8.csv](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/submission_v8.csv)

#### Selected leaderboard results (Public, selected submissions)

- Best so far (rule-based consensus, v3 anchored; 13 targeted 0→1 flips): 0.80874
  - [submission\_v13\_v3\_v8\_v5\_v2\_w3\_1\_1\_1\_thr0.416667\_d13.csv](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/blends_v13/submission_v13_v3_v8_v5_v2_w3_1_1_1_thr0.416667_d13.csv)
- Blend baseline (v3+v8+v5, rank-weighted vote): 0.80827
  - [submission\_v11\_k3\_lb\_rank\_v3\_v8\_v5.csv](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/blends/submission_v11_k3_lb_rank_v3_v8_v5.csv)
- Majority vote (v3/v8/v6/v5/v2): 0.80640
  - [submission\_v10\_majority.csv](file:///c:/Users/siruiJohn/project/Spaceship%20Titanic/submission_v10_majority.csv)

#### Base learner performance (from results\_v8)

- cat\_s202 OOF Accuracy\@0.5: 0.815829
- cat\_s42 OOF Accuracy\@0.5: 0.814218
- xgb\_s202 OOF Accuracy\@0.5: 0.812033
- xgb\_s42 OOF Accuracy\@0.5: 0.810192
- lgb\_s202 OOF Accuracy\@0.5: 0.807201
- lgb\_s42 OOF Accuracy\@0.5: 0.806051
- mlp\_dl OOF Accuracy\@0.5: 0.806626

#### Ensemble weight interpretation (meta coefficients)

- The meta coefficients suggest CatBoost and MLP-DL contribute strongly (larger coefficients), indicating that the DL learner adds complementary signal; XGBoost/LightGBM provide smaller but still positive marginal gains.

#### Current Kaggle ranking 

- Public Leaderboard Rank:217
- Public Score (best so far): 0.80874

### (6) Next Steps and Plan for Completion

- Pseudo-labeling:
  - Add high-confidence test predictions (e.g., p>=0.98 or p<=0.02) to the training set and retrain the OOF ensemble.
  - Risk control: keep thresholds strict to minimize noisy labels.
  - Consistency: if the training set is expanded, regenerate DL OOF predictions aligned to the new training rows (or temporarily skip DL in the pseudo-label stage).
- Stronger regularization for the meta-learner:
  - Replace LogisticRegression with Ridge/ElasticNet-style regularization to reduce meta overfitting.
  - Use nested CV (OOF-of-OOF) for the meta layer to further reduce leakage risk.
- Increase model diversity:
  - More seeds and slightly different sampling/hyperparameters (subsample/colsample/depth).
  - Carefully add complementary learners (e.g., linear models or ExtraTrees) without introducing weak models that drag performance.
- Additional feature engineering:
  - Family features from Name (e.g., LastName group size and family-level spending statistics).
  - Cabin interaction features (Deck+Side and crosses with CabinBin).
  - Group-level aggregate features (group spending mean/max, any spending indicator within group).

#### Reproducibility and deliverables

- Suggested one-click flow: preprocess → train\_dl → train\_model → submission + results.
- Key artifacts:
  - Preprocessed: train\_processed.csv / test\_processed.csv
  - DL preprocessed: train\_processed\_dl.csv / test\_processed\_dl.csv
  - Ensemble output: submission\_v8.csv / results\_v8.txt

