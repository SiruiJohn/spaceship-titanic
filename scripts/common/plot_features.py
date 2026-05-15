import os
import ast
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from pathlib import Path

sns.set(style="whitegrid")
ROOT = Path(__file__).resolve().parents[2]
out_dir = ROOT / "results" / "eda" / "figs"
out_dir.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(ROOT / "data" / "processed" / "train_processed.csv")
y = train_df["Transported"].astype(int)
X = train_df.drop(columns=["Transported", "PassengerId"])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
bin_cols = []
for c in ["CryoSleep", "VIP", "IsAlone", "AnySpend", "CryoSleepSpendConflict", "CabinBin"]:
    if c in X.columns:
        bin_cols.append(c)
cat_rate_cols = []
for c in ["CabinBin", "IsAlone", "CryoSleepSpendConflict", "CryoSleep", "VIP"]:
    if c in X.columns:
        cat_rate_cols.append(c)

plt.figure(figsize=(6, 4))
sns.countplot(x=y.map({0: "False", 1: "True"}))
plt.title("Transported Distribution")
plt.xlabel("Transported")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(out_dir / "target_distribution.png")
plt.close()

for c in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(train_df[c], bins=40, kde=True)
    plt.title(f"Distribution: {c}")
    plt.tight_layout()
    plt.savefig(out_dir / f"dist_{c}.png")
    plt.close()

for c in num_cols:
    if c in X.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=y.map({0: "False", 1: "True"}), y=train_df[c])
        plt.title(f"{c} by Transported")
        plt.tight_layout()
        plt.savefig(out_dir / f"{c}_by_target.png")
        plt.close()

for c in cat_rate_cols:
    if c in X.columns:
        df_tmp = pd.DataFrame({c: train_df[c], "Transported": y})
        grp = df_tmp.groupby(c)["Transported"].mean().reset_index()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=c, y="Transported", data=grp)
        plt.title(f"Transported Rate by {c}")
        plt.tight_layout()
        plt.savefig(out_dir / f"rate_by_{c}.png")
        plt.close()

corr_cols = [c for c in num_cols if c in train_df.columns]
if "Transported" not in corr_cols:
    corr_cols = corr_cols + ["Transported"]
corr = train_df[corr_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="vlag", center=0, square=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(out_dir / "corr_heatmap.png")
plt.close()

try:
    with open(ROOT / "results" / "params" / "best_lgbm_params.txt", "r", encoding="utf-8") as f:
        lgbm_params = ast.literal_eval(f.read().strip())
except Exception:
    lgbm_params = {
        "n_estimators": 180,
        "learning_rate": 0.09598232320208959,
        "num_leaves": 254,
        "max_depth": 7,
        "min_child_samples": 82,
        "subsample": 0.8773174846493815,
        "colsample_bytree": 0.5880046856820298,
        "reg_alpha": 6.237708648779934e-08,
        "reg_lambda": 1.1574035898665557e-08,
    }

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = lgb.LGBMClassifier(
    **lgbm_params,
    objective="binary",
    metric="binary_logloss",
    verbosity=-1,
    random_state=42
)
model.fit(X_train, y_train)
imp = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
top = imp.head(20)
plt.figure(figsize=(8, 10))
sns.barplot(y="feature", x="importance", data=top, orient="h")
plt.title("LightGBM Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig(out_dir / "lgb_feature_importance_top20.png")
plt.close()
