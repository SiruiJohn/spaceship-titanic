import os
import pandas as pd
import numpy as np
import ast
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from pathlib import Path

def make_cat(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=4.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        verbose=0,
        random_seed=seed,
    )

print("Starting pseudo-labeling...")
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
PSEUDO_DIR = ROOT / "data" / "pseudo"
PSEUDO_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(PROC_DIR / "train_processed.csv")
test_df = pd.read_csv(PROC_DIR / "test_processed.csv")

X = train_df.drop(columns=["Transported", "PassengerId"])
y = train_df["Transported"].astype(int).to_numpy()
X_test = test_df.drop(columns=["PassengerId"])
X_test = X_test[X.columns]

model = make_cat(42)
print("Training model...")
model.fit(X, y)
print("Predicting...")
probs = model.predict_proba(X_test)[:, 1]

threshold = 0.96
is_true = probs >= threshold
is_false = probs <= (1 - threshold)
confident_idx = np.where(is_true | is_false)[0]
print(f"Found {len(confident_idx)} confident samples.")

pseudo_df = test_df.iloc[confident_idx].copy()
pseudo_df['Transported'] = (probs[confident_idx] >= 0.5).astype(int)
combined_df = pd.concat([train_df, pseudo_df], axis=0, ignore_index=True)
out_path = PSEUDO_DIR / "train_pseudo_labeled.csv"
combined_df.to_csv(out_path, index=False)
print(f"Saved to {out_path}. Size: {len(combined_df)}")
