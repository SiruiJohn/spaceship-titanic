import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_model import make_cat, load_data

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
PSEUDO_DIR = ROOT / "data" / "pseudo"
PSEUDO_DIR.mkdir(parents=True, exist_ok=True)

def generate_pseudo_labels(threshold=0.95):
    print(f"Generating pseudo labels with fast CatBoost model (threshold={threshold})...", flush=True)
    
    X, y, X_test, test_df = load_data(use_pseudo=False)
    
    # Train a single strong CatBoost model on full train data
    print("Training base model for pseudo-labeling...", flush=True)
    model = make_cat(42)
    model.fit(X, y)
    
    # Predict probabilities on test set
    print("Predicting on test set...", flush=True)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Identify high confidence samples
    is_true = probs >= threshold
    is_false = probs <= (1 - threshold)
    
    confident_idx = np.where(is_true | is_false)[0]
    print(f"Found {len(confident_idx)} confident samples out of {len(test_df)}", flush=True)
    
    if len(confident_idx) == 0:
        print("No confident samples found. Skipping pseudo-labeling.", flush=True)
        return False
    
    # Create pseudo-labeled data
    pseudo_df = test_df.iloc[confident_idx].copy()
    pseudo_df['Transported'] = (probs[confident_idx] >= 0.5).astype(int)
    
    # Combine with original training data
    train_df = pd.read_csv(PROC_DIR / "train_processed.csv")
    combined_df = pd.concat([train_df, pseudo_df], axis=0, ignore_index=True)
    
    # Save
    out_path = PSEUDO_DIR / "train_pseudo_labeled.csv"
    combined_df.to_csv(out_path, index=False)
    print(f"Saved combined training data to {out_path} (Total size: {len(combined_df)})", flush=True)
    return True

if __name__ == "__main__":
    generate_pseudo_labels(threshold=0.96)
