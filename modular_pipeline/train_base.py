import pickle, warnings, sys
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG


def load_processed_data():
    with open(CFG.processed_data_file, 'rb') as f:
        return pickle.load(f)


def train_model(model_name, build_model_fn, needs_raw=False, needs_cat=True):
    data = load_processed_data()
    X_num = data['X_train_num']
    X_te_num = data['X_test_num']
    X_raw = data['X_train_raw']
    X_te_raw = data['X_test_raw']
    y = data['y']
    cat_indices = [X_raw.columns.get_loc(c) for c in data['CATEGORICAL_COLS'] if c in X_raw.columns]

    oof_probs = np.zeros(len(y))
    oof_counts = np.zeros(len(y))
    test_preds = []

    total = len(CFG.random_seeds) * CFG.n_splits
    fold_num = 0

    for seed in CFG.random_seeds:
        cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=seed)
        for fi, (tr_idx, val_idx) in enumerate(cv.split(X_num, y), 1):
            fold_num += 1
            tr_num = X_num.iloc[tr_idx]
            vl_num = X_num.iloc[val_idx]
            ytr = y.iloc[tr_idx]
            yvl = y.iloc[val_idx]

            model = build_model_fn(seed, fi)

            if needs_raw:
                tr_raw = X_raw.iloc[tr_idx].copy().astype(str)
                vl_raw = X_raw.iloc[val_idx].copy().astype(str)
                te_raw = X_te_raw.copy().astype(str)
                model.fit(tr_raw, ytr, cat_features=cat_indices, verbose=False)
                vp = model.predict_proba(vl_raw)[:, 1]
                tp = model.predict_proba(te_raw)[:, 1]
            else:
                model.fit(tr_num, ytr)
                vp = model.predict_proba(vl_num)[:, 1]
                tp = model.predict_proba(X_te_num)[:, 1]

            oof_probs[val_idx] += vp
            oof_counts[val_idx] += 1
            test_preds.append(tp)

            acc = accuracy_score(yvl, vp >= 0.5)
            print(f'  [{model_name}] {fold_num:2d}/{total} | s={seed} f={fi} | acc={acc:.4f}')

    oof_probs /= np.maximum(oof_counts, 1)
    avg_test = np.mean(test_preds, axis=0)

    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = CFG.output_dir / f'oof_{model_name}.npy'
    test_file = CFG.output_dir / f'test_{model_name}.npy'
    np.save(output_file, oof_probs)
    np.save(test_file, avg_test)
    print(f'\n[OK] {model_name} saved -> {output_file}, {test_file}')
