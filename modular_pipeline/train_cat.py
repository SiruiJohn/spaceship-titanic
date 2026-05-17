import pickle, warnings, sys
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG

try:
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except Exception:
    CAT_AVAILABLE = False


def load_processed_data():
    with open(CFG.processed_data_file, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__' and CAT_AVAILABLE:
    data = load_processed_data()
    X_num = data['X_train_num']
    X_te_num = data['X_test_num']
    X_raw = data['X_train_raw']
    X_te_raw = data['X_test_raw']
    y = data['y']

    if 'feature_subsets' in data and data['feature_subsets']:
        cols = data['feature_subsets'].get('cat')
        if cols is not None:
            X_num = X_num[cols]
            X_te_num = X_te_num[cols]
            X_raw = X_raw[cols]
            X_te_raw = X_te_raw[cols]
            data['CATEGORICAL_COLS'] = [c for c in data['CATEGORICAL_COLS'] if c in cols]

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
            model = CatBoostClassifier(
                iterations=500, depth=7,
                learning_rate=0.03, l2_leaf_reg=4.0,
                loss_function='Logloss', random_seed=seed * 10 + fi,
                verbose=False, allow_writing_files=False,
            )
            tr_raw = X_raw.iloc[tr_idx].copy().astype(str)
            vl_raw = X_raw.iloc[val_idx].copy().astype(str)
            te_raw = X_te_raw.copy().astype(str)

            model.fit(tr_raw, y.iloc[tr_idx], cat_features=cat_indices, verbose=False)
            vp = model.predict_proba(vl_raw)[:, 1]
            tp = model.predict_proba(te_raw)[:, 1]

            oof_probs[val_idx] += vp
            oof_counts[val_idx] += 1
            test_preds.append(tp)

            acc = accuracy_score(y.iloc[val_idx], vp >= 0.5)
            print(f'  [cat] {fold_num:2d}/{total} | s={seed} f={fi} | acc={acc:.4f}')

    oof_probs /= np.maximum(oof_counts, 1)
    avg_test = np.mean(test_preds, axis=0)

    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(CFG.output_dir / 'oof_cat.npy', oof_probs)
    np.save(CFG.output_dir / 'test_cat.npy', avg_test)
    print(f'\n[OK] cat saved -> oof_cat.npy, test_cat.npy')
elif not CAT_AVAILABLE:
    print('[WARN] CatBoost not available -- skipping.')
