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
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


def load_processed_data():
    with open(CFG.processed_data_file, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__' and XGB_AVAILABLE:
    data = load_processed_data()
    X_num = data['X_train_num']
    X_te_num = data['X_test_num']
    y = data['y']

    oof_probs = np.zeros(len(y))
    oof_counts = np.zeros(len(y))
    test_preds = []

    total = len(CFG.random_seeds) * CFG.n_splits
    fold_num = 0

    for seed in CFG.random_seeds:
        cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=seed)
        for fi, (tr_idx, val_idx) in enumerate(cv.split(X_num, y), 1):
            fold_num += 1
            model = xgb.XGBClassifier(
                n_estimators=400, max_depth=6,
                learning_rate=0.03, subsample=0.85, colsample_bytree=0.80,
                reg_alpha=0.05, reg_lambda=1.0,
                min_child_weight=3, objective='binary:logistic',
                eval_metric='logloss', tree_method='hist',
                random_state=seed * 10 + fi, n_jobs=4,
            )
            model.fit(X_num.iloc[tr_idx], y.iloc[tr_idx])
            vp = model.predict_proba(X_num.iloc[val_idx])[:, 1]
            tp = model.predict_proba(X_te_num)[:, 1]

            oof_probs[val_idx] += vp
            oof_counts[val_idx] += 1
            test_preds.append(tp)

            acc = accuracy_score(y.iloc[val_idx], vp >= 0.5)
            print(f'  [xgb] {fold_num:2d}/{total} | s={seed} f={fi} | acc={acc:.4f}')

    oof_probs /= np.maximum(oof_counts, 1)
    avg_test = np.mean(test_preds, axis=0)

    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(CFG.output_dir / 'oof_xgb.npy', oof_probs)
    np.save(CFG.output_dir / 'test_xgb.npy', avg_test)
    print(f'\n[OK] xgb saved -> oof_xgb.npy, test_xgb.npy')
elif not XGB_AVAILABLE:
    print('[WARN] XGBoost not available -- skipping.')
