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
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None
    OPTUNA_AVAILABLE = False


def load_processed_data():
    with open(CFG.processed_data_file, 'rb') as f:
        return pickle.load(f)


def find_best_params(X_num, y):
    best_params = {
        'n_estimators': 450, 'learning_rate': 0.03, 'num_leaves': 31,
        'subsample': 0.85, 'colsample_bytree': 0.80, 'min_child_samples': 18,
        'reg_alpha': 0.05, 'reg_lambda': 1.0,
    }
    if not OPTUNA_AVAILABLE:
        print('[INFO] Optuna not available -- using default LGB params.')
        return best_params

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
            'verbosity': -1,
        }
        cv = StratifiedKFold(n_splits=CFG.optuna_cv_folds, shuffle=True, random_state=42)
        accs = []
        for tr_idx, val_idx in cv.split(X_num, y):
            model = lgb.LGBMClassifier(**params, random_state=42)
            model.fit(X_num.iloc[tr_idx], y.iloc[tr_idx])
            proba = model.predict_proba(X_num.iloc[val_idx])[:, 1]
            accs.append(accuracy_score(y.iloc[val_idx], proba >= 0.5))
        return np.mean(accs)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=CFG.optuna_trials, show_progress_bar=True)

    best_params.update(study.best_params)
    print(f'\n[OK] Optuna finished  ->  best CV accuracy: {study.best_value:.5f}')
    print(f'   Best params: {study.best_params}')
    return best_params


if __name__ == '__main__' and LGB_AVAILABLE:
    data = load_processed_data()
    X_num = data['X_train_num']
    X_te_num = data['X_test_num']
    y = data['y']

    best_params = find_best_params(X_num, y)

    oof_probs = np.zeros(len(y))
    oof_counts = np.zeros(len(y))
    test_preds = []

    total = len(CFG.random_seeds) * CFG.n_splits
    fold_num = 0

    for seed in CFG.random_seeds:
        cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=seed)
        for fi, (tr_idx, val_idx) in enumerate(cv.split(X_num, y), 1):
            fold_num += 1
            model = lgb.LGBMClassifier(
                **{k: v for k, v in best_params.items() if k != 'verbosity'},
                random_state=seed * 10 + fi, verbosity=-1
            )
            model.fit(X_num.iloc[tr_idx], y.iloc[tr_idx])
            vp = model.predict_proba(X_num.iloc[val_idx])[:, 1]
            tp = model.predict_proba(X_te_num)[:, 1]

            oof_probs[val_idx] += vp
            oof_counts[val_idx] += 1
            test_preds.append(tp)

            acc = accuracy_score(y.iloc[val_idx], vp >= 0.5)
            print(f'  [lgb] {fold_num:2d}/{total} | s={seed} f={fi} | acc={acc:.4f}')

    oof_probs /= np.maximum(oof_counts, 1)
    avg_test = np.mean(test_preds, axis=0)

    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(CFG.output_dir / 'oof_lgb.npy', oof_probs)
    np.save(CFG.output_dir / 'test_lgb.npy', avg_test)
    print(f'\n[OK] lgb saved -> oof_lgb.npy, test_lgb.npy')
elif not LGB_AVAILABLE:
    print('[WARN] LightGBM not available -- skipping.')
