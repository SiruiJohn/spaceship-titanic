import pickle, warnings, sys
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG
from mixup import mixup_augment, group_swap_augment

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
    train_feat = data.get('train_feat')

    if 'feature_subsets' in data and data['feature_subsets']:
        cols = data['feature_subsets'].get('lgb')
        if cols is not None:
            X_num = X_num[cols]
            X_te_num = X_te_num[cols]

    best_params = find_best_params(X_num, y)

    all_run_oof = []
    all_run_test = []

    for run_idx in range(CFG.ensemble_runs):
        base_offset = CFG.ensemble_base_seeds[run_idx]

        oof_probs = np.zeros(len(y))
        oof_counts = np.zeros(len(y))
        test_preds = []

        total = len(CFG.random_seeds) * CFG.n_splits
        fold_num = 0

        for seed in CFG.random_seeds:
            effective_seed = seed + base_offset
            cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=effective_seed)
            for fi, (tr_idx, val_idx) in enumerate(cv.split(X_num, y), 1):
                fold_num += 1
                model = lgb.LGBMClassifier(
                    **{k: v for k, v in best_params.items() if k != 'verbosity'},
                    random_state=effective_seed * 10 + fi, verbosity=-1
                )
                tr_aug, ytr_aug = mixup_augment(
                    X_num.iloc[tr_idx].values, y.iloc[tr_idx],
                    alpha=CFG.mixup_alpha, multiplier=CFG.mixup_multiplier,
                    noise_std=CFG.noise_std, random_state=effective_seed * 10 + fi,
                )
                if CFG.group_swap_fraction > 0 and train_feat is not None:
                    tr_group_ids = train_feat.iloc[tr_idx]['GroupId'].values
                    tr_aug, ytr_aug = group_swap_augment(
                        tr_aug, ytr_aug, tr_group_ids,
                        swap_fraction=CFG.group_swap_fraction,
                        random_state=effective_seed * 10 + fi,
                    )
                model.fit(tr_aug, ytr_aug)
                vp = model.predict_proba(X_num.iloc[val_idx])[:, 1]
                tp = model.predict_proba(X_te_num)[:, 1]

                oof_probs[val_idx] += vp
                oof_counts[val_idx] += 1
                test_preds.append(tp)

                acc = accuracy_score(y.iloc[val_idx], vp >= 0.5)
                tag = f'R{run_idx+1}' if CFG.ensemble_runs > 1 else ''
                print(f'  [lgb{tag}] {fold_num:2d}/{total} | s={effective_seed} f={fi} | acc={acc:.4f}')

        oof_probs /= np.maximum(oof_counts, 1)
        avg_test = np.mean(test_preds, axis=0)
        all_run_oof.append(oof_probs)
        all_run_test.append(avg_test)

        if CFG.ensemble_runs > 1:
            CFG.output_dir.mkdir(parents=True, exist_ok=True)
            np.save(CFG.output_dir / f'oof_lgb_r{run_idx}.npy', oof_probs)
            np.save(CFG.output_dir / f'test_lgb_r{run_idx}.npy', avg_test)
            run_acc = accuracy_score(y, oof_probs >= 0.5)
            print(f'  Run {run_idx+1}/{CFG.ensemble_runs} OOF acc: {run_acc:.5f}')

    final_oof = np.mean(all_run_oof, axis=0)
    final_test = np.mean(all_run_test, axis=0)

    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(CFG.output_dir / 'oof_lgb.npy', final_oof)
    np.save(CFG.output_dir / 'test_lgb.npy', final_test)
    final_acc = accuracy_score(y, final_oof >= 0.5)
    n_runs = CFG.ensemble_runs
    print(f'\n[OK] lgb saved ({n_runs} run{"s" if n_runs > 1 else ""} avg), OOF acc: {final_acc:.5f}')
elif not LGB_AVAILABLE:
    print('[WARN] LightGBM not available -- skipping.')
