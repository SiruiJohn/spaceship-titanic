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
    train_feat = data.get('train_feat')

    if 'feature_subsets' in data and data['feature_subsets']:
        cols = data['feature_subsets'].get(model_name)
        if cols is not None:
            X_num = X_num[cols]
            X_te_num = X_te_num[cols]
            if needs_cat and 'CATEGORICAL_COLS' in data:
                data['CATEGORICAL_COLS'] = [c for c in data['CATEGORICAL_COLS'] if c in cols]

    cat_indices = [X_raw.columns.get_loc(c) for c in data['CATEGORICAL_COLS'] if c in X_raw.columns]

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
                tr_num = X_num.iloc[tr_idx]
                vl_num = X_num.iloc[val_idx]
                ytr = y.iloc[tr_idx]
                yvl = y.iloc[val_idx]

                model = build_model_fn(effective_seed, fi)

                if needs_raw:
                    tr_raw = X_raw.iloc[tr_idx].copy().astype(str)
                    vl_raw = X_raw.iloc[val_idx].copy().astype(str)
                    te_raw = X_te_raw.copy().astype(str)
                    model.fit(tr_raw, ytr, cat_features=cat_indices, verbose=False)
                    vp = model.predict_proba(vl_raw)[:, 1]
                    tp = model.predict_proba(te_raw)[:, 1]
                else:
                    tr_aug, ytr_aug = mixup_augment(
                        tr_num.values, ytr, alpha=CFG.mixup_alpha,
                        multiplier=CFG.mixup_multiplier, noise_std=CFG.noise_std,
                        random_state=effective_seed * 10 + fi,
                    )
                    if CFG.group_swap_fraction > 0 and train_feat is not None:
                        tr_group_ids = train_feat.iloc[tr_idx]['GroupId'].values
                        tr_aug, ytr_aug = group_swap_augment(
                            tr_aug, ytr_aug, tr_group_ids,
                            swap_fraction=CFG.group_swap_fraction,
                            random_state=effective_seed * 10 + fi,
                        )
                    model.fit(tr_aug, ytr_aug)
                    vp = model.predict_proba(vl_num)[:, 1]
                    tp = model.predict_proba(X_te_num)[:, 1]

                oof_probs[val_idx] += vp
                oof_counts[val_idx] += 1
                test_preds.append(tp)

                acc = accuracy_score(yvl, vp >= 0.5)
                tag = f'R{run_idx+1}' if CFG.ensemble_runs > 1 else ''
                print(f'  [{model_name}{tag}] {fold_num:2d}/{total} | s={effective_seed} f={fi} | acc={acc:.4f}')

        oof_probs /= np.maximum(oof_counts, 1)
        avg_test = np.mean(test_preds, axis=0)
        all_run_oof.append(oof_probs)
        all_run_test.append(avg_test)

        if CFG.ensemble_runs > 1:
            CFG.output_dir.mkdir(parents=True, exist_ok=True)
            np.save(CFG.output_dir / f'oof_{model_name}_r{run_idx}.npy', oof_probs)
            np.save(CFG.output_dir / f'test_{model_name}_r{run_idx}.npy', avg_test)
            run_acc = accuracy_score(y, oof_probs >= 0.5)
            print(f'  Run {run_idx+1}/{CFG.ensemble_runs} OOF acc: {run_acc:.5f}')

    final_oof = np.mean(all_run_oof, axis=0)
    final_test = np.mean(all_run_test, axis=0)

    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(CFG.output_dir / f'oof_{model_name}.npy', final_oof)
    np.save(CFG.output_dir / f'test_{model_name}.npy', final_test)
    final_acc = accuracy_score(y, final_oof >= 0.5)
    n_runs = CFG.ensemble_runs
    print(f'\n[OK] {model_name} saved ({n_runs} run{"s" if n_runs > 1 else ""} avg), OOF acc: {final_acc:.5f}')
