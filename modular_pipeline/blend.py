import pickle, warnings, sys
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG


def load_processed_data():
    with open(CFG.processed_data_file, 'rb') as f:
        return pickle.load(f)


def optimize_threshold(y_true, probs):
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.35, 0.65, 121):
        s = accuracy_score(y_true, probs >= t)
        if s > best_s:
            best_s, best_t = float(s), float(t)
    return best_t, best_s


def apply_group_consistency(test_probs, test_feat, locked_mask=None, group_confidence=0.75):
    result = test_probs.copy()
    group_ids = test_feat['GroupId'].values
    unique_groups = np.unique(group_ids)
    n_adjusted = 0

    for gid in unique_groups:
        mask = group_ids == gid
        n_members = mask.sum()
        if n_members <= 2:
            continue

        g_probs = test_probs[mask]
        g_mean = g_probs.mean()

        if g_mean > group_confidence:
            result[mask] = np.clip(g_probs * 0.7 + 0.3, 0.5, 1.0)
            n_adjusted += n_members
        elif g_mean < 1 - group_confidence:
            result[mask] = np.clip(g_probs * 0.7, 0.0, 0.5)
            n_adjusted += n_members

    if locked_mask is not None and locked_mask.any():
        result[locked_mask] = test_probs[locked_mask]

    return result, n_adjusted


def calibrate_oof(oof_probs, test_probs, y_true, n_splits=5):
    calibrated_oof = np.zeros_like(oof_probs)
    calibrated_test_parts = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in cv.split(oof_probs, y_true):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_probs[tr_idx], y_true.iloc[tr_idx])
        calibrated_oof[val_idx] = iso.transform(oof_probs[val_idx])
        calibrated_test_parts.append(iso.transform(test_probs))
    calibrated_test = np.mean(calibrated_test_parts, axis=0)
    return calibrated_oof, calibrated_test


def load_model_outputs():
    data = load_processed_data()
    model_outputs = {}
    available = []

    for name in CFG.blend_models:
        oof_file = CFG.output_dir / f'oof_{name}.npy'
        test_file = CFG.output_dir / f'test_{name}.npy'
        if oof_file.exists() and test_file.exists():
            model_outputs[name] = {
                'oof': np.load(oof_file),
                'test': np.load(test_file),
            }
            available.append(name)
            print(f'   Loaded {name} model outputs')
        else:
            print(f'   Missing {name} -- skipping')

    return data, model_outputs, available


def blend_and_submit():
    data, model_outputs, model_names = load_model_outputs()

    if len(model_names) < 2:
        print(f'[ERROR] Need at least 2 models, found {len(model_names)}')
        return

    X_train_num = data['X_train_num']
    X_test_num = data['X_test_num']
    y = data['y']
    locked_test_preds = data['locked_test_preds']
    sample_submission = data['sample_submission']
    train_feat = data['train_feat']
    test_feat = data['test_feat']

    locked_mask = locked_test_preds.notna().values
    n_locked = locked_mask.sum()
    n_original = len(y)
    print(f'[INFO] Hard-rule locked: {n_locked} / {len(test_feat)}')

    oof_mat = np.column_stack([model_outputs[n]['oof'] for n in model_names])
    test_mat = np.column_stack([model_outputs[n]['test'] for n in model_names])

    meta = LogisticRegression(C=0.5, max_iter=3000)
    meta.fit(oof_mat, y)
    oof_stack = meta.predict_proba(oof_mat)[:, 1]
    test_stack = meta.predict_proba(test_mat)[:, 1]

    simple_oof = oof_mat.mean(axis=1)
    simple_test = test_mat.mean(axis=1)

    best_w, best_cv = 0.5, -1.0
    best_oof = simple_oof
    best_test = simple_test

    for w in np.linspace(0.2, 0.8, 25):
        cand = w * oof_stack + (1 - w) * simple_oof
        t, s = optimize_threshold(y, cand)
        if s > best_cv:
            best_cv = s
            best_w = float(w)
            best_oof = cand
            best_test = w * test_stack + (1 - w) * simple_test

    best_threshold, best_cv = optimize_threshold(y, best_oof)
    print(f'\n[OK] Ensemble stage 1 (LR stacking).')
    print(f'   Models: {model_names}')
    print(f'   Stack weight: {best_w:.4f}')
    print(f'   OOF CV accuracy: {best_cv:.5f}')

    final_oof = best_oof
    final_test = best_test

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train_num.values)
    X_te_sc = scaler.transform(X_test_num.values)

    mlp_oof = np.zeros(n_original)
    mlp_count = np.zeros(n_original)
    mlp_test_parts = []

    mlp_seeds = CFG.random_seeds[:3]
    for seed in mlp_seeds:
        cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=seed)
        for fi, (tr_idx, val_idx) in enumerate(cv.split(X_sc, y), 1):
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu', alpha=0.005,
                learning_rate_init=1e-3, max_iter=300,
                early_stopping=True, validation_fraction=0.1,
                random_state=seed * 10 + fi, batch_size=256,
            )
            mlp.fit(X_sc[tr_idx], y.iloc[tr_idx])
            mlp_oof[val_idx] += mlp.predict_proba(X_sc[val_idx])[:, 1]
            mlp_count[val_idx] += 1
            mlp_test_parts.append(mlp.predict_proba(X_te_sc)[:, 1])

    mlp_oof /= np.maximum(mlp_count, 1)
    mlp_test = np.mean(mlp_test_parts, axis=0)
    mlp_cv_acc = accuracy_score(y, mlp_oof >= 0.5)
    print(f'\n[OK] MLP blending OOF accuracy: {mlp_cv_acc:.5f}')

    best_final_acc = best_cv
    best_final_oof = final_oof
    best_final_test = final_test
    best_final_label = 'GBDT ensemble'

    oof_2 = np.column_stack([final_oof, mlp_oof])
    test_2 = np.column_stack([final_test, mlp_test])
    meta2 = LogisticRegression(C=0.3, max_iter=1000)
    meta2.fit(oof_2, y)
    lr_oof = meta2.predict_proba(oof_2)[:, 1]
    lr_test = meta2.predict_proba(test_2)[:, 1]
    _, lr_acc = optimize_threshold(y, lr_oof)
    print(f'  LR meta-blend          : {lr_acc:.5f}  weights={meta2.coef_[0].round(3)}')
    if lr_acc > best_final_acc:
        best_final_acc, best_final_oof, best_final_test = lr_acc, lr_oof, lr_test
        best_final_label = 'LR meta-blend (GBDT + MLP)'

    for mlp_w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        w_oof = (1 - mlp_w) * final_oof + mlp_w * mlp_oof
        w_test = (1 - mlp_w) * final_test + mlp_w * mlp_test
        _, w_acc = optimize_threshold(y, w_oof)
        if w_acc > best_final_acc:
            best_final_acc, best_final_oof, best_final_test = w_acc, w_oof, w_test
            best_final_label = f'Fixed blend mlp_w={mlp_w:.2f}'

    _, gbdt_acc = optimize_threshold(y, final_oof)
    if gbdt_acc >= best_final_acc:
        best_final_acc, best_final_oof, best_final_test = gbdt_acc, final_oof, final_test
        best_final_label = 'GBDT ensemble (MLP did not help)'

    final_threshold, final_cv_acc = optimize_threshold(y, best_final_oof)
    print(f'\n[OK] Winner: {best_final_label}  |  OOF acc: {final_cv_acc:.5f}')

    pre_cal_acc = final_cv_acc
    cal_oof, cal_test = calibrate_oof(best_final_oof, best_final_test, y)
    cal_threshold, cal_acc = optimize_threshold(y, cal_oof)

    if cal_acc > pre_cal_acc:
        print(f'[OK] Calibration improved: {pre_cal_acc:.5f} -> {cal_acc:.5f}')
        use_probs = cal_test
        use_threshold = cal_threshold
    else:
        print(f'[SKIP] Calibration did NOT help ({cal_acc:.5f} <= {pre_cal_acc:.5f})')
        use_probs = best_final_test
        use_threshold = final_threshold

    final_preds = (use_probs >= use_threshold).astype(float)
    for idx in locked_test_preds.dropna().index:
        final_preds[int(idx)] = locked_test_preds[idx]

    submission = sample_submission.copy()
    submission[CFG.target] = final_preds.astype(bool)
    submission.to_csv('submission_no_group.csv', index=False)

    group_probs, n_group_adjusted = apply_group_consistency(
        use_probs, test_feat, locked_mask=locked_mask, group_confidence=0.75
    )
    group_preds = (group_probs >= use_threshold).astype(float)
    for idx in locked_test_preds.dropna().index:
        group_preds[int(idx)] = locked_test_preds[idx]

    sub_group = sample_submission.copy()
    sub_group[CFG.target] = group_preds.astype(bool)
    sub_group.to_csv(CFG.submission_file, index=False)

    print(f'\n[OK] Submissions created:')
    print(f'   submission_no_group.csv     (no group consistency)')
    print(f'   {CFG.submission_file}       (with group consistency)')
    print(f'   Positive rate (no group) : {final_preds.mean():.5f}')
    print(f'   Positive rate (w/ group) : {group_preds.mean():.5f}')
    print(f'   Hard-rule locked  : {n_locked}')
    print(f'   Group-adjusted    : {n_group_adjusted}')
    print(f'   Pre-calibration   : {pre_cal_acc:.5f}')
    print(f'   Post-calibration  : {cal_acc:.5f}')
    print(f'   Winner            : {best_final_label}')


if __name__ == '__main__':
    blend_and_submit()
