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


def apply_group_consistency(test_probs, test_feat, locked_mask=None, group_confidence=0.65):
    result = test_probs.copy()
    group_ids = test_feat['GroupId'].values
    group_sizes = test_feat['GroupSize'].values
    unique_groups = np.unique(group_ids)

    n_adjusted = 0

    for gid in unique_groups:
        mask = group_ids == gid
        n_members = mask.sum()
        if n_members <= 1:
            continue

        g_probs = test_probs[mask]
        g_mean = g_probs.mean()

        if g_mean > group_confidence:
            result[mask] = np.clip(g_probs * 0.4 + 0.6, 0.5, 1.0)
            n_adjusted += n_members
        elif g_mean < 1 - group_confidence:
            result[mask] = np.clip(g_probs * 0.4, 0.0, 0.5)
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

    for name in ['extra_trees', 'hist_gb', 'xgb', 'lgb', 'cat', 'nn']:
        oof_file = CFG.output_dir / f'oof_{name}.npy'
        test_file = CFG.output_dir / f'test_{name}.npy'
        if oof_file.exists() and test_file.exists():
            model_outputs[name] = {
                'oof': np.load(oof_file),
                'test': np.load(test_file),
            }
            available.append(name)
            print(f'   Loaded {name} model outputs')

    return data, model_outputs, available


def ensemble_stage1_bootstrap(model_outputs, model_names, y_true, seed):
    oof_mat = np.column_stack([model_outputs[n]['oof'] for n in model_names])
    test_mat = np.column_stack([model_outputs[n]['test'] for n in model_names])

    rng = np.random.RandomState(seed)
    n = len(y_true)
    bootstrap_idx = rng.choice(n, size=n, replace=True)
    bootstrap_oof = oof_mat[bootstrap_idx]
    bootstrap_y = y_true.iloc[bootstrap_idx]

    meta = LogisticRegression(C=0.5, max_iter=3000)
    meta.fit(bootstrap_oof, bootstrap_y)
    oof_stack = meta.predict_proba(oof_mat)[:, 1]
    test_stack = meta.predict_proba(test_mat)[:, 1]

    simple_oof = oof_mat.mean(axis=1)
    simple_test = test_mat.mean(axis=1)

    best_w, best_cv = 0.5, -1.0
    best_oof = simple_oof
    best_test = simple_test

    for w in np.linspace(0.2, 0.8, 25):
        cand = w * oof_stack + (1 - w) * simple_oof
        t, s = optimize_threshold(y_true, cand)
        if s > best_cv:
            best_cv = s
            best_w = float(w)
            best_oof = cand
            best_test = w * test_stack + (1 - w) * simple_test
            best_t = float(t)

    return best_oof, best_test, best_t, best_cv, best_w


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
    print(f'[INFO] Hard-rule locked (CryoSleep+NoSpend): {n_locked} / {len(test_feat)}')

    all_run_test_probs = []
    all_run_oof_probs = []
    all_run_labels = []

    for run_idx in range(CFG.ensemble_runs):
        base_seed = CFG.ensemble_base_seeds[run_idx]
        print(f'\n{"="*50}')
        print(f'  ENSEMBLE RUN {run_idx+1}/{CFG.ensemble_runs}  (base_seed={base_seed})')
        print(f'{"="*50}')

        oof_probs, test_probs, threshold, cv_acc, stack_w = ensemble_stage1_bootstrap(
            model_outputs, model_names, y, seed=base_seed
        )
        print(f'  Stage1 CV acc: {cv_acc:.5f}  |  threshold: {threshold:.4f}  |  stack_w: {stack_w:.4f}')

        current_oof = oof_probs
        current_test = test_probs
        current_acc = cv_acc

        aug_X_num = X_train_num.copy()
        aug_y = y.copy()
        n_original = len(y)

        for pseudo_round in range(1, CFG.pseudo_rounds + 1):
            high_mask = (current_test >= CFG.pseudo_threshold) | \
                        (current_test <= 1 - CFG.pseudo_threshold)
            if locked_mask.any():
                high_mask[locked_mask] = False

            n_pseudo = high_mask.sum()
            if n_pseudo < 100:
                print(f'  Pseudo round {pseudo_round}: only {n_pseudo} confident samples, stopping.')
                break

            pseudo_labels = (current_test[high_mask] >= 0.5).astype(int)
            pseudo_X = X_test_num.iloc[high_mask].reset_index(drop=True)

            aug_X = pd.concat([aug_X_num, pseudo_X], ignore_index=True)
            aug_y_full = pd.concat([aug_y, pd.Series(pseudo_labels)], ignore_index=True)

            aug_oofs = {}
            aug_tests = {}
            for name in model_names:
                clf = LogisticRegression(C=0.3, max_iter=2000)
                orig_oof = model_outputs[name]['oof']
                orig_test = model_outputs[name]['test']
                clf.fit(orig_oof.reshape(-1, 1), y.values)
                pseudo_pseudo_train = clf.predict_proba(
                    model_outputs[name]['test'][high_mask].reshape(-1, 1)
                )[:, 1]

                aug_oofs[name] = np.concatenate([orig_oof, pseudo_pseudo_train])
                aug_tests[name] = orig_test

            augmented_outputs = {
                n: {'oof': aug_oofs[n], 'test': aug_tests[n]}
                for n in model_names
            }

            new_oof, new_test, new_t, new_acc, new_w = ensemble_stage1_bootstrap(
                augmented_outputs, model_names, aug_y_full, seed=base_seed + pseudo_round * 100
            )

            oof_orig_only = new_oof[:n_original]
            orig_t, orig_acc = optimize_threshold(y, oof_orig_only)

            print(f'  Pseudo round {pseudo_round} | +{n_pseudo} samples | '
                  f'reported acc: {new_acc:.5f} | true acc (orig only): {orig_acc:.5f}')

            if orig_acc > current_acc:
                print(f'  [OK] Real improvement ({current_acc:.5f} -> {orig_acc:.5f}), keeping.')
                current_oof = oof_orig_only
                current_test = new_test
                current_acc = orig_acc
                aug_X_num = aug_X
                aug_y = aug_y_full
            else:
                print(f'  [SKIP] No real improvement ({orig_acc:.5f} <= {current_acc:.5f}), reverting.')
                break

        all_run_oof_probs.append(current_oof)
        all_run_test_probs.append(current_test)
        all_run_labels.append(f'run{run_idx+1}_acc={current_acc:.5f}')
        print(f'  Run {run_idx+1} final OOF acc: {current_acc:.5f}')

    final_oof = np.mean(all_run_oof_probs, axis=0)
    final_test = np.mean(all_run_test_probs, axis=0)
    final_threshold, final_cv_acc = optimize_threshold(y, final_oof)

    print(f'\n[OK] Multi-run ensemble ({CFG.ensemble_runs} runs) complete.')
    print(f'   Per-run accs: {", ".join(all_run_labels)}')
    print(f'   Averaged OOF acc: {final_cv_acc:.5f}')

    pre_cal_acc = final_cv_acc
    cal_oof, cal_test = calibrate_oof(final_oof, final_test, y)
    cal_threshold, cal_acc = optimize_threshold(y, cal_oof)

    if cal_acc > pre_cal_acc:
        print(f'[OK] Calibration improved: {pre_cal_acc:.5f} -> {cal_acc:.5f}')
        use_probs = cal_test
        use_threshold = cal_threshold
    else:
        print(f'[SKIP] Calibration did NOT help ({cal_acc:.5f} <= {pre_cal_acc:.5f})')
        use_probs = final_test
        use_threshold = final_threshold

    print(f'\n[OK] Applying group consistency post-processing...')
    group_final, n_group_adjusted = apply_group_consistency(
        use_probs, test_feat, locked_mask=locked_mask, group_confidence=0.65
    )
    final_preds = (group_final >= use_threshold).astype(float)

    for idx in locked_test_preds.dropna().index:
        final_preds[int(idx)] = locked_test_preds[idx]
        group_final[int(idx)] = locked_test_preds[idx]

    submission = sample_submission.copy()
    submission[CFG.target] = final_preds.astype(bool)
    submission.to_csv(CFG.submission_file, index=False)

    print(f'\n[OK] Submission created: {CFG.submission_file}')
    print(f'   Positive rate: {final_preds.mean():.5f}')
    print(f'   Hard-rule locked: {n_locked} / {len(final_preds)}')
    print(f'   Group-adjusted: {n_group_adjusted}')
    print(f'   Pre-calibration OOF acc: {pre_cal_acc:.5f}')
    print(f'   Post-calibration OOF acc: {cal_acc:.5f}')

    sub_raw = sample_submission.copy()
    sub_raw[CFG.target] = (use_probs >= 0.5).astype(bool)
    sub_raw.to_csv('submission_no_group.csv', index=False)
    raw_rate = sub_raw['Transported'].mean()
    print(f'   Raw model positive rate (no group): {raw_rate:.5f}')
    print(f'   Group-adjusted positive rate: {final_preds.mean():.5f}')


if __name__ == '__main__':
    blend_and_submit()
