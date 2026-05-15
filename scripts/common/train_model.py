import ast
import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
PSEUDO_DIR = DATA_DIR / "pseudo"
RESULTS_DIR = ROOT / "results"
PARAMS_DIR = RESULTS_DIR / "params"
PRED_DIR = RESULTS_DIR / "preds" / "dl"
SUB_DIR = RESULTS_DIR / "submissions"
METRICS_DIR = RESULTS_DIR / "metrics"
PROC_TAG = os.environ.get("PROC_TAG", "").strip()

def proc_path(name: str) -> Path:
    fname = f"{name}_{PROC_TAG}.csv" if PROC_TAG else f"{name}.csv"
    return PROC_DIR / fname

def rank_transform(mat: np.ndarray) -> np.ndarray:
    out = np.empty_like(mat, dtype=float)
    n = mat.shape[0]
    for j in range(mat.shape[1]):
        order = np.argsort(mat[:, j], kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        if n > 1:
            ranks /= (n - 1)
        out[:, j] = ranks
    return out

def load_data(use_pseudo=False):
    print("Loading data...", flush=True)
    pseudo_path = PSEUDO_DIR / "train_pseudo_labeled.csv"
    if use_pseudo and pseudo_path.exists():
        print("Using pseudo-labeled data!", flush=True)
        train_df = pd.read_csv(pseudo_path)
    else:
        train_df = pd.read_csv(proc_path("train_processed"))
    
    test_df = pd.read_csv(proc_path("test_processed"))
    
    X = train_df.drop(columns=["Transported", "PassengerId"])
    y = train_df["Transported"].astype(int).to_numpy()
    X_test = test_df.drop(columns=["PassengerId"])
    X_test = X_test[X.columns]
    
    return X, y, X_test, test_df

def make_xgb(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        eval_metric="logloss",
        verbosity=0,
        random_state=seed,
    )

def make_lgb(seed: int) -> lgb.LGBMClassifier:
    try:
        with open(PARAMS_DIR / "best_lgbm_params.txt", "r", encoding="utf-8") as f:
            lgbm_params = ast.literal_eval(f.read().strip())
    except Exception:
        lgbm_params = {
            "n_estimators": 180,
            "learning_rate": 0.09598232320208959,
            "num_leaves": 254,
            "max_depth": 7,
            "min_child_samples": 82,
            "subsample": 0.8773174846493815,
            "colsample_bytree": 0.5880046856820298,
            "reg_alpha": 6.237708648779934e-08,
            "reg_lambda": 1.1574035898665557e-08,
        }
    return lgb.LGBMClassifier(
        **lgbm_params,
        objective="binary",
        metric="binary_logloss",
        verbosity=-1,
        random_state=seed,
    )

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

def run_stacking(use_pseudo=False):
    X, y, X_test, test_df = load_data(use_pseudo)
    
    fast = os.environ.get("FAST", "0").strip() == "1"
    seeds_env = os.environ.get("SEEDS", "").strip()
    if seeds_env:
        seeds = [int(s.strip()) for s in seeds_env.split(",") if s.strip()]
    else:
        seeds = [42] if fast else [42, 202, 555]

    models_env = os.environ.get("MODELS", "").strip().lower()
    requested_models = [m.strip() for m in models_env.split(",") if m.strip()] if models_env else []

    def wrap_factory(model_name: str, factory):
        def f(seed: int):
            model = factory(seed)
            if not fast:
                return model
            fast_lgb_n = int(os.environ.get("FAST_LGB_N", "120").strip())
            fast_xgb_n = int(os.environ.get("FAST_XGB_N", "200").strip())
            fast_cat_n = int(os.environ.get("FAST_CAT_N", "200").strip())
            if model_name == "lgb":
                params = model.get_params()
                params["n_estimators"] = min(int(params.get("n_estimators", fast_lgb_n)), fast_lgb_n)
                params["learning_rate"] = min(float(params.get("learning_rate", 0.05)), 0.05)
                params["n_jobs"] = 1
                return lgb.LGBMClassifier(**params)
            if model_name == "xgb":
                params = model.get_params()
                params["n_estimators"] = min(int(params.get("n_estimators", fast_xgb_n)), fast_xgb_n)
                params["learning_rate"] = min(float(params.get("learning_rate", 0.05)), 0.05)
                params["n_jobs"] = 1
                return xgb.XGBClassifier(**params)
            if model_name == "cat":
                params = model.get_params()
                params["iterations"] = min(int(params.get("iterations", fast_cat_n)), fast_cat_n)
                params["learning_rate"] = min(float(params.get("learning_rate", 0.05)), 0.05)
                params["thread_count"] = 1
                return CatBoostClassifier(**params)
            return model

        return f

    all_factories = [
        ("lgb", wrap_factory("lgb", make_lgb)),
        ("cat", wrap_factory("cat", make_cat)),
        ("xgb", wrap_factory("xgb", make_xgb)),
    ]
    model_factories = [mf for mf in all_factories if (not requested_models or mf[0] in requested_models)]

    n_splits = int(os.environ.get("N_SPLITS", "3" if fast else "5").strip())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    feature_names = []
    oof_matrix = []
    test_matrix = []
    base_scores = {}

    for model_name, factory in model_factories:
        for seed in seeds:
            col_name = f"{model_name}_s{seed}"
            feature_names.append(col_name)
            oof_pred = np.zeros(X.shape[0], dtype=float)
            test_pred = np.zeros(X_test.shape[0], dtype=float)

            print(f"Training {col_name}...", flush=True)
            for train_idx, val_idx in skf.split(X, y):
                X_tr = X.iloc[train_idx]
                y_tr = y[train_idx]
                X_va = X.iloc[val_idx]

                model = factory(seed)
                model.fit(X_tr, y_tr)

                oof_pred[val_idx] = model.predict_proba(X_va)[:, 1]
                test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits

            oof_matrix.append(oof_pred)
            test_matrix.append(test_pred)
            base_scores[col_name] = float(accuracy_score(y, (oof_pred >= 0.5).astype(int)))
            print(f"{col_name} OOF Accuracy (thr=0.5): {base_scores[col_name]:.4f}", flush=True)

    use_dl = os.environ.get("USE_DL", "1").strip() == "1"
    if use_dl:
        oof_path = PRED_DIR / "oof_preds_dl.npy"
        test_path = PRED_DIR / "test_preds_dl.npy"
        if oof_path.exists() and test_path.exists():
            print("Loading DL predictions...", flush=True)
            oof_dl = np.load(oof_path).ravel()
            test_dl = np.load(test_path).ravel()
            
            if len(oof_dl) == len(X):
                oof_matrix.append(oof_dl)
                test_matrix.append(test_dl)
                feature_names.append("mlp_dl")
                base_scores["mlp_dl"] = float(accuracy_score(y, (oof_dl >= 0.5).astype(int)))
                print(f"mlp_dl OOF Accuracy (thr=0.5): {base_scores['mlp_dl']:.4f}", flush=True)
            else:
                print(f"DL OOF size mismatch ({len(oof_dl)} vs {len(X)}), skipping MLP DL features.", flush=True)

    oof_matrix = np.vstack(oof_matrix).T
    test_matrix = np.vstack(test_matrix).T

    print(f"Final ensemble features: {feature_names}", flush=True)

    def best_threshold_for_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
        thresholds = np.linspace(0.05, 0.95, 181)
        best_thr = 0.5
        best_acc = -1.0
        for thr in thresholds:
            acc = accuracy_score(y_true, (y_prob >= thr).astype(int))
            if acc > best_acc:
                best_acc = float(acc)
                best_thr = float(thr)
        return best_thr, best_acc

    use_optuna_blend = os.environ.get("USE_OPTUNA_BLEND", "0").strip() == "1"
    if use_optuna_blend:
        n_trials = int(os.environ.get("OPTUNA_TRIALS", "200").strip())
        sampler_name = os.environ.get("OPTUNA_SAMPLER", "tpe").strip().lower()
        print(f"Running Optuna weight optimizer (trials={n_trials})...", flush=True)

        def objective(trial: optuna.Trial) -> float:
            weights = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(oof_matrix.shape[1])], dtype=float)
            if weights.sum() == 0:
                weights = np.ones_like(weights)
            oof_blend = np.average(oof_matrix, axis=1, weights=weights)
            _, acc = best_threshold_for_accuracy(y, oof_blend)
            return acc

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if sampler_name == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(seed=42)
        else:
            sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        best_w = np.array([study.best_params[f"w{i}"] for i in range(oof_matrix.shape[1])], dtype=float)
        if best_w.sum() == 0:
            best_w = np.ones_like(best_w)
        oof_blend = np.average(oof_matrix, axis=1, weights=best_w)
        thr, acc = best_threshold_for_accuracy(y, oof_blend)
        test_blend = np.average(test_matrix, axis=1, weights=best_w)
        best = {"mode": "optuna_blend", "oof_acc": float(acc), "thr": float(thr), "weights": best_w.tolist(), "features": feature_names}
        return test_blend, thr, acc, test_df, best, feature_names

    skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    thr_grid = np.linspace(0.3, 0.7, 81)

    candidates = []
    for use_rank in [False, True]:
        X_meta = rank_transform(oof_matrix) if use_rank else oof_matrix
        for C in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            meta = LogisticRegression(max_iter=5000, C=C)
            meta_oof_cv = np.zeros(X_meta.shape[0], dtype=float)
            for tr_idx, va_idx in skf_meta.split(X_meta, y):
                meta.fit(X_meta[tr_idx], y[tr_idx])
                meta_oof_cv[va_idx] = meta.predict_proba(X_meta[va_idx])[:, 1]

            best_thr = 0.5
            best_acc = -1.0
            for thr in thr_grid:
                acc = accuracy_score(y, (meta_oof_cv >= thr).astype(int))
                if acc > best_acc:
                    best_acc = float(acc)
                    best_thr = float(thr)

            candidates.append(
                {
                    "use_rank": use_rank,
                    "C": float(C),
                    "oof_acc": best_acc,
                    "thr": best_thr,
                }
            )

    best = max(candidates, key=lambda d: (d["oof_acc"], -abs(d["thr"] - 0.5), d["C"]))
    print(
        f"Meta Selected: acc={best['oof_acc']:.4f}, thr={best['thr']:.3f}, "
        f"C={best['C']}, rank={best['use_rank']}",
        flush=True
    )

    X_meta = rank_transform(oof_matrix) if best["use_rank"] else oof_matrix
    T_meta = rank_transform(test_matrix) if best["use_rank"] else test_matrix
    meta = LogisticRegression(max_iter=5000, C=best["C"])
    meta.fit(X_meta, y)
    meta_test = meta.predict_proba(T_meta)[:, 1]

    return meta_test, best["thr"], best["oof_acc"], test_df, best, feature_names

if __name__ == "__main__":
    use_pseudo = os.environ.get("USE_PSEUDO", "0").strip() == "1"
    meta_test, best_thr, best_acc, test_df, best_meta, feature_names = run_stacking(use_pseudo=use_pseudo)
    
    pred = (meta_test >= best_thr).astype(bool)
    use_optuna_blend = os.environ.get("USE_OPTUNA_BLEND", "0").strip() == "1"
    if use_optuna_blend:
        version = "v10_optuna"
        version_dir = "v10"
    else:
        version = "v9" if use_pseudo else "v10_meta"
        version_dir = "v9" if use_pseudo else "v10"
    out_sub_dir = SUB_DIR / version_dir
    out_sub_dir.mkdir(parents=True, exist_ok=True)
    submission_file = out_sub_dir / f"submission_{version}.csv"
    
    submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": pred})
    submission.to_csv(submission_file, index=False)
    print(f"Saved {submission_file}", flush=True)

    out_metrics_dir = METRICS_DIR / version_dir
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_dir / f"results_{version}.txt", "w", encoding="utf-8") as f:
        f.write(f"OOF Meta Accuracy: {best_acc:.6f}\n")
        f.write(f"Meta Threshold: {best_thr:.6f}\n")
        if use_optuna_blend:
            f.write(f"Weights: {best_meta['weights']}\n")
        else:
            f.write(f"Meta C: {best_meta['C']}\n")
            f.write(f"Meta Rank: {best_meta['use_rank']}\n")
        f.write(f"Features: {feature_names}\n")
