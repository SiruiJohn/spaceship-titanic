import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import logging
from sklearn.metrics import log_loss
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load processed data
logger.info("Loading data...")
ROOT = Path(__file__).resolve().parents[2]
train_df = pd.read_csv(ROOT / "data" / "processed" / "train_processed.csv")

# Prepare X and y
X = train_df.drop(columns=['Transported', 'PassengerId'])
y = train_df['Transported']

# Split validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def objective_logloss(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42
    }

    try:
        from optuna.integration import LightGBMPruningCallback
        pruning_callback = LightGBMPruningCallback(trial, "binary_logloss")
    except ImportError:
         from optuna_integration import LightGBMPruningCallback
         pruning_callback = LightGBMPruningCallback(trial, "binary_logloss")

    model = lgb.LGBMClassifier(**param)
    
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)],
        callbacks=[pruning_callback]
    )

    preds_proba = model.predict_proba(X_val)[:, 1]
    ll = log_loss(y_val, preds_proba)
    return ll

if __name__ == '__main__':
    logger.info("Starting Optuna optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_logloss, n_trials=50, timeout=600)

    logger.info("Number of finished trials: {}".format(len(study.trials)))
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))
    
    out_dir = ROOT / "results" / "params"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "best_lgbm_params.txt", "w", encoding="utf-8") as f:
        f.write(str(trial.params))
