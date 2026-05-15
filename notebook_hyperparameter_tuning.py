"""Hyperparameter tuning helpers extracted from the notebook export.

This module serves two purposes:
1. keep the notebook's hyperparameter search logic in one place;
2. expose the already tuned parameter sets used by the main pipeline.
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


TRAINED_PARAMS = {
    "xgb": {
        "colsample_bytree": 0.8498791800104656,
        "learning_rate": 0.020233442882782587,
        "max_depth": 4,
        "n_estimators": 469,
        "subsample": 0.746529796772373,
    },
    "lgb": {
        "colsample_bytree": 0.7774799983649324,
        "learning_rate": 0.007653648135411494,
        "max_depth": 5,
        "n_estimators": 350,
        "reg_alpha": 0.14326300616140863,
        "reg_lambda": 0.9310129332502252,
        "subsample": 0.6189257947519665,
    },
    "cat": {
        "random_strength": 0.1,
        "one_hot_max_size": 10,
        "max_bin": 100,
        "learning_rate": 0.01,
        "l2_leaf_reg": 0.5,
        "grow_policy": "Lossguide",
        "depth": 5,
        "bootstrap_type": "Bernoulli",
    },
    "logreg": {
        "C": 10,
        "penalty": "l1",
        "solver": "liblinear",
    },
    "rf": {
        "n_estimators": 400,
        "min_samples_split": 2,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "max_depth": 6,
        "bootstrap": True,
    },
    "gbm": {
        "min_samples_split": 2,
        "min_samples_leaf": 4,
        "max_features": "log2",
        "max_depth": 5,
        "learning_rate": 0.0004430621457583882,
    },
    "dtc": {
        "min_samples_split": 4,
        "min_samples_leaf": 9,
        "max_depth": 4,
        "criterion": "gini",
    },
}


def get_trained_params(name: str) -> dict:
    """Return a copy of the tuned parameters for a given model name."""
    return dict(TRAINED_PARAMS[name])


def tune_xgb(X_train, y_train, random_state: int = 1, n_iter: int = 50, cv: int = 3, n_jobs: int = -1):
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.7,
        objective="binary:logistic",
        verbosity=0,
        random_state=random_state,
    )
    param_dist = {
        "n_estimators": np.arange(50, 1000, 50),
        "max_depth": np.arange(3, 15, 2),
        "learning_rate": np.arange(0.001, 0.05, 0.004),
        "subsample": [0.1, 0.3, 0.5, 0.7, 0.9],
        "colsample_bytree": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    search = RandomizedSearchCV(model, param_distributions=param_dist, cv=cv, n_iter=n_iter, random_state=random_state, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    return search


def tune_lgb(X_train, y_train, random_state: int = 1, n_iter: int = 20, cv: int = 3, n_jobs: int = -1):
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.0,
        reg_lambda=0.0,
        device="cpu",
        random_state=random_state,
    )
    param_dist = {
        "n_estimators": np.arange(50, 1000, 50),
        "max_depth": np.arange(3, 15, 2),
        "learning_rate": np.arange(0.001, 0.02, 0.002),
        "subsample": [0.1, 0.3, 0.5, 0.7, 0.9],
        "colsample_bytree": [0.1, 0.3, 0.5, 0.7, 0.9],
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 1),
    }
    search = RandomizedSearchCV(model, param_distributions=param_dist, cv=cv, n_iter=n_iter, random_state=random_state, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    return search


def tune_cat(X_train, y_train, random_state: int = 1, n_iter: int = 50, cv: int = 3, n_jobs: int = -1):
    model = CatBoostClassifier(iterations=200, eval_metric="Accuracy", loss_function="Logloss", task_type="CPU", verbose=False)
    param_dist = {
        "depth": np.arange(3, 15, 2),
        "learning_rate": np.arange(0.001, 0.02, 0.002),
        "l2_leaf_reg": [0.1, 0.5, 0.7],
        "random_strength": [0.1, 0.2, 0.5],
        "max_bin": [50, 100, 150],
        "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
        "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"],
        "one_hot_max_size": [10, 50, 70],
    }
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_log_loss",
        cv=cv,
        verbose=1,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    search.fit(X_train, y_train)
    return search


def tune_logreg(X_train, y_train, cv: int = 5, n_jobs: int = -1):
    model = LogisticRegression(max_iter=500, random_state=2023)
    param_grid = {
        "penalty": ["l1", "l2", "elasticnet"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    }
    search = GridSearchCV(estimator=model, param_grid=param_grid, scoring="roc_auc", cv=cv, verbose=1, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    return search


def tune_rf(X_train, y_train, random_state: int = 42, n_iter: int = 15, cv: int = 5, n_jobs: int = -1):
    model = RandomForestClassifier(
        bootstrap=False,
        max_depth=4,
        max_features="sqrt",
        min_samples_leaf=2,
        min_samples_split=5,
        n_estimators=341,
        random_state=random_state,
    )
    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, None],
        "max_features": ["sqrt", "log2", None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
    search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=n_iter, scoring="accuracy", cv=cv, verbose=1, n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    return search


def tune_hist_gbm(X_train, y_train, random_state: int = 42, n_iter: int = 30, cv: int = 3, n_jobs: int = -1):
    model = HistGradientBoostingClassifier(max_iter=2000)
    param_dist = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [1, 3, 5, 7],
        "l2_regularization": np.logspace(-4, 1, 6),
        "max_bins": [32, 64, 128, 256],
        "random_state": [random_state],
    }
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring="accuracy", n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    return search


def tune_gbm(X_train, y_train, random_state: int = 42, n_iter: int = 100, cv: int = 5, n_jobs: int = -1):
    model = GradientBoostingClassifier(
        max_depth=4,
        max_features="sqrt",
        min_samples_leaf=2,
        min_samples_split=5,
        n_estimators=341,
        random_state=random_state,
    )
    param_dist = {
        "n_estimators": np.arange(100, 1000, 50),
        "learning_rate": np.logspace(-4, 0, num=100),
        "max_depth": [2, 3, 4, 5, 6],
        "min_samples_split": [2, 3, 4, 5, 6],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring="accuracy", n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    return search


def tune_svc(X_train, y_train, random_state: int = 1, n_iter: int = 5, cv: int = 3, n_jobs: int = -1):
    model = SVC(probability=True)
    param_dist = {
        "C": uniform(0.1, 10),
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": randint(1, 10),
        "gamma": ["scale", "auto"] + list(uniform(0.01, 1).rvs(10)),
    }
    search = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    return search


def tune_knn(X_train, y_train, random_state: int = 1, n_iter: int = 20, cv: int = 5, n_jobs: int = -1):
    model = KNeighborsClassifier()
    param_dist = {
        "n_neighbors": np.arange(2, 20, 2),
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": randint(1, 100),
        "p": [1, 2],
    }
    search = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    return search


def tune_mlp(X_train, y_train, random_state: int = 42, n_iter: int = 20, cv: int = 5, n_jobs: int = -1):
    model = MLPClassifier(random_state=random_state, max_iter=1000)
    param_dist = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["lbfgs", "adam"],
        "alpha": uniform(0.0001, 0.1),
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "learning_rate_init": uniform(0.0001, 0.1),
    }
    search = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    return search


def tune_gpc(X_train, y_train, random_state: int = 1, n_iter: int = 20, cv: int = 5, n_jobs: int = -1):
    model = GaussianProcessClassifier(random_state=random_state)
    param_dist = {
        "kernel": [1.0 * RBF(l) for l in uniform(0.01, 10).rvs(10)],
        "optimizer": ["fmin_l_bfgs_b", "fmin_tnc", "fmin_powell"],
        "max_iter_predict": randint(10, 500),
    }
    search = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    return search


def tune_extra_trees(X_train, y_train, random_state: int = 1, n_iter: int = 20, cv: int = 5, n_jobs: int = -1):
    model = ExtraTreesClassifier(random_state=random_state)
    param_dist = {
        "n_estimators": np.arange(100, 1000, 100),
        "max_depth": [None, 5, 10, 15],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": np.arange(2, 10, 2),
        "min_samples_leaf": np.arange(1, 5, 1),
        "bootstrap": [True, False],
    }
    search = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    return search


def tune_decision_tree(X_train, y_train, random_state: int = 42, n_iter: int = 50, cv: int = 5, n_jobs: int = -1):
    model = DecisionTreeClassifier(random_state=random_state)
    param_dist = {
        "max_depth": np.arange(2, 50, 1),
        "min_samples_split": np.arange(2, 20, 2),
        "min_samples_leaf": np.arange(1, 10, 1),
        "criterion": ["gini", "entropy"],
    }
    search = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    return search


def tune_adaboost(X_train, y_train, random_state: int = 42, n_iter: int = 50, cv: int = 5, n_jobs: int = -1):
    model = AdaBoostClassifier(random_state=random_state)
    param_dist = {
        "n_estimators": np.arange(50, 500, 50),
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1],
        "algorithm": ["SAMME", "SAMME.R"],
    }
    search = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=random_state)
    search.fit(X_train, y_train)
    return search


def tune_gaussian_nb(X_train, y_train, random_state: int = 1, n_iter: int = 50, cv: int = 5, n_jobs: int = -1):
    del random_state
    model = GaussianNB()
    param_dist = {
        "var_smoothing": np.arange(1e-10, 1e-8, 1e-9),
    }
    search = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=1)
    search.fit(X_train, y_train)
    return search


if __name__ == "__main__":
    for model_name, params in TRAINED_PARAMS.items():
        print(f"{model_name}: {params}")
