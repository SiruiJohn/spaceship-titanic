import argparse
import json
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, calinski_harabasz_score, precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "notebook_repro"
EDA_DIR = RESULTS_DIR / "eda"
METRICS_DIR = RESULTS_DIR / "metrics"
SUB_DIR = RESULTS_DIR / "submissions"
DATA_DIR = RESULTS_DIR / "data"
IMPORTANCE_DIR = RESULTS_DIR / "feature_importance"


def ensure_dirs() -> None:
    for path in [RESULTS_DIR, EDA_DIR, METRICS_DIR, SUB_DIR, DATA_DIR, IMPORTANCE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def bool_c(value):
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    if pd.isna(value):
        return np.nan
    return float(value)


def extract_last_name(value: str) -> str:
    value = str(value).strip().lower()
    parts = value.split()
    return parts[-1] if parts else "no_name"


def cabin_deck(value):
    if pd.isna(value):
        return "Missing_Deck"
    return str(value).split("/")[0]


def cabin_num(value):
    if pd.isna(value):
        return np.nan
    return float(str(value).split("/")[1])


def cabin_side(value):
    if pd.isna(value):
        return "Missing_Side"
    return str(value).split("/")[2]


def acc_cutoff(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    _, _, thresholds = stats_metrics_roc_curve(y_true, y_prob)
    pred_labels = (y_prob[None, :] > thresholds[:, None]).astype(int)
    acc_scores = (pred_labels == y_true).mean(axis=1)
    return float(thresholds[np.argmax(acc_scores)])


def stats_metrics_roc_curve(y_true: np.ndarray, y_prob: np.ndarray):
    from sklearn.metrics import roc_curve
    return roc_curve(y_true, y_prob)


def single_feature_cv_accuracy(train: pd.DataFrame, feature: str, target: str, n_splits: int = 5, random_state: int = 42) -> float:
    X = train[[feature]].to_numpy()
    y = train[target].to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        cutoff = acc_cutoff(y_val, y_prob)
        y_pred = (y_prob > cutoff).astype(int)
        scores.append(accuracy_score(y_val, y_pred))
    return float(np.mean(scores))


def choose_best_k(x: np.ndarray, k_min: int, k_max: int, random_state: int = 42):
    best_k = None
    best_score = -np.inf
    upper = min(k_max, max(k_min, len(x) - 1))
    for k in range(k_min, upper + 1):
        if len(x) <= k:
            break
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(x)
        if len(np.unique(labels)) < 2:
            continue
        score = calinski_harabasz_score(x, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, float(best_score) if np.isfinite(best_score) else None


def safe_woe_map(feature: pd.Series, target: pd.Series) -> dict:
    df = pd.DataFrame({"feature": feature, "target": target})
    grouped = df.groupby("feature")["target"].agg(["sum", "count"])
    pos = grouped["sum"] + 0.5
    neg = (grouped["count"] - grouped["sum"]) + 0.5
    woe = np.log1p(pos / neg)
    return woe.to_dict()


def ohe_like_notebook(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], target: str):
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    for col in cols:
        one_hot = pd.get_dummies(combined[col], prefix="", prefix_sep="")
        counts = combined[col].value_counts()
        if not counts.empty:
            min_count_category = counts.idxmin()
            if min_count_category in one_hot.columns:
                one_hot = one_hot.drop(columns=[min_count_category])
        combined = pd.concat([combined, one_hot], axis=1)
        combined = combined.drop(columns=[col])
        combined = combined.loc[:, ~combined.columns.duplicated()]
    train_ohe = combined.iloc[: len(train)].copy()
    test_ohe = combined.iloc[len(train):].copy()
    test_ohe = test_ohe.reset_index(drop=True)
    if target in test_ohe.columns:
        test_ohe = test_ohe.drop(columns=[target])
    return train_ohe, test_ohe


def load_data():
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    return train, test


def load_cached_features():
    train_path = DATA_DIR / "train_features.csv"
    test_path = DATA_DIR / "test_features.csv"
    if not train_path.exists() or not test_path.exists():
        return None, None
    return pd.read_csv(train_path), pd.read_csv(test_path)


def run_eda(train: pd.DataFrame, test: pd.DataFrame, fast: bool) -> None:
    eda_summary = {}
    cont_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() > 3]
    cat_cols = [c for c in train.columns if c not in cont_cols + ["PassengerId", "Transported"]]

    miss_train = train.isna().mean().sort_values(ascending=False)
    miss_test = test.isna().mean().sort_values(ascending=False)
    pd.concat([miss_train.rename("train_missing_ratio"), miss_test.rename("test_missing_ratio")], axis=1).to_csv(
        EDA_DIR / "missing_summary.csv"
    )

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Transported", data=train)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "target_distribution.png", dpi=200)
    plt.close()

    if cont_cols:
        train[cont_cols].describe().to_csv(EDA_DIR / "numeric_describe_train.csv")
        test[cont_cols].describe().to_csv(EDA_DIR / "numeric_describe_test.csv")

        fig, axs = plt.subplots(len(cont_cols), 2, figsize=(10, 4 * len(cont_cols)))
        if len(cont_cols) == 1:
            axs = np.array([axs])
        for i, col in enumerate(cont_cols):
            sns.histplot(train[col].dropna(), ax=axs[i][0], kde=True, color="steelblue")
            sns.histplot(test[col].dropna(), ax=axs[i][1], kde=True, color="indianred")
            axs[i][0].set_title(f"Train - {col}")
            axs[i][1].set_title(f"Test - {col}")
        fig.tight_layout()
        fig.savefig(EDA_DIR / "train_test_distributions.png", dpi=200)
        plt.close(fig)

        fig, axs = plt.subplots(len(cont_cols), 1, figsize=(8, 4 * len(cont_cols)))
        if len(cont_cols) == 1:
            axs = [axs]
        for i, col in enumerate(cont_cols):
            sns.boxplot(x="Transported", y=col, data=train, ax=axs[i])
            axs[i].set_title(f"{col} vs Target")
        fig.tight_layout()
        fig.savefig(EDA_DIR / "numeric_vs_target_boxplots.png", dpi=200)
        plt.close(fig)

        fig, axs = plt.subplots(len(cont_cols), 1, figsize=(8, 4 * len(cont_cols)))
        if len(cont_cols) == 1:
            axs = [axs]
        for i, col in enumerate(cont_cols):
            sns.violinplot(x="Transported", y=col, data=train, ax=axs[i])
            axs[i].set_title(f"{col} vs Target Violin")
        fig.tight_layout()
        fig.savefig(EDA_DIR / "numeric_vs_target_violins.png", dpi=200)
        plt.close(fig)

        if not fast and len(cont_cols) <= 8:
            pair = sns.pairplot(data=train[cont_cols + ["Transported"]], vars=cont_cols, hue="Transported")
            pair.savefig(EDA_DIR / "pairplot.png", dpi=150)
            plt.close("all")

        ttest_rows = []
        anova_rows = []
        for col in cont_cols:
            zero = train.loc[train["Transported"] == 0, col].dropna()
            one = train.loc[train["Transported"] == 1, col].dropna()
            if len(zero) > 1 and len(one) > 1:
                t_stat, t_p = stats.ttest_ind(zero, one, equal_var=False)
                f_stat, f_p = stats.f_oneway(zero, one)
                ttest_rows.append({"feature": col, "t_stat": t_stat, "p_value": t_p})
                anova_rows.append({"feature": col, "f_stat": f_stat, "p_value": f_p})
        pd.DataFrame(ttest_rows).sort_values("p_value").to_csv(EDA_DIR / "ttest_results.csv", index=False)
        pd.DataFrame(anova_rows).sort_values("p_value").to_csv(EDA_DIR / "anova_results.csv", index=False)

        pair_rows = []
        if len(cont_cols) >= 2:
            imputed = train[cont_cols].fillna(train[cont_cols].median())
            pair_candidates = []
            for i in range(len(cont_cols)):
                for j in range(i + 1, len(cont_cols)):
                    pair_candidates.append((cont_cols[i], cont_cols[j]))
            for col1, col2 in pair_candidates:
                X = imputed[[col1, col2]].to_numpy()
                y = train["Transported"].astype(int).to_numpy()
                kf = KFold(n_splits=5 if fast else 10, shuffle=True, random_state=42)
                scores = []
                for train_idx, val_idx in kf.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    model = SVC(probability=True, gamma="scale", random_state=42)
                    model.fit(X_train, y_train)
                    y_prob = model.predict_proba(X_val)[:, 1]
                    cutoff = acc_cutoff(y_val, y_prob)
                    scores.append(accuracy_score(y_val, (y_prob > cutoff).astype(int)))
                pair_rows.append({"pair": f"{col1}__{col2}", "cv_accuracy": float(np.mean(scores))})
            pd.DataFrame(pair_rows).sort_values("cv_accuracy", ascending=False).to_csv(
                EDA_DIR / "pair_feature_svc_accuracy.csv", index=False
            )

    if cat_cols:
        rows = []
        for col in cat_cols:
            ct = pd.crosstab(train[col].fillna("Missing"), train["Transported"], normalize="index")
            ct.to_csv(EDA_DIR / f"cat_target_dist_{col}.csv")
            rows.append({"feature": col, "unique_count": int(train[col].nunique(dropna=False))})
        pd.DataFrame(rows).to_csv(EDA_DIR / "categorical_summary.csv", index=False)

    corr_features = [c for c in train.columns if c not in ["Transported"] and pd.api.types.is_numeric_dtype(train[c])]
    if corr_features:
        plt.figure(figsize=(14, 12))
        corr = train[corr_features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "correlation_heatmap.png", dpi=200)
        plt.close()

    eda_summary["cont_cols"] = cont_cols
    eda_summary["cat_cols"] = cat_cols
    save_json(EDA_DIR / "eda_summary.json", eda_summary)


def preprocess_base(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()

    cont_cols = [f for f in train.columns if pd.api.types.is_numeric_dtype(train[f]) and train[f].nunique() > 3]

    train["group"] = train["PassengerId"].str[:-3].astype(int)
    test["group"] = test["PassengerId"].str[:-3].astype(int)
    train["cabin_deck"] = train["Cabin"].apply(cabin_deck)
    test["cabin_deck"] = test["Cabin"].apply(cabin_deck)
    train["cabin_num"] = train["Cabin"].apply(cabin_num)
    test["cabin_num"] = test["Cabin"].apply(cabin_num)
    train["cabin_side"] = train["Cabin"].apply(cabin_side)
    test["cabin_side"] = test["Cabin"].apply(cabin_side)
    train = train.drop(columns=["Cabin"])
    test = test.drop(columns=["Cabin"])

    train["Name"] = train["Name"].fillna("No_Name")
    test["Name"] = test["Name"].fillna("No_Name")
    train["Last_Name"] = train["Name"].apply(extract_last_name)
    test["Last_Name"] = test["Name"].apply(extract_last_name)
    train["VIP"] = train["VIP"].apply(bool_c)
    test["VIP"] = test["VIP"].apply(bool_c)
    train["CryoSleep"] = train["CryoSleep"].apply(bool_c)
    test["CryoSleep"] = test["CryoSleep"].apply(bool_c)
    train["Transported"] = train["Transported"].astype(int)

    miss_cat = [feature for feature in train.columns if train[feature].isnull().sum() > 0 and train[feature].dtype == "O"]
    for feature in miss_cat:
        train[feature] = train[feature].fillna(f"missing_{feature}")
        test[feature] = test[feature].fillna(f"missing_{feature}")

    exp_features = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    train["Expenditure"] = train[exp_features].sum(axis=1)
    test["Expenditure"] = test[exp_features].sum(axis=1)
    train["CryoSleep"] = np.where(train["Expenditure"] == 0, 1, 0)
    test["CryoSleep"] = np.where(test["Expenditure"] == 0, 1, 0)
    train["VIP"] = np.where(train["CryoSleep"] == 0, 1, 0)
    test["VIP"] = np.where(test["CryoSleep"] == 0, 1, 0)
    train = train.drop(columns=["Expenditure"])
    test = test.drop(columns=["Expenditure"])

    for col in exp_features:
        train[col] = np.where(train["CryoSleep"] == 1, 0, train[col])
        test[col] = np.where(test["CryoSleep"] == 1, 0, test[col])

    miss_cont = [feature for feature in train.columns if train[feature].isnull().sum() > 0 and train[feature].dtype != "O" and feature != "Transported"]
    if miss_cont:
        imputer = KNNImputer(n_neighbors=5)
        train[miss_cont] = imputer.fit_transform(train[miss_cont])
        test[miss_cont] = imputer.transform(test[miss_cont])

    train["expenditure"] = train["VRDeck"] + train["Spa"] + train["RoomService"]
    test["expenditure"] = test["VRDeck"] + test["Spa"] + test["RoomService"]
    num_feat = [f for f in train.columns if train[f].dtype != "O" and train[f].nunique() > 10]

    return train, test, cont_cols, num_feat


def select_numeric_transformations(train: pd.DataFrame, test: pd.DataFrame, num_feat: list[str], fast: bool):
    sc = MinMaxScaler()
    unimportant_features = []
    rows = []
    for col in num_feat:
        train[f"log_{col}"] = np.log1p(sc.fit_transform(train[[col]]))
        test[f"log_{col}"] = np.log1p(sc.transform(test[[col]]))

        train[f"sqrt_{col}"] = np.sqrt(sc.fit_transform(train[[col]]))
        test[f"sqrt_{col}"] = np.sqrt(sc.transform(test[[col]]))

        transformer = PowerTransformer(method="box-cox")
        train[f"bx_cx_{col}"] = transformer.fit_transform(sc.fit_transform(train[[col]]) + 1.0)
        test[f"bx_cx_{col}"] = transformer.transform(sc.transform(test[[col]]) + 1.0)

        transformer = PowerTransformer(method="yeo-johnson")
        train[f"y_J_{col}"] = transformer.fit_transform(train[[col]])
        test[f"y_J_{col}"] = transformer.transform(test[[col]])

        transformer = FunctionTransformer(lambda x: np.power(x, 0.25))
        train[f"pow_{col}"] = transformer.fit_transform(sc.fit_transform(train[[col]]))
        test[f"pow_{col}"] = transformer.transform(sc.transform(test[[col]]))

        transformer = FunctionTransformer(lambda x: np.power(x, 0.1))
        train[f"pow2_{col}"] = transformer.fit_transform(sc.fit_transform(train[[col]]))
        test[f"pow2_{col}"] = transformer.transform(sc.transform(test[[col]]))

        train[f"log_pow2{col}"] = np.log1p(train[f"pow2_{col}"])
        test[f"log_pow2{col}"] = np.log1p(test[f"pow2_{col}"])

        temp_cols = [
            col,
            f"log_{col}",
            f"sqrt_{col}",
            f"bx_cx_{col}",
            f"y_J_{col}",
            f"pow_{col}",
            f"pow2_{col}",
            f"log_pow2{col}",
        ]
        train[temp_cols] = train[temp_cols].fillna(0)
        test[temp_cols] = test[temp_cols].fillna(0)

        pca = TruncatedSVD(n_components=1, random_state=42)
        pca_train = pd.DataFrame(pca.fit_transform(train[temp_cols]), columns=[f"{col}_pca_comb"])
        pca_test = pd.DataFrame(pca.transform(test[temp_cols]), columns=[f"{col}_pca_comb"])
        train = pd.concat([train, pca_train], axis=1)
        test = pd.concat([test, pca_test], axis=1)
        temp_cols.append(f"{col}_pca_comb")

        acc_rows = []
        orig_acc = None
        for feature in temp_cols:
            score = single_feature_cv_accuracy(train, feature, "Transported", n_splits=5 if fast else 10)
            acc_rows.append((feature, score))
            if feature == col:
                orig_acc = score
        best_col, best_acc = sorted(acc_rows, key=lambda x: x[1], reverse=True)[0]
        cols_to_drop = [f for f in temp_cols if f != best_col]
        unimportant_features.extend(cols_to_drop)
        rows.append(
            {
                "original_feature": col,
                "original_accuracy": orig_acc,
                "selected_feature": best_col,
                "selected_accuracy": best_acc,
            }
        )
    pd.DataFrame(rows).to_csv(METRICS_DIR / "numeric_transformation_selection.csv", index=False)
    return train, test, unimportant_features


def add_tfidf_features(train: pd.DataFrame, test: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=1000)
    vectors_train = vectorizer.fit_transform(train["Last_Name"])
    vectors_test = vectorizer.transform(test["Last_Name"])
    svd = TruncatedSVD(n_components=5, random_state=42)
    tfidf_train = pd.DataFrame(svd.fit_transform(vectors_train))
    tfidf_test = pd.DataFrame(svd.transform(vectors_test))
    cols = [f"Last_Name_tfidf_{i}" for i in range(tfidf_train.shape[1])]
    tfidf_train.columns = cols
    tfidf_test.columns = cols
    train = pd.concat([train.reset_index(drop=True), tfidf_train], axis=1)
    test = pd.concat([test.reset_index(drop=True), tfidf_test], axis=1)
    train = train.drop(columns=["Name", "Last_Name"])
    test = test.drop(columns=["Name", "Last_Name"])
    return train, test


def select_categorical_encodings(train: pd.DataFrame, test: pd.DataFrame, fast: bool):
    cat_features = ["HomePlanet", "cabin_deck", "Destination", "cabin_side"]
    rows = []
    for feature in cat_features:
        cat_labels = train.groupby(feature)["Transported"].mean().sort_values().index
        cat_map = {k: i for i, k in enumerate(cat_labels, 0)}
        train[f"{feature}_target"] = train[feature].map(cat_map)
        test[f"{feature}_target"] = test[feature].map(cat_map)

        count_map = train[feature].value_counts().to_dict()
        train[f"{feature}_count"] = np.log1p(train[feature].map(count_map))
        test[f"{feature}_count"] = np.log1p(test[feature].map(count_map).fillna(0))

        count_label_map = dict(zip(count_map.keys(), np.arange(len(count_map), 0, -1)))
        train[f"{feature}_count_label"] = train[feature].replace(count_label_map)
        test[f"{feature}_count_label"] = test[feature].replace(count_label_map)

        woe_map = safe_woe_map(train[feature], train["Transported"])
        train[f"{feature}_WOE"] = train[feature].map(woe_map)
        test[f"{feature}_WOE"] = test[feature].map(woe_map)

        temp_cols = [f"{feature}_target", f"{feature}_count", f"{feature}_count_label", f"{feature}_WOE"]
        imputer = KNNImputer(n_neighbors=5)
        train[temp_cols] = imputer.fit_transform(train[temp_cols])
        test[temp_cols] = imputer.transform(test[temp_cols])

        if train[feature].dtype != "O":
            temp_cols.append(feature)
        else:
            train = train.drop(columns=[feature])
            test = test.drop(columns=[feature])

        temp_train = train[temp_cols].copy()
        temp_test = test[temp_cols].copy()
        sc = StandardScaler()
        temp_train_scaled = sc.fit_transform(temp_train)
        temp_test_scaled = sc.transform(temp_test)
        ideal_k, ch_score = choose_best_k(temp_train_scaled, 3, 10 if fast else 15)

        encoded_candidates = temp_cols.copy()
        if ideal_k is not None:
            kmeans = KMeans(n_clusters=ideal_k, random_state=42, n_init=10)
            train[f"{feature}_cat_cluster_WOE"] = kmeans.fit_predict(temp_train_scaled)
            test[f"{feature}_cat_cluster_WOE"] = kmeans.predict(temp_test_scaled)
            train[f"{feature}_cat_OHE_cluster"] = feature + "_OHE_" + train[f"{feature}_cat_cluster_WOE"].astype(str)
            test[f"{feature}_cat_OHE_cluster"] = feature + "_OHE_" + test[f"{feature}_cat_cluster_WOE"].astype(str)
            train, test = ohe_like_notebook(train, test, [f"{feature}_cat_OHE_cluster"], "Transported")
            cluster_map = np.log1p(train.groupby(f"{feature}_cat_cluster_WOE")["Transported"].mean()).to_dict()
            train[f"{feature}_cat_cluster_WOE"] = train[f"{feature}_cat_cluster_WOE"].map(cluster_map)
            test[f"{feature}_cat_cluster_WOE"] = test[f"{feature}_cat_cluster_WOE"].map(cluster_map)
            encoded_candidates.append(f"{feature}_cat_cluster_WOE")

        best_feature = None
        best_score = -np.inf
        for candidate in encoded_candidates:
            score = single_feature_cv_accuracy(train, candidate, "Transported", n_splits=5 if fast else 10)
            if score > best_score:
                best_score = score
                best_feature = candidate
        rows.append(
            {
                "feature": feature,
                "selected_feature": best_feature,
                "selected_accuracy": best_score,
                "cluster_k": ideal_k,
                "cluster_score": ch_score,
            }
        )
        drop_cols = [c for c in encoded_candidates if c != best_feature]
        existing = [c for c in drop_cols if c in train.columns and c in test.columns]
        if existing:
            train = train.drop(columns=existing)
            test = test.drop(columns=existing)
    pd.DataFrame(rows).to_csv(METRICS_DIR / "categorical_encoding_selection.csv", index=False)
    return train, test


def reduce_unimportant_features(train: pd.DataFrame, test: pd.DataFrame, num_feat: list[str], unimportant_features: list[str], fast: bool):
    rows = []
    kf_splits = 5 if fast else 10
    for col in num_feat:
        sub_set = [f for f in unimportant_features if col in f and f in train.columns]
        if not sub_set:
            continue
        temp_train = train[sub_set].copy()
        temp_test = test[sub_set].copy()
        sc = StandardScaler()
        temp_train_scaled = sc.fit_transform(temp_train)
        temp_test_scaled = sc.transform(temp_test)
        ideal_k, _ = choose_best_k(temp_train_scaled, 3, 15 if fast else 25)
        if ideal_k is None:
            ideal_k = min(10, max(3, len(temp_train_scaled) - 1))
        kmeans = KMeans(n_clusters=ideal_k, random_state=42, n_init=10)
        train[f"{col}_OHE_cluster"] = kmeans.fit_predict(temp_train_scaled)
        test[f"{col}_OHE_cluster"] = kmeans.predict(temp_test_scaled)
        train[f"{col}_unimp_cluster_WOE"] = train[f"{col}_OHE_cluster"]
        test[f"{col}_unimp_cluster_WOE"] = test[f"{col}_OHE_cluster"]
        cluster_map = np.log1p(train.groupby(f"{col}_unimp_cluster_WOE")["Transported"].mean()).to_dict()
        train[f"{col}_unimp_cluster_WOE"] = train[f"{col}_unimp_cluster_WOE"].map(cluster_map)
        test[f"{col}_unimp_cluster_WOE"] = test[f"{col}_unimp_cluster_WOE"].map(cluster_map)
        score = single_feature_cv_accuracy(train, f"{col}_unimp_cluster_WOE", "Transported", n_splits=kf_splits)
        rows.append({"feature": f"{col}_unimp_cluster_WOE", "cv_accuracy": score, "cluster_k": ideal_k})
        train[f"{col}_OHE_cluster"] = col + "_OHE_" + train[f"{col}_OHE_cluster"].astype(str)
        test[f"{col}_OHE_cluster"] = col + "_OHE_" + test[f"{col}_OHE_cluster"].astype(str)
        train, test = ohe_like_notebook(train, test, [f"{col}_OHE_cluster"], "Transported")
    pd.DataFrame(rows).to_csv(METRICS_DIR / "unimportant_feature_cluster_summary.csv", index=False)
    return train, test


def final_numeric_feature_selection(train: pd.DataFrame, test: pd.DataFrame, cont_cols: list[str], num_feat: list[str], fast: bool):
    num_derived_list = []
    for derived in train.columns:
        for base in num_feat:
            if base in derived:
                num_derived_list.append(derived)
    num_derived_list = sorted(set(num_derived_list))

    final_drop_list = []
    best_cols = []
    rows = []
    threshold = 0.8

    for col in num_feat:
        sub_set = [f for f in num_derived_list if col in f and f in train.columns]
        correlated_features = []
        for i, feature in enumerate(sub_set):
            for j in range(i + 1, len(sub_set)):
                correlation = abs(train[feature].corr(train[sub_set[j]]))
                if correlation > threshold:
                    correlated_features.append(sub_set[j])
        correlated_features = sorted(set(correlated_features))
        if len(correlated_features) > 1:
            temp_train = train[correlated_features].copy()
            temp_test = test[correlated_features].copy()
            sc = StandardScaler()
            temp_train_scaled = sc.fit_transform(temp_train)
            temp_test_scaled = sc.transform(temp_test)

            pca = TruncatedSVD(n_components=1, random_state=42)
            train[f"{col}_pca_comb_final"] = pca.fit_transform(temp_train_scaled)
            test[f"{col}_pca_comb_final"] = pca.transform(temp_test_scaled)

            ideal_k, _ = choose_best_k(temp_train_scaled, 10 if not fast else 3, 10 if fast else 25)
            if ideal_k is None:
                ideal_k = 10 if not fast else 5
            kmeans = KMeans(n_clusters=ideal_k, random_state=42, n_init=10)
            train[f"{col}_final_cluster"] = kmeans.fit_predict(temp_train_scaled)
            test[f"{col}_final_cluster"] = kmeans.predict(temp_test_scaled)
            cluster_map = np.log1p(train.groupby(f"{col}_final_cluster")["Transported"].mean()).to_dict()
            train[f"{col}_final_cluster"] = train[f"{col}_final_cluster"].map(cluster_map)
            test[f"{col}_final_cluster"] = test[f"{col}_final_cluster"].map(cluster_map)

            correlated_features = correlated_features + [f"{col}_pca_comb_final", f"{col}_final_cluster"]
            scores = []
            for feature in correlated_features:
                if feature in best_cols:
                    continue
                score = single_feature_cv_accuracy(train, feature, "Transported", n_splits=5 if fast else 10)
                scores.append((feature, score))
            best_col, best_acc = sorted(scores, key=lambda x: x[1], reverse=True)[0]
            best_cols.append(best_col)
            cols_to_drop = [f for f in correlated_features if f not in best_cols]
            final_drop_list.extend(cols_to_drop)
            rows.append({"original": col, "final_transformed_feature": best_col, "cv_accuracy": best_acc})
        else:
            rows.append({"original": col, "final_transformed_feature": "All features selected", "cv_accuracy": None})

    final_drop_list = [f for f in final_drop_list if f not in cont_cols]
    final_drop_list = sorted(set([f for f in final_drop_list if f in train.columns]))
    if final_drop_list:
        train = train.drop(columns=final_drop_list)
        test = test.drop(columns=[f for f in final_drop_list if f in test.columns])

    pd.DataFrame(rows).to_csv(METRICS_DIR / "final_numeric_feature_selection.csv", index=False)
    return train, test, num_derived_list, final_drop_list


def scale_and_split(train: pd.DataFrame, test: pd.DataFrame):
    feature_scale = [feature for feature in train.columns if feature not in ["PassengerId", "Transported"]]
    scaler = StandardScaler()
    train[feature_scale] = scaler.fit_transform(train[feature_scale])
    test[feature_scale] = scaler.transform(test[feature_scale])
    ids = test[["PassengerId"]].copy()
    train = train.drop(columns=["PassengerId"])
    test = test.drop(columns=["PassengerId"])
    X_train = train.drop(columns=["Transported"])
    y_train = train["Transported"].astype(int)
    X_test = test.copy()
    return X_train, y_train, X_test, ids


class NotebookClassifier:
    def __init__(self, n_estimators: int, random_state: int, fast: bool, early_stopping_rounds: int):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.fast = fast
        self.early_stopping_rounds = early_stopping_rounds

    def build(self):
        # Match the tuned notebook parameters, while still allowing CLI overrides
        # for total boosting rounds during full reproduction runs.
        xgb_params = {
            "colsample_bytree": 0.8498791800104656,
            "learning_rate": 0.020233442882782587,
            "max_depth": 4,
            "subsample": 0.746529796772373,
            "n_estimators": self.n_estimators,
            "early_stopping_rounds": self.early_stopping_rounds,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": self.random_state,
            "n_jobs": 1 if self.fast else -1,
        }
        lgb_params = {
            "colsample_bytree": 0.7774799983649324,
            "learning_rate": 0.007653648135411494,
            "max_depth": 5,
            "reg_alpha": 0.14326300616140863,
            "reg_lambda": 0.9310129332502252,
            "subsample": 0.6189257947519665,
            "n_estimators": self.n_estimators,
            "objective": "binary",
            "random_state": self.random_state,
            "verbosity": -1,
            "n_jobs": 1 if self.fast else -1,
        }
        cat_params = {
            "random_strength": 0.1,
            "one_hot_max_size": 10,
            "max_bin": 100,
            "learning_rate": 0.01,
            "l2_leaf_reg": 0.5,
            "grow_policy": "Lossguide",
            "depth": 5,
            "bootstrap_type": "Bernoulli",
            "n_estimators": self.n_estimators,
            "task_type": "CPU",
            "random_state": self.random_state,
            "verbose": False,
            "thread_count": 1 if self.fast else -1,
        }
        return {
            "xgb": xgb.XGBClassifier(**xgb_params),
            "lgb": lgb.LGBMClassifier(**lgb_params),
            "cat": CatBoostClassifier(**cat_params),
        }


class OptunaWeights:
    def __init__(self, random_state: int):
        self.random_state = random_state
        self.weights = None

    def _objective(self, trial, y_true, y_preds):
        weights = [trial.suggest_float(f"weight{n}", 0.0, 1.0) for n in range(len(y_preds))]
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)
        cutoff = acc_cutoff(y_true, weighted_pred)
        y_weight_pred = np.where(weighted_pred > cutoff, 1, 0)
        return accuracy_score(y_true, y_weight_pred)

    def fit(self, y_true, y_preds, n_trials: int):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(sampler=sampler, study_name="OptunaWeights", direction="maximize")
        study.optimize(lambda trial: self._objective(trial, y_true, y_preds), n_trials=n_trials)
        self.weights = [study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        return np.average(np.array(y_preds).T, axis=1, weights=self.weights)

    def fit_predict(self, y_true, y_preds, n_trials: int):
        self.fit(y_true, y_preds, n_trials)
        return self.predict(y_preds)


def visualize_importance(models: list, feature_cols: list[str], title: str):
    feature_importance = pd.DataFrame()
    for i, model in enumerate(models):
        if not hasattr(model, "feature_importances_"):
            continue
        fold_df = pd.DataFrame({"importance": model.feature_importances_, "feature": feature_cols, "fold": i})
        feature_importance = pd.concat([feature_importance, fold_df], ignore_index=True)
    if feature_importance.empty:
        return
    feature_importance = feature_importance.groupby("feature", as_index=False)["importance"].mean().sort_values(
        "importance", ascending=False
    )
    top_df = feature_importance.head(20)
    plt.figure(figsize=(12, 10))
    sns.barplot(x="importance", y="feature", data=top_df, color="skyblue")
    plt.title(f"{title} Feature Importance Top 20")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / f"{title}_feature_importance.png", dpi=200)
    plt.close()


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    fast: bool,
    n_splits_override: int | None = None,
    n_estimators_override: int | None = None,
    optuna_trials_override: int | None = None,
):
    kfold = True
    n_splits = n_splits_override if n_splits_override is not None else (3 if fast else 10)
    random_state = 2023
    random_state_list = [2140]
    n_estimators = n_estimators_override if n_estimators_override is not None else (300 if fast else 9999)
    early_stopping_rounds = 50 if fast else 200
    optuna_trials = optuna_trials_override if optuna_trials_override is not None else (80 if fast else 2000)

    splitter = StratifiedKFold(n_splits=n_splits, random_state=random_state_list[0], shuffle=True)
    test_predss = np.zeros(X_test.shape[0], dtype=float)
    ensemble_score = []
    weights = []
    trained_models = {"xgb": [], "lgb": [], "cat": []}
    oof_ensemble = np.zeros(len(X_train), dtype=float)
    last_val_idx = None
    last_val_pred = None

    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        models = NotebookClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            fast=fast,
            early_stopping_rounds=early_stopping_rounds,
        ).build()
        oof_preds = []
        test_preds = []
        for name, model in models.items():
            if name == "cat":
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            elif name == "lgb":
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
                )
            elif name == "xgb":
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            else:
                model.fit(X_tr, y_tr)

            y_val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(X_test)[:, 1]
            score = accuracy_score(y_val, (y_val_pred > acc_cutoff(y_val, y_val_pred)).astype(int))
            fold_rows.append({"fold": fold, "model": name, "accuracy": score})
            oof_preds.append(y_val_pred)
            test_preds.append(test_pred)
            trained_models[name].append(model)

        optweights = OptunaWeights(random_state=random_state)
        ensemble_val_pred = optweights.fit_predict(y_val.to_numpy(), oof_preds, n_trials=optuna_trials)
        ensemble_acc = accuracy_score(y_val, (ensemble_val_pred > acc_cutoff(y_val, ensemble_val_pred)).astype(int))
        ensemble_score.append(ensemble_acc)
        weights.append(optweights.weights)
        test_predss += optweights.predict(test_preds) / n_splits
        oof_ensemble[val_idx] = ensemble_val_pred
        last_val_idx = val_idx
        last_val_pred = ensemble_val_pred
        fold_rows.append({"fold": fold, "model": "ensemble", "accuracy": ensemble_acc})
        pd.DataFrame(fold_rows).to_csv(METRICS_DIR / "fold_scores_in_progress.csv", index=False)
        save_json(
            METRICS_DIR / "ensemble_progress.json",
            {
                "completed_folds": int(fold + 1),
                "total_folds": int(n_splits),
                "current_mean_accuracy": float(np.mean(ensemble_score)),
                "current_std_accuracy": float(np.std(ensemble_score)) if len(ensemble_score) > 1 else 0.0,
                "latest_fold_accuracy": float(ensemble_acc),
                "latest_weights": {name: float(w) for name, w in zip(["xgb", "lgb", "cat"], optweights.weights)},
                "n_estimators": int(n_estimators),
                "optuna_trials": int(optuna_trials),
            },
        )

    mean_weights = np.mean(weights, axis=0).tolist()
    model_names = ["xgb", "lgb", "cat"]
    cutoff = acc_cutoff(y_train.to_numpy(), oof_ensemble) if np.any(oof_ensemble) else 0.5
    summary = {
        "ensemble_accuracy_mean": float(np.mean(ensemble_score)),
        "ensemble_accuracy_std": float(np.std(ensemble_score)),
        "cutoff": float(cutoff),
        "weights_mean": {name: float(w) for name, w in zip(model_names, mean_weights)},
        "n_splits": int(n_splits),
        "n_estimators": int(n_estimators),
        "optuna_trials": int(optuna_trials),
    }
    pd.DataFrame(fold_rows).to_csv(METRICS_DIR / "fold_scores.csv", index=False)
    save_json(METRICS_DIR / "ensemble_summary.json", summary)

    for name, models in trained_models.items():
        visualize_importance(models, list(X_train.columns), name)

    return test_predss, cutoff


def combine_or_gate(ids: pd.DataFrame, base_submission: pd.DataFrame, submission_paths: list[str] | None):
    if not submission_paths or len(submission_paths) == 0:
        return None
    subs = []
    for path in submission_paths:
        sub_path = Path(path)
        if sub_path.exists():
            subs.append(pd.read_csv(sub_path))
    if not subs:
        return None
    combined = base_submission.copy()
    value = combined["Transported"].astype(bool)
    for sub in subs:
        value = value | sub["Transported"].astype(bool)
    combined["Transported"] = value
    combined.to_csv(SUB_DIR / "submission_or_gate.csv", index=False)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Reproduce the full Spaceship Titanic notebook pipeline.")
    parser.add_argument("--fast", action="store_true", help="Run a lighter version for quick validation.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA artifact generation.")
    parser.add_argument("--or-submission", action="append", default=[], help="Optional submission paths for OR-gate experiment.")
    parser.add_argument("--n-splits", type=int, default=None, help="Override CV fold count.")
    parser.add_argument("--n-estimators", type=int, default=None, help="Override tree estimator count.")
    parser.add_argument("--optuna-trials", type=int, default=None, help="Override Optuna weight search trials.")
    parser.add_argument("--reuse-features", action="store_true", help="Reuse cached engineered features if available.")
    args = parser.parse_args()

    ensure_dirs()
    save_json(
        METRICS_DIR / "last_run.json",
        {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "fast": bool(args.fast),
            "skip_eda": bool(args.skip_eda),
            "n_splits": args.n_splits,
            "n_estimators": args.n_estimators,
            "optuna_trials": args.optuna_trials,
        },
    )
    train_raw, test_raw = load_data()

    if not args.skip_eda:
        run_eda(train_raw.copy(), test_raw.copy(), fast=args.fast)

    train = test = None
    if args.reuse_features:
        train, test = load_cached_features()

    if train is None or test is None:
        train, test, cont_cols, num_feat = preprocess_base(train_raw, test_raw)
        train, test, unimportant_features = select_numeric_transformations(train, test, num_feat, fast=args.fast)
        train, test = add_tfidf_features(train, test)
        train, test = select_categorical_encodings(train, test, fast=args.fast)
        train, test = reduce_unimportant_features(train, test, num_feat, unimportant_features, fast=args.fast)
        train, test, num_derived_list, final_drop_list = final_numeric_feature_selection(
            train, test, cont_cols, num_feat, fast=args.fast
        )

        train.to_csv(DATA_DIR / "train_features.csv", index=False)
        test.to_csv(DATA_DIR / "test_features.csv", index=False)
        save_json(
            METRICS_DIR / "feature_artifacts.json",
            {
                "cont_cols": cont_cols,
                "num_feat": num_feat,
                "num_derived_list_count": len(num_derived_list),
                "unimportant_features_count": len(unimportant_features),
                "final_drop_list": final_drop_list,
                "train_shape_before_scaling": list(train.shape),
                "test_shape_before_scaling": list(test.shape),
            },
        )

    X_train, y_train, X_test, ids = scale_and_split(train.copy(), test.copy())
    pd.DataFrame({"train_rows": [len(X_train)], "train_cols": [X_train.shape[1]], "test_rows": [len(X_test)], "test_cols": [X_test.shape[1]]}).to_csv(
        METRICS_DIR / "final_shapes.csv", index=False
    )

    test_predss, cutoff = train_ensemble(
        X_train,
        y_train,
        X_test,
        fast=args.fast,
        n_splits_override=args.n_splits,
        n_estimators_override=args.n_estimators,
        optuna_trials_override=args.optuna_trials,
    )

    submission = ids.copy()
    submission["Transported"] = np.where(test_predss > cutoff, 1, 0).astype(bool)
    submission.to_csv(SUB_DIR / "submission_model.csv", index=False)
    save_json(
        METRICS_DIR / "submission_distribution.json",
        (submission["Transported"].value_counts(normalize=True).rename_axis("Transported").to_dict()),
    )

    combine_or_gate(ids, submission, args.or_submission)


if __name__ == "__main__":
    main()
