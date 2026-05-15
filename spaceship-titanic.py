# Auto-exported from Jupyter notebook
# Source: spaceship-titanic.ipynb
from __future__ import annotations


# %% [code] cell 1 (notebook cell 3)
# execution_count = 1

import builtins, os, random, re, warnings, json, math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings('ignore')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

try:
    import xgboost as xgb;   XGB_AVAILABLE = True
except Exception:            XGB_AVAILABLE = False
try:
    import lightgbm as lgb;  LGB_AVAILABLE = True
except Exception:            LGB_AVAILABLE = False
try:
    from catboost import CatBoostClassifier; CAT_AVAILABLE = True
except Exception:            CAT_AVAILABLE = False

# -- Reproducibility ----------------------------------------------------------
GLOBAL_RANDOM_STATE = 42

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_global_seed(GLOBAL_RANDOM_STATE)

# -- Artifact saving ---------------------------------------------------------
ARTIFACT_ROOT = Path("results/notebook_export")
FIGURE_DIR = ARTIFACT_ROOT / "figures"
TABLE_DIR = ARTIFACT_ROOT / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
_figure_counter = 0
_table_counter = 0
_ORIGINAL_PRINT = builtins.print

def _slugify(text, fallback):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or fallback

def save_table_artifact(name, obj):
    global _table_counter
    _table_counter += 1
    base_name = f"{_table_counter:03d}_{_slugify(name, 'table')}"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(TABLE_DIR / f"{base_name}.csv", index=True)
        obj.to_html(TABLE_DIR / f"{base_name}.html", index=True)
    elif isinstance(obj, pd.Series):
        obj.to_frame(name=obj.name or "value").to_csv(TABLE_DIR / f"{base_name}.csv", index=True)

def _guess_figure_name(fig):
    if getattr(fig, "_suptitle", None) and fig._suptitle.get_text():
        return fig._suptitle.get_text()
    for ax in fig.axes:
        if ax.get_title():
            return ax.get_title()
    return "figure"

def save_figure_artifact(fig):
    global _figure_counter
    _figure_counter += 1
    figure_name = _guess_figure_name(fig)
    base_name = f"{_figure_counter:03d}_{_slugify(figure_name, 'figure')}"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{base_name}.png", dpi=300, bbox_inches="tight")

def finalize_figures():
    for fig_num in plt.get_fignums():
        save_figure_artifact(plt.figure(fig_num))
    plt.close('all')

def tracked_print(*args, **kwargs):
    for arg in args:
        if isinstance(arg, (pd.DataFrame, pd.Series)):
            save_table_artifact(type(arg).__name__, arg)
    return _ORIGINAL_PRINT(*args, **kwargs)

# -- Dark theme ---------------------------------------------------------------
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',   'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',      'ytick.color': '#8b949e',
    'grid.color': '#21262d',       'text.color': '#e6edf3',
    'font.size': 11,
})
BLUE, RED, GREEN = '#58a6ff', '#f78166', '#3fb950'
PURPLE, ORANGE   = '#d2a8ff', '#ffa657'
PALETTE          = [BLUE, RED, GREEN, PURPLE, ORANGE, '#79c0ff']

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:,.5f}'.format

print('[OK] All libraries loaded.')
print(
    f'   XGBoost={XGB_AVAILABLE} | LightGBM={LGB_AVAILABLE} | '
    f'CatBoost={CAT_AVAILABLE} | Optuna={OPTUNA_AVAILABLE} | SHAP={SHAP_AVAILABLE}'
)


# %% [code] cell 2 (notebook cell 4)
# execution_count = 2

class CFG:
    # -- Paths -------------------------------------------------------------
    competition_name = 'spaceship-titanic'
    local_data_dir   = Path('data/raw')
    input_root       = Path('/kaggle/input/competitions/spaceship-titanic')
    input_dir        = None
    target           = 'Transported'

    # -- CV ----------------------------------------------------------------
    random_seeds     = [42, 2024, 7, 2025, 88]
    n_splits         = 5              # -> 25 total folds/model
    pseudo_threshold = 0.92           # confidence required for pseudo-labels
    pseudo_rounds    = 2              # how many pseudo-label iterations

    # -- Optuna ------------------------------------------------------------
    optuna_trials    = 200
    optuna_cv_folds  = 3

    # -- Multi-run ensemble ------------------------------------------------
    ensemble_runs       = 3
    ensemble_base_seeds = [42, 123, 888]

    # -- Features ----------------------------------------------------------
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    categorical_base = [
        'HomePlanet', 'CryoSleep', 'Destination', 'VIP',
        'CabinDeck', 'CabinSide', 'HomeDest', 'DeckSide',
        'CabinZone', 'AgeBand', 'Surname'
    ]

    # Feature cols -- extended vs v1
    feature_cols = [
        # Original categoricals
        'HomePlanet', 'CryoSleep', 'Destination', 'VIP',
        'CabinDeck', 'CabinSide', 'HomeDest', 'DeckSide', 'CabinZone', 'AgeBand', 'Surname',
        # Group features (structural -- don't require train/test overlap)
        'GroupSize', 'Solo', 'FamilySize',
        # Numeric
        'Age', 'CabinNum', 'CryoFlag', 'VipFlag',
        'IsChild', 'IsTeen', 'IsSenior', 'SpendPositiveCount', 'NoSpend',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'TotalSpend', 'AvgSpendPerService', 'SpendPerGroupMember',
        'Log_RoomService', 'Log_FoodCourt', 'Log_ShoppingMall', 'Log_Spa', 'Log_VRDeck',
        'Log_TotalSpend', 'Log_AvgSpendPerService', 'Log_SpendPerGroupMember',
        'AgeSpendInteraction',
        # NEW: advanced features
        'CabinNumParity', 'CabinNumBucket', 'SurnameGroupSize',
        'SpendEntropy', 'MaxSpendCategory',
        'CryoNoSpend', 'NotCryoHasSpend',
        # Existing generated features that should be audited in-model
        'CabinAgreementScore', 'CabinMean',
        'Surname_tfidf_0', 'Surname_tfidf_1', 'Surname_tfidf_2',
        'Surname_tfidf_3', 'Surname_tfidf_4',
        # OOF target-encoded columns (added dynamically)
        'TE_HomePlanet', 'TE_CabinDeck', 'TE_HomeDest', 'TE_DeckSide',
        'TE_AgeBand', 'TE_CryoSleep', 'TE_CabinZone',
    ]

    submission_file = 'submission_90plus.csv'

    # R2/R3 group rules disabled -- GroupIDs have zero overlap between train/test
    group_propagation_enabled = False

print('[OK] CFG ready.')
print(f'   Seeds: {CFG.random_seeds}  |  Folds: {CFG.n_splits}  |  Total folds/model: {len(CFG.random_seeds)*CFG.n_splits}')
print(f'   Optuna trials: {CFG.optuna_trials}  |  Pseudo rounds: {CFG.pseudo_rounds}')
print(f'   Ensemble runs: {CFG.ensemble_runs}  |  Feature count: {len(CFG.feature_cols)}')


# %% [code] cell 3 (notebook cell 6)
# execution_count = 3

def discover_input_dir() -> Path:
    files = ['train.csv', 'test.csv', 'sample_submission.csv']
    local_candidates = [CFG.local_data_dir]
    if CFG.input_root.exists():
        local_candidates.append(CFG.input_root)
    for candidate in local_candidates:
        if all((candidate / f).exists() for f in files):
            return candidate
    for p in sorted(CFG.input_root.rglob('train.csv')):
        parent = p.parent
        if all((parent / f).exists() for f in files):
            return parent
    raise FileNotFoundError(f'Attach the spaceship-titanic dataset. Looked in: {local_candidates}')

CFG.input_dir     = discover_input_dir()
train_raw         = pd.read_csv(CFG.input_dir / 'train.csv')
test_raw          = pd.read_csv(CFG.input_dir / 'test.csv')
sample_submission = pd.read_csv(CFG.input_dir / 'sample_submission.csv')
y_raw             = train_raw[CFG.target].astype(int)

print(f'[OK] Loaded  ->  train {train_raw.shape}  |  test {test_raw.shape}')
print(f'   Data dir: {CFG.input_dir}')
tracked_print(train_raw.head(3))





# %% [code] cell 4 (notebook cell 8)
# execution_count = 4

def mode_or_nan(s: pd.Series) -> Any:
    non_null = s.dropna()
    if non_null.empty: return np.nan
    m = non_null.mode(dropna=True)
    return m.iloc[0] if not m.empty else non_null.iloc[0]

def fill_group_mode(df: pd.DataFrame, key: str, col: str) -> None:
    mapping = df.groupby(key)[col].agg(mode_or_nan)
    df[col]  = df[col].fillna(df[key].map(mapping))

def parse_cabin(s: pd.Series):
    c = s.fillna('U/9999/U').astype(str).str.split('/', expand=True)
    return c[0].replace('nan','U'), pd.to_numeric(c[1], errors='coerce'), c[2].replace('nan','U')


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train = train_df.copy(); test = test_df.copy()
    train['_is_train'] = 1; test['_is_train'] = 0
    test[CFG.target] = np.nan
    full = pd.concat([train, test], ignore_index=True)

    # -- Group / ID ----------------------------------------------------------
    gp = full['PassengerId'].str.split('_', expand=True)
    full['GroupId']     = pd.to_numeric(gp[0], errors='coerce')
    full['GroupMember'] = pd.to_numeric(gp[1], errors='coerce')
    full['GroupSize']   = full.groupby('GroupId')['PassengerId'].transform('size').astype(int)
    full['Solo']        = (full['GroupSize'] == 1).astype(int)

    # -- Cabin ---------------------------------------------------------------
    full['CabinDeck'], full['CabinNum'], full['CabinSide'] = parse_cabin(full['Cabin'])
    full['CabinNumParity'] = (full['CabinNum'] % 2).fillna(-1).astype(int)  # NEW
    full['CabinNumBucket'] = pd.cut(full['CabinNum'], bins=10,
                                     labels=False, duplicates='drop').fillna(-1).astype(int)  # NEW

    # -- Name / family -------------------------------------------------------
    nm = full['Name'].fillna('Unknown Unknown').astype(str).str.split(' ', n=1, expand=True)
    full['Surname']      = nm[1].fillna('Unknown')
    full['FamilySize']   = full.groupby('Surname')['PassengerId'].transform('size').astype(int)
    full['SurnameGroupSize'] = full.groupby('Surname')['GroupId'].transform('nunique').astype(int)  # NEW

    # -- CryoSleep inference from spend --------------------------------------
    sp0 = full[CFG.spend_cols].fillna(0).sum(axis=1)
    full.loc[full['CryoSleep'].isna() & (sp0 > 0), 'CryoSleep'] = False
    full.loc[full['CryoSleep'].isna() & (sp0 == 0), 'CryoSleep'] = True

    # -- Group-mode imputation ------------------------------------------------
    for c in ['HomePlanet','Destination','CabinDeck','CabinSide','Surname']:
        fill_group_mode(full, 'GroupId', c)

    # -- Cross-column imputation ----------------------------------------------
    for src, tgt, fallback_col in [
        ('CabinDeck','HomePlanet', None),
        ('HomePlanet','Destination', None),
        ('HomePlanet','CabinDeck', None),
    ]:
        mapping = full.groupby(src)[tgt].agg(mode_or_nan)
        full[tgt] = full[tgt].fillna(full[src].map(mapping))
    full['HomePlanet'] = full['HomePlanet'].fillna(mode_or_nan(full['HomePlanet']))
    full['Destination']= full['Destination'].fillna(mode_or_nan(full['Destination']))
    full['CabinDeck']  = full['CabinDeck'].fillna('U')
    full['CabinSide']  = full['CabinSide'].fillna(mode_or_nan(full['CabinSide']))

    # -- Numeric imputation ---------------------------------------------------
    full['CabinNum'] = (full['CabinNum']
                        .fillna(full.groupby('GroupId')['CabinNum'].transform('median'))
                        .fillna(full['CabinNum'].median()))
    full['Age']      = (full['Age']
                        .fillna(full.groupby('GroupId')['Age'].transform('median'))
                        .fillna(full.groupby('HomePlanet')['Age'].transform('median'))
                        .fillna(full['Age'].median()))
    full['VIP']      = full['VIP'].fillna(False)

    # -- Spend imputation -----------------------------------------------------
    for col in CFG.spend_cols:
        full.loc[full['CryoSleep'] == True, col] = 0.0
        hp_med = full.groupby('HomePlanet')[col].transform('median')
        full[col] = full[col].fillna(hp_med).fillna(full[col].median())
        full.loc[full['CryoSleep'] == True, col] = 0.0

    # -- Spend aggregate features ---------------------------------------------
    full['TotalSpend']          = full[CFG.spend_cols].sum(axis=1)
    full['SpendPositiveCount']  = (full[CFG.spend_cols] > 0).sum(axis=1).astype(int)
    full['NoSpend']             = (full['TotalSpend'] == 0).astype(int)
    full['AvgSpendPerService']  = full['TotalSpend'] / full['SpendPositiveCount'].replace(0, 1)
    full['SpendPerGroupMember'] = full['TotalSpend'] / full['GroupSize'].replace(0, 1)

    # -- Spend entropy (NEW) --------------------------------------------------
    spend_probs = full[CFG.spend_cols].div(full['TotalSpend'].replace(0, 1), axis=0).clip(0, 1)
    spend_probs  = spend_probs.replace(0, 1e-9)
    full['SpendEntropy']    = -(spend_probs * np.log(spend_probs)).sum(axis=1)  # NEW
    full['MaxSpendCategory']= full[CFG.spend_cols].idxmax(axis=1).fillna('None')  # NEW

    # -- Log-transform spending -----------------------------------------------
    for col in CFG.spend_cols + ['TotalSpend','AvgSpendPerService','SpendPerGroupMember']:
        full[f'Log_{col}'] = np.log1p(full[col])

    # -- Demographic flags ----------------------------------------------------
    full['CryoFlag']  = full['CryoSleep'].astype(int)
    full['VipFlag']   = full['VIP'].astype(int)
    full['IsChild']   = (full['Age'] < 13).astype(int)
    full['IsTeen']    = ((full['Age'] >= 13) & (full['Age'] < 18)).astype(int)
    full['IsSenior']  = (full['Age'] >= 60).astype(int)
    full['AgeSpendInteraction'] = full['Age'] * full['Log_TotalSpend']

    # -- Deterministic rule flags (NEW) ---------------------------------------
    full['CryoNoSpend']     = ((full['CryoFlag'] == 1) & (full['NoSpend'] == 1)).astype(int)
    full['NotCryoHasSpend'] = ((full['CryoFlag'] == 0) & (full['TotalSpend'] > 0)).astype(int)

    # -- Binning & interactions -----------------------------------------------
    full['AgeBand']   = pd.cut(full['Age'], bins=[-1,12,18,25,40,60,120],
                               labels=['child','teen','young_adult','adult','midlife','senior']).astype(str)
    full['CabinZone'] = pd.qcut(full['CabinNum'], q=6, duplicates='drop').astype(str)
    full['HomeDest']  = full['HomePlanet'].astype(str) + '_' + full['Destination'].astype(str)
    full['DeckSide']  = full['CabinDeck'].astype(str) + '_' + full['CabinSide'].astype(str)

    full['CryoSleep'] = full['CryoSleep'].map({True:'True', False:'False'}).fillna('False')
    full['VIP']       = full['VIP'].map({True:'True', False:'False'}).fillna('False')

    # -- TF-IDF on Surnames --------------------------------------------------
    tfidf = TfidfVectorizer(max_features=1000)
    vec_all = tfidf.fit_transform(full['Surname'])
    svd = TruncatedSVD(n_components=5, random_state=42)
    tfidf_arr = svd.fit_transform(vec_all)
    for i in range(5):
        full[f'Surname_tfidf_{i}'] = tfidf_arr[:, i]

    # -- Cabin-Number Propagation --------------------------------------------
    train_mask = full['_is_train'] == 1
    test_mask  = full['_is_train'] == 0
    y_full = full.loc[train_mask, CFG.target].copy()

    cabin_stats = (
        full.loc[train_mask, ['CabinNum', CFG.target]]
        .groupby('CabinNum')[CFG.target]
        .agg(cabin_mean='mean', cabin_count='count')
        .reset_index()
    )
    cabin_stats['cabin_agreement'] = np.where(
        cabin_stats['cabin_mean'] >= 0.5,
        cabin_stats['cabin_mean'],
        1 - cabin_stats['cabin_mean']
    )

    train_cabin = full.loc[train_mask, ['CabinNum']].copy()
    train_cabin['_y'] = y_full.values
    train_cabin['_cabin_sum']   = train_cabin['CabinNum'].map(
        cabin_stats.set_index('CabinNum')['cabin_mean']) * \
        train_cabin['CabinNum'].map(cabin_stats.set_index('CabinNum')['cabin_count'])
    train_cabin['_cabin_count'] = train_cabin['CabinNum'].map(
        cabin_stats.set_index('CabinNum')['cabin_count'])
    loo_sum   = (train_cabin['_cabin_sum'] - train_cabin['_y']).clip(0)
    loo_count = (train_cabin['_cabin_count'] - 1).clip(1)
    loo_mean  = (loo_sum / loo_count).fillna(y_full.mean())
    full.loc[train_mask, 'CabinAgreementScore'] = np.maximum(loo_mean, 1 - loo_mean)
    full.loc[train_mask, 'CabinMean']           = loo_mean

    cabin_map_mean = cabin_stats.set_index('CabinNum')['cabin_mean']
    cabin_map_agr  = cabin_stats.set_index('CabinNum')['cabin_agreement']
    test_cabins = full.loc[test_mask, 'CabinNum']
    full.loc[test_mask, 'CabinAgreementScore'] = (
        test_cabins.map(cabin_map_agr).fillna(0.5)
    )
    full.loc[test_mask, 'CabinMean'] = (
        test_cabins.map(cabin_map_mean).fillna(y_full.mean())
    )

    # -- Split back -----------------------------------------------------------
    train_out = full[full['_is_train'] == 1].drop(columns=['_is_train']).reset_index(drop=True)
    test_out  = full[full['_is_train'] == 0].drop(columns=['_is_train']).drop(columns=[CFG.target]).reset_index(drop=True)
    return train_out, test_out


train_feat, test_feat = engineer_features(train_raw, test_raw)
y = train_feat[CFG.target].astype(int)

# Cabin overlap diagnostic
train_cabins = set(train_feat['CabinNum'].dropna().astype(int))
test_cabins  = set(test_feat['CabinNum'].dropna().astype(int))
cabin_overlap = train_cabins & test_cabins
cabin_overlap_test_pct = test_feat['CabinNum'].dropna().isin(train_cabins).mean() * 100
print(f'[OK] Feature engineering done  ->  train {train_feat.shape}  |  test {test_feat.shape}')
print(f'   CabinNum overlap: {len(cabin_overlap):,} cabins  ->  {cabin_overlap_test_pct:.1f}% test covered')



# %% [code] cell 5 (notebook cell 10)
# execution_count = 5

def compute_group_label_propagation(
    train_f: pd.DataFrame,
    test_f:  pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enrich train and test with group-level label signal features.
    Returns modified copies of both frames.
    """
    train_out = train_f.copy()
    test_out  = test_f.copy()

    # Build group -> label mapping from training data only
    # Using leave-one-out to avoid direct leakage for train rows
    train_out['_y'] = y_train.values

    group_stats = (
        train_out.groupby('GroupId')['_y']
        .agg(group_mean='mean', group_count='count', group_sum='sum')
        .reset_index()
    )
    group_stats['group_agreement'] = np.where(
        group_stats['group_mean'] >= 0.5,
        group_stats['group_mean'],
        1 - group_stats['group_mean']
    )  # how unanimous is the group? 0.5 = split, 1.0 = all same

    # For TRAIN: leave-one-out group mean target
    train_out['_group_sum']   = train_out['GroupId'].map(group_stats.set_index('GroupId')['group_sum'])
    train_out['_group_count'] = train_out['GroupId'].map(group_stats.set_index('GroupId')['group_count'])
    train_out['GroupTrainKnownRate']  = 1.0  # all train group members are known
    # Leave-one-out: exclude current row from group mean
    loo_sum   = (train_out['_group_sum']   - train_out['_y']).clip(0)
    loo_count = (train_out['_group_count'] - 1).clip(1)
    loo_mean                        = (loo_sum / loo_count).fillna(0.5)
    train_out['GroupAgreementScore'] = np.maximum(loo_mean, 1 - loo_mean)
    train_out['GroupMean']           = loo_mean
    train_out['GroupTrainKnown']     = (loo_count > 0).astype(int)
    train_out = train_out.drop(columns=['_y','_group_sum','_group_count'])

    # For TEST: use full group stats from train
    total_group_size = pd.concat([
        train_out[['GroupId','PassengerId']],
        test_out[['GroupId','PassengerId']]
    ]).groupby('GroupId')['PassengerId'].count().rename('total_size')

    group_stats_map = group_stats.set_index('GroupId')

    test_out['GroupAgreementScore'] = (
        test_out['GroupId'].map(group_stats_map['group_agreement'])
    ).fillna(0.5)
    test_out['GroupMean'] = (
        test_out['GroupId'].map(group_stats_map['group_mean'])
    ).fillna(0.5)
    test_out['GroupTrainKnown'] = (
        test_out['GroupId'].isin(group_stats['GroupId'])
    ).astype(int)
    train_known_count = group_stats_map['group_count']
    test_out['GroupTrainKnownRate'] = (
        test_out['GroupId'].map(train_known_count).fillna(0) /
        test_out['GroupId'].map(total_group_size).fillna(1)
    ).clip(0, 1)

    return train_out, test_out


train_feat, test_feat = compute_group_label_propagation(train_feat, test_feat, y)

# Diagnostic
print(f'[OK] Group Label Propagation complete.')
print(f'   [WARN]  Group propagation DISABLED -- zero overlap between train/test GroupIds')
print(f'       All test GroupAgreementScore = 0.5 (fill value), GroupTrainKnown = 0')

# Visualisation: skipped -- features not used in this data version
print(f'   Group propagation visualisation skipped (features not applicable).')



# %% [code] cell 6 (notebook cell 12)
# execution_count = 6

def apply_hard_rules_train(
    df: pd.DataFrame, y_true: pd.Series
) -> dict:
    """Measure hard-rule accuracy on training data."""
    results = {}

    # R1: CryoSleep + NoSpend
    mask_r1  = (df['CryoFlag'] == 1) & (df['NoSpend'] == 1)
    if mask_r1.sum() > 0:
        acc_r1 = accuracy_score(y_true[mask_r1], np.ones(mask_r1.sum(), dtype=int))
        results['R1_CryoNoSpend'] = {'n': mask_r1.sum(), 'acc': acc_r1, 'label': 1}

    # R2/R3 group rules DISABLED -- GroupIds have zero overlap between train/test in this data version
    return results


rule_stats = apply_hard_rules_train(train_feat, y)
print('[OK] Hard Rule Analysis (on training data):')
print(f'   {"Rule":<30} {"N samples":>10}  {"Train Accuracy":>15}  {"Label":>8}')
print('   ' + '-' * 68)
for rule_name, stats in rule_stats.items():
    print(f'   {rule_name:<30} {stats["n"]:>10,}  {stats["acc"]:>14.4f}  {stats["label"]:>8}')


def generate_locked_test_predictions(
    test_df: pd.DataFrame, agreement_threshold: float = 0.95
) -> pd.Series:
    """
    Returns a Series of 'locked' predictions for test rows where
    we are highly confident without running the ML model.
    NaN = not locked, must be predicted by model.
    """
    locked = pd.Series(np.nan, index=test_df.index)

    # R1: CryoSleep + NoSpend -> Transported
    r1_mask  = (test_df['CryoFlag'] == 1) & (test_df['NoSpend'] == 1)
    locked[r1_mask] = 1

    # R2/R3 group rules removed -- GroupIDs have zero overlap between train/test
    return locked


locked_test_preds = generate_locked_test_predictions(test_feat)
n_locked = locked_test_preds.notna().sum()
n_unlocked = locked_test_preds.isna().sum()
print(f'\n   Test passengers locked   (hard rules) : {n_locked:,} ({n_locked/len(test_feat)*100:.1f}%)')
print(f'   Test passengers unlocked (needs model): {n_unlocked:,} ({n_unlocked/len(test_feat)*100:.1f}%)')



# %% [code] cell 7 (notebook cell 14)
# execution_count = 7

def oof_target_encode(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    y_train:  pd.Series,
    cols:     list[str],
    n_splits: int = 5,
    seed:     int = 42,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leakage-safe OOF target encoding with Laplace smoothing.
    Formula: (count_i * mean_i + smoothing * global_mean) / (count_i + smoothing)
    """
    train_enc = train_df.copy()
    test_enc  = test_df.copy()
    global_mean = y_train.mean()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in cols:
        te_col = f'TE_{col}'
        train_enc[te_col] = np.nan

        for tr_idx, val_idx in cv.split(train_df, y_train):
            tr_part  = train_df.iloc[tr_idx].copy()
            tr_part['_y'] = y_train.iloc[tr_idx].values
            val_part = train_df.iloc[val_idx]

            stats = tr_part.groupby(col)['_y'].agg(['mean','count'])
            smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / \
                     (stats['count'] + smoothing)

            train_enc.loc[val_idx, te_col] = val_part[col].map(smooth).fillna(global_mean).values

        # Full-train encoding for test
        tmp = train_df[[col]].copy(); tmp['_y'] = y_train.values
        stats_full = tmp.groupby(col)['_y'].agg(['mean','count'])
        smooth_full = (stats_full['count'] * stats_full['mean'] + smoothing * global_mean) / \
                      (stats_full['count'] + smoothing)
        test_enc[te_col] = test_df[col].map(smooth_full).fillna(global_mean)

        train_enc[te_col] = train_enc[te_col].fillna(global_mean)

    return train_enc, test_enc


TE_COLS = ['HomePlanet', 'CabinDeck', 'HomeDest', 'DeckSide', 'AgeBand', 'CryoSleep', 'CabinZone']

train_feat, test_feat = oof_target_encode(
    train_feat, test_feat, y, TE_COLS,
    n_splits=CFG.n_splits, seed=42, smoothing=10.0
)

print('[OK] OOF Target Encoding complete.')
te_added = [f'TE_{c}' for c in TE_COLS]
tracked_print(train_feat[te_added].describe().T)
save_table_artifact("target_encoding_stats", train_feat[te_added].describe().T)

# NOTE: Surname Exact Match disabled -- causes data leakage (not OOF)
# To use safely, implement OOF target encoding for Surname instead.

train_surnames = set(train_feat['Surname'])
test_surnames  = set(test_feat['Surname'])
surname_overlap = train_surnames & test_surnames
test_surname_covered = test_feat['Surname'].isin(train_surnames).sum()
print(f'   Surname overlap: {len(surname_overlap):,} surnames  ->  {test_surname_covered:,} / {len(test_feat):,} test ({test_surname_covered/len(test_feat)*100:.1f}%)')

# -- Adversarial Validation --------------------------------------------
adv_train = train_feat.drop(columns=[CFG.target]).copy()
adv_test  = test_feat.copy()
adv_train['_is_train'] = 1
adv_test['_is_train']  = 0
adv_full = pd.concat([adv_train, adv_test], ignore_index=True, sort=False)
adv_y = adv_full['_is_train']

adv_feats = adv_full.select_dtypes(include=[np.number]).fillna(0).drop(
    columns=['_is_train'], errors='ignore')
from sklearn.model_selection import cross_val_score
adv_clf = HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)
adv_auc = cross_val_score(adv_clf, adv_feats, adv_y, cv=5, scoring='roc_auc').mean()
print(f'   Adversarial Validation AUC: {adv_auc:.4f}  ({"[WARN] high shift" if adv_auc > 0.70 else "[OK] normal" if adv_auc > 0.60 else "[OK] very low shift"})')



# %% [code] cell 8 (notebook cell 16)
# execution_count = 8

# Ensure all feature_cols exist; add any that were missing from advanced features
all_available_features = [c for c in CFG.feature_cols if c in train_feat.columns]
print(f'Features available: {len(all_available_features)} / {len(CFG.feature_cols)}')
missing = [c for c in CFG.feature_cols if c not in train_feat.columns]
if missing:
    print(f'Missing features (will skip): {missing}')
FEAT_COLS = all_available_features

feature_audit = pd.DataFrame({
    'feature': CFG.feature_cols,
    'generated_in_train': [c in train_feat.columns for c in CFG.feature_cols],
    'used_in_model': [c in FEAT_COLS for c in CFG.feature_cols],
    'dtype': [str(train_feat[c].dtype) if c in train_feat.columns else None for c in CFG.feature_cols],
})
save_table_artifact("feature_audit", feature_audit)
print('Feature audit saved.')

CATEGORICAL_COLS = [c for c in CFG.categorical_base + ['MaxSpendCategory'] if c in FEAT_COLS]


def encode_ordinal(
    train_x: pd.DataFrame, test_x: pd.DataFrame, cat_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enc = OrdinalEncoder(handle_unknown='use_encoded_value',
                         unknown_value=-1, encoded_missing_value=-1)
    tr = train_x.copy(); te = test_x.copy()
    tc = train_x[cat_cols].fillna('__MISSING__').astype(str)
    ec = test_x[cat_cols].fillna('__MISSING__').astype(str)
    enc.fit(pd.concat([tc, ec], ignore_index=True))
    tr[cat_cols] = enc.transform(tc)
    te[cat_cols] = enc.transform(ec)
    return tr.astype(float), te.astype(float)


def optimize_threshold(y_true: pd.Series, probs: np.ndarray) -> tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.35, 0.65, 121):
        s = accuracy_score(y_true, probs >= t)
        if s > best_s:
            best_s, best_t = float(s), float(t)
    return best_t, best_s


X_train_raw  = train_feat[FEAT_COLS].copy()
X_test_raw   = test_feat[FEAT_COLS].copy()
X_train_num, X_test_num = encode_ordinal(X_train_raw, X_test_raw, CATEGORICAL_COLS)
print(f'\n[OK] Encoding done  ->  X_train {X_train_num.shape}  |  X_test {X_test_num.shape}')


# -- SHAP dead feature scan (run BEFORE ensemble to filter noise) ------------
DEAD_FEATURES = []
if LGB_AVAILABLE and SHAP_AVAILABLE:
    lgb_scan = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        random_state=42, verbosity=-1
    )
    lgb_scan.fit(X_train_num, y)
    explainer_scan = shap.TreeExplainer(lgb_scan)
    sample_idx = np.random.RandomState(42).choice(
        len(X_train_num), size=min(1000, len(X_train_num)), replace=False
    )
    shap_vals = explainer_scan.shap_values(X_train_num.iloc[sample_idx])
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_imp = pd.DataFrame({
        'feature': X_train_num.columns,
        'mean_shap': np.abs(shap_vals).mean(axis=0)
    }).sort_values('mean_shap', ascending=False)

    cutoff = shap_imp['mean_shap'].quantile(0.10)
    DEAD_FEATURES = shap_imp[shap_imp['mean_shap'] < cutoff]['feature'].tolist()
    print(f'   SHAP dead features (bottom 10%, cutoff={cutoff:.5f}): {DEAD_FEATURES}')

    if DEAD_FEATURES:
        FEAT_COLS = [c for c in FEAT_COLS if c not in DEAD_FEATURES]
        X_train_raw  = X_train_raw[FEAT_COLS]
        X_test_raw   = X_test_raw[FEAT_COLS]
        X_train_num  = X_train_num[FEAT_COLS]
        X_test_num   = X_test_num[FEAT_COLS]
        CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in FEAT_COLS]
        print(f'   Removed {len(DEAD_FEATURES)} dead features -> {len(FEAT_COLS)} remaining')
else:
    print('   SHAP dead feature scan skipped (LGB/SHAP unavailable)')



# %% [code] cell 9 (notebook cell 18)
# execution_count = 9

best_lgb_params = {  # fallback defaults
    'n_estimators': 450, 'learning_rate': 0.03, 'num_leaves': 31,
    'subsample': 0.85, 'colsample_bytree': 0.80, 'min_child_samples': 18,
    'reg_alpha': 0.05, 'reg_lambda': 1.0,
}

if LGB_AVAILABLE and OPTUNA_AVAILABLE:
    def lgb_objective(trial):
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators', 300, 800),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves'       : trial.suggest_int('num_leaves', 20, 100),
            'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
            'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
            'verbosity'        : -1,
        }
        cv  = StratifiedKFold(n_splits=CFG.optuna_cv_folds, shuffle=True, random_state=42)
        accs = []
        for tr_idx, val_idx in cv.split(X_train_num, y):
            model = lgb.LGBMClassifier(**params, random_state=42)
            model.fit(X_train_num.iloc[tr_idx], y.iloc[tr_idx])
            proba = model.predict_proba(X_train_num.iloc[val_idx])[:, 1]
            accs.append(accuracy_score(y.iloc[val_idx], proba >= 0.5))
        return np.mean(accs)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lgb_objective, n_trials=CFG.optuna_trials,
                   show_progress_bar=True)

    best_lgb_params = study.best_params
    print(f'\n[OK] Optuna finished  ->  best CV accuracy: {study.best_value:.5f}')
    print(f'   Best params: {best_lgb_params}')

    # Visualise optimisation history
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Optuna Hyperparameter Optimisation', fontsize=14, fontweight='bold')

    trial_values = [t.value for t in study.trials if t.value is not None]
    axes[0].plot(trial_values, color=BLUE, linewidth=0.8, alpha=0.7)
    axes[0].plot(np.maximum.accumulate(trial_values), color=GREEN, linewidth=2, label='Best so far')
    axes[0].set_title('Optimisation History', color='white')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('CV Accuracy')
    axes[0].legend()

    # Param importance
    try:
        importance = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importance.keys())[:8], list(importance.values())[:8],
                     color=PURPLE, edgecolor='none')
        axes[1].set_title('Hyperparameter Importance', color='white')
        axes[1].set_xlabel('Importance')
    except Exception:
        axes[1].text(0.5, 0.5, 'Not enough trials for importance', ha='center', color='white')

    plt.tight_layout()
    finalize_figures()
elif LGB_AVAILABLE and not OPTUNA_AVAILABLE:
    print('[INFO]  Optuna not available -- using default LightGBM params.')
else:
    print('[INFO]  LightGBM not available -- using default params.')



# %% [code] cell 10 (notebook cell 20)
# execution_count = 10

def run_ensemble_cv(
    X_tr_num: pd.DataFrame, X_te_num: pd.DataFrame,
    X_tr_raw: pd.DataFrame, X_te_raw: pd.DataFrame,
    y_true:   pd.Series,
    cat_indices: list[int],
    lgb_params: dict,
    label: str = 'main',
) -> dict:

    model_names = ['extra_trees', 'hist_gb']
    if XGB_AVAILABLE: model_names.append('xgb')
    if LGB_AVAILABLE: model_names.append('lgb')
    if CAT_AVAILABLE: model_names.append('cat')

    oof_store   = {n: np.zeros(len(y_true)) for n in model_names}
    count_store = {n: np.zeros(len(y_true)) for n in model_names}
    test_store  = {n: []                    for n in model_names}
    fold_rows   = []

    total = len(CFG.random_seeds) * CFG.n_splits
    fold_num = 0

    for seed in CFG.random_seeds:
        cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=seed)
        for fi, (tr_idx, val_idx) in enumerate(cv.split(X_tr_num, y_true), 1):
            fold_num += 1
            xtr, xvl = X_tr_num.iloc[tr_idx], X_tr_num.iloc[val_idx]
            xtr_r     = X_tr_raw.iloc[tr_idx].copy().astype(str)
            xvl_r     = X_tr_raw.iloc[val_idx].copy().astype(str)
            xte_r     = X_te_raw.copy().astype(str)
            ytr, yvl  = y_true.iloc[tr_idx], y_true.iloc[val_idx]
            accs = []

            # ExtraTrees
            et = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2,
                                       random_state=seed*10+fi, n_jobs=4)
            et.fit(xtr, ytr)
            vp = et.predict_proba(xvl)[:, 1]
            tp = et.predict_proba(X_te_num)[:, 1]
            oof_store['extra_trees'][val_idx]  += vp
            count_store['extra_trees'][val_idx]+= 1
            test_store['extra_trees'].append(tp)
            a = accuracy_score(yvl, vp >= 0.5)
            fold_rows.append({'seed':seed,'fold':fi,'model':'extra_trees','acc':a})
            accs.append(f'ET={a:.4f}')

            # HistGB
            hgb = HistGradientBoostingClassifier(max_depth=7, learning_rate=0.035,
                                                  max_iter=400, min_samples_leaf=10,
                                                  random_state=seed*10+fi)
            hgb.fit(xtr, ytr)
            vp = hgb.predict_proba(xvl)[:, 1]
            tp = hgb.predict_proba(X_te_num)[:, 1]
            oof_store['hist_gb'][val_idx]  += vp
            count_store['hist_gb'][val_idx]+= 1
            test_store['hist_gb'].append(tp)
            a = accuracy_score(yvl, vp >= 0.5)
            fold_rows.append({'seed':seed,'fold':fi,'model':'hist_gb','acc':a})
            accs.append(f'HGB={a:.4f}')

            if XGB_AVAILABLE:
                xgb_m = xgb.XGBClassifier(
                    n_estimators=400, max_depth=6,
                    learning_rate=lgb_params.get('learning_rate', 0.03),
                    subsample=lgb_params.get('subsample', 0.85),
                    colsample_bytree=lgb_params.get('colsample_bytree', 0.80),
                    reg_alpha=lgb_params.get('reg_alpha', 0.05),
                    reg_lambda=lgb_params.get('reg_lambda', 1.0),
                    min_child_weight=3, objective='binary:logistic',
                    eval_metric='logloss', tree_method='hist',
                    random_state=seed*10+fi, n_jobs=4,
                )
                xgb_m.fit(xtr, ytr)
                vp = xgb_m.predict_proba(xvl)[:, 1]
                tp = xgb_m.predict_proba(X_te_num)[:, 1]
                oof_store['xgb'][val_idx]  += vp
                count_store['xgb'][val_idx]+= 1
                test_store['xgb'].append(tp)
                a = accuracy_score(yvl, vp >= 0.5)
                fold_rows.append({'seed':seed,'fold':fi,'model':'xgb','acc':a})
                accs.append(f'XGB={a:.4f}')

            if LGB_AVAILABLE:
                lgb_m = lgb.LGBMClassifier(
                    **{k: v for k, v in lgb_params.items() if k != 'verbosity'},
                    random_state=seed*10+fi, verbosity=-1
                )
                lgb_m.fit(xtr, ytr)
                vp = lgb_m.predict_proba(xvl)[:, 1]
                tp = lgb_m.predict_proba(X_te_num)[:, 1]
                oof_store['lgb'][val_idx]  += vp
                count_store['lgb'][val_idx]+= 1
                test_store['lgb'].append(tp)
                a = accuracy_score(yvl, vp >= 0.5)
                fold_rows.append({'seed':seed,'fold':fi,'model':'lgb','acc':a})
                accs.append(f'LGB={a:.4f}')
                
            if CAT_AVAILABLE:
                cat_m = CatBoostClassifier(
                    iterations=500, depth=7,
                    learning_rate=lgb_params.get('learning_rate', 0.03),
                    l2_leaf_reg=lgb_params.get('reg_lambda', 4.0),
                    loss_function='Logloss', random_seed=seed*10+fi,
                    verbose=False, allow_writing_files=False,
                )
                cat_m.fit(xtr_r, ytr, cat_features=cat_indices, verbose=False)
                vp = cat_m.predict_proba(xvl_r)[:, 1]
                tp = cat_m.predict_proba(xte_r)[:, 1]
                oof_store['cat'][val_idx]  += vp
                count_store['cat'][val_idx]+= 1
                test_store['cat'].append(tp)
                a = accuracy_score(yvl, vp >= 0.5)
                fold_rows.append({'seed':seed,'fold':fi,'model':'cat','acc':a})
                accs.append(f'CAT={a:.4f}')

            print(f'  [{label}] {fold_num:2d}/{total} | s={seed} f={fi} | {" | ".join(accs)}')

    for n in model_names:
        oof_store[n] /= np.maximum(count_store[n], 1)

    oof_mat  = np.column_stack([oof_store[n]                    for n in model_names])
    test_mat = np.column_stack([np.mean(test_store[n], axis=0) for n in model_names])

    meta = LogisticRegression(C=0.5, max_iter=3000)
    meta.fit(oof_mat, y_true)
    oof_stack  = meta.predict_proba(oof_mat)[:, 1]
    test_stack = meta.predict_proba(test_mat)[:, 1]

    simple_oof  = oof_mat.mean(axis=1)
    simple_test = test_mat.mean(axis=1)

    best_w, best_t, best_cv = 0.5, 0.5, -1.0
    best_oof = simple_oof; best_test = simple_test

    for w in np.linspace(0.2, 0.8, 25):
        cand = w * oof_stack + (1-w) * simple_oof
        t, s = optimize_threshold(y_true, cand)
        if s > best_cv:
            best_cv = s; best_w = float(w); best_t = float(t)
            best_oof  = cand
            best_test = w * test_stack + (1-w) * simple_test

    return {
        'model_names' : model_names,
        'oof_probs'   : best_oof,
        'test_probs'  : best_test,
        'threshold'   : best_t,
        'cv_accuracy' : best_cv,
        'stack_weight': best_w,
        'fold_scores' : pd.DataFrame(fold_rows),
        'oof_matrix'  : oof_mat,
        'test_matrix' : test_mat,
        'meta_model'  : meta,
    }


def train_mlp_oof(
    X_num:   pd.DataFrame,
    X_te:    pd.DataFrame,
    y_true:  pd.Series,
    seeds:   list[int],
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    scaler     = StandardScaler()
    X_sc       = scaler.fit_transform(X_num.values)
    X_te_sc    = scaler.transform(X_te.values)

    oof_probs  = np.zeros(len(y_true))
    oof_counts = np.zeros(len(y_true))
    test_preds = []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fi, (tr_idx, val_idx) in enumerate(cv.split(X_sc, y_true), 1):
            mlp = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                alpha=0.01,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed * 10 + fi,
                batch_size=256,
            )
            mlp.fit(X_sc[tr_idx], y_true.iloc[tr_idx])
            vp = mlp.predict_proba(X_sc[val_idx])[:, 1]
            tp = mlp.predict_proba(X_te_sc)[:, 1]
            oof_probs[val_idx]  += vp
            oof_counts[val_idx] += 1
            test_preds.append(tp)
            acc = accuracy_score(y_true.iloc[val_idx], vp >= 0.5)
            print(f'  MLP | seed={seed} fold={fi} | acc={acc:.4f}')

    oof_probs /= np.maximum(oof_counts, 1)
    return oof_probs, np.mean(test_preds, axis=0)


# -- Save original data for multi-run reset ----------------------------------
_orig_X_train_num = X_train_num.copy()
_orig_X_test_num  = X_test_num.copy()
_orig_X_train_raw = X_train_raw.copy()
_orig_X_test_raw  = X_test_raw.copy()
_orig_y           = y.copy()

all_final_test_probs = []
all_final_oof_probs  = []
all_cv_accs          = []
all_best_labels      = []

for run_idx in range(CFG.ensemble_runs):
    base_seed = CFG.ensemble_base_seeds[run_idx]
    print(f'\n{"="*60}')
    print(f'  ENSEMBLE RUN {run_idx+1}/{CFG.ensemble_runs}  (base_seed={base_seed})')
    print(f'{"="*60}')

    X_train_num = _orig_X_train_num.copy()
    X_test_num  = _orig_X_test_num.copy()
    X_train_raw = _orig_X_train_raw.copy()
    X_test_raw  = _orig_X_test_raw.copy()
    y           = _orig_y.copy()

    CFG.random_seeds = [base_seed + i * 1000 for i in range(5)]
    cat_idxs = [X_train_raw.columns.get_loc(c) for c in CATEGORICAL_COLS if c in X_train_raw.columns]

    print('[RUN] Running ensemble CV training...')
    results = run_ensemble_cv(
        X_train_num, X_test_num,
        X_train_raw, X_test_raw,
        y, cat_idxs, best_lgb_params,
        label=f'run{run_idx+1}'
    )
    print(f'\n[OK] Ensemble training complete!')
    print(f'   OOF CV Accuracy : {results["cv_accuracy"]:.5f}')
    print(f'   Best threshold  : {results["threshold"]:.4f}')

    # -- Pseudo-labelling --------------------------------------------------
    current_results     = results.copy()
    current_X_train_num = X_train_num.copy()
    current_X_train_raw = X_train_raw.copy()
    current_y           = y.copy()

    n_original = len(y)
    pseudo_history = [{'round': 0, 'cv_acc_reported': results['cv_accuracy'],
                       'cv_acc_original_only': results['cv_accuracy'],
                       'n_train': n_original, 'n_pseudo': 0}]

    for pseudo_round in range(1, CFG.pseudo_rounds + 1):
        test_probs = current_results['test_probs']
        threshold  = current_results['threshold']

        high_conf_mask = (test_probs >= CFG.pseudo_threshold) | \
                         (test_probs <= 1 - CFG.pseudo_threshold)
        n_pseudo = high_conf_mask.sum()

        if n_pseudo == 0:
            print(f'Round {pseudo_round}: no confident pseudo-labels. Stopping.')
            break

        pseudo_labels = (test_probs[high_conf_mask] >= threshold).astype(int)
        pseudo_X_num  = X_test_num[high_conf_mask].reset_index(drop=True)
        pseudo_X_raw  = X_test_raw[high_conf_mask].reset_index(drop=True)
        pseudo_y      = pd.Series(pseudo_labels, name=CFG.target)

        aug_X_num = pd.concat([current_X_train_num, pseudo_X_num], ignore_index=True)
        aug_X_raw = pd.concat([current_X_train_raw, pseudo_X_raw], ignore_index=True)
        aug_y     = pd.concat([current_y,           pseudo_y],     ignore_index=True)

        new_results = run_ensemble_cv(
            aug_X_num, X_test_num,
            aug_X_raw, X_test_raw,
            aug_y, cat_idxs, best_lgb_params,
            label=f'pseudo_r{pseudo_round}_run{run_idx+1}'
        )

        oof_original_only = new_results['oof_probs'][:n_original]
        true_threshold, true_cv_acc = optimize_threshold(y, oof_original_only)

        reported_acc = new_results['cv_accuracy']
        print(f'\n  Round {pseudo_round} | pseudo rows added: {n_pseudo:,}')
        print(f'  Reported CV acc (all rows, inflated) : {reported_acc:.5f}')
        print(f'  TRUE CV acc     (original 8693 only) : {true_cv_acc:.5f}')
        print(f'  Previous best true CV acc            : '
              f'{pseudo_history[-1]["cv_acc_original_only"]:.5f}')

        pseudo_history.append({
            'round'                : pseudo_round,
            'cv_acc_reported'      : reported_acc,
            'cv_acc_original_only' : true_cv_acc,
            'n_train'              : len(aug_y),
            'n_pseudo'             : n_pseudo,
        })

        if true_cv_acc > pseudo_history[-2]['cv_acc_original_only']:
            print(f'  [OK] Real improvement confirmed -- keeping pseudo-label round.')
            new_results['oof_probs']  = oof_original_only
            new_results['threshold']  = true_threshold
            new_results['cv_accuracy'] = true_cv_acc
            current_results     = new_results
            current_X_train_num = aug_X_num
            current_X_train_raw = aug_X_raw
            current_y           = aug_y
        else:
            print(f'  [WARN]  No real improvement -- reverting.')

    # -- MLP blending (always on original 8693 rows) ----------------------
    print('[MLP] Training MLP neural layer...')
    mlp_oof, mlp_test = train_mlp_oof(
        _orig_X_train_num, X_test_num, y,
        seeds=CFG.random_seeds[:2],
        n_splits=CFG.n_splits,
    )
    mlp_cv_acc = accuracy_score(y, mlp_oof >= 0.5)
    print(f'\n[OK] MLP standalone OOF accuracy: {mlp_cv_acc:.5f}')

    gbdt_oof_orig  = current_results['oof_probs']
    gbdt_test_best = current_results['test_probs']

    print('\n  Testing blend strategies...')
    best_final_acc   = -1.0
    best_final_oof   = gbdt_oof_orig
    best_final_test  = gbdt_test_best
    best_final_label = 'GBDT-only'

    oof_mat  = np.column_stack([gbdt_oof_orig,  mlp_oof])
    test_mat = np.column_stack([gbdt_test_best, mlp_test])
    meta = LogisticRegression(C=0.3, max_iter=1000)
    meta.fit(oof_mat, y)
    lr_oof  = meta.predict_proba(oof_mat)[:, 1]
    lr_test = meta.predict_proba(test_mat)[:, 1]
    _, lr_acc = optimize_threshold(y, lr_oof)
    print(f'  LR meta-blend          : {lr_acc:.5f}  weights={meta.coef_[0].round(3)}')
    if lr_acc > best_final_acc:
        best_final_acc, best_final_oof, best_final_test = lr_acc, lr_oof, lr_test
        best_final_label = 'LR meta-blend (GBDT + MLP)'

    for mlp_w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        w_oof  = (1 - mlp_w) * gbdt_oof_orig  + mlp_w * mlp_oof
        w_test = (1 - mlp_w) * gbdt_test_best + mlp_w * mlp_test
        _, w_acc = optimize_threshold(y, w_oof)
        print(f'  Fixed blend mlp_w={mlp_w:.2f}   : {w_acc:.5f}')
        if w_acc > best_final_acc:
            best_final_acc, best_final_oof, best_final_test = w_acc, w_oof, w_test
            best_final_label = f'Fixed blend mlp_w={mlp_w:.2f}'

    _, gbdt_acc = optimize_threshold(y, gbdt_oof_orig)
    print(f'  GBDT-only              : {gbdt_acc:.5f}')
    if gbdt_acc >= best_final_acc:
        best_final_acc, best_final_oof, best_final_test = gbdt_acc, gbdt_oof_orig, gbdt_test_best
        best_final_label = 'GBDT-only (MLP did not help)'

    final_threshold, final_cv_acc = optimize_threshold(y, best_final_oof)
    print(f'\n[OK] Winner : {best_final_label}')
    print(f'   Final OOF accuracy : {final_cv_acc:.5f}')

    all_final_test_probs.append(best_final_test)
    all_final_oof_probs.append(best_final_oof)
    all_cv_accs.append(final_cv_acc)
    all_best_labels.append(best_final_label)

# -- Multi-run averaging -----------------------------------------------------
avg_test_probs   = np.mean(all_final_test_probs, axis=0)
final_oof_probs  = all_final_oof_probs[-1]
final_test_probs  = avg_test_probs
final_threshold, final_cv_acc = optimize_threshold(y, final_oof_probs)
best_final_label = f'Multi-run avg ({CFG.ensemble_runs} runs)'

print(f'\n{"="*60}')
print(f'  MULTI-RUN AVERAGING COMPLETE')
print(f'{"="*60}')
print(f'   Individual OOF accuracies : {[f"{a:.5f}" for a in all_cv_accs]}')
print(f'   Best labels per run       : {all_best_labels}')
print(f'   Mean OOF accuracy         : {np.mean(all_cv_accs):.5f}')
print(f'   Final OOF (last run ref)  : {final_cv_acc:.5f}')

# Restore original data for downstream use (SHAP, calibration)
X_train_num = _orig_X_train_num.copy()
X_test_num  = _orig_X_test_num.copy()
X_train_raw = _orig_X_train_raw.copy()
X_test_raw  = _orig_X_test_raw.copy()
y           = _orig_y.copy()


# %% [code] cell 13 (notebook cell 26)
# execution_count = 13

if LGB_AVAILABLE and SHAP_AVAILABLE:
    # Train a single LGB on full training data for SHAP
    lgb_shap = lgb.LGBMClassifier(
        **{k: v for k, v in best_lgb_params.items() if k != 'verbosity'},
        random_state=42, verbosity=-1
    )
    lgb_shap.fit(X_train_num, y)

    explainer  = shap.TreeExplainer(lgb_shap)
    # Use a sample of 1000 for speed
    sample_idx = np.random.RandomState(42).choice(len(X_train_num), size=min(1000, len(X_train_num)), replace=False)
    shap_values = explainer.shap_values(X_train_num.iloc[sample_idx])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # binary: take class-1 SHAP

    # Mean absolute SHAP per feature
    shap_importance = pd.DataFrame({
        'feature'   : X_train_num.columns,
        'mean_shap' : np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_shap', ascending=False).reset_index(drop=True)

    print('[OK] SHAP analysis complete. Top 20 features:')
    tracked_print(shap_importance.head(20))
    save_table_artifact("shap_importance_top20", shap_importance.head(20))

    # Identify dead features (bottom 10%)
    cutoff       = shap_importance['mean_shap'].quantile(0.10)
    dead_features = shap_importance[shap_importance['mean_shap'] < cutoff]['feature'].tolist()
    print(f'\n[WARN]  Low-SHAP features (bottom 10%, cutoff={cutoff:.4f}):')
    print(f'   {dead_features}')

    # Visualise top 25 features
    top25 = shap_importance.head(25)
    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(
        top25['feature'][::-1], top25['mean_shap'][::-1],
        color=[PALETTE[i % len(PALETTE)] for i in range(len(top25))],
        edgecolor='none'
    )
    ax.set_title('SHAP Feature Importance (Top 25)', fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('Mean |SHAP Value|', color='white')
    plt.tight_layout()
    finalize_figures()
elif LGB_AVAILABLE and not SHAP_AVAILABLE:
    print('[INFO]  SHAP not available -- skipping SHAP analysis.')
    dead_features = []
else:
    print('[INFO]  LightGBM not available -- skipping SHAP analysis.')
    dead_features = []


# %% [code] cell 14 (notebook cell 28)
# execution_count = 14

from sklearn.isotonic import IsotonicRegression

def calibrate_oof(
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    y_true: pd.Series,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    OOF isotonic calibration.
    Fits a calibrator on each fold's OOF; averages test calibrations.
    """
    calibrated_oof  = np.zeros_like(oof_probs)
    calibrated_test_parts = []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in cv.split(oof_probs, y_true):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_probs[tr_idx], y_true.iloc[tr_idx])
        calibrated_oof[val_idx] = iso.transform(oof_probs[val_idx])
        calibrated_test_parts.append(iso.transform(test_probs))

    calibrated_test = np.mean(calibrated_test_parts, axis=0)
    return calibrated_oof, calibrated_test

# -- BLOCK 13 FIX -- only apply calibration if it strictly improves OOF -------

cal_oof, cal_test = calibrate_oof(final_oof_probs, final_test_probs, y)
cal_threshold, cal_acc = optimize_threshold(y, cal_oof)
pre_cal_acc = accuracy_score(y, final_oof_probs >= final_threshold)

print(f'Pre-calibration  OOF accuracy : {pre_cal_acc:.5f}')
print(f'Post-calibration OOF accuracy : {cal_acc:.5f}')

if cal_acc > pre_cal_acc:
    print('[OK] Calibration improved accuracy -- applying.')
else:
    print('[WARN]  Calibration hurt accuracy -- reverting to uncalibrated probs.')
    cal_oof       = final_oof_probs
    cal_test      = final_test_probs
    cal_threshold = final_threshold
    cal_acc       = pre_cal_acc

# Calibration curve
from sklearn.calibration import calibration_curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Probability Calibration', fontsize=14, fontweight='bold')

for probs, label, color in [
    (final_oof_probs, 'Before calibration', RED),
    (cal_oof,         'After calibration',  GREEN),
]:
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=15)
    axes[0].plot(mean_pred, frac_pos, marker='o', linewidth=2,
                 color=color, label=label, markersize=4)

axes[0].plot([0, 1], [0, 1], 'w--', linewidth=1, label='Perfect calibration')
axes[0].set_title('Calibration Curve', color='white')
axes[0].set_xlabel('Mean Predicted Probability')
axes[0].set_ylabel('Fraction of Positives')
axes[0].legend(fontsize=9)

axes[1].hist(final_oof_probs, bins=50, alpha=0.6, color=RED,   label='Pre-calibration',  edgecolor='none')
axes[1].hist(cal_oof,         bins=50, alpha=0.6, color=GREEN, label='Post-calibration', edgecolor='none')
axes[1].axvline(cal_threshold, color='yellow', linestyle='--', linewidth=2,
                label=f'Threshold={cal_threshold:.3f}')
axes[1].set_title('OOF Probability Distribution', color='white')
axes[1].set_xlabel('Predicted Probability')
axes[1].legend(fontsize=9)

plt.tight_layout()
finalize_figures()


# %% [code] cell 15 (notebook cell 30)
# execution_count = 15

# Group propagation DISABLED -- GroupIDs have zero overlap between train/test in this data version
# All predictions come from the model + hard rules (CryoSleep+NoSpend only)

final_preds = np.full(len(test_feat), np.nan)

# Priority 3: Model (base layer for all rows)
model_preds = (cal_test >= cal_threshold).astype(float)
final_preds = model_preds.copy()

# Priority 1: Hard-rule locks (overrides everything)
for idx in locked_test_preds.dropna().index:
    final_preds[idx] = locked_test_preds[idx]

final_bool = final_preds.astype(bool)

# Build submission
submission = sample_submission.copy()
submission[CFG.target] = final_bool
submission.to_csv(CFG.submission_file, index=False)
save_table_artifact("final_submission", submission)

# Breakdown
n_locked_final = locked_test_preds.notna().sum()
n_model_only   = len(test_feat) - n_locked_final

summary = pd.DataFrame({
    'Stage': [
        'Hard rules (Priority 1: CryoSleep+NoSpend)',
        'Model prediction (Priority 2)',
        'TOTAL TEST SET',
    ],
    'Passengers': [
        n_locked_final,
        n_model_only,
        len(test_feat),
    ],
    '%': [
        f'{n_locked_final/len(test_feat)*100:.1f}%',
        f'{n_model_only/len(test_feat)*100:.1f}%',
        '100%',
    ],
})

print('[OK] Final submission created!')
print(f'   File: {CFG.submission_file}')
print(f'   Predicted positive rate: {final_bool.mean():.4f}')
print()
tracked_print(summary)
tracked_print(submission.head(10))
save_table_artifact("prediction_sources", summary)
save_table_artifact("submission_head", submission.head(10))

# -- DIAGNOSTIC: save raw model-only submission (no hard rules, no group propagation) --
sub_model_only = sample_submission.copy()
sub_model_only[CFG.target] = (cal_test >= 0.5).astype(bool)
sub_model_only.to_csv('submission_model_only.csv', index=False)
model_only_rate = sub_model_only['Transported'].mean()

sub_calibrated = sample_submission.copy()
sub_calibrated[CFG.target] = model_preds.astype(bool)  # cal_test >= cal_threshold
sub_calibrated.to_csv('submission_calibrated.csv', index=False)

print(f'\n   --- Diagnostic ---')
print(f'   Raw model (cal_test >= 0.5) positive rate : {model_only_rate:.4f}')
print(f'   Calibrated model positive rate            : {model_preds.mean():.4f}')
print(f'   Final (model + hard rules) rate           : {final_bool.mean():.4f}')
print(f'   Calibration threshold                     : {cal_threshold:.4f}')
print(f'   OOF CV accuracy (best pipeline)           : {final_cv_acc:.5f}')
print(f'   Pre-calibration OOF accuracy              : {pre_cal_acc:.5f}')
print(f'   Post-calibration OOF accuracy             : {cal_acc:.5f}')
print(f'   Winners: {best_final_label}')
print(f'   Hard-rule locked (CryoSleep+NoSpend): {n_locked_final}')
print(f'   Group propagation: DISABLED (no train/test GroupId overlap)')

# -- Save pre-pseudo baseline submission (for OOF<->LB gap diagnosis) ------
sub_pre_pseudo = sample_submission.copy()
baseline_test_probs = results['test_probs']
sub_pre_pseudo[CFG.target] = (baseline_test_probs >= 0.5).astype(bool)
sub_pre_pseudo.to_csv('submission_pre_pseudo.csv', index=False)
print(f'   Pre-pseudo baseline positive rate        : {sub_pre_pseudo["Transported"].mean():.4f}')
print(f'   Pre-pseudo baseline saved as submission_pre_pseudo.csv')


# %% [code] cell 16 (notebook cell 32)
# execution_count = 16

fig = plt.figure(figsize=(20, 16))
fig.suptitle('[RUN] Spaceship Titanic -- 0.90+ Pipeline Dashboard',
             fontsize=18, fontweight='bold', color='white', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# -- 1. Pipeline score progression -------------------------------------------
ax1 = fig.add_subplot(gs[0, :])
stages = [
    ('Baseline\n(v1)', 0.81709),
    ('+ Group\nPropagation', results['cv_accuracy']),
    (f'+ Pseudo\nLabelling', current_results['cv_accuracy']),
    ('+ MLP\nBlend', final_cv_acc),
    ('+ Isotonic\nCalibration', cal_acc),
    ('Target\n0.90+', 0.90),
]
stage_names   = [s[0] for s in stages]
stage_values  = [s[1] for s in stages]
bar_colors    = [PALETTE[i % len(PALETTE)] for i in range(len(stages) - 1)] + ['yellow']

bars = ax1.bar(stage_names, stage_values, color=bar_colors, edgecolor='none', width=0.6)
ax1.axhline(0.90, color='yellow', linestyle='--', linewidth=2, alpha=0.8, label='0.90 target')
ax1.set_ylim(0.80, 0.92)
ax1.set_title('Score Progression Through Pipeline Stages', color='white', fontsize=13)
ax1.set_ylabel('OOF CV Accuracy')
ax1.legend()
for bar, v in zip(bars, stage_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{v:.5f}', ha='center', fontsize=9, color='white', fontweight='bold')

# -- 2. Final OOF prob distribution ------------------------------------------
ax2 = fig.add_subplot(gs[1, 0])
y_np = y.to_numpy()
for cls, color, lbl in [(0, RED, 'Not Transported'), (1, GREEN, 'Transported')]:
    ax2.hist(cal_oof[y_np == cls], bins=50, alpha=0.6, color=color,
             label=lbl, edgecolor='none')
ax2.axvline(cal_threshold, color='yellow', linestyle='--', linewidth=2)
ax2.set_title('Final Calibrated OOF Probabilities', color='white')
ax2.set_xlabel('P(Transported)')
ax2.legend(fontsize=8)

# -- 3. Model comparison ------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 1])
fold_scores  = current_results['fold_scores']
model_means  = fold_scores.groupby('model')['acc'].mean().sort_values(ascending=False)
bars3 = ax3.bar(model_means.index, model_means.values,
                color=PALETTE[:len(model_means)], edgecolor='none')
ax3.set_ylim(0.79, 0.86)
ax3.set_title('Mean CV Accuracy by Model', color='white')
ax3.set_ylabel('Accuracy')
for b, v in zip(bars3, model_means.values):
    ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0005,
             f'{v:.4f}', ha='center', fontsize=9, color='white')

# -- 4. OOF confusion matrix --------------------------------------------------
ax4 = fig.add_subplot(gs[1, 2])
cm   = confusion_matrix(y, (cal_oof >= cal_threshold).astype(int))
disp = ConfusionMatrixDisplay(cm, display_labels=['Not Transported', 'Transported'])
disp.plot(ax=ax4, colorbar=False, cmap='Blues')
ax4.set_title(f'Confusion Matrix (Acc={cal_acc:.5f})', color='white')

# -- 5. Prediction source breakdown ------------------------------------------
ax5 = fig.add_subplot(gs[2, 0])
sources = ['Hard Rules', 'Model Only']
counts  = [n_locked_final, n_model_only]
ax5.pie(counts, labels=sources, autopct='%1.1f%%',
        colors=[GREEN, BLUE], startangle=90,
        textprops={'color':'white','fontsize':10},
        wedgeprops={'edgecolor':'#0e1117','linewidth':2})
ax5.set_title('Test Prediction Sources', color='white')

# -- 6. Submission class distribution ----------------------------------------
ax6 = fig.add_subplot(gs[2, 1])
sub_vc = submission[CFG.target].value_counts()
bars6 = ax6.bar(['Not Transported', 'Transported'], sub_vc.values,
                color=[RED, GREEN], edgecolor='none', width=0.5)
for b, v in zip(bars6, sub_vc.values):
    ax6.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
             f'{v:,}\n({v/len(submission)*100:.1f}%)',
             ha='center', fontsize=11, color='white', fontweight='bold')
ax6.set_title('Submission: Predicted Class Split', color='white')
ax6.set_ylabel('Count')

# -- 7. SHAP summary (if available) ------------------------------------------
ax7 = fig.add_subplot(gs[2, 2])
if 'shap_importance' in dir():
    top10 = shap_importance.head(10)
    ax7.barh(top10['feature'][::-1], top10['mean_shap'][::-1],
             color=PURPLE, edgecolor='none')
    ax7.set_title('Top 10 SHAP Features', color='white')
    ax7.set_xlabel('Mean |SHAP|')
else:
    ax7.text(0.5, 0.5, 'SHAP not run\n(LightGBM required)',
             ha='center', va='center', color='white', transform=ax7.transAxes)

plt.tight_layout()
finalize_figures()
print('[OK] Dashboard saved to pipeline_dashboard.png')


# %% [code] cell 17 (notebook cell 34)
# execution_count = 17

print('=' * 70)
print('  [RUN]  PIPELINE COMPLETE -- SCORE SUMMARY')
print('=' * 70)
print()
print(f"  Baseline (v1 5-model ensemble)    : ~0.81709")
print(f"  + Group label propagation         :  {results['cv_accuracy']:.5f}")
print(f"  + Pseudo-labelling                :  {current_results['cv_accuracy']:.5f}")
print(f"  + MLP neural blending             :  {final_cv_acc:.5f}")
print(f"  + Isotonic calibration            :  {cal_acc:.5f}")
print()
print(f"  Submission file                   :  {CFG.submission_file}")
print(f"  Predicted positive rate           :  {final_bool.mean():.4f}")
print()
print('-' * 70)
print('  [TARGET] REMAINING CEILING BREAKERS (To Truly Hit 0.90+):')
print('-' * 70)
print()
print('  1 FT-Transformer / TabNet   -- Deep learning tabular architectures')
print('     that model feature interactions without manual engineering.')
print('     Libraries: pytorch-tabnet, tab-transformer-pytorch')
print()
print('  2 Adversarial Validation    -- Detect train/test distribution shift.')
print('     Train a classifier to predict "is this train or test?"')
print('     Remove train rows that look too different from test.')
print()
print('  3 Surname Exact Match       -- Passengers with identical surnames')
print('     in train+test almost always share the same label.')
print('     Add a surname-level OOF target encoded feature.')
print()
print('  4 Cabin-Number Propagation  -- Passengers in same cabin share label.')
print('     Build cabin-level group agreement score (not just group_id).')
print()
print('  5 Larger Optuna Search      -- Run 500+ trials, tune all 5 models.')
print(f'     Current: {CFG.optuna_trials} trials on LGB only.')
print()
print('  6 Noise Injection           -- Add small Gaussian noise to numeric')
print('     features during each fold to regularise and reduce overfitting.')
print()
print('  7 Stochastic Ensemble       -- Run 10+ seeds, 10-fold CV.')
print(f'     Current: {len(CFG.random_seeds)} seeds x {CFG.n_splits}-fold = {len(CFG.random_seeds)*CFG.n_splits} runs/model.')
print('     10 seeds x 10-fold = 100 runs/model -> much more stable OOF.')
print()
print('=' * 70)


# %% [code] cell 18 (notebook cell 35)


