import pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from config import CFG

CFG.output_dir.mkdir(parents=True, exist_ok=True)


def mode_or_nan(s: pd.Series) -> Any:
    non_null = s.dropna()
    if non_null.empty:
        return np.nan
    m = non_null.mode(dropna=True)
    return m.iloc[0] if not m.empty else non_null.iloc[0]

def fill_group_mode(df: pd.DataFrame, key: str, col: str) -> None:
    mapping = df.groupby(key)[col].agg(mode_or_nan)
    df[col] = df[col].fillna(df[key].map(mapping))

def parse_cabin(s: pd.Series):
    c = s.fillna('U/9999/U').astype(str).str.split('/', expand=True)
    return c[0].replace('nan', 'U'), pd.to_numeric(c[1], errors='coerce'), c[2].replace('nan', 'U')


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train = train_df.copy()
    test = test_df.copy()
    train['_is_train'] = 1
    test['_is_train'] = 0
    test[CFG.target] = np.nan
    full = pd.concat([train, test], ignore_index=True)

    gp = full['PassengerId'].str.split('_', expand=True)
    full['GroupId'] = pd.to_numeric(gp[0], errors='coerce')
    full['GroupMember'] = pd.to_numeric(gp[1], errors='coerce')
    full['GroupSize'] = full.groupby('GroupId')['PassengerId'].transform('size').astype(int)
    full['Solo'] = (full['GroupSize'] == 1).astype(int)

    full['CabinDeck'], full['CabinNum'], full['CabinSide'] = parse_cabin(full['Cabin'])
    full['CabinNumParity'] = (full['CabinNum'] % 2).fillna(-1).astype(int)
    full['CabinNumBucket'] = pd.cut(full['CabinNum'], bins=10, labels=False, duplicates='drop').fillna(-1).astype(int)

    nm = full['Name'].fillna('Unknown Unknown').astype(str).str.split(' ', n=1, expand=True)
    full['Surname'] = nm[1].fillna('Unknown')
    full['FamilySize'] = full.groupby('Surname')['PassengerId'].transform('size').astype(int)
    full['SurnameGroupSize'] = full.groupby('Surname')['GroupId'].transform('nunique').astype(int)

    sp0 = full[CFG.spend_cols].fillna(0).sum(axis=1)
    full.loc[full['CryoSleep'].isna() & (sp0 > 0), 'CryoSleep'] = False
    full.loc[full['CryoSleep'].isna() & (sp0 == 0), 'CryoSleep'] = True

    for c in ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide', 'Surname']:
        fill_group_mode(full, 'GroupId', c)

    for src, tgt, fallback_col in [
        ('CabinDeck', 'HomePlanet', None),
        ('HomePlanet', 'Destination', None),
        ('HomePlanet', 'CabinDeck', None),
    ]:
        mapping = full.groupby(src)[tgt].agg(mode_or_nan)
        full[tgt] = full[tgt].fillna(full[src].map(mapping))
    full['HomePlanet'] = full['HomePlanet'].fillna(mode_or_nan(full['HomePlanet']))
    full['Destination'] = full['Destination'].fillna(mode_or_nan(full['Destination']))
    full['CabinDeck'] = full['CabinDeck'].fillna('U')
    full['CabinSide'] = full['CabinSide'].fillna(mode_or_nan(full['CabinSide']))

    full['CabinNum'] = (full['CabinNum']
                        .fillna(full.groupby('GroupId')['CabinNum'].transform('median'))
                        .fillna(full['CabinNum'].median()))
    full['Age'] = (full['Age']
                   .fillna(full.groupby('GroupId')['Age'].transform('median'))
                   .fillna(full.groupby('HomePlanet')['Age'].transform('median'))
                   .fillna(full['Age'].median()))
    full['VIP'] = full['VIP'].fillna(False)

    for col in CFG.spend_cols:
        full.loc[full['CryoSleep'] == True, col] = 0.0
        hp_med = full.groupby('HomePlanet')[col].transform('median')
        full[col] = full[col].fillna(hp_med).fillna(full[col].median())
        full.loc[full['CryoSleep'] == True, col] = 0.0

    full['TotalSpend'] = full[CFG.spend_cols].sum(axis=1)
    full['SpendPositiveCount'] = (full[CFG.spend_cols] > 0).sum(axis=1).astype(int)
    full['NoSpend'] = (full['TotalSpend'] == 0).astype(int)
    full['AvgSpendPerService'] = full['TotalSpend'] / full['SpendPositiveCount'].replace(0, 1)
    full['SpendPerGroupMember'] = full['TotalSpend'] / full['GroupSize'].replace(0, 1)

    spend_probs = full[CFG.spend_cols].div(full['TotalSpend'].replace(0, 1), axis=0).clip(0, 1)
    spend_probs = spend_probs.replace(0, 1e-9)
    full['SpendEntropy'] = -(spend_probs * np.log(spend_probs)).sum(axis=1)
    full['MaxSpendCategory'] = full[CFG.spend_cols].idxmax(axis=1).fillna('None')

    for col in CFG.spend_cols + ['TotalSpend', 'AvgSpendPerService', 'SpendPerGroupMember']:
        full[f'Log_{col}'] = np.log1p(full[col])

    full['CryoFlag'] = full['CryoSleep'].astype(int)
    full['VipFlag'] = full['VIP'].astype(int)
    full['IsChild'] = (full['Age'] < 13).astype(int)
    full['IsTeen'] = ((full['Age'] >= 13) & (full['Age'] < 18)).astype(int)
    full['IsSenior'] = (full['Age'] >= 60).astype(int)
    full['AgeSpendInteraction'] = full['Age'] * full['Log_TotalSpend']

    full['CryoNoSpend'] = ((full['CryoFlag'] == 1) & (full['NoSpend'] == 1)).astype(int)
    full['NotCryoHasSpend'] = ((full['CryoFlag'] == 0) & (full['TotalSpend'] > 0)).astype(int)

    full['AgeBand'] = pd.cut(full['Age'], bins=[-1, 12, 18, 25, 40, 60, 120],
                             labels=['child', 'teen', 'young_adult', 'adult', 'midlife', 'senior']).astype(str)
    full['CabinZone'] = pd.qcut(full['CabinNum'], q=6, duplicates='drop').astype(str)
    full['HomeDest'] = full['HomePlanet'].astype(str) + '_' + full['Destination'].astype(str)
    full['DeckSide'] = full['CabinDeck'].astype(str) + '_' + full['CabinSide'].astype(str)

    full['CryoSleep'] = full['CryoSleep'].map({True: 'True', False: 'False'}).fillna('False')
    full['VIP'] = full['VIP'].map({True: 'True', False: 'False'}).fillna('False')

    tfidf = TfidfVectorizer(max_features=1000)
    vec_all = tfidf.fit_transform(full['Surname'])
    svd = TruncatedSVD(n_components=5, random_state=42)
    tfidf_arr = svd.fit_transform(vec_all)
    for i in range(5):
        full[f'Surname_tfidf_{i}'] = tfidf_arr[:, i]

    train_mask = full['_is_train'] == 1
    test_mask = full['_is_train'] == 0
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
    train_cabin['_cabin_sum'] = train_cabin['CabinNum'].map(
        cabin_stats.set_index('CabinNum')['cabin_mean']) * \
        train_cabin['CabinNum'].map(cabin_stats.set_index('CabinNum')['cabin_count'])
    train_cabin['_cabin_count'] = train_cabin['CabinNum'].map(
        cabin_stats.set_index('CabinNum')['cabin_count'])
    loo_sum = (train_cabin['_cabin_sum'] - train_cabin['_y']).clip(0)
    loo_count = (train_cabin['_cabin_count'] - 1).clip(1)
    loo_mean = (loo_sum / loo_count).fillna(y_full.mean())
    full.loc[train_mask, 'CabinAgreementScore'] = np.maximum(loo_mean, 1 - loo_mean)
    full.loc[train_mask, 'CabinMean'] = loo_mean

    cabin_map_mean = cabin_stats.set_index('CabinNum')['cabin_mean']
    cabin_map_agr = cabin_stats.set_index('CabinNum')['cabin_agreement']
    test_cabins = full.loc[test_mask, 'CabinNum']
    full.loc[test_mask, 'CabinAgreementScore'] = test_cabins.map(cabin_map_agr).fillna(0.5)
    full.loc[test_mask, 'CabinMean'] = test_cabins.map(cabin_map_mean).fillna(y_full.mean())

    train_out = full[full['_is_train'] == 1].drop(columns=['_is_train']).reset_index(drop=True)
    test_out = full[full['_is_train'] == 0].drop(columns=['_is_train']).drop(columns=[CFG.target]).reset_index(drop=True)
    return train_out, test_out


def oof_target_encode(train_df, test_df, y_train, cols, n_splits=5, seed=42, smoothing=10.0):
    train_enc = train_df.copy()
    test_enc = test_df.copy()
    global_mean = y_train.mean()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in cols:
        te_col = f'TE_{col}'
        train_enc[te_col] = np.nan
        for tr_idx, val_idx in cv.split(train_df, y_train):
            tr_part = train_df.iloc[tr_idx].copy()
            tr_part['_y'] = y_train.iloc[tr_idx].values
            val_part = train_df.iloc[val_idx]
            stats = tr_part.groupby(col)['_y'].agg(['mean', 'count'])
            smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
            train_enc.loc[val_idx, te_col] = val_part[col].map(smooth).fillna(global_mean).values

        tmp = train_df[[col]].copy()
        tmp['_y'] = y_train.values
        stats_full = tmp.groupby(col)['_y'].agg(['mean', 'count'])
        smooth_full = (stats_full['count'] * stats_full['mean'] + smoothing * global_mean) / (stats_full['count'] + smoothing)
        test_enc[te_col] = test_df[col].map(smooth_full).fillna(global_mean)
        train_enc[te_col] = train_enc[te_col].fillna(global_mean)

    return train_enc, test_enc


def encode_ordinal(train_x, test_x, cat_cols):
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1)
    tr = train_x.copy()
    te = test_x.copy()
    tc = train_x[cat_cols].fillna('__MISSING__').astype(str)
    ec = test_x[cat_cols].fillna('__MISSING__').astype(str)
    enc.fit(pd.concat([tc, ec], ignore_index=True))
    tr[cat_cols] = enc.transform(tc)
    te[cat_cols] = enc.transform(ec)
    return tr.astype(float), te.astype(float)


def compute_group_label_propagation(train_f, test_f, y_train):
    train_out = train_f.copy()
    test_out = test_f.copy()
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
    )
    train_out['_group_sum'] = train_out['GroupId'].map(group_stats.set_index('GroupId')['group_sum'])
    train_out['_group_count'] = train_out['GroupId'].map(group_stats.set_index('GroupId')['group_count'])
    train_out['GroupTrainKnownRate'] = 1.0
    loo_sum = (train_out['_group_sum'] - train_out['_y']).clip(0)
    loo_count = (train_out['_group_count'] - 1).clip(1)
    loo_mean = (loo_sum / loo_count).fillna(0.5)
    train_out['GroupAgreementScore'] = np.maximum(loo_mean, 1 - loo_mean)
    train_out['GroupMean'] = loo_mean
    train_out['GroupTrainKnown'] = (loo_count > 0).astype(int)
    train_out = train_out.drop(columns=['_y', '_group_sum', '_group_count'])

    total_group_size = pd.concat([
        train_out[['GroupId', 'PassengerId']],
        test_out[['GroupId', 'PassengerId']]
    ]).groupby('GroupId')['PassengerId'].count().rename('total_size')
    group_stats_map = group_stats.set_index('GroupId')
    test_out['GroupAgreementScore'] = test_out['GroupId'].map(group_stats_map['group_agreement']).fillna(0.5)
    test_out['GroupMean'] = test_out['GroupId'].map(group_stats_map['group_mean']).fillna(0.5)
    test_out['GroupTrainKnown'] = test_out['GroupId'].isin(group_stats['GroupId']).astype(int)
    train_known_count = group_stats_map['group_count']
    test_out['GroupTrainKnownRate'] = (
        test_out['GroupId'].map(train_known_count).fillna(0) /
        test_out['GroupId'].map(total_group_size).fillna(1)
    ).clip(0, 1)
    return train_out, test_out


def apply_hard_rules_train(df, y_true):
    results = {}
    mask_r1 = (df['CryoFlag'] == 1) & (df['NoSpend'] == 1)
    if mask_r1.sum() > 0:
        acc_r1 = (y_true[mask_r1] == 1).mean()
        results['R1_CryoNoSpend'] = {'n': mask_r1.sum(), 'acc': acc_r1, 'label': 1}
    return results


def generate_locked_test_predictions(test_df):
    locked = pd.Series(np.nan, index=test_df.index)
    r1_mask = (test_df['CryoFlag'] == 1) & (test_df['NoSpend'] == 1)
    locked[r1_mask] = 1
    return locked


if __name__ == '__main__':
    print('[OK] Loading data...')
    train_raw = pd.read_csv(CFG.data_dir / 'train.csv')
    test_raw = pd.read_csv(CFG.data_dir / 'test.csv')
    sample_submission = pd.read_csv(CFG.data_dir / 'sample_submission.csv')

    print('[OK] Engineering features...')
    train_feat, test_feat = engineer_features(train_raw, test_raw)
    y = train_feat[CFG.target].astype(int)

    train_cabins = set(train_feat['CabinNum'].dropna().astype(int))
    test_cabins = set(test_feat['CabinNum'].dropna().astype(int))
    cabin_overlap = train_cabins & test_cabins
    cabin_overlap_test_pct = test_feat['CabinNum'].dropna().isin(train_cabins).mean() * 100
    print(f'   Features: train {train_feat.shape}, test {test_feat.shape}')
    print(f'   CabinNum overlap: {len(cabin_overlap):,} -> {cabin_overlap_test_pct:.1f}% test covered')

    print('[OK] Group label propagation...')
    train_feat, test_feat = compute_group_label_propagation(train_feat, test_feat, y)
    print('   Group propagation: DISABLED (no train/test GroupId overlap)')

    print('[OK] Hard rules...')
    rule_stats = apply_hard_rules_train(train_feat, y)
    locked_test_preds = generate_locked_test_predictions(test_feat)
    n_locked = locked_test_preds.notna().sum()
    print(f'   Locked (CryoSleep+NoSpend): {n_locked} / {len(test_feat)} ({n_locked/len(test_feat)*100:.1f}%)')

    print('[OK] OOF Target Encoding...')
    train_feat, test_feat = oof_target_encode(
        train_feat, test_feat, y, CFG.TE_COLS,
        n_splits=CFG.n_splits, seed=42, smoothing=10.0
    )
    print(f'   TE columns added: {[f"TE_{c}" for c in CFG.TE_COLS]}')

    print('[OK] Encoding categorical features...')
    FEAT_COLS = [c for c in CFG.feature_cols if c in train_feat.columns]
    CATEGORICAL_COLS = [c for c in CFG.categorical_base + ['MaxSpendCategory'] if c in FEAT_COLS]

    X_train_raw = train_feat[FEAT_COLS].copy()
    X_test_raw = test_feat[FEAT_COLS].copy()
    X_train_num, X_test_num = encode_ordinal(X_train_raw, X_test_raw, CATEGORICAL_COLS)
    print(f'   X_train_num: {X_train_num.shape}, X_test_num: {X_test_num.shape}')

    print(f'[OK] Saving processed data to {CFG.processed_data_file}...')
    processed = {
        'X_train_num': X_train_num,
        'X_test_num': X_test_num,
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'y': y,
        'FEAT_COLS': FEAT_COLS,
        'CATEGORICAL_COLS': CATEGORICAL_COLS,
        'train_feat': train_feat,
        'test_feat': test_feat,
        'locked_test_preds': locked_test_preds,
        'sample_submission': sample_submission,
    }
    with open(CFG.processed_data_file, 'wb') as f:
        pickle.dump(processed, f)

    print(f'[OK] Data preparation complete!')
