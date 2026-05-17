from pathlib import Path
import random, os, numpy as np

GLOBAL_RANDOM_STATE = 42

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_global_seed(GLOBAL_RANDOM_STATE)


class CFG:
    target = 'Transported'
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    categorical_base = [
        'HomePlanet', 'CryoSleep', 'Destination', 'VIP',
        'CabinDeck', 'CabinSide', 'HomeDest', 'DeckSide',
        'CabinZone', 'AgeBand', 'Surname'
    ]

    feature_cols = [
        'HomePlanet', 'CryoSleep', 'Destination', 'VIP',
        'CabinDeck', 'CabinSide', 'HomeDest', 'DeckSide', 'CabinZone', 'AgeBand', 'Surname',
        'GroupSize', 'Solo', 'FamilySize',
        'Age', 'CabinNum', 'CryoFlag', 'VipFlag',
        'IsChild', 'IsTeen', 'IsSenior', 'SpendPositiveCount', 'NoSpend',
        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
        'TotalSpend', 'AvgSpendPerService', 'SpendPerGroupMember',
        'Log_RoomService', 'Log_FoodCourt', 'Log_ShoppingMall', 'Log_Spa', 'Log_VRDeck',
        'Log_TotalSpend', 'Log_AvgSpendPerService', 'Log_SpendPerGroupMember',
        'AgeSpendInteraction',
        'CabinNumParity', 'CabinNumBucket', 'SurnameGroupSize',
        'SpendEntropy', 'MaxSpendCategory',
        'CryoNoSpend', 'NotCryoHasSpend',
        'CabinAgreementScore', 'CabinMean',
        'Surname_tfidf_0', 'Surname_tfidf_1', 'Surname_tfidf_2',
        'Surname_tfidf_3', 'Surname_tfidf_4',
        'TE_HomePlanet', 'TE_CabinDeck', 'TE_HomeDest', 'TE_DeckSide',
        'TE_AgeBand', 'TE_CryoSleep', 'TE_CabinZone',
    ]

    TE_COLS = ['HomePlanet', 'CabinDeck', 'HomeDest', 'DeckSide', 'AgeBand', 'CryoSleep', 'CabinZone']

    random_seeds    = [42, 2024, 7, 2025, 88]
    n_splits        = 5

    optuna_trials   = 200
    optuna_cv_folds = 3

    ensemble_runs       = 1
    ensemble_base_seeds = [42, 123, 888]

    submission_file    = 'submission_90plus.csv'
    blend_models       = ['hist_gb', 'xgb', 'lgb']
    shap_keep_percentile = 10
    feature_sample_ratio = 1.0
    mixup_alpha         = 0.4
    mixup_multiplier    = 1
    noise_std           = 0.01
    group_swap_fraction = 0.0
    data_dir           = Path('data/raw')
    output_dir         = Path('modular_pipeline/output')
    processed_data_file  = output_dir / 'processed_data.pkl'
