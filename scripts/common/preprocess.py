import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(RAW_DIR / "train.csv")
test_df = pd.read_csv(RAW_DIR / "test.csv")

# Combine for preprocessing
train_df['is_train'] = 1
test_df['is_train'] = 0
data = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

proc_tag = os.environ.get("PROC_TAG", "").strip()
use_tfidf = os.environ.get("USE_TFIDF", "0").strip() == "1"

def with_tag(name: str) -> str:
    return f"{name}_{proc_tag}" if proc_tag else name

def add_tfidf_svd(train_text: pd.Series, test_text: pd.Series, n_features: int = 1000, n_components: int = 5, prefix: str = "Last_Name") -> tuple[pd.DataFrame, pd.DataFrame]:
    train_text = train_text.fillna("UNK").astype(str)
    test_text = test_text.fillna("UNK").astype(str)
    vectorizer = TfidfVectorizer(max_features=n_features)
    X_tr = vectorizer.fit_transform(train_text)
    X_te = vectorizer.transform(test_text)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z_tr = svd.fit_transform(X_tr)
    Z_te = svd.transform(X_te)
    cols = [f"{prefix}_tfidf_{i}" for i in range(n_components)]
    return pd.DataFrame(Z_tr, columns=cols), pd.DataFrame(Z_te, columns=cols)

# 1. Feature Engineering: Cabin
def split_cabin(x):
    if pd.isna(x):
        return np.nan, np.nan, np.nan
    else:
        try:
            deck, num, side = x.split('/')
            return deck, float(num), side
        except ValueError:
            return np.nan, np.nan, np.nan

print("Processing Cabin...")
data[['Deck', 'Cabin_Num', 'Side']] = data['Cabin'].apply(lambda x: pd.Series(split_cabin(x)))

# 2. Feature Engineering: Group Size
print("Processing Group Size...")
data['GroupId'] = data['PassengerId'].apply(lambda x: x.split('_')[0])
group_counts = data['GroupId'].value_counts()
data['GroupSize'] = data['GroupId'].map(group_counts)
data['IsAlone'] = (data['GroupSize'] == 1).astype(int)

data["Last_Name"] = (
    data["Name"]
    .fillna("UNK")
    .astype(str)
    .str.split()
    .str[-1]
    .fillna("UNK")
)

# 3. Impute Missing Values
print("Imputing Missing Values...")
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_Num']
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Fill numerical with median
for col in numerical_cols:
    data[col] = data[col].fillna(data[col].median())

# Fill categorical with mode
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# 4. Feature Engineering: Total Spend & Log Transform
print("Processing Total Spend & Log Transform...")
data['TotalSpend'] = data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck']
data['LogTotalSpend'] = np.log1p(data['TotalSpend'])
data['AnySpend'] = (data[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1) > 0).astype(int)
data['NumSpendCats'] = (data[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] > 0).sum(axis=1)
data['LogRoomService'] = np.log1p(data['RoomService'])
data['LogFoodCourt'] = np.log1p(data['FoodCourt'])
data['LogShoppingMall'] = np.log1p(data['ShoppingMall'])
data['LogSpa'] = np.log1p(data['Spa'])
data['LogVRDeck'] = np.log1p(data['VRDeck'])

# 5. Encoding
# Binary Encoding
data['CryoSleep'] = data['CryoSleep'].astype(bool).astype(int)
data['VIP'] = data['VIP'].astype(bool).astype(int)
data['CryoSleepSpendConflict'] = ((data['CryoSleep'] == 1) & (data['TotalSpend'] > 0)).astype(int)
data['CabinBin'] = pd.qcut(data['Cabin_Num'], q=10, labels=False, duplicates='drop')

# One-Hot Encoding
data = pd.get_dummies(data, columns=['HomePlanet', 'Destination', 'Deck', 'Side'], drop_first=True)

if use_tfidf:
    print("Adding TFIDF-SVD features for Last_Name...")
    tr_mask = data["is_train"] == 1
    te_mask = data["is_train"] == 0
    tf_tr, tf_te = add_tfidf_svd(
        data.loc[tr_mask, "Last_Name"],
        data.loc[te_mask, "Last_Name"],
        n_features=1000,
        n_components=5,
        prefix="Last_Name",
    )
    tf_tr.index = data.index[tr_mask]
    tf_te.index = data.index[te_mask]
    tf_all = pd.concat([tf_tr, tf_te], axis=0).sort_index()
    data = pd.concat([data, tf_all], axis=1)

# Drop unnecessary columns
# Keep PassengerId for test submission, but drop for training (will handle later)
# Dropping GroupId as we have GroupSize now
cols_to_drop = ['Name', 'Last_Name', 'Cabin', 'GroupId'] 
data = data.drop(columns=cols_to_drop)

# Split back
train_processed = data[data['is_train'] == 1].drop(columns=['is_train'])
test_processed = data[data['is_train'] == 0].drop(columns=['is_train', 'Transported'])

# Convert Transported to int in train
train_processed['Transported'] = train_processed['Transported'].astype(int)

# Save
train_processed.to_csv(OUT_DIR / f"{with_tag('train_processed')}.csv", index=False)
test_processed.to_csv(OUT_DIR / f"{with_tag('test_processed')}.csv", index=False)

print("Preprocessing complete!")
print("Train processed shape:", train_processed.shape)
print("Test processed shape:", test_processed.shape)
