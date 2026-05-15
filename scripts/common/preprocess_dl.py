
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from pathlib import Path

print("Starting DL preprocessing...")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(RAW_DIR / "train.csv")
test_df = pd.read_csv(RAW_DIR / "test.csv")

# Combine for preprocessing
data = pd.concat([train_df.drop('Transported', axis=1), test_df], ignore_index=True)

# --- Feature Engineering (same as before, but adapted for DL) ---
def split_cabin(x):
    if pd.isna(x):
        return 'UNK', -1.0, 'UNK' # Use UNK for unknown string, -1 for num
    else:
        try:
            deck, num, side = x.split('/')
            return deck, float(num), side
        except ValueError:
            return 'UNK', -1.0, 'UNK'

data[['Deck', 'Cabin_Num', 'Side']] = data['Cabin'].apply(lambda x: pd.Series(split_cabin(x)))

data['GroupId'] = data['PassengerId'].apply(lambda x: x.split('_')[0])
group_counts = data['GroupId'].value_counts()
data['GroupSize'] = data['GroupId'].map(group_counts)
data['IsAlone'] = (data['GroupSize'] == 1).astype(int)

# --- Imputation ---
# For DL, we can be more explicit with imputation
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_Num']
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

for col in numerical_cols:
    data[col] = data[col].fillna(data[col].median())

for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# --- More Feature Engineering ---
data['TotalSpend'] = data[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1)
data['AnySpend'] = (data['TotalSpend'] > 0).astype(int)
data['CryoSleepSpendConflict'] = ((data['CryoSleep'] == 1) & (data['TotalSpend'] > 0)).astype(int)

# --- Encoding & Scaling ---
# Identify categorical and numerical features for DL
# We will use the original categorical features for embedding layers
categorical_features_for_dl = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Label Encoding for categorical features
label_encoders = {}
for col in categorical_features_for_dl:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Identify numerical features for scaling
numerical_features_for_dl = [
    'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 
    'Cabin_Num', 'GroupSize', 'IsAlone', 'TotalSpend', 'AnySpend', 'CryoSleepSpendConflict'
]

# StandardScaler for numerical features
scaler = StandardScaler()
data[numerical_features_for_dl] = scaler.fit_transform(data[numerical_features_for_dl])

# --- Final Touches ---
# Drop columns not needed for the model
data = data.drop(columns=['PassengerId', 'Name', 'Cabin', 'GroupId'])

# Split back into train and test
train_processed_dl = data.iloc[:len(train_df)]
test_processed_dl = data.iloc[len(train_df):]

# Add target back to train
train_processed_dl['Transported'] = train_df['Transported'].astype(int)

# Save processed data
train_processed_dl.to_csv(OUT_DIR / "train_processed_dl.csv", index=False)
test_processed_dl.to_csv(OUT_DIR / "test_processed_dl.csv", index=False)

# Save encoders and scaler for inference
with open(OUT_DIR / "dl_preproc_artifacts.pkl", "wb") as f:
    pickle.dump({'label_encoders': label_encoders, 'scaler': scaler}, f)

print("DL preprocessing complete!")
print(f"Train DL shape: {train_processed_dl.shape}")
print(f"Test DL shape: {test_processed_dl.shape}")
