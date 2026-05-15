
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
train_df = pd.read_csv(RAW_DIR / "train.csv")
test_df = pd.read_csv(RAW_DIR / "test.csv")

print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)

print("\n--- Train Data Info ---")
print(train_df.info())

print("\n--- Train Data Head ---")
print(train_df.head())

print("\n--- Missing Values in Train ---")
print(train_df.isnull().sum())

print("\n--- Missing Values in Test ---")
print(test_df.isnull().sum())

# Identify numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=[np.number]).columns
categorical_cols = train_df.select_dtypes(exclude=[np.number]).columns

print("\nNumerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)

# Basic stats
print("\n--- Numerical Stats ---")
print(train_df.describe())

print("\n--- Categorical Stats ---")
for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(train_df[col].value_counts().head())
