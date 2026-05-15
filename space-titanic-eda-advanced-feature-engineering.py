# Auto-exported from Jupyter notebook
# Source: space-titanic-eda-advanced-feature-engineering.ipynb

# %% [code] cell 1 (notebook cell 4)

import sklearn
import numpy as np
import os
import datetime
import builtins
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from prettytable import PrettyTable
from pathlib import Path
# %matplotlib inline
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
tqdm_notebook.get_lock().locks = []
# !pip install sweetviz
# import sweetviz as sv
import concurrent.futures
from copy import deepcopy       
from functools import partial
from itertools import combinations
import random
from random import randint, uniform
import gc
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler,PowerTransformer, FunctionTransformer
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from itertools import combinations
from sklearn.impute import SimpleImputer
import xgboost as xg
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,mean_squared_log_error, roc_auc_score, accuracy_score, f1_score, precision_recall_curve, log_loss
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from gap_statistic.optimalK import OptimalK
from scipy import stats
import statsmodels.api as sm
from scipy.stats import ttest_ind
from scipy.stats import boxcox
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
import optuna
import xgboost as xgb
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder, CatBoostEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier,ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from catboost import Pool
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from notebook_hyperparameter_tuning import (
    get_trained_params,
)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
pd.pandas.set_option('display.max_columns',None)

ARTIFACT_ROOT = Path("results/notebook_export")
FIGURE_DIR = ARTIFACT_ROOT / "figures"
TABLE_DIR = ARTIFACT_ROOT / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
USE_EXTERNAL_BLEND = False
GLOBAL_RANDOM_STATE = 2023
NUM_ENSEMBLE_RUNS = 3

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

_ORIGINAL_PRINT = builtins.print
_figure_counter = 0
_table_counter = 0


def _slugify(text, fallback):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or fallback


def save_table_artifact(name, obj):
    global _table_counter
    _table_counter += 1
    base_name = f"{_table_counter:03d}_{_slugify(name, 'table')}"

    if isinstance(obj, pd.DataFrame):
        obj.to_csv(TABLE_DIR / f"{base_name}.csv", index=True)
        obj.to_html(TABLE_DIR / f"{base_name}.html", index=True)
    elif isinstance(obj, pd.Series):
        obj.to_frame(name=obj.name or "value").to_csv(TABLE_DIR / f"{base_name}.csv", index=True)
    elif isinstance(obj, PrettyTable):
        (TABLE_DIR / f"{base_name}.txt").write_text(str(obj), encoding="utf-8")


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
    fig.savefig(FIGURE_DIR / f"{base_name}.png", dpi=300, bbox_inches="tight")


def tracked_print(*args, **kwargs):
    for arg in args:
        if isinstance(arg, (pd.DataFrame, pd.Series, PrettyTable)):
            save_table_artifact(type(arg).__name__, arg)
    return _ORIGINAL_PRINT(*args, **kwargs)


def finalize_figures(*args, **kwargs):
    for fig_num in plt.get_fignums():
        save_figure_artifact(plt.figure(fig_num))
    plt.close("all")
    return None


builtins.print = tracked_print
_ORIGINAL_PRINT(f"Artifacts will be saved under: {ARTIFACT_ROOT.resolve()}")

# %% [code] cell 2 (notebook cell 5)
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

save_table_artifact("train_head_initial", train.head())

# %% [code] cell 3 (notebook cell 8)
table = PrettyTable()

table.field_names = ['Column Name', 'Data Type', 'Non-Null Count']
for column in train.columns:
    data_type = str(train[column].dtype)
    non_null_count = train[column].count()
    table.add_row([column, data_type, non_null_count])
print(table)

# %% [code] cell 4 (notebook cell 9)
msno.matrix(train)
finalize_figures()

# %% [code] cell 5 (notebook cell 10)
msno.matrix(test)
finalize_figures()

# %% [code] cell 6 (notebook cell 14)
# Calculate the proportion of each class
class_counts = train['Transported'].value_counts()
class_proportions = class_counts / train.shape[0]
class_proportions = class_proportions.values.tolist()
class_proportions_str = [f'{prop:.2%}' for prop in class_proportions]

# Set the color palette
colors = sns.color_palette('pastel')[0:len(class_counts)]

# Plot the distribution of the target variable
plt.figure(figsize=(8, 4))
sns.countplot(x='Transported', data=train, palette=colors)
plt.title('Distribution of Target Variable', fontsize=16)
plt.xlabel('Transported', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.ylim([0, len(train)])
for i, count in enumerate(class_counts):
    plt.text(i, count + 50, class_proportions_str[i], ha='center', fontsize=14, color='black')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()
finalize_figures()

# %% [code] cell 7 (notebook cell 18)
cont_cols=[f for f in train.columns if train[f].dtype in [float,int] and train[f].nunique()>3]

# Create subplots for each continuous column
fig, axs = plt.subplots(len(cont_cols), 2, figsize=(10,20))

# Loop through each continuous column and plot the histograms
for i, col in enumerate(cont_cols):
    # Determine the range of values to plot
    max_val = max(train[col].max(), test[col].max())
    min_val = min(train[col].min(), test[col].min())
    range_val = max_val - min_val
    
    # Determine the bin size and number of bins
    bin_size = range_val / 20
    num_bins_train = round(range_val / bin_size)
    num_bins_test = round(range_val / bin_size)
    
    # Plot the histograms
    sns.histplot(train[col], ax=axs[i][0], color='blue', kde=True, label='Train', bins=num_bins_train)
    sns.histplot(test[col], ax=axs[i][1], color='red', kde=True, label='Test', bins=num_bins_test)
    axs[i][0].set_title(f'Train - {col}')
    axs[i][0].set_xlabel('Value')
    axs[i][0].set_ylabel('Frequency')
    axs[i][1].set_title(f'Test - {col}')
    axs[i][1].set_xlabel('Value')
    axs[i][1].set_ylabel('Frequency')
    axs[i][0].legend()
    axs[i][1].legend()

plt.tight_layout()
finalize_figures()

# %% [code] cell 8 (notebook cell 21)
# Create subplots for each continuous feature
fig, axs = plt.subplots(nrows=len(cont_cols), figsize=(8, 4 * len(cont_cols)))
for i, col in enumerate(cont_cols):
    sns.boxplot(x='Transported', y=col, data=train, ax=axs[i], palette='pastel')
    axs[i].set_title(f'{col.title()} vs Target', fontsize=16)
    axs[i].set_xlabel('Transported', fontsize=14)
    axs[i].set_ylabel(col.title(), fontsize=14)
    axs[i].tick_params(axis='both', labelsize=14)
    sns.despine()

# Adjust spacing between subplots
fig.tight_layout()

# Display the plot
finalize_figures()

# %% [code] cell 9 (notebook cell 25)
sns.pairplot(data=train, vars=cont_cols, hue='Transported')
finalize_figures()

# %% [code] cell 10 (notebook cell 28)
# Define the numerical features to plot
features=["Spa","VRDeck","RoomService"]

# Create a figure with multiple subplots
fig, axs = plt.subplots(1, len(features), figsize=(16, 5))

# Loop through each feature and plot a violin plot on a separate subplot
for i, col in enumerate(features):
    sns.violinplot(x='Transported', y=col, data=train, ax=axs[i])
    axs[i].set_title(f'{col.title()} Distribution by Target', fontsize=14)
    axs[i].set_xlabel('Transported', fontsize=12)
    axs[i].set_ylabel(col.title(), fontsize=12)
    sns.despine()

# Adjust spacing between subplots
fig.tight_layout()

# Display the plot
finalize_figures()

# %% [code] cell 11 (notebook cell 32)

def perform_ttest(train, feature_list, target):
    """
    Performs t-test on a list of independent features for a binary classification problem
    
    :param train: pandas dataframe containing the training data
    :param feature_list: list of feature names to perform t-test on
    :param target: name of the target variable (binary)
    :return: dictionary containing t-test results
    """
    ttest_results = {}
    table = PrettyTable()

    table.field_names = ['Feature', 't_stat', 'p_val']
    
    for feature in feature_list:
        group_0 = train[train[target] == 0][feature]
        group_1 = train[train[target] == 1][feature]
        
        t_stat, p_val = ttest_ind(group_0, group_1, nan_policy='omit')
        table.add_row([feature,t_stat, p_val ])
        
    return print(table)
perform_ttest(train, cont_cols, 'Transported')

# %% [code] cell 12 (notebook cell 35)
from scipy.stats import f_oneway

def perform_anova(train, feature_list, target):
    """
    Performs ANOVA on a list of independent features for a binary classification problem
    
    :param train: pandas dataframe containing the training data
    :param feature_list: list of feature names to perform ANOVA on
    :param target: name of the target variable (binary)
    :return: dictionary containing ANOVA results
    """
    anova_results = {}
    table = PrettyTable()
    
    table.field_names = ['Feature', 'F-statistic', 'p-value']
    
    for feature in feature_list:
        groups = []
        for group_value in train[target].unique():
            group = train[train[target] == group_value][feature].dropna()
            groups.append(group)
        
        f_stat, p_val = f_oneway(*groups)
        table.add_row([feature, f_stat, p_val])
        
    return print(table)

perform_anova(train, cont_cols, 'Transported')

# %% [code] cell 13 (notebook cell 38)
feature_pairs = list(combinations(cont_cols, 2))
table = PrettyTable()
table.field_names = ['Feature Pair', 'Accuracy']

# Fill missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
train_imputed = imputer.fit_transform(train[cont_cols])

for pair in feature_pairs:
    # Using the entire train data to fit, not a CV because it is time consuming
    x_temp = train_imputed[:, [cont_cols.index(pair[0]), cont_cols.index(pair[1])]]
    y_temp = train['Transported']
    model = SVC(gamma='auto')
    model.fit(x_temp, y_temp)
    y_pred = model.predict(x_temp)
    acc = accuracy_score(y_temp, y_pred)
    table.add_row([pair, acc])
table.sortby = 'Accuracy'
table.reversesort = True
print(table)

# %% [code] cell 14 (notebook cell 41)
cat_features=[f for f in train.columns if f not in cont_cols+["PassengerId","Name","Transported"] and train[f].nunique()<50]
cat_features

# %% [code] cell 15 (notebook cell 43)
import matplotlib.pyplot as plt

target = 'Transported'

# Create subplots for each categorical feature
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))

# Loop through each categorical feature and plot the contingency table in a subplot
for i, col in enumerate(cat_features):
    contingency_table = pd.crosstab(train[col], train[target], normalize='index')
    contingency_table.plot(kind='bar', stacked=True, ax=axs[i//2, i%2])
    axs[i//2, i%2].set_title(f"{col.title()} Distribution by Target")
    axs[i//2, i%2].set_xlabel(col.title())
    axs[i//2, i%2].set_ylabel("Proportion")
    
# Adjust spacing between subplots
fig.tight_layout()

# Show the plot
finalize_figures()

# %% [code] cell 16 (notebook cell 46)
features=[f for f in train.columns if train[f].dtype!='O' and f not in ['Transported']]
corr = train[features].corr()
plt.figure(figsize = (10, 10), dpi = 300)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask = mask, cmap = sns.diverging_palette(500, 10, as_cmap=True), annot = True, annot_kws = {'size' : 7})
plt.title('Train Feature Correlation Matrix\n', fontsize = 25, weight = 'bold')
finalize_figures()

# %% [code] cell 17 (notebook cell 51)
# Extract passenger Groups
train["group"]=train["PassengerId"].str[:-3].astype(int)
test["group"]=test["PassengerId"].str[:-3].astype(int)

# Extract Deck 
def deck(x):
    x=str(x)
    if x=='nan':
        return 'Missing_Deck'
    else:
        x=x.split('/')
        return x[0]
train['cabin_deck']=train["Cabin"].apply(deck)
test['cabin_deck']=test["Cabin"].apply(deck)

# Extract the cabin number
def num(x):
    x=str(x)
    if x=='nan':
        return None
    else:
        x=x.split('/')
        return int(x[1])
train['cabin_num']=train["Cabin"].apply(num)
test['cabin_num']=test["Cabin"].apply(num)

# Extract the Cabin Side
def side(x):
    x=str(x)
    if x=='nan':
        return "Missing_Side"
    else:
        x=x.split('/')
        return x[2]

train['cabin_side']=train["Cabin"].apply(side)
test['cabin_side']=test["Cabin"].apply(side)

train.drop(columns=["Cabin"],inplace=True)
test.drop(columns=["Cabin"],inplace=True)

# %% [code] cell 18 (notebook cell 53)
# Extract the last names 
def name(x):
    x=str(x)
    x=x.lower()
    x=x.strip()
    x=x.split(" ")
    if len(x)>1:
        return x[-1]
    else:
        return (x[0])
# We first fill the missing names and then extract the last names
train['Name']=train['Name'].fillna("No_Name")
test['Name']=test['Name'].fillna("No_Name")
train['Last_Name']=train['Name'].apply(name)
test['Last_Name']=test['Name'].apply(name)
print(train['Last_Name'].isna().sum(),test['Last_Name'].isna().sum())

# %% [code] cell 19 (notebook cell 54)
# Bool to Num
def bool_c(x):
    if x==True:
        return 1
    elif x==False:
        return 0
    else:
        return np.nan
train["VIP"]=train["VIP"].apply(bool_c)
test["VIP"]=test["VIP"].apply(bool_c)
train["CryoSleep"]=train["CryoSleep"].apply(bool_c)
test["CryoSleep"]=test["CryoSleep"].apply(bool_c)
train["Transported"]=train["Transported"].astype(int)

# %% [code] cell 20 (notebook cell 57)
# Categorical features

miss_cat=[feature for feature in train.columns if train[feature].isnull().sum()>0 and train[feature].dtype=='O']
miss_cat

# %% [code] cell 21 (notebook cell 58)
# Calculate the missing percentages for both train and test data
train_missing_pct = train[miss_cat].isnull().mean() * 100
test_missing_pct = test[miss_cat].isnull().mean() * 100

# Combine the missing percentages for train and test data into a single dataframe
missing_pct_df = pd.concat([train_missing_pct, test_missing_pct], axis=1, keys=['Train %', 'Test%'])

# Print the missing percentage dataframe
print(missing_pct_df)

# %% [code] cell 22 (notebook cell 61)
save_table_artifact("train_head_after_missing_summary", train.head())

# %% [code] cell 23 (notebook cell 62)
for feature in miss_cat:
    train[feature].fillna("missing_"+feature,inplace=True)
    test[feature].fillna("missing_"+feature,inplace=True)
    
# Calculate the missing percentages for both train and test data
train_missing_pct = train[miss_cat].isnull().mean() * 100
test_missing_pct = test[miss_cat].isnull().mean() * 100

# Combine the missing percentages for train and test data into a single dataframe
missing_pct_df = pd.concat([train_missing_pct, test_missing_pct], axis=1, keys=['Train %', 'Test%'])

# Print the missing percent
print(missing_pct_df)

# cat_params={
#             'depth': 7,
#             'learning_rate': 0.1,
#             'l2_leaf_reg': 0.7,
#             'random_strength': 0.2,
#             'max_bin': 200,
#             'od_wait': 65,
#             'one_hot_max_size': 70,
#             'grow_policy': 'Depthwise',
#             'bootstrap_type': 'Bayesian',
#             'od_type': 'Iter',
#             'eval_metric': 'MultiClass',
#             'loss_function': 'MultiClass',
# }
# def store_missing_rows(df, features):
#     missing_rows = {}
    
#     for feature in features:
#         missing_rows[feature] = df[df[feature].isnull()]
    
#     return missing_rows


# def fill_missing_categorical(train,test,target, features, max_iterations=10):
    
#     df=pd.concat([train.drop(columns=[target,"PassengerId"]),test.drop(columns=['PassengerId'])],axis="rows")
#     df=df.reset_index(drop=True)
    
#     # Step 1: Store the instances with missing values in each feature
#     missing_rows = store_missing_rows(df, features)
    
#     # Step 2: Initially fill all missing values with "Missing"
#     for f in features:
#         df[f]=df[f].fillna("Missing_"+f)
# #     df[features] = df[features].fillna("Missing")
    
#     for iteration in tqdm(range(max_iterations), desc="Iterations"):
#         for feature in features:
#             # Skip features with no missing values
#             rows_miss = missing_rows[feature].index
            
#             missing_temp = df.loc[rows_miss].copy()
#             non_missing_temp = df.drop(index=rows_miss).copy()
#             missing_temp = missing_temp.drop(columns=[feature])
            
#             cat_features = [x for x in df.columns if df[x].dtype=="O" and x != feature]
            
#             # Step 3: Use the remaining features to predict missing values using Random Forests
#             X_train = non_missing_temp.drop(columns=[feature])
#             y_train = non_missing_temp[[feature]]
            
#             catboost_classifier = CatBoostClassifier(**cat_params)
#             catboost_classifier.fit(X_train, y_train, cat_features=cat_features, verbose=False)
            
#             # Step 4: Predict missing values for the feature and update all N features
#             y_pred = catboost_classifier.predict(missing_temp)
#             df.loc[rows_miss, feature] = y_pred
#     train[features] = np.array(df.iloc[:train.shape[0]][features])
#     test[features] = np.array(df.iloc[train.shape[0]:][features])
    
#     return train,test


# train ,test= fill_missing_categorical(train,test,"Transported",miss_cat,10)

# %% [code] cell 24 (notebook cell 64)
miss_cont=[feature for feature in train.columns if train[feature].isnull().sum()>0 and train[feature].dtype!='O' and feature not in ['Transported']]
miss_cont

# %% [code] cell 25 (notebook cell 65)
# Calculate the missing percentages for both train and test data
train_missing_pct = train[miss_cont].isnull().mean() * 100
test_missing_pct = test[miss_cont].isnull().mean() * 100

# Combine the missing percentages for train and test data into a single dataframe
missing_pct_df = pd.concat([train_missing_pct, test_missing_pct], axis=1, keys=['Train %', 'Test%'])

# Print the missing percentage dataframe
print(missing_pct_df)

# %% [code] cell 26 (notebook cell 67)
# First lets fill CryoSleep, based on totdal expenditure
exp_features=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train["Expenditure"]=train[exp_features].sum(axis="columns")
test["Expenditure"]=test[exp_features].sum(axis="columns")

# Zero expenditure indicate that they are in CryoSleep
train['CryoSleep']=np.where(train['Expenditure']==0,1,0)
test['CryoSleep']=np.where(test['Expenditure']==0,1,0)

# Also, if they are VIPs, they probably would not choose to be in CryoSleep
train['VIP']=np.where(train['CryoSleep']==0,1,0)
test['VIP']=np.where(test['CryoSleep']==0,1,0)

train.drop(columns=["Expenditure"],inplace=True)
test.drop(columns=["Expenditure"],inplace=True)

# %% [code] cell 27 (notebook cell 68)
for col in exp_features:
    train[col]=np.where(train["CryoSleep"]==1,0,train[col])
    test[col]=np.where(test["CryoSleep"]==1,0,test[col])    
    
# Calculate the missing percentages for both train and test data
train_missing_pct = train[miss_cont].isnull().mean() * 100
test_missing_pct = test[miss_cont].isnull().mean() * 100

# Combine the missing percentages for train and test data into a single dataframe
missing_pct_df = pd.concat([train_missing_pct, test_missing_pct], axis=1, keys=['Train %', 'Test%'])

# Print the missing percentage dataframe
print(missing_pct_df)

# %% [code] cell 28 (notebook cell 70)
miss_cont=[feature for feature in train.columns if train[feature].isnull().sum()>0 and train[feature].dtype!='O' and feature not in ['Transported']]
miss_cont
imputer=KNNImputer(n_neighbors=5)
train[miss_cont]=imputer.fit_transform(train[miss_cont])
test[miss_cont]=imputer.transform(test[miss_cont])

# Calculate the missing percentages for both train and test data
train_missing_pct = train[miss_cont].isnull().mean() * 100
test_missing_pct = test[miss_cont].isnull().mean() * 100

# Combine the missing percentages for train and test data into a single dataframe
missing_pct_df = pd.concat([train_missing_pct, test_missing_pct], axis=1, keys=['Train %', 'Test%'])

# Print the missing percentage dataframe
print(missing_pct_df)

# %% [code] cell 29 (notebook cell 71)
# cb_params = {
#             'iterations': 500,
#             'depth': 6,
#             'learning_rate': 0.02,
#             'l2_leaf_reg': 0.5,
#             'random_strength': 0.2,
#             'max_bin': 150,
#             'od_wait': 80,
#             'one_hot_max_size': 70,
#             'grow_policy': 'Depthwise',
#             'bootstrap_type': 'Bayesian',
#             'od_type': 'IncToDec',
#             'eval_metric': 'RMSE',
#             'loss_function': 'RMSE',
#             'random_state': 42,
#         }
# def rmse(y1,y2):
#     return(np.sqrt(mean_squared_error(y1,y2)))

# def fill_missing_numerical(train,test,target, features, max_iterations=10):
    
#     df=pd.concat([train.drop(columns=[target,"PassengerId"]),test.drop(columns="PassengerId")],axis="rows")
#     df=df.reset_index(drop=True)
    
#     # Step 1: Store the instances with missing values in each feature
#     missing_rows = store_missing_rows(df, features)
    
#     # Step 2: Initially fill all missing values with "Missing"
#     for f in features:
#         df[f]=df[f].fillna(df[f].mean())
    
#     cat_features=[f for f in df.columns if df[f].dtype=="O"]
#     dictionary = {feature: [] for feature in features}
    
#     for iteration in tqdm(range(max_iterations), desc="Iterations"):
#         for feature in features:
#             # Skip features with no missing values
#             rows_miss = missing_rows[feature].index
            
#             missing_temp = df.loc[rows_miss].copy()
#             non_missing_temp = df.drop(index=rows_miss).copy()
#             y_pred_prev=missing_temp[feature]
#             missing_temp = missing_temp.drop(columns=[feature])
            
            
#             # Step 3: Use the remaining features to predict missing values using Random Forests
#             X_train = non_missing_temp.drop(columns=[feature])
#             y_train = non_missing_temp[[feature]]
            
#             catboost_classifier = CatBoostRegressor(**cb_params)
#             catboost_classifier.fit(X_train, y_train,cat_features=cat_features, verbose=False)
            
#             # Step 4: Predict missing values for the feature and update all N features
#             y_pred = catboost_classifier.predict(missing_temp)
#             df.loc[rows_miss, feature] = y_pred
#             error_minimize=rmse(y_pred,y_pred_prev)
#             dictionary[feature].append(error_minimize)  # Append the error_minimize value

#     for feature, values in dictionary.items():
#         iterations = range(1, len(values) + 1)  # x-axis values (iterations)
#         plt.plot(iterations, values, label=feature)  # plot the values
#         plt.xlabel('Iterations')
#         plt.ylabel('RMSE')
#         plt.title('Minimization of RMSE with iterations')
#         plt.legend()
#         finalize_figures()
#     train[features] = np.array(df.iloc[:train.shape[0]][features])
#     test[features] = np.array(df.iloc[train.shape[0]:][features])

#     return train,test


# train,test = fill_missing_numerical(train,test,"Transported",miss_cont,20)

# %% [code] cell 30 (notebook cell 74)
train["expenditure"]=train["VRDeck"]+train["Spa"]+train["RoomService"]
test["expenditure"]=test["VRDeck"]+test["Spa"]+test["RoomService"]

# train["total_exp"]=train["VRDeck"]+train["Spa"]+train["RoomService"]+train['FoodCourt']+train['ShoppingMall']
# test["total_exp"]=test["VRDeck"]+test["Spa"]+test["RoomService"]+train['FoodCourt']+train['ShoppingMall']

# %% [code] cell 31 (notebook cell 75)
num_feat=[f for f in train.columns if train[f].dtype!="O" and train[f].nunique()>10] # The rest are discrete/categorical
num_feat

# %% [code] cell 32 (notebook cell 79)
# Below are the functions to decide the decision boundaries in order to maximize Accuracy/ f1-score
def f1_cutoff(precisions, recalls, thresholds):
    a=precisions*recalls/(recalls+precisions)
    b=sorted(zip(a,thresholds))
    return b[-1][1]
def acc_cutoff(y_valid, y_pred_valid):
    y_valid=np.array(y_valid)
    y_pred_valid=np.array(y_pred_valid)
    fpr, tpr, threshold = metrics.roc_curve(y_valid, y_pred_valid)
    pred_valid = pd.DataFrame({'label': y_pred_valid})
    thresholds = np.array(threshold)
    pred_labels = (pred_valid['label'].values > thresholds[:, None]).astype(int)
    acc_scores = (pred_labels == y_valid).mean(axis=1)
    acc_df = pd.DataFrame({'threshold': threshold, 'test_acc': acc_scores})
    acc_df.sort_values(by='test_acc', ascending=False, inplace=True)
    cutoff = acc_df.iloc[0, 0]
    return cutoff
    
sc=MinMaxScaler()
unimportant_features=[]
table = PrettyTable()
dt_params= {'min_samples_split': 80, 'min_samples_leaf': 30, 'max_depth': 8, 'criterion': 'absolute_error'}

table.field_names = ['Original Feature', 'Original Accuracy(CV-TRAIN)', 'Transformed Feature', 'Tranformed Accuracy(CV-TRAIN)']
for col in num_feat:
    
    # Log Transformation after MinMax Scaling(keeps data between 0 and 1)
    train["log_"+col]=np.log1p(sc.fit_transform(train[[col]]))
    test["log_"+col]=np.log1p(sc.transform(test[[col]]))
    
    # Square Root Transformation
    train["sqrt_"+col]=np.sqrt(sc.fit_transform(train[[col]]))
    test["sqrt_"+col]=np.sqrt(sc.transform(test[[col]]))
    
    # Box-Cox transformation
    transformer = PowerTransformer(method='box-cox')
    train["bx_cx_"+col] = transformer.fit_transform(sc.fit_transform(train[[col]])+1) # adjusted to make it +ve
    test["bx_cx_"+col] = transformer.transform(sc.transform(test[[col]])+1)
    
    # Yeo-Johnson transformation
    transformer = PowerTransformer(method='yeo-johnson')
    train["y_J_"+col] = transformer.fit_transform(train[[col]])
    test["y_J_"+col] = transformer.transform(test[[col]])
    
    # Power transformation, 0.25
    power_transform = lambda x: np.power(x, 0.25) 
    transformer = FunctionTransformer(power_transform)
    train["pow_"+col] = transformer.fit_transform(sc.fit_transform(train[[col]]))
    test["pow_"+col] = transformer.transform(sc.transform(test[[col]]))
    
    # Power transformation, 0.1
    power_transform = lambda x: np.power(x, 0.1) 
    transformer = FunctionTransformer(power_transform)
    train["pow2_"+col] = transformer.fit_transform(sc.fit_transform(train[[col]]))
    test["pow2_"+col] = transformer.transform(sc.transform(test[[col]]))
    
    # log to power transformation
    train["log_pow2"+col]=np.log1p(train["pow2_"+col])
    test["log_pow2"+col]=np.log1p(test["pow2_"+col])
    
    temp_cols=[col,"log_"+col,"sqrt_"+col, "bx_cx_"+col,"y_J_"+col ,"pow_"+col,"pow2_"+col,"log_pow2"+col ]
    
    # Fill na becaue, it would be Nan if the vaues are negative and a transformation applied on it
    train[temp_cols]=train[temp_cols].fillna(0)
    test[temp_cols]=test[temp_cols].fillna(0)
    
    #Apply PCA on  the features and compute an additional column
    pca=TruncatedSVD(n_components=1)
    x_pca_train=pca.fit_transform(train[temp_cols])
    x_pca_test=pca.transform(test[temp_cols])
    x_pca_train=pd.DataFrame(x_pca_train, columns=[col+"_pca_comb"])
    x_pca_test=pd.DataFrame(x_pca_test, columns=[col+"_pca_comb"])
    temp_cols.append(col+"_pca_comb")
    #print(temp_cols)
    
    train=pd.concat([train,x_pca_train],axis='columns')
    test=pd.concat([test,x_pca_test],axis='columns')
    
    # See which transformation along with the original is giving you the best univariate fit with target
    kf=KFold(n_splits=10, shuffle=True, random_state=42)
    
    ACC=[]
    
    for f in temp_cols:
        X=train[[f]].values
        y=train["Transported"].values
        
        acc=[]
        for train_idx, val_idx in kf.split(X,y):
            X_train,y_train=X[train_idx],y[train_idx]
            x_val,y_val=X[val_idx],y[val_idx]
            
            model=LogisticRegression()
#             model=DecisionTreeRegressor(**dt_params)
            model.fit(X_train,y_train)
            y_pred=model.predict_proba(x_val)[:,1]
            precisions,recalls, thresholds=precision_recall_curve(y_val,y_pred)
#             cutoff=f1_cutoff(precisions,recalls, thresholds)
            cutoff=acc_cutoff(y_val,y_pred)
#             print(cutoff)
            predicted =pd.DataFrame()
            predicted["Transported"] = y_pred
            y_pred=np.where(predicted["Transported"]>float(cutoff),1,0)
            acc.append(accuracy_score(y_val,y_pred))
        ACC.append((f,np.mean(acc)))
        if f==col:
            orig_acc=np.mean(acc)
    best_col, best_acc=sorted(ACC, key=lambda x:x[1], reverse=True)[0]
    
    cols_to_drop = [f for f in temp_cols if  f!= best_col]
#     print(cols_to_drop)
    final_selection=[f for f in temp_cols if f not in cols_to_drop]
    if cols_to_drop:
        unimportant_features=unimportant_features+cols_to_drop
    table.add_row([col,orig_acc,best_col ,best_acc])
print(table)    

# %% [code] cell 33 (notebook cell 82)
cat_features=[*set([feature for feature in train.columns if train[feature].nunique()<=10 or train[feature].dtype=='O'])-set(["PassengerId","Transported","VIP","CryoSleep"])]
train[cat_features].nunique()

# %% [code] cell 34 (notebook cell 85)
def tf_idf(train, test, column,n,p):
    vectorizer=TfidfVectorizer(max_features=n)
    vectors_train=vectorizer.fit_transform(train[column])
    vectors_test=vectorizer.transform(test[column])
    
    svd=TruncatedSVD(p)
    x_pca_train=svd.fit_transform(vectors_train)
    x_pca_test=svd.transform(vectors_test)
    tfidf_df_train=pd.DataFrame(x_pca_train)
    tfidf_df_test=pd.DataFrame(x_pca_test)

    
    cols=[(column+"_tfidf_"+str(f)) for f in tfidf_df_train.columns]
    tfidf_df_train.columns=cols
    tfidf_df_test.columns=cols
    train=pd.concat([train,tfidf_df_train], axis="columns")
    test=pd.concat([test,tfidf_df_test], axis="columns")
    
    return (train, test)

(train,test)=tf_idf(train,test,"Last_Name",1000,5)
train.drop(columns=["Name","Last_Name"], inplace=True)
test.drop(columns=["Name","Last_Name"], inplace=True)

# %% [code] cell 35 (notebook cell 87)
cat_features=['HomePlanet', 'cabin_deck', 'Destination', 'cabin_side']
table = PrettyTable()
table.field_names = ['Feature', 'Encoded Feature', "Accuracy (CV)- Logistic regression"]

def OHE(train,test,cols,target):
    combined = pd.concat([train, test], axis=0)
    for col in cols:
        one_hot = pd.get_dummies(combined[col])
        counts = combined[col].value_counts()
        min_count_category = counts.idxmin()
        one_hot = one_hot.drop(min_count_category, axis=1)
        combined = pd.concat([combined, one_hot], axis="columns")
        combined = combined.drop(col, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # split back to train and test dataframes
    train_ohe = combined[:len(train)]
    test_ohe = combined[len(train):]
    test_ohe.reset_index(inplace=True,drop=True)
    test_ohe.drop(columns=[target],inplace=True)
    
    return train_ohe, test_ohe

for feature in cat_features:
    ## Target Guided Mean --Data Leakage Possible
    
    cat_labels=train.groupby([feature])['Transported'].mean().sort_values().index
    cat_labels2={k:i for i,k in enumerate(cat_labels,0)}
    train[feature+"_target"]=train[feature].map(cat_labels2)
    test[feature+"_target"]=test[feature].map(cat_labels2)
    
    ## Count Encoding
    
    dic=train[feature].value_counts().to_dict()
    train[feature+"_count"]=np.log1p(train[feature].map(dic))
    test[feature+"_count"]=np.log1p(test[feature].map(dic))

    
    ## Count Labeling
    
    dic2=train[feature].value_counts().to_dict()
    list1=np.arange(len(dic2.values()),0,-1) # Higher rank for high count
    # list1=np.arange(len(dic2.values())) # Higher rank for low count
    dic3=dict(zip(list(dic2.keys()),list1))
    train[feature+"_count_label"]=train[feature].replace(dic3)
    test[feature+"_count_label"]=test[feature].replace(dic3)

    
    ## WOE Binning
    cat_labels=np.log1p(train.groupby([feature])['Transported'].sum()/(train.groupby([feature])['Transported'].count()-train.groupby([feature])['Transported'].sum()))#.sort_values().index
    cat_labels2=cat_labels.to_dict()
    train[feature+"_WOE"]=train[feature].map(cat_labels2)
    test[feature+"_WOE"]=test[feature].map(cat_labels2)
    
    
    temp_cols=[feature+"_target", feature+"_count", feature+"_count_label",feature+"_WOE"]
    
    
    # It is possible to have NaN values in the test data when new categories are seen
    imputer=KNNImputer(n_neighbors=5)
    train[temp_cols]=imputer.fit_transform(train[temp_cols])
    test[temp_cols]=imputer.transform(test[temp_cols])
    
    
    if train[feature].dtype!="O":
        temp_cols.append(feature)
    else:
        train.drop(columns=[feature],inplace=True)
        test.drop(columns=[feature],inplace=True)
    # Also, doing a group clustering on all encoding types and an additional one-hot on the clusters
    
    temp_train=train[temp_cols]
    temp_test=test[temp_cols]
    
    sc=StandardScaler()
    temp_train=sc.fit_transform(temp_train)
    temp_test=sc.transform(temp_test)
    model = KMeans()


    # Initialize the KElbowVisualizer with the KMeans model and desired range of clusters
    visualizer = KElbowVisualizer(model, k=(3, 15), metric='calinski_harabasz', timings=False)

    # Fit the visualizer to the data
    visualizer.fit(np.array(temp_train))

    ideal_clusters = visualizer.elbow_value_
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Calinski-Harabasz Index')
    plt.title("Clustering on encoded featured from "+feature)
    finalize_figures()
    print(ideal_clusters)
    if ideal_clusters is not None:
        
        kmeans = KMeans(n_clusters=ideal_clusters)
        kmeans.fit(np.array(temp_train))
        labels_train = kmeans.labels_

        train[feature+'_cat_cluster_WOE'] = labels_train
        test[feature+'_cat_cluster_WOE'] = kmeans.predict(np.array(temp_test))

        train[feature+'_cat_OHE_cluster']=feature+"_OHE_"+train[feature+'_cat_cluster_WOE'].astype(str)
        test[feature+'_cat_OHE_cluster']=feature+"_OHE_"+test[feature+'_cat_cluster_WOE'].astype(str)

        train, test=OHE(train,test, [feature+'_cat_OHE_cluster'],"Transported")

        cat_labels=cat_labels=np.log1p(train.groupby([feature+'_cat_cluster_WOE'])['Transported'].mean())
        cat_labels2=cat_labels.to_dict()
        train[feature+'_cat_cluster_WOE']=train[feature+'_cat_cluster_WOE'].map(cat_labels2)
        test[feature+'_cat_cluster_WOE']=test[feature+'_cat_cluster_WOE'].map(cat_labels2)
        
        temp_cols=temp_cols+[feature+'_cat_cluster_WOE']
    else:
        print("No good clusters were found, skipped without clustering and OHE")
        

    
    
    
    # See which transformation along with the original is giving you the best univariate fit with target
    skf=StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    accuaries=[]
    
    for f in temp_cols:
        X=train[[f]].values
        y=train["Transported"].values
        
        acc=[]
        for train_idx, val_idx in skf.split(X,y):
            X_train,y_train=X[train_idx],y[train_idx]
            x_val,y_val=X[val_idx],y[val_idx]
            
            model=LogisticRegression()
            model.fit(X_train,y_train)
            y_pred=model.predict_proba(x_val)[:,1]
            precisions,recalls, thresholds=precision_recall_curve(y_val,y_pred)
#             cutoff=f1_cutoff(precisions,recalls, thresholds)
            cutoff=acc_cutoff(y_val,y_pred)
#             print(cutoff)
            predicted =pd.DataFrame()
            predicted["Transported"] = y_pred
            y_pred=np.where(predicted["Transported"]>float(cutoff),1,0)
            acc.append(accuracy_score(y_val,y_pred))
        accuaries.append((f,np.mean(acc)))
    best_col, best_acc=sorted(accuaries, key=lambda x:x[1], reverse=True)[0]
    
    # check correlation between best_col and other columns and drop if correlation >0.75
    corr = train[temp_cols].corr(method='pearson')
    corr_with_best_col = corr[best_col]
    cols_to_drop = [f for f in temp_cols if corr_with_best_col[f] > 0.75 and f != best_col]
    final_selection=[f for f in temp_cols if f not in cols_to_drop]
    if cols_to_drop:
        train = train.drop(columns=cols_to_drop)
        test = test.drop(columns=cols_to_drop)
    table.add_row([feature,best_col ,best_acc])
print(table)

# %% [code] cell 36 (notebook cell 91)
table = PrettyTable()
table.field_names = ['Cluster WOE Feature', 'MAE(CV-TRAIN)']
for col in num_feat:
    sub_set=[f for f in unimportant_features if col in f]
    print(sub_set)
    temp_train=train[sub_set]
    temp_test=test[sub_set]
    sc=StandardScaler()
    temp_train=sc.fit_transform(temp_train)
    temp_test=sc.transform(temp_test)
    model = KMeans()


    # Initialize the KElbowVisualizer with the KMeans model and desired range of clusters
    visualizer = KElbowVisualizer(model, k=(3, 25), metric='calinski_harabasz', timings=False)

    # Fit the visualizer to the data
    visualizer.fit(np.array(temp_train))
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Calinski-Harabasz Index')
    finalize_figures()

    ideal_clusters = visualizer.elbow_value_
    if ideal_clusters is None:
        ideal_clusters=25

    # print(ideal_clusters)
    kmeans = KMeans(n_clusters=ideal_clusters)
    kmeans.fit(np.array(temp_train))
    labels_train = kmeans.labels_

    train[col+'_OHE_cluster'] = labels_train
    test[col+'_OHE_cluster'] = kmeans.predict(np.array(temp_test))
    # Also, making a copy to do mean encoding followed by a log transformation
    
    train[col+"_unimp_cluster_WOE"]=train[col+'_OHE_cluster']
    test[col+"_unimp_cluster_WOE"]=test[col+'_OHE_cluster'] 
    cat_labels=cat_labels=np.log1p(train.groupby([col+"_unimp_cluster_WOE"])['Transported'].mean())
    cat_labels2=cat_labels.to_dict()
    train[col+"_unimp_cluster_WOE"]=train[col+"_unimp_cluster_WOE"].map(cat_labels2)
    test[col+"_unimp_cluster_WOE"]=test[col+"_unimp_cluster_WOE"].map(cat_labels2)

    X=train[[col+"_unimp_cluster_WOE"]].values
    y=train["Transported"].values

    ACC=[]
    for train_idx, val_idx in kf.split(X,y):
        X_train,y_train=X[train_idx],y[train_idx]
        x_val,y_val=X[val_idx],y[val_idx]

        model=LogisticRegression()
        model.fit(X_train,y_train)
        y_pred=model.predict_proba(x_val)[:,1]
        precisions,recalls, thresholds=precision_recall_curve(y_val,y_pred)
#             cutoff=f1_cutoff(precisions,recalls, thresholds)
        cutoff=acc_cutoff(y_val,y_pred)
#             print(cutoff)
        predicted =pd.DataFrame()
        predicted["Transported"] = y_pred
        y_pred=np.where(predicted["Transported"]>float(cutoff),1,0)
        ACC.append(accuracy_score(y_val,y_pred))
    table.add_row([col+"_unimp_cluster_WOE",np.mean(ACC)])
    
    train[col+'_OHE_cluster']=col+"_OHE_"+train[col+'_OHE_cluster'].astype(str)
    test[col+'_OHE_cluster']=col+"_OHE_"+test[col+'_OHE_cluster'].astype(str)
    train, test=OHE(train,test,[col+'_OHE_cluster'],"Transported")
print(table)

# %% [code] cell 37 (notebook cell 94)
# from itertools import combinations
# # num_features=[f for f in train.columns if train[f].nunique()>100 and f not in ['Transported',"PassengerId"]]
# feature_pairs = list(combinations(num_feat, 2))

# table = PrettyTable()
# table.field_names = ['Pair Features', 'Accuracy(CV-TRAIN)', "Selected"]


# selected_features=[]
# max_product=float('-inf')
# for pair in feature_pairs:
#     col1, col2 = pair
# #     print(pair)
#     product_col_train = train[col1] * train[col2]
#     product_col_test= test[col1] * test[col2]
#     name=f'{col1}_{col2}_product'
#     train[name] = product_col_train
#     test[name] = product_col_test
#     max_product = max(max_product, product_col_train.max())

#     kf=KFold(n_splits=5, shuffle=True, random_state=42)
#     MAE=[]
#     X=train[[name]].values
#     y=train["Transported"].values

#     ACC=[]
#     for train_idx, val_idx in kf.split(X,y):
#         X_train,y_train=X[train_idx],y[train_idx]
#         x_val,y_val=X[val_idx],y[val_idx]

#         model=LogisticRegression()
#         model.fit(X_train,y_train)
#         y_pred=model.predict_proba(x_val)[:,1]
#         precisions,recalls, thresholds=precision_recall_curve(y_val,y_pred)
# #             cutoff=f1_cutoff(precisions,recalls, thresholds)
#         cutoff=acc_cutoff(y_val,y_pred)
# #             print(cutoff)
#         predicted =pd.DataFrame()
#         predicted["Transported"] = y_pred
#         y_pred=np.where(predicted["Transported"]>float(cutoff),1,0)
#         ACC.append(accuracy_score(y_val,y_pred))
#     if np.mean(ACC)<0.7:
#         unimportant_features.append(name)
#         selected="No"
#     else:
#         selected_features.append(pair)
#         selected="Yes"
#     table.add_row([pair,np.mean(ACC),selected ])
# table.sortby = 'Accuracy(CV-TRAIN)'
# table.reversesort = True
# print(table)

# %% [code] cell 38 (notebook cell 97)
print("Number of Unimportant Features are ",len(unimportant_features))

# %% [code] cell 39 (notebook cell 98)
test.reset_index(inplace=True,drop=True)
for col in cont_cols:
    sub_set=[f for f in unimportant_features if col in f]
    
    existing=[f for f in train.columns if f in sub_set]
    temp_train=train[existing]
    temp_test=test[existing]
    sc=StandardScaler()
    temp_train=sc.fit_transform(temp_train)
    temp_test=sc.transform(temp_test)
    
    pca=TruncatedSVD(n_components=1)
    x_pca_train=pca.fit_transform(temp_train)
    x_pca_test=pca.transform(temp_test)
    x_pca_train=pd.DataFrame(x_pca_train, columns=[col+"_pca_comb_unimp"])
    x_pca_test=pd.DataFrame(x_pca_test, columns=[col+"_pca_comb_unimp"])
    
    train=pd.concat([train,x_pca_train],axis='columns')
    test=pd.concat([test,x_pca_test],axis='columns')
    for f in sub_set:
        if f in train.columns and f not in cont_cols:
            train=train.drop(columns=[f])
            test=test.drop(columns=[f])

# %% [code] cell 40 (notebook cell 101)
num_derived_list=[]
for f1 in train.columns:
    for f2 in num_feat:
        if f2 in f1:
            num_derived_list.append(f1)
num_derived_list=[*set(num_derived_list)]      
                       
corr = train[num_derived_list].corr()
plt.figure(figsize = (40, 40), dpi = 300)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask = mask, cmap = sns.diverging_palette(500, 10, as_cmap=True), annot = True, annot_kws = {'size' : 8})
plt.title('Post-Feature Engineering Correlation Matrix\n', fontsize = 10, weight = 'bold')
finalize_figures()

# %% [code] cell 41 (notebook cell 103)
final_drop_list=[]

table = PrettyTable()
table.field_names = ['Original', 'Final Transformed feature', "Accuray(CV)- Logistic Regression"]

threshold=0.8
# It is possible that multiple parent features share same child features, so storing selected features to avoid selecting the same feature again
best_cols=[]

for col in num_feat:
    sub_set=[f for f in num_derived_list if col in f]
    # print(sub_set)
    
    correlated_features = []

    # Loop through each feature
    for i, feature in enumerate(sub_set):
        # Check correlation with all remaining features
        for j in range(i+1, len(sub_set)):
            correlation = np.abs(train[feature].corr(train[sub_set[j]]))
            # If correlation is greater than threshold, add to list of highly correlated features
            if correlation > threshold:
                correlated_features.append(sub_set[j])

    # Remove duplicate features from the list
    correlated_features = list(set(correlated_features))
    if len(correlated_features)>1:

        temp_train=train[correlated_features]
        temp_test=test[correlated_features]
        #Scale before applying PCA
        sc=StandardScaler()
        temp_train=sc.fit_transform(temp_train)
        temp_test=sc.transform(temp_test)

        # Initiate PCA
        pca=TruncatedSVD(n_components=1)
        x_pca_train=pca.fit_transform(temp_train)
        x_pca_test=pca.transform(temp_test)
        x_pca_train=pd.DataFrame(x_pca_train, columns=[col+"_pca_comb_final"])
        x_pca_test=pd.DataFrame(x_pca_test, columns=[col+"_pca_comb_final"])
        train=pd.concat([train,x_pca_train],axis='columns')
        test=pd.concat([test,x_pca_test],axis='columns')

        # Clustering
        model = KMeans()


        # Initialize the KElbowVisualizer with the KMeans model and desired range of clusters
        visualizer = KElbowVisualizer(model, k=(10, 25), metric='calinski_harabasz', timings=False)

        # Fit the visualizer to the data
        visualizer.fit(np.array(temp_train))
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Calinski-Harabasz Index')
        plt.title("Clustering on features from "+col)
        finalize_figures()

        ideal_clusters = visualizer.elbow_value_
        
        if ideal_clusters is None:
            ideal_clusters=10

        # print(ideal_clusters)
        kmeans = KMeans(n_clusters=ideal_clusters)
        kmeans.fit(np.array(temp_train))
        labels_train = kmeans.labels_

        train[col+'_final_cluster'] = labels_train
        test[col+'_final_cluster'] = kmeans.predict(np.array(temp_test))

        cat_labels=cat_labels=np.log1p(train.groupby([col+"_final_cluster"])['Transported'].mean())
        cat_labels2=cat_labels.to_dict()
        train[col+"_final_cluster"]=train[col+"_final_cluster"].map(cat_labels2)
        test[col+"_final_cluster"]=test[col+"_final_cluster"].map(cat_labels2)

        correlated_features=correlated_features+[col+"_pca_comb_final",col+"_final_cluster"]
        # See which transformation along with the original is giving you the best univariate fit with target
        kf=KFold(n_splits=5, shuffle=True, random_state=42)

        ACC=[]

        for f in correlated_features:
            X=train[[f]].values
            y=train["Transported"].values

            acc=[]
            for train_idx, val_idx in kf.split(X,y):
                X_train,y_train=X[train_idx],y[train_idx]
                x_val,y_val=X[val_idx],y[val_idx]

                model=LogisticRegression()
                model.fit(X_train,y_train)
                y_pred=model.predict_proba(x_val)[:,1]
                precisions,recalls, thresholds=precision_recall_curve(y_val,y_pred)
                cutoff=acc_cutoff(y_val,y_pred)
                predicted =pd.DataFrame()
                predicted["Transported"] = y_pred
                y_pred=np.where(predicted["Transported"]>float(cutoff),1,0)
                acc.append(accuracy_score(y_val,y_pred))

            if f not in best_cols:
                ACC.append((f,np.mean(acc)))
        best_col, best_acc=sorted(ACC, key=lambda x:x[1], reverse=True)[0]
        best_cols.append(best_col)

        cols_to_drop = [f for f in correlated_features if  f not in  best_cols]
        if cols_to_drop:
            final_drop_list=final_drop_list+cols_to_drop
        table.add_row([col,best_col ,best_acc])
    else:
        print(f"All features for {col} have correlation less than threshold")
        table.add_row([col,"All features selected" ,"--"])
print(table)      

# %% [code] cell 42 (notebook cell 104)
final_drop_list=[f for f in final_drop_list if f not in cont_cols]
train.drop(columns=[*set(final_drop_list)],inplace=True)
test.drop(columns=[*set(final_drop_list)],inplace=True)

# %% [code] cell 43 (notebook cell 106)
feature_scale=[feature for feature in train.columns if feature not in ['PassengerId','Transported']]
scaler=StandardScaler()

train[feature_scale]=scaler.fit_transform(train[feature_scale])
test[feature_scale]=scaler.transform(test[feature_scale])

# %% [code] cell 44 (notebook cell 107)
ID=test[['PassengerId']]
train.drop(columns=['PassengerId'],inplace=True)
test.drop(columns=['PassengerId'],inplace=True)

# %% [code] cell 45 (notebook cell 108)
X_train=train.drop(['Transported'],axis=1)
y_train=train['Transported']

X_test=test.copy()
print(X_train.shape,X_test.shape)

# %% [code] cell 46 (notebook cell 113)
# !pip install tensorflow
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout

# %% [code] cell 47 (notebook cell 114)
sgd=tensorflow.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=True)
rms = tensorflow.keras.optimizers.RMSprop()
nadam=tensorflow.keras.optimizers.Nadam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam"
)
lrelu = lambda x: tensorflow.keras.activations.relu(x, alpha=0.1)

# %% [code] cell 48 (notebook cell 115)
ann = Sequential()
ann.add(Dense(64, input_dim=X_train.shape[1], kernel_initializer='he_uniform', activation=lrelu))
ann.add(Dropout(0.1))
ann.add(Dense(16,  kernel_initializer='he_uniform', activation=lrelu))
ann.add(Dropout(0.1))
# model.add(Dense(32,  kernel_initializer='he_uniform', activation='relu'))
# model.add(Dropout(0.1))

ann.add(Dense(1,  kernel_initializer='he_uniform', activation='sigmoid'))
ann.compile(loss="binary_crossentropy", optimizer=nadam,metrics=['accuracy'])

# %% [code] cell 49-70 (notebook cells 117-152)
# Tuned hyperparameters are maintained in `notebook_hyperparameter_tuning.py`.
xgb_params = get_trained_params("xgb")
lgb_params = get_trained_params("lgb")
cat_params = get_trained_params("cat")
lg_params = get_trained_params("logreg")
rf_params = get_trained_params("rf")
gbm_params = get_trained_params("gbm")
dtc_params = get_trained_params("dtc")

# %% [code] cell 71 (notebook cell 155)
class Splitter:
    def __init__(self, kfold=True, n_splits=5):
        self.n_splits = n_splits
        self.kfold = kfold

    def split_data(self, X, y, random_state_list):
        if self.kfold:
            for random_state in random_state_list:
                kf = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        else:
            split_idx = int(X.shape[0] / 10)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            yield X_train, X_val, y_train, y_val

class Classifier:
    def __init__(self, n_estimators=100, device="cpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.len_models = len(self.models)
        
    def _define_model(self):
        xgb_params.update({
            'n_estimators': self.n_estimators,
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'random_state': self.random_state,
        })
        if self.device == 'gpu':
            xgb_params.update({
            'tree_method' :'gpu_hist',
            'predictor': 'gpu_predictor',
          })

        lgb_params.update({
            'n_estimators': self.n_estimators,
            'objective': 'binary',
            'random_state': self.random_state,
        })

        cat_params.update({
            'n_estimators': self.n_estimators,
            'task_type': self.device.upper(),
            'random_state': self.random_state,
        })
        
        cat_sym_params = cat_params.copy()
        cat_sym_params['grow_policy'] = 'SymmetricTree'
        cat_dep_params = cat_params.copy()
        cat_dep_params['grow_policy'] = 'Depthwise'
        dt_params= {'min_samples_split': 80, 'min_samples_leaf': 30, 'max_depth': 8, 'criterion': 'gini'}
#         rf_params.update({
#             'n_estimators': self.n_estimators,
#         })
        models = {
            'xgb': xgb.XGBClassifier(**xgb_params),
            'lgb': lgb.LGBMClassifier(**lgb_params),
            'cat': CatBoostClassifier(**cat_params),
#             "cat_sym": CatBoostClassifier(**cat_sym_params),
#             "cat_dep": CatBoostClassifier(**cat_dep_params),
#             'lr': LogisticRegression(),
#             'rf': RandomForestClassifier(max_depth= 9,max_features= 'auto',min_samples_split= 10,
#                                                            min_samples_leaf= 4,  n_estimators=500,random_state=self.random_state),
#             'hgb': HistGradientBoostingClassifier(max_iter=self.n_estimators,learning_rate=0.01, loss="binary_crossentropy", 
#                                                   n_iter_no_change=300,random_state=self.random_state),
#             'gbdt': GradientBoostingClassifier(**gbm_params,random_state=self.random_state),
#             'svc': SVC(gamma="auto", probability=True),
#             'knn': KNeighborsClassifier(n_neighbors=5),
#             'mlp': MLPClassifier(random_state=self.random_state, max_iter=1000),
#             'gpc': GaussianProcessClassifier(**gpc_params, random_state=self.random_state),
#             'etr':ExtraTreesClassifier(min_samples_split=55, min_samples_leaf= 15, max_depth=10,
#                                        n_estimators=200,random_state=self.random_state),
#             'dt' :DecisionTreeClassifier(**dt_params,random_state=self.random_state),
#             'ada': AdaBoostClassifier(random_state=self.random_state),
#             'GNB': GaussianNB(**nb_params),
#             'ann':ann,
        }
        
        return models

# %% [code] cell 72 (notebook cell 157)
class OptunaWeights:
    def __init__(self, random_state):
        self.study = None
        self.weights = None
        self.random_state = random_state

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 0, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        # Calculate the Recall score for the weighted prediction
        precisions,recalls, thresholds=precision_recall_curve(y_true,weighted_pred)
#         cutoff=f1_cutoff(precisions,recalls, thresholds)
        cutoff=acc_cutoff(y_true,weighted_pred)
        y_weight_pred=np.where(weighted_pred>float(cutoff),1,0)        
        score = metrics.accuracy_score(y_true, y_weight_pred)
        return score

    def fit(self, y_true, y_preds, n_trials=2000):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        self.study = optuna.create_study(sampler=sampler, study_name="OptunaWeights", direction='maximize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds, n_trials=2000):
        self.fit(y_true, y_preds, n_trials=n_trials)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights
    
def acc_cutoff_class(y_valid, y_pred_valid):
    y_valid=np.array(y_valid)
    y_pred_valid=np.array(y_pred_valid)
    fpr, tpr, threshold = metrics.roc_curve(y_valid, y_pred_valid)
    pred_valid = pd.DataFrame({'label': y_pred_valid})
    thresholds = np.array(threshold)
    pred_labels = (pred_valid['label'].values > thresholds[:, None]).astype(int)
    acc_scores = (pred_labels == y_valid).mean(axis=1)
    acc_df = pd.DataFrame({'threshold': threshold, 'test_acc': acc_scores})
    acc_df.sort_values(by='test_acc', ascending=False, inplace=True)
    cutoff = acc_df.iloc[0, 0]
    y_pred_valid=np.where(y_pred_valid<float(cutoff),0,1)
    return y_pred_valid

# %% [code] cell 73-76 (notebook cells 159-163)
def fit_and_predict_model(
    name,
    model,
    X_train_fold,
    y_train_fold,
    X_val_fold,
    y_val_fold,
    X_test_data,
    early_stopping_rounds,
    verbose,
):
    if name == 'xgb':
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=verbose,
        )
    elif name == 'lgb':
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=verbose)],
        )
    elif name == 'cat':
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
    elif name == 'ann':
        model.fit(
            X_train_fold,
            y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            batch_size=5,
            epochs=50,
            verbose=verbose,
        )
    else:
        model.fit(X_train_fold, y_train_fold)

    if name == 'ann':
        test_pred = np.array(model.predict(X_test_data))[:, 0]
        y_val_pred = np.array(model.predict(X_val_fold))[:, 0]
    else:
        test_pred = model.predict_proba(X_test_data)[:, 1]
        y_val_pred = model.predict_proba(X_val_fold)[:, 1]

    return model, y_val_pred, test_pred


def train_single_fold(
    fold_index,
    split_data,
    splitter,
    X_test_data,
    n_splits,
    random_state_list,
    n_estimators,
    device,
    random_state,
    early_stopping_rounds,
    verbose,
    trained_models,
):
    X_train_fold, X_val_fold, y_train_fold, y_val_fold = split_data
    n = fold_index % n_splits
    m = fold_index // n_splits

    print(f'Entering fold {fold_index}')

    classifier = Classifier(n_estimators, device, random_state)
    models = classifier.models
    oof_preds = []
    test_preds = []
    model_scores = {}

    for name, model in models.items():
        print(f'training model: {name}')
        model, y_val_pred, test_pred = fit_and_predict_model(
            name,
            model,
            X_train_fold,
            y_train_fold,
            X_val_fold,
            y_val_fold,
            X_test_data,
            early_stopping_rounds,
            verbose,
        )

        score = accuracy_score(y_val_fold, acc_cutoff_class(y_val_fold, y_val_pred))
        model_scores[name] = score
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] Accuracy score: {score:.5f}')

        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)

        if name in trained_models:
            trained_models[name].append(deepcopy(model))

    optweights = OptunaWeights(random_state=random_state)
    ensemble_val_pred = optweights.fit_predict(y_val_fold.values, oof_preds)
    ensemble_score = accuracy_score(y_val_fold, acc_cutoff_class(y_val_fold, ensemble_val_pred))
    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] Accuracy score {ensemble_score:.5f}')

    averaged_test_pred = optweights.predict(test_preds) / (splitter.n_splits * len(random_state_list))
    return {
        'models': models,
        'y_val': y_val_fold,
        'y_val_pred': ensemble_val_pred,
        'ensemble_score': ensemble_score,
        'weights': optweights.weights,
        'test_pred': averaged_test_pred,
        'model_scores': model_scores,
        'fold_index': fold_index,
        'seed': random_state_list[m],
        'val_size': len(y_val_fold),
    }


def train_ensemble_pipeline(
    X_train_data,
    y_train_data,
    X_test_data,
    *,
    kfold=True,
    random_state=2023,
    random_state_list=None,
    n_estimators=9999,
    early_stopping_rounds=200,
    verbose=False,
    device='cpu',
):
    if random_state_list is None:
        random_state_list = [2140]

    n_splits = 1 if not kfold else 10
    splitter = Splitter(kfold=kfold, n_splits=n_splits)

    results = {
        'test_predss': np.zeros(X_test_data.shape[0]),
        'ensemble_score': [],
        'weights': [],
        'trained_models': {'xgb': [], 'lgb': [], 'cat': []},
        'models': None,
        'last_y_val': None,
        'last_y_val_pred': None,
        'fold_records': [],
        'oof_true': [],
        'oof_pred': [],
    }

    for i, split_data in enumerate(splitter.split_data(X_train_data, y_train_data, random_state_list=random_state_list)):
        fold_result = train_single_fold(
            i,
            split_data,
            splitter,
            X_test_data,
            n_splits,
            random_state_list,
            n_estimators,
            device,
            random_state,
            early_stopping_rounds,
            verbose,
            results['trained_models'],
        )
        results['models'] = fold_result['models']
        results['last_y_val'] = fold_result['y_val']
        results['last_y_val_pred'] = fold_result['y_val_pred']
        results['ensemble_score'].append(fold_result['ensemble_score'])
        results['weights'].append(fold_result['weights'])
        results['test_predss'] += fold_result['test_pred']
        results['oof_true'].append(np.asarray(fold_result['y_val']))
        results['oof_pred'].append(np.asarray(fold_result['y_val_pred']))
        results['fold_records'].append({
            'fold_index': fold_result['fold_index'],
            'seed': fold_result['seed'],
            'val_size': fold_result['val_size'],
            'ensemble_score': fold_result['ensemble_score'],
            **fold_result['model_scores'],
        })
        gc.collect()

    if results['oof_true']:
        results['oof_true'] = np.concatenate(results['oof_true'])
        results['oof_pred'] = np.concatenate(results['oof_pred'])
    else:
        results['oof_true'] = np.array([])
        results['oof_pred'] = np.array([])

    return results


def summarize_ensemble_results(ensemble_score, weights, models):
    print('len(ensemble_score):', len(ensemble_score))
    print('len(weights):', len(weights))
    print('ensemble_score:', ensemble_score[:3] if len(ensemble_score) else ensemble_score)
    print('weights:', weights[:1] if len(weights) else weights)

    if len(ensemble_score) == 0 or len(weights) == 0 or models is None:
        print('ensemble_score 或 weights 为空，说明训练单元没有成功运行到结果汇总阶段。')
        return

    mean_score = np.mean(ensemble_score)
    std_score = np.std(ensemble_score)
    print(f'Ensemble Accuracy score {mean_score:.5f} ± {std_score:.5f}')

    print('--- Model Weights ---')
    mean_weights = np.mean(weights, axis=0)
    std_weights = np.std(weights, axis=0)
    for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
        print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')


def compute_final_predictions(oof_true, oof_pred, test_predss, fixed_cutoff=0.5):
    precisions, recalls, thresholds = precision_recall_curve(oof_true, oof_pred)
    del precisions, recalls, thresholds
    cutoff = acc_cutoff(oof_true, oof_pred)
    y_test_pred_oof = np.where(test_predss > float(cutoff), 1, 0)
    y_test_pred_fixed = np.where(test_predss > float(fixed_cutoff), 1, 0)
    return cutoff, y_test_pred_oof, y_test_pred_fixed


def load_optional_submission_candidates():
    candidate_paths = [
        Path("results/submissions/misc/XGB_best.csv"),
        Path("results/submissions/misc/submission.csv"),
        Path("results/submissions/misc/misaelcribeiro_submission.csv"),
    ]
    submissions = []
    for path in candidate_paths:
        if path.exists():
            submissions.append(pd.read_csv(path))
    return submissions


def save_single_submission(pred_flags, file_name, artifact_name):
    sub = pd.read_csv('data/raw/sample_submission.csv')
    sub['Transported'] = np.asarray(pred_flags).astype(bool)
    sub.to_csv(file_name, index=False)
    save_table_artifact(artifact_name, sub)
    return sub


def save_submissions(y_test_pred_oof, y_test_pred_fixed, enable_external_blend=USE_EXTERNAL_BLEND):
    sub_fixed = save_single_submission(y_test_pred_fixed, 'submission_model.csv', "submission_model")
    save_single_submission(y_test_pred_fixed, 'submission_cutoff_05.csv', "submission_cutoff_05")
    sub_oof = save_single_submission(y_test_pred_oof, 'submission_oof_cutoff.csv', "submission_oof_cutoff")

    sub_combined = sub_fixed.copy()
    if enable_external_blend:
        additional_submissions = load_optional_submission_candidates()
        if additional_submissions:
            combined_pred = sub_fixed['Transported'].copy()
            for extra_sub in additional_submissions:
                combined_pred = combined_pred | extra_sub['Transported'].astype(bool)
            sub_combined['Transported'] = combined_pred

    sub_combined.to_csv('submission.csv', index=False)
    save_table_artifact("submission_combined", sub_combined)
    return sub_fixed, sub_oof, sub_combined


def collect_runtime_versions():
    import catboost
    import lightgbm
    import numpy
    import optuna as optuna_pkg
    import pandas
    import sklearn as sklearn_pkg
    import xgboost

    return {
        'python': sys.version,
        'numpy': numpy.__version__,
        'pandas': pandas.__version__,
        'scikit_learn': sklearn_pkg.__version__,
        'xgboost': xgboost.__version__,
        'lightgbm': lightgbm.__version__,
        'catboost': catboost.__version__,
        'optuna': optuna_pkg.__version__,
    }


def save_run_diagnostics(training_results, cutoff, sub_fixed, sub_oof, sub_combined):
    versions = collect_runtime_versions()
    fold_records = pd.DataFrame(training_results['fold_records'])
    if not fold_records.empty:
        save_table_artifact("fold_scores", fold_records)

    submission_same_ratio = float((sub_fixed['Transported'] == sub_combined['Transported']).mean())
    submission_oof_same_ratio = float((sub_fixed['Transported'] == sub_oof['Transported']).mean())
    summary = pd.DataFrame([
        {
            'ensemble_mean_accuracy': float(np.mean(training_results['ensemble_score'])) if training_results['ensemble_score'] else np.nan,
            'ensemble_std_accuracy': float(np.std(training_results['ensemble_score'])) if training_results['ensemble_score'] else np.nan,
            'cutoff': float(cutoff),
            'submission_model_positive_rate': float(sub_fixed['Transported'].mean()),
            'submission_oof_cutoff_positive_rate': float(sub_oof['Transported'].mean()),
            'submission_cutoff_05_positive_rate': float(sub_fixed['Transported'].mean()),
            'submission_final_positive_rate': float(sub_combined['Transported'].mean()),
            'submission_same_ratio': submission_same_ratio,
            'submission_vs_oof_same_ratio': submission_oof_same_ratio,
            'n_folds': len(training_results['fold_records']),
            'n_models': len(training_results['models']) if training_results['models'] is not None else 0,
            'oof_size': int(training_results['oof_true'].shape[0]),
            'external_blend_enabled': bool(USE_EXTERNAL_BLEND),
            'default_submission_strategy': 'fixed_0_5',
        }
    ])
    save_table_artifact("run_summary", summary)

    diagnostics_payload = {
        'versions': versions,
        'summary': summary.iloc[0].to_dict(),
        'fold_records': training_results['fold_records'],
    }
    diagnostics_path = TABLE_DIR / "run_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print("Saved diagnostics to:", diagnostics_path)


RUN_TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = TABLE_DIR / f"run_{RUN_TS}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

all_test_predss = []
all_run_summaries = []
all_trained_models = {'xgb': [], 'lgb': [], 'cat': []}
best_run_idx = 0
best_run_cv = -1

for run_idx in range(NUM_ENSEMBLE_RUNS):
    run_seed = GLOBAL_RANDOM_STATE + run_idx * 100
    set_global_seed(run_seed)
    print(f"\n{'='*60}")
    print(f"Ensemble run {run_idx + 1}/{NUM_ENSEMBLE_RUNS}  (seed={run_seed})")
    print(f"{'='*60}\n")

    training_results = train_ensemble_pipeline(
        X_train,
        y_train,
        X_test,
        random_state=run_seed,
    )
    test_predss_run = training_results['test_predss']
    ensemble_score = training_results['ensemble_score']
    weights_list = training_results['weights']
    trained_models_run = training_results['trained_models']

    summarize_ensemble_results(ensemble_score, weights_list, training_results['models'])

    all_test_predss.append(test_predss_run)
    for model_name in all_trained_models:
        all_trained_models[model_name].extend(trained_models_run.get(model_name, []))

    oof_true_run = training_results['oof_true']
    oof_pred_run = training_results['oof_pred']
    cutoff_run, _, _ = compute_final_predictions(oof_true_run, oof_pred_run, test_predss_run)

    sub_run = save_single_submission(
        np.where(test_predss_run > 0.5, 1, 0),
        'submission_model.csv',
        f"submission_run_{run_idx:02d}",
    )
    sub_run.to_csv(RUN_DIR / f'submission_run_{run_idx:02d}.csv', index=False)

    fold_records_df = pd.DataFrame(training_results['fold_records'])
    if not fold_records_df.empty:
        fold_records_df.to_csv(RUN_DIR / f"fold_scores_run_{run_idx:02d}.csv", index=False)

    run_cv_mean = float(np.mean(ensemble_score)) if ensemble_score else float('nan')
    run_summary = pd.DataFrame([{
        'run_index': run_idx,
        'run_seed': run_seed,
        'ensemble_mean_accuracy': run_cv_mean,
        'ensemble_std_accuracy': float(np.std(ensemble_score)) if ensemble_score else float('nan'),
        'cutoff': float(cutoff_run),
        'submission_positive_rate': float(sub_run['Transported'].mean()),
    }])
    all_run_summaries.append(run_summary)

    if run_cv_mean > best_run_cv:
        best_run_cv = run_cv_mean
        best_run_idx = run_idx

    gc.collect()

avg_test_predss = np.mean(all_test_predss, axis=0)
test_predss = avg_test_predss

runs_summary = pd.concat(all_run_summaries, ignore_index=True)
runs_summary.to_csv(RUN_DIR / "all_runs_summary.csv", index=False)
save_table_artifact("all_runs_summary", runs_summary)
print(f"\nBest single run: run {best_run_idx} (CV={best_run_cv:.5f})")

oof_true = training_results['oof_true']
oof_pred = training_results['oof_pred']
models = training_results['models']
trained_models = all_trained_models
ensemble_score = runs_summary['ensemble_mean_accuracy'].tolist()
weights = [training_results['weights']]

cutoff, y_test_pred_oof, y_test_pred_fixed = compute_final_predictions(oof_true, oof_pred, test_predss)
sub, sub_oof, sub_combined = save_submissions(y_test_pred_oof, y_test_pred_fixed)

versions = collect_runtime_versions()
summary = pd.DataFrame([{
    'ensemble_mean_accuracy': float(runs_summary['ensemble_mean_accuracy'].mean()),
    'ensemble_std_accuracy': float(runs_summary['ensemble_mean_accuracy'].std()),
    'best_single_run_cv': float(best_run_cv),
    'best_single_run_idx': int(best_run_idx),
    'cutoff': float(cutoff),
    'submission_model_positive_rate': float(sub['Transported'].mean()),
    'submission_oof_cutoff_positive_rate': float(sub_oof['Transported'].mean()),
    'submission_cutoff_05_positive_rate': float(y_test_pred_fixed.mean()),
    'submission_final_positive_rate': float(sub_combined['Transported'].mean()),
    'submission_same_ratio': float((sub['Transported'] == sub_combined['Transported']).mean()),
    'submission_vs_oof_same_ratio': float((sub['Transported'] == sub_oof['Transported']).mean()),
    'num_ensemble_runs': NUM_ENSEMBLE_RUNS,
    'n_folds_per_run': len(training_results['fold_records']),
    'n_models': len(models) if models is not None else 0,
    'oof_size': int(oof_true.shape[0]),
    'external_blend_enabled': bool(USE_EXTERNAL_BLEND),
    'default_submission_strategy': 'fixed_0_5_ensemble',
}])
save_table_artifact("run_summary", summary)

diagnostics_payload = {
    'versions': versions,
    'summary': summary.iloc[0].to_dict(),
    'runs_summary': runs_summary.to_dict('records'),
    'fold_records': training_results['fold_records'],
}
diagnostics_path = RUN_DIR / "run_diagnostics.json"
diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2, ensure_ascii=False), encoding='utf-8')
print("Saved diagnostics to:", diagnostics_path)

# %% [code] cell 77 (notebook cell 165)
def visualize_importance(models, feature_cols, title, top=20):
    importances = []
    feature_importance = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["importance"] = model.feature_importances_
        _df["feature"] = pd.Series(feature_cols)
        _df["fold"] = i
        _df = _df.sort_values('importance', ascending=False)
        _df = _df.head(top)
        feature_importance = pd.concat([feature_importance, _df], axis=0, ignore_index=True)
        
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    save_table_artifact(f"{title}_feature_importance", feature_importance)
    # display(feature_importance.groupby(["feature"]).mean().reset_index().drop('fold', axis=1))
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance, color='skyblue', errorbar='sd')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'{title} Feature Importance [Top {top}]', fontsize=18)
    plt.grid(True, axis='x')
    finalize_figures()
    
for name, models in trained_models.items():
    visualize_importance(models, list(X_train.columns), name)
