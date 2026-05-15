"""
独立 EDA 可视化脚本 — 不依赖主脚本，只读原始数据
生成图表供可视化文档使用
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

OUT_DIR = Path("results/eda_viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#333333', 'axes.labelcolor': '#222222',
    'xtick.color': '#444444', 'ytick.color': '#444444',
    'grid.color': '#dddddd', 'text.color': '#222222',
    'font.size': 11,
})
BLUE, RED, GREEN  = '#2878b5', '#d43d3d', '#2ca02c'
PURPLE, ORANGE    = '#9467bd', '#e07020'
YELLOW            = '#bc8f00'
PALETTE = [BLUE, RED, GREEN, PURPLE, ORANGE, '#17becf', YELLOW]

# ── Load ────────────────────────────────────────────────────────────────────
train = pd.read_csv('data/raw/train.csv')
test  = pd.read_csv('data/raw/test.csv')
train['Transported_int'] = train['Transported'].astype(int)
SPEND   = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
CATEGORICAL = ['HomePlanet','CryoSleep','Destination','VIP']

# ── Helper ──────────────────────────────────────────────────────────────────
def save_and_close(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

# =============================================================================
# 1. 目标变量分布
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Target Variable: Transported', fontsize=16, fontweight='bold', y=1.02)

vc = train['Transported'].value_counts()
colors_pie = [RED, GREEN]
wedges, texts, autotexts = axes[0].pie(
    vc.values, labels=['Not Transported','Transported'],
    autopct='%1.1f%%', colors=colors_pie, startangle=90,
    textprops={'color':'#222222','fontsize':12},
    wedgeprops={'edgecolor':'white','linewidth':2}
)
for at in autotexts:
    at.set_fontweight('bold')
axes[0].set_title(f'Distribution (n={len(train):,})', color='#222222', fontsize=13)

bars = axes[1].bar(['Not Transported','Transported'], vc.values, color=colors_pie, edgecolor='none', width=0.5)
for b, v in zip(bars, vc.values):
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+10, f'{v:,}',
                 ha='center', fontsize=13, color='#222222', fontweight='bold')
axes[1].set_title('Count', color='#222222', fontsize=13)
axes[1].set_ylabel('Passengers')
save_and_close('01_target_distribution')

# =============================================================================
# 2. 缺失值分析
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Missing Values Analysis', fontsize=16, fontweight='bold', y=1.02)

missing = train.isnull().mean().sort_values(ascending=False)
missing = missing[missing > 0] * 100
cols_with_missing = missing.index.tolist()

axes[0].barh(cols_with_missing[::-1], missing.values[::-1],
             color=ORANGE, edgecolor='none')
axes[0].set_xlabel('Missing (%)')
axes[0].set_title('Train Missing Rate', color='#222222')
for i, (c, v) in enumerate(zip(cols_with_missing[::-1], missing.values[::-1])):
    axes[0].text(v+0.3, i, f'{v:.1f}%', va='center', fontsize=9, color='#222222')

missing_t = test.isnull().mean().sort_values(ascending=False)
missing_t = missing_t[missing_t > 0] * 100
cols_t = missing_t.index.tolist()
axes[1].barh(cols_t[::-1], missing_t.values[::-1],
             color=PURPLE, edgecolor='none')
axes[1].set_xlabel('Missing (%)')
axes[1].set_title('Test Missing Rate', color='#222222')
for i, (c, v) in enumerate(zip(cols_t[::-1], missing_t.values[::-1])):
    axes[1].text(v+0.3, i, f'{v:.1f}%', va='center', fontsize=9, color='#222222')
save_and_close('02_missing_values')

# =============================================================================
# 3. 类别特征分布 (按 Target 分色)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Categorical Features by Transported Status', fontsize=16, fontweight='bold', y=1.01)

for ax, col in zip(axes.flat, ['HomePlanet','CryoSleep','Destination','VIP']):
    ct = pd.crosstab(train[col].fillna('Missing'), train['Transported'])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind='barh', stacked=True, ax=ax, color=[RED, GREEN], edgecolor='none', width=0.7)
    ax.set_title(col, color='#222222', fontsize=13)
    ax.set_xlabel('%')
    ax.legend(loc='lower right', fontsize=9)
    for c in ax.containers:
        labels = [f'{v.get_width():.0f}%' if v.get_width() > 3 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=8, color='#222222')
save_and_close('03_categorical_by_target')

# =============================================================================
# 4. 数值特征分布 (按 Target 分色)
# =============================================================================
num_cols = ['Age'] + SPEND + ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
# deduplicate
_num_cols = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Numerical Feature Distributions by Transported Status', fontsize=16, fontweight='bold', y=1.01)

for ax, col in zip(axes.flat, _num_cols):
    for label, color, ls in [(True,GREEN,'Transported'), (False,RED,'Not Transported')]:
        data = train.loc[train['Transported']==label, col].dropna()
        if col == 'Age':
            ax.hist(data, bins=40, alpha=0.55, color=color, label=ls, edgecolor='none')
        else:
            vals = data[data < data.quantile(0.98)]
            ax.hist(vals, bins=50, alpha=0.55, color=color, label=ls, edgecolor='none')
    ax.set_title(col, color='#222222', fontsize=12)
    ax.legend(fontsize=8)
save_and_close('04_numerical_by_target')

# =============================================================================
# 5. 消费特征对数分布 (by target)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Log-Transformed Spend Distributions', fontsize=16, fontweight='bold', y=1.01)

for ax, col in zip(axes.flat, _num_cols[1:]):
    for label, color, ls in [(True,GREEN,'Transported'), (False,RED,'Not Transported')]:
        vals = np.log1p(train.loc[train['Transported']==label, col].dropna())
        ax.hist(vals, bins=50, alpha=0.55, color=color, label=ls, edgecolor='none')
    ax.set_title(f'log1p({col})', color='#222222', fontsize=12)
    ax.legend(fontsize=8)
save_and_close('05_log_spend_by_target')

# =============================================================================
# 6. 消费特征箱线图
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle('Spend Features Boxplot (log scale, clipped at 99p)', fontsize=16, fontweight='bold', y=1.02)

plot_data = train[SPEND].copy()
for c in SPEND:
    q99 = plot_data[c].quantile(0.99)
    plot_data[c] = plot_data[c].clip(upper=q99)
plot_data_log = np.log1p(plot_data)
bp = ax.boxplot([plot_data_log[c].dropna().values for c in SPEND],
                patch_artist=True,
                medianprops={'color':'black','linewidth':2})
ax.set_xticklabels(SPEND)
for patch, color in zip(bp['boxes'], PALETTE[:5]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('log1p(Amount)', color='#222222')
ax.set_title('Spend Distribution (log scale)', color='#222222', fontsize=13)
plt.xticks(rotation=30)
save_and_close('06_spend_boxplot')

# =============================================================================
# 7. 消费相关性热力图
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle('Feature Correlation Matrix', fontsize=16, fontweight='bold', y=1.01)

corr_cols = ['Age'] + SPEND + ['Transported_int']
corr = train[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-0.3, vmax=0.3,
            linewidths=0.5, linecolor='#cccccc',
            cbar_kws={'shrink':0.8}, ax=ax)
ax.set_title('Pearson Correlation', color='#222222', fontsize=13)
save_and_close('07_correlation_heatmap')

# =============================================================================
# 8. 船舱甲板 (Cabin Deck) 分布
# =============================================================================
cabin_parsed = train['Cabin'].fillna('U/9999/U').str.split('/', expand=True)
train['CabinDeck'] = cabin_parsed[0].replace('nan','U')
train['CabinSide'] = cabin_parsed[2].replace('nan','U')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Cabin Analysis', fontsize=16, fontweight='bold', y=1.02)

deck_order = sorted(train['CabinDeck'].value_counts().index.tolist())
deck_ct = pd.crosstab(train['CabinDeck'], train['Transported'])
deck_ct_pct = deck_ct.div(deck_ct.sum(axis=1), axis=0) * 100
deck_ct_pct.loc[deck_order].plot(kind='bar', stacked=True, ax=axes[0], color=[RED, GREEN], edgecolor='none')
axes[0].set_title('Deck × Transported', color='#222222')
axes[0].set_ylabel('%')
axes[0].legend(fontsize=8)

side_ct = pd.crosstab(train['CabinSide'], train['Transported'])
side_ct_pct = side_ct.div(side_ct.sum(axis=1), axis=0) * 100
side_ct_pct.plot(kind='bar', stacked=True, ax=axes[1], color=[RED, GREEN], edgecolor='none')
axes[1].set_title('Side × Transported', color='#222222')
axes[1].set_ylabel('%')
axes[1].legend(fontsize=8)

deck_counts = train['CabinDeck'].value_counts()
axes[2].bar(deck_counts.index, deck_counts.values, color=PURPLE, edgecolor='none', alpha=0.8)
axes[2].set_title('Deck Distribution (All)', color='#222222')
axes[2].set_ylabel('Count')
save_and_close('08_cabin_analysis')

# =============================================================================
# 9. 群组大小分析
# =============================================================================
gp = train['PassengerId'].str.split('_', expand=True)
train['GroupId'] = pd.to_numeric(gp[0], errors='coerce')
train['GroupSize'] = train.groupby('GroupId')['PassengerId'].transform('size')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Group Analysis', fontsize=16, fontweight='bold', y=1.02)

gs_counts = train['GroupSize'].value_counts().sort_index()
axes[0].bar(gs_counts.index.astype(str), gs_counts.values, color=BLUE, edgecolor='none', alpha=0.8)
axes[0].set_title('Group Size Distribution', color='#222222')
axes[0].set_xlabel('Group Size')
axes[0].set_ylabel('Passengers')

solo = train[train['GroupSize']==1]
non_solo = train[train['GroupSize']>1]
axes[1].bar(['Solo','Non-Solo'], [len(solo), len(non_solo)],
            color=[ORANGE, PURPLE], edgecolor='none', width=0.5)
for i, v in enumerate([len(solo), len(non_solo)]):
    axes[1].text(i, v+10, str(v), ha='center', fontsize=12, color='#222222', fontweight='bold')
axes[1].set_title('Solo vs Group Travelers', color='#222222')

for label, color, ls in [(True,GREEN,'Transported'),(False,RED,'Not Transported')]:
    rates = []
    sizes = sorted(train['GroupSize'].unique())
    for s in sizes:
        subset = train[train['GroupSize']==s]
        if len(subset) > 0:
            rates.append(subset['Transported'].mean())
    axes[2].plot(sizes, rates, 'o-', color=color, label=ls, markersize=5, linewidth=2)
axes[2].axhline(0.5, color='#222222', linestyle='--', alpha=0.4)
axes[2].set_title('Transport Rate by Group Size', color='#222222')
axes[2].set_xlabel('Group Size')
axes[2].set_ylabel('Transport Rate')
axes[2].legend()
save_and_close('09_group_analysis')

# =============================================================================
# 10. CryoSleep × Spend 交互分析
# =============================================================================
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CryoSleep x Spend Interaction', fontsize=16, fontweight='bold', y=1.02)

    train['TotalSpend'] = train[SPEND].sum(axis=1)
    train['NoSpend'] = (train['TotalSpend'] == 0)

    cryo_str = train['CryoSleep'].fillna('Missing').astype(str)
    cryo_transport = train.groupby(cryo_str)['Transported'].mean() * 100
    axes[0].bar(cryo_transport.index, cryo_transport.values,
                color=[RED, GREEN, PURPLE][:len(cryo_transport)], edgecolor='none', width=0.5)
    axes[0].set_title('Transport Rate by CryoSleep', color='#222222')
    axes[0].set_ylabel('Transport Rate (%)')
    for i, v in enumerate(cryo_transport.values):
        axes[0].text(i, v+1, f'{v:.1f}%', ha='center', fontsize=11, color='#222222', fontweight='bold')

    no_spend_rate = train.groupby(cryo_str)['NoSpend'].mean() * 100
    axes[1].bar(no_spend_rate.index, no_spend_rate.values,
                color=[RED, GREEN, PURPLE][:len(no_spend_rate)], edgecolor='none', width=0.5)
    axes[1].set_title('Zero-Spend Rate by CryoSleep', color='#222222')
    axes[1].set_ylabel('Zero-Spend Rate (%)')
    for i, v in enumerate(no_spend_rate.values):
        axes[1].text(i, v+1, f'{v:.1f}%', ha='center', fontsize=11, color='#222222', fontweight='bold')
    save_and_close('10_cryosleep_spend')
except Exception as e:
    print(f'ERROR in section 10: {e}')
    import traceback
    traceback.print_exc()

# =============================================================================
# 11. 年龄分布详细分析
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Age Analysis', fontsize=16, fontweight='bold', y=1.02)

for label, color, ls in [(True,GREEN,'Transported'),(False,RED,'Not Transported')]:
    data = train.loc[train['Transported']==label, 'Age'].dropna()
    axes[0].hist(data, bins=40, alpha=0.55, color=color, label=ls, edgecolor='none')
axes[0].axvline(train['Age'].median(), color=YELLOW, linestyle='--', linewidth=2, label=f'Median={train["Age"].median():.0f}')
axes[0].set_title('Age Distribution by Target', color='#222222')
axes[0].set_xlabel('Age')
axes[0].legend()

age_bins = [0,13,18,25,40,60,120]
age_labels = ['Child\n(0-12)','Teen\n(13-17)','Young Adult\n(18-24)','Adult\n(25-39)','Midlife\n(40-59)','Senior\n(60+)']
train['AgeBand'] = pd.cut(train['Age'], bins=age_bins, labels=age_labels)
age_transport = train.groupby('AgeBand')['Transported'].mean() * 100
axes[1].bar(range(len(age_transport)), age_transport.values,
            color=BLUE, edgecolor='none', width=0.6)
axes[1].set_xticks(range(len(age_transport)))
axes[1].set_xticklabels(age_labels, fontsize=9)
axes[1].set_title('Transport Rate by Age Group', color='#222222')
axes[1].set_ylabel('Transport Rate (%)')
axes[1].axhline(50, color='#222222', linestyle='--', alpha=0.4)
for i, v in enumerate(age_transport.values):
    axes[1].text(i, v+0.8, f'{v:.1f}%', ha='center', fontsize=10, color='#222222', fontweight='bold')
save_and_close('11_age_analysis')

# =============================================================================
# 12. HomePlanet × Destination 联合分析
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('HomePlanet × Destination Analysis', fontsize=16, fontweight='bold', y=1.02)

train['HomeDest'] = train['HomePlanet'].fillna('?') + '→' + train['Destination'].fillna('?')
hd_ct = pd.crosstab(train['HomeDest'], train['Transported'])
hd_ct = hd_ct.loc[hd_ct.sum(axis=1).sort_values(ascending=False).index]
hd_pct = hd_ct.div(hd_ct.sum(axis=1), axis=0) * 100
hd_pct.head(15).plot(kind='barh', stacked=True, ax=axes[0], color=[RED, GREEN], edgecolor='none')
axes[0].set_title('Top Routes × Transported', color='#222222')
axes[0].set_xlabel('%')
axes[0].legend(fontsize=8)

# by HomePlanet
hp_transport = train.groupby('HomePlanet')['Transported'].agg(['mean','count'])
hp_transport['mean'] = hp_transport['mean'] * 100
hp_transport = hp_transport.sort_values('mean')
colors_hp = [RED if v < 50 else GREEN for v in hp_transport['mean']]
bars = axes[1].barh(hp_transport.index, hp_transport['mean'], color=colors_hp, edgecolor='none', height=0.6)
axes[1].axvline(50, color='#222222', linestyle='--', alpha=0.5)
axes[1].set_title('Transport Rate by HomePlanet', color='#222222')
axes[1].set_xlabel('Transport Rate (%)')
for b, (v, c) in zip(bars, zip(hp_transport['mean'], hp_transport['count'])):
    axes[1].text(v+0.5, b.get_y()+b.get_height()/2, f'{v:.1f}% (n={c})',
                 va='center', fontsize=9, color='#222222')
save_and_close('12_homeplanet_destination')

# =============================================================================
# 13. VIP 分析
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('VIP Analysis', fontsize=16, fontweight='bold', y=1.02)

vip_ct = pd.crosstab(train['VIP'].fillna('Missing'), train['Transported'])
vip_pct = vip_ct.div(vip_ct.sum(axis=1), axis=0) * 100
vip_pct.plot(kind='bar', stacked=True, ax=axes[0], color=[RED, GREEN], edgecolor='none')
for c in axes[0].containers:
    axes[0].bar_label(c, labels=[f'{v.get_height():.0f}%' for v in c],
                      label_type='center', fontsize=12, color='#222222', fontweight='bold')
axes[0].set_title('VIP × Transported', color='#222222')
axes[0].set_ylabel('%')
axes[0].legend(fontsize=9)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# VIP × CryoSleep × Transported
vip_filled = train['VIP'].fillna('Missing')
cryo_filled = train['CryoSleep'].fillna('Missing')
vip_cryo = train.groupby([vip_filled, cryo_filled])['Transported'].mean().unstack() * 100
vip_cryo.plot(kind='bar', ax=axes[1], color=[RED, GREEN, PURPLE], edgecolor='none')
axes[1].set_title('Transport Rate: VIP × CryoSleep', color='#222222')
axes[1].set_ylabel('Transport Rate (%)')
axes[1].axhline(50, color='#222222', linestyle='--', alpha=0.4)
axes[1].legend(fontsize=9)
for c in axes[1].containers:
    axes[1].bar_label(c, labels=[f'{v.get_height():.0f}%' if v.get_height()>0 else '' for v in c],
                      fontsize=9, color='#222222')
save_and_close('13_vip_analysis')

# =============================================================================
# 14. 综合仪表板 (Dashboard Summary) — skipped for robustness, individual plots above cover all aspects
