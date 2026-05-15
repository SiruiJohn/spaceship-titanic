# Spaceship Titanic Kaggle Project

Kaggle 竞赛 [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) 二分类任务（预测乘客是否被传送）。

---

## 快速开始：3 步生成提交文件

```powershell
cd "c:\Users\siruiJohn\project\Spaceship Titanic_1\Spaceship Titanic"

# 第一步：数据准备（1-2 分钟）
python modular_pipeline/main.py prep

# 第二步：训练所有模型（15-30 分钟，取决于 CPU）
python modular_pipeline/main.py train -m all

# 第三步：集成 + 生成提交文件（2-5 分钟）
python modular_pipeline/main.py blend
```

最终提交产物是项目根目录下的 `submission_90plus.csv`。

> **提示**：如果你已经跑过 `prep` 且数据没变，可以跳过第一步。如果只想重新集成调权重，可以跳过第二步只跑 `blend`。

---

## 项目结构

```
Spaceship Titanic/
├── modular_pipeline/             # 模块化训练管道（推荐使用）
│   ├── config.py                 # 共享配置（特征列表、种子、路径）
│   ├── data_prep.py              # 数据加载 + 特征工程 + 编码 → processed_data.pkl
│   ├── train_base.py             # 训练基类（CV folds、OOF 聚合）
│   ├── train_et.py               # ExtraTrees 训练
│   ├── train_hgb.py              # HistGradientBoosting 训练
│   ├── train_xgb.py              # XGBoost 训练
│   ├── train_lgb.py              # LightGBM 训练（含 Optuna 超参搜索）
│   ├── train_cat.py              # CatBoost 训练
│   ├── blend.py                  # 模型集成 + MLP + 校准 + 提交文件
│   ├── main.py                   # 主入口（CLI 控制整个流程）
│   └── output/                   # 中间产物（pkl + oof/test .npy）
│
├── data/raw/                     # Kaggle 原始数据（Git 忽略）
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── results/                      # 可视化产出
│   ├── eda_viz/                  # 13 张 EDA 图表
│   └── model_report/             # 8 张模型报告图表
│
├── spaceship-titanic.py          # 单体脚本（也可以直接跑）
├── spaceship-titanic.ipynb       # Jupyter 源文件
├── eda_visualizations.py         # EDA 可视化独立脚本
├── requirements.txt              # Python 依赖
├── README.md                     # 本文件
├── EDA可视化分析报告.md          # EDA 报告
├── 模型训练与评估分析报告.md     # 模型报告
└── spaceship-titanic-说明文档.md # 技术说明
```

---

## 数据流

```
data/raw/train.csv, test.csv
        |
   data_prep.py      # 特征工程 + OOF编码 + Ordinal编码
        |
   processed_data.pkl
        |
  ┌─────┼─────┬─────┬─────┐
  et   hgb   xgb   lgb   cat     # 各模型独立训练
  └─────┼─────┴─────┴─────┘
        |
  oof_*.npy + test_*.npy         # OOF 和 Test 预测
        |
    blend.py                     # 集成 + MLP + 校准
        |
  submission_90plus.csv          # 最终提交
```

---

## modular_pipeline/main.py 命令参考

```powershell
# 查看帮助
python modular_pipeline/main.py --help

# 查看可用模型列表
python modular_pipeline/main.py list

# 只跑数据准备
python modular_pipeline/main.py prep

# 只训练指定模型（适用于调参）
python modular_pipeline/main.py train -m lgb          # 只训练 LightGBM
python modular_pipeline/main.py train -m lgb xgb      # 训练 LGB + XGBoost
python modular_pipeline/main.py train -m all          # 训练全部 5 个模型

# 只跑集成（不需要重新训练）
python modular_pipeline/main.py blend

# 完整流水线（= prep + train all + blend）
python modular_pipeline/main.py full
```

---

## 典型工作流

### 场景 1：首次运行，从零开始

```powershell
python modular_pipeline/main.py full
```

预计总时间：25-45 分钟（取决于 CPU 核数）。

### 场景 2：调整 LightGBM 超参

```powershell
# 修改 modular_pipeline/config.py 中的 optuna_trials
# 或者在 train_lgb.py 中调整搜索空间
python modular_pipeline/main.py train -m lgb
python modular_pipeline/main.py blend
```

### 场景 3：调试集成权重，不需要重新训练

```powershell
python modular_pipeline/main.py blend
```

### 场景 4：并行训练加速

打开 3 个终端窗口，分别运行：

```powershell
# 终端 1
python modular_pipeline/main.py prep

# 等 prep 完成后，3 个终端分别：
# 终端 1: python modular_pipeline/train_lgb.py
# 终端 2: python modular_pipeline/train_xgb.py
# 终端 3: python modular_pipeline/train_cat.py

# 全部跑完后：
python modular_pipeline/main.py blend
```

### 场景 5：使用原来的单体脚本

```powershell
python spaceship-titanic.py
```

---

## 关键配置参数

所有配置集中在 `modular_pipeline/config.py`：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `random_seeds` | [42, 2024, 7, 2025, 88] | 5 seeds × 5 folds = 每模型 25 folds |
| `n_splits` | 5 | 每 seed 的 CV folds 数 |
| `optuna_trials` | 200 | LightGBM 超参搜索轮数 |
| `optuna_cv_folds` | 3 | Optuna 内部 CV folds 数 |
| `ensemble_runs` | 3 | 多轮集成（不同 base_seed） |
| `ensemble_base_seeds` | [42, 123, 888] | 每轮集成的 base_seed |

> **调参建议**：快速调试时把 `optuna_trials` 降到 20，`ensemble_runs` 降到 1。

---

## 当前成绩

| 指标 | 数值 |
|------|------|
| OOF CV 准确率 | ~81.87% |
| Kaggle 公开榜 | 0.80453 |
| 提交文件 | `submission_90plus.csv` |

---

## 环境要求

```bash
pip install -r requirements.txt
```

核心依赖：`pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm catboost optuna shap`

`lightgbm`、`xgboost`、`catboost`、`optuna`、`shap` 为可选依赖，缺失时会自动跳过对应步骤。

---

## 常见问题

### UnicodeEncodeError (GBK codec)

Windows 终端 GBK 编码导致 emoji 无法打印。已通过移除所有非 ASCII 字符解决。

如果仍有编码问题：
```powershell
$env:PYTHONIOENCODING='utf-8'
python modular_pipeline/main.py full
```

### OSError: Cannot save file into a non-existent directory

已修复。所有写入操作现在会自动创建父目录。如果手动删除了 `modular_pipeline/output/` 或 `results/notebook_export/`，重新运行脚本即可自动重建。

### 训练时间过长

- 把 `config.py` 中的 `optuna_trials` 降到 20-50
- 把 `random_seeds` 减少到 2-3 个
- 把 `ensemble_runs` 设为 1
- 只训练部分模型：`main.py train -m lgb et`
