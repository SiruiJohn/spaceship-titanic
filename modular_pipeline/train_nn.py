import pickle, warnings, sys
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG


def load_processed_data():
    with open(CFG.processed_data_file, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    data = load_processed_data()
    X_num = data['X_train_num']
    X_te_num = data['X_test_num']
    y = data['y']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num.values)
    X_te_scaled = scaler.transform(X_te_num.values)

    oof_probs = np.zeros(len(y))
    oof_counts = np.zeros(len(y))
    test_preds = []

    total = len(CFG.random_seeds) * CFG.n_splits
    fold_num = 0

    for seed in CFG.random_seeds:
        cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=seed)
        for fi, (tr_idx, val_idx) in enumerate(cv.split(X_scaled, y), 1):
            fold_num += 1
            mlp = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                alpha=0.005,
                learning_rate_init=1e-3,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed * 10 + fi,
                batch_size=256,
            )
            mlp.fit(X_scaled[tr_idx], y.iloc[tr_idx])
            vp = mlp.predict_proba(X_scaled[val_idx])[:, 1]
            tp = mlp.predict_proba(X_te_scaled)[:, 1]

            oof_probs[val_idx] += vp
            oof_counts[val_idx] += 1
            test_preds.append(tp)

            acc = accuracy_score(y.iloc[val_idx], vp >= 0.5)
            print(f'  [nn] {fold_num:2d}/{total} | s={seed} f={fi} | acc={acc:.4f}')

    oof_probs /= np.maximum(oof_counts, 1)
    avg_test = np.mean(test_preds, axis=0)

    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(CFG.output_dir / 'oof_nn.npy', oof_probs)
    np.save(CFG.output_dir / 'test_nn.npy', avg_test)

    final_acc = accuracy_score(y, oof_probs >= 0.5)
    print(f'\n[OK] nn saved -> oof_nn.npy, test_nn.npy')
    print(f'   OOF accuracy: {final_acc:.5f}')
