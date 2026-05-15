from sklearn.ensemble import HistGradientBoostingClassifier
from train_base import train_model

def build_hgb(seed, fold_i):
    return HistGradientBoostingClassifier(
        max_depth=7, learning_rate=0.035,
        max_iter=400, min_samples_leaf=10,
        random_state=seed * 10 + fold_i
    )

if __name__ == '__main__':
    train_model('hist_gb', build_hgb)
