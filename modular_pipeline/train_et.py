from sklearn.ensemble import ExtraTreesClassifier
from train_base import train_model

def build_et(seed, fold_i):
    return ExtraTreesClassifier(
        n_estimators=600, min_samples_leaf=2,
        random_state=seed * 10 + fold_i, n_jobs=4
    )

if __name__ == '__main__':
    train_model('extra_trees', build_et)
