import numpy as np


def mixup_augment(X, y, alpha=0.4, multiplier=1, noise_std=0.01, random_state=None):
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    X_aug = X.copy()
    y_aug = y.copy()

    if multiplier > 0 and alpha > 0:
        for _ in range(multiplier):
            idx = rng.permutation(n)
            lam = rng.beta(alpha, alpha, size=n)
            lam_mat = lam.reshape(-1, 1)
            X_mix = lam_mat * X + (1 - lam_mat) * X[idx]
            y_mix = np.where(lam >= 0.5, y, y.iloc[idx].values)
            X_aug = np.vstack([X_aug, X_mix])
            y_aug = np.concatenate([y_aug, y_mix])

    if noise_std > 0 and multiplier >= 0:
        n_total = X_aug.shape[0]
        noise = rng.normal(0, noise_std, size=(n_total, X_aug.shape[1]))
        X_aug = X_aug + noise

    return X_aug, y_aug


def group_swap_augment(X, y, group_ids, swap_fraction=0.3, random_state=None):
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    n_feat = X.shape[1]
    X_aug = X.copy()
    y_aug = y.copy() if isinstance(y, np.ndarray) else np.asarray(y)

    unique_groups = np.unique(group_ids)
    X_swapped = X.copy()

    for gid in unique_groups:
        mask = np.where(group_ids == gid)[0]
        n_g = len(mask)
        if n_g < 2:
            continue
        for i_idx in mask:
            candidates = mask[mask != i_idx]
            if len(candidates) == 0:
                continue
            partner = rng.choice(candidates)
            swap_mask = rng.rand(n_feat) < swap_fraction
            X_swapped[i_idx, swap_mask] = X[partner, swap_mask]

    X_aug = np.vstack([X_aug, X_swapped])
    y_aug = np.concatenate([y_aug, y_aug[:n]])
    return X_aug, y_aug
