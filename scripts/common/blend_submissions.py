import os
import pandas as pd
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUB_DIR = ROOT / "results" / "submissions"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    s = df["Transported"]
    if s.dtype == bool:
        y = s.astype(int)
    elif s.dtype == object:
        y = s.map({"True": 1, "False": 0, True: 1, False: 0}).astype(int)
    else:
        y = s.astype(int)
    out = pd.DataFrame({"PassengerId": df["PassengerId"].astype(str), "y": y.values})
    out = out.sort_values("PassengerId").reset_index(drop=True)
    return out


def majority_vote(dfs: list[pd.DataFrame], tie_break: pd.Series | None = None) -> np.ndarray:
    mat = np.vstack([d["y"].to_numpy() for d in dfs]).T
    votes = mat.sum(axis=1)
    half = mat.shape[1] / 2.0
    pred = (votes > half).astype(int)
    ties = votes == half
    if tie_break is not None and ties.any():
        pred[ties] = tie_break.to_numpy()[ties].astype(int)
    return pred


def weighted_vote(dfs: list[pd.DataFrame], weights: list[float], threshold: float = 0.5) -> np.ndarray:
    mat = np.vstack([d["y"].to_numpy() for d in dfs]).T.astype(float)
    w = np.asarray(weights, dtype=float)
    score = (mat * w.reshape(1, -1)).sum(axis=1) / w.sum()
    return (score >= threshold).astype(int)


def main() -> None:
    candidates = ["v3", "v8", "v6", "v5", "v2"]
    paths = []
    for v in candidates:
        p = SUB_DIR / v / f"submission_{v}.csv"
        if p.exists():
            paths.append(str(p))
    if len(paths) < 2:
        raise SystemExit("Not enough submission files found to blend.")

    dfs = [load(p) for p in paths]
    pid = dfs[0]["PassengerId"]
    for d in dfs[1:]:
        if not pid.equals(d["PassengerId"]):
            raise SystemExit("PassengerId mismatch between submissions.")

    tie_break = dfs[0]["y"]
    mv = majority_vote(dfs, tie_break=tie_break)
    out_mv = pd.DataFrame({"PassengerId": pid, "Transported": mv.astype(bool)})
    out_dir = SUB_DIR / "v10"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mv.to_csv(out_dir / "submission_v10_majority.csv", index=False)

    weights = [1.0 for _ in dfs]
    wv = weighted_vote(dfs, weights, threshold=0.5)
    out_wv = pd.DataFrame({"PassengerId": pid, "Transported": wv.astype(bool)})
    out_wv.to_csv(out_dir / "submission_v10_weighted.csv", index=False)

    print("Saved submission_v10_majority.csv and submission_v10_weighted.csv")


if __name__ == "__main__":
    main()
