import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


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


def weighted_vote(mat: np.ndarray, weights: np.ndarray, thr: float = 0.5) -> np.ndarray:
    score = (mat * weights.reshape(1, -1)).sum(axis=1) / weights.sum()
    return (score >= thr).astype(int)


@dataclass(frozen=True)
class Candidate:
    name: str
    path: str
    lb_score: float


def main() -> None:
    sub_dir = ROOT / "results" / "submissions"
    pool = [
        Candidate("v3", str(sub_dir / "v3" / "submission_v3.csv"), 0.80851),
        Candidate("v8", str(sub_dir / "v8" / "submission_v8.csv"), 0.80687),
        Candidate("v6", str(sub_dir / "v6" / "submission_v6.csv"), 0.80547),
        Candidate("v5", str(sub_dir / "v5" / "submission_v5.csv"), 0.80547),
        Candidate("v2", str(sub_dir / "v2" / "submission_v2.csv"), 0.80523),
        Candidate("v7", str(sub_dir / "v7" / "submission_v7.csv"), 0.80430),
        Candidate("base", str(sub_dir / "misc" / "submission.csv"), 0.79728),
        Candidate("v9", str(sub_dir / "v9" / "submission_v9.csv"), 0.80079),
    ]
    pool = [c for c in pool if os.path.exists(c.path)]
    if len(pool) < 3:
        raise SystemExit("Need at least 3 existing submissions to search blends.")

    dfs = {c.name: load(c.path) for c in pool}
    pid = next(iter(dfs.values()))["PassengerId"]
    for k, d in dfs.items():
        if not pid.equals(d["PassengerId"]):
            raise SystemExit(f"PassengerId mismatch: {k}")

    ref = dfs["v3"] if "v3" in dfs else next(iter(dfs.values()))
    refy = ref["y"].to_numpy()

    out_dir = ROOT / "results" / "blends" / "blends"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [c for c in pool if c.name != "base"]
    results = []

    subset_sizes = [3, 4, 5]
    weight_modes = ["equal", "lb_rank", "lb_linear"]

    for k in subset_sizes:
        for subset in itertools.combinations(candidates, k):
            names = [c.name for c in subset]
            mat = np.vstack([dfs[n]["y"].to_numpy() for n in names]).T.astype(float)
            lbs = np.array([c.lb_score for c in subset], dtype=float)

            for mode in weight_modes:
                if mode == "equal":
                    w = np.ones(k, dtype=float)
                elif mode == "lb_rank":
                    order = np.argsort(-lbs, kind="mergesort")
                    ranks = np.empty(k, dtype=float)
                    ranks[order] = np.arange(k, 0, -1, dtype=float)
                    w = ranks
                else:
                    base = np.min(lbs)
                    w = (lbs - base) + 1e-6
                    w = w / np.mean(w)

                pred = weighted_vote(mat, w, thr=0.5)
                true_rate = float(pred.mean())
                diff_rows = int((pred != refy).sum()) if len(refy) == len(pred) else None
                diff_rate = float(diff_rows / len(pred)) if diff_rows is not None else None

                tag = f"k{k}_{mode}_" + "_".join(names)
                out_path = out_dir / f"submission_v11_{tag}.csv"
                out_df = pd.DataFrame({"PassengerId": pid, "Transported": pred.astype(bool)})
                out_df.to_csv(out_path, index=False)

                results.append(
                    {
                        "file": out_path.name,
                        "mode": mode,
                        "members": ",".join(names),
                        "true_rate": true_rate,
                        "diff_rows_vs_v3": diff_rows,
                        "diff_rate_vs_v3": diff_rate,
                        "members_lb": ",".join(f"{c.name}:{c.lb_score:.5f}" for c in subset),
                    }
                )

    res = pd.DataFrame(results)
    if "diff_rows_vs_v3" in res.columns:
        res = res.sort_values(
            ["diff_rows_vs_v3", "true_rate", "members"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    report = out_dir / "blend_candidates_v11.csv"
    res.to_csv(report, index=False)
    print(f"Wrote: {report}")


if __name__ == "__main__":
    main()
