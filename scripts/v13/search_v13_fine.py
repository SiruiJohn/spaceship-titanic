import hashlib
import itertools
import os
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


def md5_labels(y: np.ndarray) -> str:
    return hashlib.md5(y.tobytes()).hexdigest()


def threshold_grid_from_weights(weights: list[int]) -> list[float]:
    total = sum(weights)
    sums = set()
    for r in range(len(weights) + 1):
        for idxs in itertools.combinations(range(len(weights)), r):
            sums.add(sum(weights[i] for i in idxs))
    vals = sorted({s / total for s in sums})
    thrs = []
    for a, b in zip(vals, vals[1:]):
        thrs.append((a + b) / 2.0)
    return thrs


def main() -> None:
    sub_dir = ROOT / "results" / "submissions"
    pool = {
        "v3": sub_dir / "v3" / "submission_v3.csv",
        "v8": sub_dir / "v8" / "submission_v8.csv",
        "v5": sub_dir / "v5" / "submission_v5.csv",
        "v6": sub_dir / "v6" / "submission_v6.csv",
        "v2": sub_dir / "v2" / "submission_v2.csv",
        "v7": sub_dir / "v7" / "submission_v7.csv",
    }
    pool = {k: v for k, v in pool.items() if v.exists()}
    if "v3" not in pool:
        raise SystemExit("submission_v3.csv is required for v13 search.")

    dfs = {k: load(str(p)) for k, p in pool.items()}
    pid = dfs["v3"]["PassengerId"]
    for k, d in dfs.items():
        if not pid.equals(d["PassengerId"]):
            raise SystemExit(f"PassengerId mismatch: {k}")

    y = {k: dfs[k]["y"].to_numpy().astype(float) for k in dfs.keys()}
    y3 = y["v3"].astype(int)

    out_dir = ROOT / "results" / "blends" / "blends_v13"
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = []
    if all(k in y for k in ["v3", "v8", "v5"]):
        combos.append(["v3", "v8", "v5"])
    if all(k in y for k in ["v3", "v6", "v2"]):
        combos.append(["v3", "v6", "v2"])
    if all(k in y for k in ["v3", "v8", "v5", "v6"]):
        combos.append(["v3", "v8", "v5", "v6"])
    if all(k in y for k in ["v3", "v8", "v5", "v2"]):
        combos.append(["v3", "v8", "v5", "v2"])

    w3_list = [2, 3, 4, 5, 6, 8, 10]
    w_other_list = [1, 2, 3, 4, 5]

    min_diff = 5
    max_diff = 80
    max_files = 40

    seen = set()
    records = []

    for members in combos:
        k = len(members)
        for w3 in w3_list:
            for others in itertools.product(w_other_list, repeat=k - 1):
                weights = [w3] + list(others)
                total = sum(weights)

                score = np.zeros_like(y3, dtype=float)
                for name, wv in zip(members, weights):
                    score += wv * y[name]
                score /= total

                thrs = threshold_grid_from_weights(weights)
                for thr in thrs:
                    pred = (score >= thr).astype(int)
                    h = md5_labels(pred)
                    if h in seen:
                        continue
                    seen.add(h)

                    diff = int((pred != y3).sum())
                    if diff < min_diff or diff > max_diff:
                        continue

                    true_rate = float(pred.mean())
                    flip_0_to_1 = int(((y3 == 0) & (pred == 1)).sum())
                    flip_1_to_0 = int(((y3 == 1) & (pred == 0)).sum())

                    records.append(
                        {
                            "file": "",
                            "members": ",".join(members),
                            "weights": ",".join(str(x) for x in weights),
                            "thr": float(thr),
                            "diff_rows_vs_v3": diff,
                            "true_rate": true_rate,
                            "flip_0_to_1": flip_0_to_1,
                            "flip_1_to_0": flip_1_to_0,
                            "md5": h[:10],
                        }
                    )

    if not records:
        raise SystemExit("No candidates found. Try widening diff range or weights.")

    df = pd.DataFrame(records)
    df = df.sort_values(
        ["diff_rows_vs_v3", "members", "true_rate", "weights", "thr"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    df = df.head(max_files).copy()
    for i, row in df.iterrows():
        members_tag = row["members"].replace(",", "_")
        weights_tag = row["weights"].replace(",", "_")
        tag = f"{members_tag}_w{weights_tag}_thr{row.thr:.6f}_d{int(row.diff_rows_vs_v3)}"
        out_path = out_dir / f"submission_v13_{tag}.csv"
        members = row["members"].split(",")
        weights = [int(x) for x in row["weights"].split(",")]
        total = sum(weights)
        score = np.zeros_like(y3, dtype=float)
        for name, wv in zip(members, weights):
            score += wv * y[name]
        score /= total
        pred = (score >= float(row.thr)).astype(int)
        out_df = pd.DataFrame({"PassengerId": pid, "Transported": pred.astype(bool)})
        out_df.to_csv(out_path, index=False)
        df.loc[i, "file"] = os.path.basename(out_path)

    report_path = out_dir / "v13_candidates.csv"
    df.to_csv(report_path, index=False)
    print(f"Wrote: {report_path}")
    print(f"Wrote {len(df)} submission_v13_*.csv files into {out_dir}")


if __name__ == "__main__":
    main()
