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


def threshold_midpoints(weights: list[int]) -> list[float]:
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
    p_v3 = sub_dir / "v3" / "submission_v3.csv"
    p_v8 = sub_dir / "v8" / "submission_v8.csv"
    p_v5 = sub_dir / "v5" / "submission_v5.csv"
    p_v2 = sub_dir / "v2" / "submission_v2.csv"

    for p in [p_v3, p_v8, p_v5, p_v2]:
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    d3 = load(str(p_v3))
    d8 = load(str(p_v8))
    d5 = load(str(p_v5))
    d2 = load(str(p_v2))

    pid = d3["PassengerId"]
    for name, d in [("v8", d8), ("v5", d5), ("v2", d2)]:
        if not pid.equals(d["PassengerId"]):
            raise SystemExit(f"PassengerId mismatch: {name}")

    y3 = d3["y"].to_numpy().astype(int)
    y8 = d8["y"].to_numpy().astype(float)
    y5 = d5["y"].to_numpy().astype(float)
    y2 = d2["y"].to_numpy().astype(float)

    # Optional: compare to best-known public winner file (if present)
    best_path = None
    for cand in [
        ROOT / "results" / "blends" / "blends_v13" / "submission_v13_v3_v8_v5_v2_w3_1_1_1_thr0.416667_d13.csv",
        ROOT / "results" / "blends" / "blends" / "submission_v11_k3_lb_rank_v3_v8_v5.csv",
    ]:
        if cand.exists():
            best_path = cand
            break
    y_best = load(str(best_path))["y"].to_numpy().astype(int) if best_path else None

    out_dir = ROOT / "results" / "blends" / "blends_v14"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = np.array([3, 1, 1, 1], dtype=int)

    # Neighborhood weights around base (positive integers only)
    weight_set = set()
    weight_set.add(tuple(base.tolist()))
    for i in range(4):
        for delta in [-2, -1, 1, 2]:
            w = base.copy()
            w[i] += delta
            if (w > 0).all():
                weight_set.add(tuple(w.tolist()))

    # Extra v3-heavy variants (keep others small)
    for w3 in [4, 5, 6, 7, 8, 10]:
        weight_set.add((w3, 1, 1, 1))
        weight_set.add((w3, 1, 2, 1))
        weight_set.add((w3, 1, 1, 2))

    weights_list = sorted(weight_set)

    min_diff = 1
    max_diff = 40
    max_files = 50

    seen = set()
    records = []

    for w3, w8, w5, w2 in weights_list:
        weights = [w3, w8, w5, w2]
        total = sum(weights)
        score = (w3 * y3.astype(float) + w8 * y8 + w5 * y5 + w2 * y2) / total
        for thr in threshold_midpoints(weights):
            pred = (score >= thr).astype(int)
            h = md5_labels(pred)
            if h in seen:
                continue
            seen.add(h)

            diff_v3 = int((pred != y3).sum())
            if diff_v3 < min_diff or diff_v3 > max_diff:
                continue

            flip_0_to_1 = int(((y3 == 0) & (pred == 1)).sum())
            flip_1_to_0 = int(((y3 == 1) & (pred == 0)).sum())
            diff_best = int((pred != y_best).sum()) if y_best is not None else None

            records.append(
                {
                    "file": "",
                    "weights": f"{w3},{w8},{w5},{w2}",
                    "thr": float(thr),
                    "diff_rows_vs_v3": diff_v3,
                    "diff_rows_vs_best": diff_best,
                    "true_rate": float(pred.mean()),
                    "flip_0_to_1": flip_0_to_1,
                    "flip_1_to_0": flip_1_to_0,
                    "md5": h[:10],
                }
            )

    if not records:
        raise SystemExit("No candidates found. Try increasing max_diff or expanding neighborhood.")

    df = pd.DataFrame(records)
    sort_cols = ["diff_rows_vs_v3", "diff_rows_vs_best", "true_rate", "weights", "thr"]
    # If diff_rows_vs_best is None for all, drop it from sorting
    if df["diff_rows_vs_best"].isna().all():
        sort_cols = ["diff_rows_vs_v3", "true_rate", "weights", "thr"]
    df = df.sort_values(sort_cols, ascending=[True] * len(sort_cols)).reset_index(drop=True)
    df = df.head(max_files).copy()

    for i, row in df.iterrows():
        w3, w8, w5, w2 = [int(x) for x in row["weights"].split(",")]
        weights = [w3, w8, w5, w2]
        total = sum(weights)
        score = (w3 * y3.astype(float) + w8 * y8 + w5 * y5 + w2 * y2) / total
        pred = (score >= float(row["thr"])).astype(int)
        tag = f"w{w3}_{w8}_{w5}_{w2}_thr{float(row['thr']):.6f}_d{int(row['diff_rows_vs_v3'])}"
        out_path = out_dir / f"submission_v14_{tag}.csv"
        out_df = pd.DataFrame({"PassengerId": pid, "Transported": pred.astype(bool)})
        out_df.to_csv(out_path, index=False)
        df.loc[i, "file"] = os.path.basename(out_path)

    report_path = out_dir / "v14_candidates.csv"
    df.to_csv(report_path, index=False)
    print(f"Wrote: {report_path}")
    print(f"Wrote {len(df)} submission_v14_*.csv files into {out_dir}")


if __name__ == "__main__":
    main()
