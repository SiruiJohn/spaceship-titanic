import hashlib
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


def md5_labels(y: np.ndarray) -> str:
    return hashlib.md5(y.tobytes()).hexdigest()


@dataclass(frozen=True)
class GridConfig:
    w3: int
    w8: int
    w5: int
    thr: float


def main() -> None:
    sub_dir = ROOT / "results" / "submissions"
    p3 = sub_dir / "v3" / "submission_v3.csv"
    p8 = sub_dir / "v8" / "submission_v8.csv"
    p5 = sub_dir / "v5" / "submission_v5.csv"
    for p in [p3, p8, p5]:
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    d3 = load(str(p3))
    d8 = load(str(p8))
    d5 = load(str(p5))

    pid = d3["PassengerId"]
    if not pid.equals(d8["PassengerId"]) or not pid.equals(d5["PassengerId"]):
        raise SystemExit("PassengerId mismatch among v3/v8/v5.")

    y3 = d3["y"].to_numpy().astype(float)
    y8 = d8["y"].to_numpy().astype(float)
    y5 = d5["y"].to_numpy().astype(float)

    out_dir = ROOT / "results" / "blends" / "blends_v12"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grid: keep it moderate to avoid too many submissions.
    w3_list = [1, 2, 3, 4, 5]
    w8_list = [1, 2, 3, 4]
    w5_list = [1, 2, 3, 4]
    thr_list = [0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53]

    seen = {}
    records = []

    for w3 in w3_list:
        for w8 in w8_list:
            for w5 in w5_list:
                score = (w3 * y3 + w8 * y8 + w5 * y5) / (w3 + w8 + w5)
                for thr in thr_list:
                    pred = (score >= thr).astype(int)
                    h = md5_labels(pred)
                    if h in seen:
                        continue
                    seen[h] = True

                    diff = int((pred != y3.astype(int)).sum())
                    true_rate = float(pred.mean())

                    records.append(
                        {
                            "w3": w3,
                            "w8": w8,
                            "w5": w5,
                            "thr": thr,
                            "diff_rows_vs_v3": diff,
                            "diff_rate_vs_v3": diff / len(pred),
                            "true_rate": true_rate,
                            "md5": h[:10],
                        }
                    )

    res = pd.DataFrame(records)
    # Prefer small perturbations around v3 and reasonable class balance
    res = res.sort_values(
        ["diff_rows_vs_v3", "true_rate", "w3", "w8", "w5", "thr"],
        ascending=[True, True, False, True, True, True],
    ).reset_index(drop=True)

    # Save top N candidates (excluding exact v3)
    top_n = 30
    chosen = res[res["diff_rows_vs_v3"] > 0].head(top_n).copy()

    for i, row in chosen.iterrows():
        tag = f"w3{int(row.w3)}_w8{int(row.w8)}_w5{int(row.w5)}_thr{row.thr:.2f}_d{int(row.diff_rows_vs_v3)}"
        out_path = out_dir / f"submission_v12_{tag}.csv"
        score = (row.w3 * y3 + row.w8 * y8 + row.w5 * y5) / (row.w3 + row.w8 + row.w5)
        pred = (score >= float(row.thr)).astype(int)
        out_df = pd.DataFrame({"PassengerId": pid, "Transported": pred.astype(bool)})
        out_df.to_csv(out_path, index=False)

    report_path = out_dir / "v12_candidates.csv"
    chosen.to_csv(report_path, index=False)
    print(f"Wrote: {report_path}")
    print(f"Wrote {len(chosen)} submission_v12_*.csv files into {out_dir}")


if __name__ == "__main__":
    main()
