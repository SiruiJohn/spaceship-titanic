import os
import pandas as pd
import numpy as np
from pathlib import Path


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


def save(pid: pd.Series, pred: np.ndarray, out_path: str) -> None:
    out = pd.DataFrame({"PassengerId": pid, "Transported": pred.astype(bool)})
    out.to_csv(out_path, index=False)


def summarize(name: str, pred: np.ndarray, y3: np.ndarray, y_best: np.ndarray | None) -> dict:
    diff_v3 = int((pred != y3).sum())
    flip_0_to_1 = int(((y3 == 0) & (pred == 1)).sum())
    flip_1_to_0 = int(((y3 == 1) & (pred == 0)).sum())
    diff_best = int((pred != y_best).sum()) if y_best is not None else None
    return {
        "name": name,
        "diff_rows_vs_v3": diff_v3,
        "flip_0_to_1": flip_0_to_1,
        "flip_1_to_0": flip_1_to_0,
        "true_rate": float(pred.mean()),
        "diff_rows_vs_best": diff_best,
    }


def main() -> None:
    sub_dir = ROOT / "results" / "submissions"
    paths = {
        "v3": sub_dir / "v3" / "submission_v3.csv",
        "v8": sub_dir / "v8" / "submission_v8.csv",
        "v5": sub_dir / "v5" / "submission_v5.csv",
        "v2": sub_dir / "v2" / "submission_v2.csv",
        "v6": sub_dir / "v6" / "submission_v6.csv",
    }

    required = ["v3", "v8", "v5", "v2"]
    for k in required:
        if not paths[k].exists():
            raise SystemExit(f"Missing required file: {paths[k]}")

    d = {k: load(str(p)) for k, p in paths.items() if p.exists()}
    pid = d["v3"]["PassengerId"]
    for k, df in d.items():
        if not pid.equals(df["PassengerId"]):
            raise SystemExit(f"PassengerId mismatch: {k}")

    y3 = d["v3"]["y"].to_numpy().astype(int)
    y8 = d["v8"]["y"].to_numpy().astype(int)
    y5 = d["v5"]["y"].to_numpy().astype(int)
    y2 = d["v2"]["y"].to_numpy().astype(int)
    y6 = d["v6"]["y"].to_numpy().astype(int) if "v6" in d else None

    best_path = ROOT / "results" / "blends" / "blends_v13" / "submission_v13_v3_v8_v5_v2_w3_1_1_1_thr0.416667_d13.csv"
    y_best = load(str(best_path))["y"].to_numpy().astype(int) if best_path.exists() else None

    out_dir = ROOT / "results" / "rules" / "rules_v17"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = {}

    # Baseline: keep v3
    base = y3.copy()

    # R1: v8=v5=v2=1 and v3=0 (same as the current best A)
    r1 = base.copy()
    r1[(y3 == 0) & (y8 == 1) & (y5 == 1) & (y2 == 1)] = 1
    preds["r1_v8v5v2_all1"] = r1

    # R2: v8=v5=1 and v2=1 (equivalent to r1 but kept explicit)
    r2 = base.copy()
    r2[(y3 == 0) & (y8 == 1) & (y5 == 1) & (y2 == 1)] = 1
    preds["r2_v8v5_1_and_v2_1"] = r2

    # R3: v8=v5=1 (drop v2 requirement) - larger flip set
    r3 = base.copy()
    r3[(y3 == 0) & (y8 == 1) & (y5 == 1)] = 1
    preds["r3_v8v5_both1"] = r3

    # R4: majority of {v8,v5,v2} says 1 (>=2) while v3=0
    r4 = base.copy()
    r4[(y3 == 0) & ((y8 + y5 + y2) >= 2)] = 1
    preds["r4_majority_v8v5v2_1"] = r4

    # R5: v8=1 and v2=1 (drop v5 requirement)
    r5 = base.copy()
    r5[(y3 == 0) & (y8 == 1) & (y2 == 1)] = 1
    preds["r5_v8v2_both1"] = r5

    # R6: v5=1 and v2=1 (drop v8 requirement)
    r6 = base.copy()
    r6[(y3 == 0) & (y5 == 1) & (y2 == 1)] = 1
    preds["r6_v5v2_both1"] = r6

    if y6 is not None:
        # R7: v8=v5=v2=v6=1 while v3=0 (strongest consensus)
        r7 = base.copy()
        r7[(y3 == 0) & (y8 == 1) & (y5 == 1) & (y2 == 1) & (y6 == 1)] = 1
        preds["r7_v8v5v2v6_all1"] = r7

        # R8: at least 3 of {v8,v5,v2,v6} are 1 while v3=0
        r8 = base.copy()
        r8[(y3 == 0) & ((y8 + y5 + y2 + y6) >= 3)] = 1
        preds["r8_majority_v8v5v2v6_1"] = r8

        # R9: v6 acts as tie-breaker for {v8,v5,v2} majority
        r9 = base.copy()
        s3 = y8 + y5 + y2
        cond = (y3 == 0) & (s3 == 1) & (y6 == 1)
        r9[cond] = 1
        preds["r9_when_v8v5v2_sum1_and_v6_1"] = r9

    rows = []
    for key, pred in preds.items():
        out_name = f"submission_v17_{key}.csv"
        out_path = out_dir / out_name
        save(pid, pred, str(out_path))
        rows.append(summarize(out_name, pred, y3, y_best))

    rep = pd.DataFrame(rows).sort_values(
        ["diff_rows_vs_v3", "diff_rows_vs_best", "flip_0_to_1", "name"], ascending=[True, True, True, True], na_position="last"
    )
    rep_path = out_dir / "v17_rules_report.csv"
    rep.to_csv(rep_path, index=False)

    print(f"Wrote: {rep_path}")
    print(f"Wrote {len(rows)} submission_v17_*.csv files into {out_dir}")


if __name__ == "__main__":
    main()
