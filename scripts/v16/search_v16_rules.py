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


def save_submission(pid: pd.Series, pred: np.ndarray, out_path: str) -> None:
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
    }
    for k, p in paths.items():
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    d3 = load(str(paths["v3"]))
    d8 = load(str(paths["v8"]))
    d5 = load(str(paths["v5"]))
    d2 = load(str(paths["v2"]))

    pid = d3["PassengerId"]
    for name, d in [("v8", d8), ("v5", d5), ("v2", d2)]:
        if not pid.equals(d["PassengerId"]):
            raise SystemExit(f"PassengerId mismatch: {name}")

    y3 = d3["y"].to_numpy().astype(int)
    y8 = d8["y"].to_numpy().astype(int)
    y5 = d5["y"].to_numpy().astype(int)
    y2 = d2["y"].to_numpy().astype(int)

    best_path = ROOT / "results" / "blends" / "blends_v13" / "submission_v13_v3_v8_v5_v2_w3_1_1_1_thr0.416667_d13.csv"
    y_best = load(str(best_path))["y"].to_numpy().astype(int) if best_path.exists() else None

    out_dir = ROOT / "results" / "rules" / "rules_v16"
    out_dir.mkdir(parents=True, exist_ok=True)

    others_sum = y8 + y5 + y2
    others_unanimous_1 = others_sum == 3
    others_unanimous_0 = others_sum == 0
    others_majority_1 = others_sum >= 2
    others_majority_0 = others_sum <= 1

    preds = {}

    # Rule A: current best logic (derived from w3=3,w8=1,w5=1,w2=1, thr=0.416667)
    pred_a = y3.copy()
    pred_a[(y3 == 0) & others_unanimous_1] = 1
    preds["rule_A_unanimous1_only"] = pred_a

    # Rule B: symmetric correction (flip 1->0 only when all other models say 0)
    pred_b = y3.copy()
    pred_b[(y3 == 1) & others_unanimous_0] = 0
    preds["rule_B_unanimous0_only"] = pred_b

    # Rule C: apply both unanimous corrections
    pred_c = y3.copy()
    pred_c[(y3 == 0) & others_unanimous_1] = 1
    pred_c[(y3 == 1) & others_unanimous_0] = 0
    preds["rule_C_unanimous_both"] = pred_c

    # Rule D: flip 0->1 when majority of others say 1
    pred_d = y3.copy()
    pred_d[(y3 == 0) & others_majority_1] = 1
    preds["rule_D_majority1_only"] = pred_d

    # Rule E: flip 1->0 when majority of others say 0
    pred_e = y3.copy()
    pred_e[(y3 == 1) & others_majority_0] = 0
    preds["rule_E_majority0_only"] = pred_e

    # Rule F: majority corrections in both directions
    pred_f = y3.copy()
    pred_f[(y3 == 0) & others_majority_1] = 1
    pred_f[(y3 == 1) & others_majority_0] = 0
    preds["rule_F_majority_both"] = pred_f

    # Rule G: "trust v8 & v5" when they agree and v3 disagrees (tie-break by v8/v5 consensus)
    pred_g = y3.copy()
    agree_85 = y8 == y5
    pred_g[(agree_85) & (y3 != y8)] = y8[(agree_85) & (y3 != y8)]
    preds["rule_G_follow_v8v5_agree"] = pred_g

    # Rule H: "trust v8 & v5 & v2" when they agree (strong consensus)
    pred_h = y3.copy()
    agree_all = (y8 == y5) & (y5 == y2)
    pred_h[(agree_all) & (y3 != y8)] = y8[(agree_all) & (y3 != y8)]
    preds["rule_H_follow_all3_agree"] = pred_h

    rows = []
    for name, pred in preds.items():
        out_path = out_dir / f"submission_v16_{name}.csv"
        save_submission(pid, pred, str(out_path))
        rows.append(summarize(out_path.name, pred, y3, y_best))

    rep = pd.DataFrame(rows).sort_values(["diff_rows_vs_best", "diff_rows_vs_v3", "name"], na_position="last")
    rep_path = out_dir / "v16_rules_report.csv"
    rep.to_csv(rep_path, index=False)

    print(f"Wrote: {rep_path}")
    print(f"Wrote {len(rows)} submission_v16_*.csv files into {out_dir}")


if __name__ == "__main__":
    main()
