import glob
import hashlib
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PATTERN = str(ROOT / "results" / "submissions" / "**" / "submission*.csv")


def load_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "PassengerId" not in df.columns or "Transported" not in df.columns:
        raise ValueError(f"Invalid submission format: {path}")

    s = df["Transported"]
    if s.dtype == bool:
        y = s.astype(int)
    elif s.dtype == object:
        y = s.map({"True": 1, "False": 0, True: 1, False: 0})
        if y.isna().any():
            raise ValueError(f"Unrecognized Transported values in {path}")
        y = y.astype(int)
    else:
        y = s.astype(int)

    out = pd.DataFrame({"PassengerId": df["PassengerId"].astype(str), "y": y})
    out = out.sort_values("PassengerId").reset_index(drop=True)
    return out


def md5_of_labels(y: pd.Series) -> str:
    b = y.to_numpy().tobytes()
    return hashlib.md5(b).hexdigest()

def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = df.values.tolist()
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in rows)
    return "\n".join([header, sep, body]) + "\n"

def main() -> None:
    files = sorted(glob.glob(PATTERN, recursive=True))
    if not files:
        raise SystemExit("No submission*.csv found.")

    loaded: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    groups: dict[str, list[str]] = {}

    for f in files:
        name = os.path.basename(f)
        df = load_submission(f)
        loaded[name] = df
        h = md5_of_labels(df["y"])
        groups.setdefault(h, []).append(name)
        rows.append(
            {
                "file": name,
                "rows": int(len(df)),
                "true_count": int(df["y"].sum()),
                "true_rate": float(df["y"].mean()),
                "md5": h[:10],
            }
        )

    summary = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)

    ref = "submission_v3.csv" if "submission_v3.csv" in loaded else summary.loc[0, "file"]
    refdf = loaded[ref]

    diffs_vs_ref = []
    for name, df in sorted(loaded.items()):
        m = refdf.merge(df, on="PassengerId", suffixes=("_ref", "_cur"))
        diff = int((m["y_ref"] != m["y_cur"]).sum())
        diffs_vs_ref.append(
            {
                "file": name,
                "diff_rows_vs_ref": diff,
                "diff_rate_vs_ref": diff / len(m),
            }
        )
    diffs_vs_ref = pd.DataFrame(diffs_vs_ref).sort_values("file").reset_index(drop=True)

    key_cands = [
        n
        for n in [
            "submission.csv",
            "submission_v2.csv",
            "submission_v3.csv",
            "submission_v5.csv",
            "submission_v6.csv",
            "submission_v7.csv",
            "submission_v8.csv",
            "submission_v9.csv",
        ]
        if n in loaded
    ]

    pairwise = []
    for i in range(len(key_cands)):
        for j in range(i + 1, len(key_cands)):
            a, b = key_cands[i], key_cands[j]
            ma = loaded[a]
            mb = loaded[b]
            m = ma.merge(mb, on="PassengerId", suffixes=("_a", "_b"))
            diff = int((m["y_a"] != m["y_b"]).sum())
            pairwise.append(
                {
                    "a": a,
                    "b": b,
                    "diff_rows": diff,
                    "diff_rate": diff / len(m),
                }
            )
    pairwise = pd.DataFrame(pairwise).sort_values(["diff_rows", "a", "b"], ascending=[False, True, True])

    out_md = ROOT / "docs" / "submissions" / "submission_comparison.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Submission Comparison\n\n")
        f.write("## Per-file Summary\n\n")
        f.write(df_to_md(summary))
        f.write("\n\n")

        f.write("## Identical Prediction Groups\n\n")
        any_group = False
        for h, names in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
            if len(names) > 1:
                any_group = True
                f.write(f"- {h[:10]}: {', '.join(sorted(names))}\n")
        if not any_group:
            f.write("- (None)\n")
        f.write("\n")

        f.write(f"## Diffs vs Reference ({ref})\n\n")
        f.write(df_to_md(diffs_vs_ref))
        f.write("\n\n")

        f.write("## Pairwise Diffs (Key Candidates)\n\n")
        if len(pairwise) == 0:
            f.write("- (Not enough submissions)\n")
        else:
            f.write(df_to_md(pairwise))
        f.write("\n")

    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
