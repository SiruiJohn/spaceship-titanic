import argparse
import sys

import scripts.common.reproduce_notebook_pipeline as pipeline


def build_argv(args: argparse.Namespace) -> list[str]:
    argv = ["reproduce_notebook_pipeline.py"]
    if args.fast:
        argv.append("--fast")
    if not args.with_eda:
        argv.append("--skip-eda")
    if not args.rebuild_features:
        argv.append("--reuse-features")
    if args.n_splits is not None:
        argv.extend(["--n-splits", str(args.n_splits)])
    if args.n_estimators is not None:
        argv.extend(["--n-estimators", str(args.n_estimators)])
    if args.optuna_trials is not None:
        argv.extend(["--optuna-trials", str(args.optuna_trials)])
    for path in args.or_submission:
        argv.extend(["--or-submission", path])
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Spaceship Titanic notebook reproduction with notebook-like defaults."
    )
    parser.add_argument("--fast", action="store_true", help="Run a lighter validation version.")
    parser.add_argument(
        "--with-eda",
        action="store_true",
        help="Generate EDA artifacts instead of using the default skip-EDA mode.",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Ignore cached features and rebuild the full feature pipeline instead of reusing cache.",
    )
    parser.add_argument("--n-splits", type=int, default=10, help="CV fold count. Defaults to 10.")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=9999,
        help="Max tree rounds before early stopping. Defaults to 9999.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=200,
        help="Optuna blending trials. Defaults to 200 for a practical full run.",
    )
    parser.add_argument(
        "--or-submission",
        action="append",
        default=[],
        help="Optional extra submission path for OR-gate experiments.",
    )
    args = parser.parse_args()

    sys.argv = build_argv(args)
    pipeline.main()


if __name__ == "__main__":
    main()
