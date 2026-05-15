import subprocess, sys, argparse, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODULAR = Path(__file__).parent

MODELS = {
    'et':   {'script': 'train_et.py',  'desc': 'ExtraTrees (600 estimators)'},
    'hgb':  {'script': 'train_hgb.py', 'desc': 'HistGradientBoosting (400 iter)'},
    'xgb':  {'script': 'train_xgb.py', 'desc': 'XGBoost (400 rounds)'},
    'lgb':  {'script': 'train_lgb.py', 'desc': 'LightGBM (Optuna tuned)'},
    'cat':  {'script': 'train_cat.py', 'desc': 'CatBoost (500 iter)'},
    'nn':   {'script': 'train_nn.py',  'desc': 'MLP Neural Network (256-128-64)'},
}

DEFAULT_MODELS = ['et', 'hgb', 'xgb', 'lgb', 'cat', 'nn']


def run_script(script_name, extra_args=None):
    script_path = MODULAR / script_name
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    print(f'\n{"="*60}')
    print(f'  Running: {" ".join(cmd)}')
    print(f'{"="*60}')
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f'[ERROR] {script_name} failed with exit code {result.returncode}')
        return False
    return True


def cmd_prep():
    run_script('data_prep.py')


def cmd_train(models):
    for m in models:
        if m not in MODELS:
            print(f'[ERROR] Unknown model: {m}')
            continue
        t0 = time.time()
        success = run_script(MODELS[m]['script'])
        elapsed = time.time() - t0
        if success:
            print(f'[OK] {m} finished in {elapsed:.0f}s')
        else:
            print(f'[FAIL] {m} failed')


def cmd_blend():
    run_script('blend.py')


def cmd_full():
    steps = [
        ('Data Preparation', cmd_prep),
        ('Model Training',   lambda: cmd_train(DEFAULT_MODELS)),
        ('Blending',         cmd_blend),
    ]
    t0 = time.time()
    for name, fn in steps:
        print(f'\n{"#"*60}')
        print(f'  STEP: {name}')
        print(f'{"#"*60}')
        fn()
    total = time.time() - t0
    print(f'\n[OK] Full pipeline complete in {total:.0f}s ({total/60:.1f} min)')


def main():
    parser = argparse.ArgumentParser(description='Spaceship Titanic Modular Pipeline')
    parser.add_argument('action', nargs='?', default='full',
                        choices=['full', 'prep', 'train', 'blend', 'list'],
                        help='Action to run (default: full)')
    parser.add_argument('--models', '-m', nargs='+', default=DEFAULT_MODELS,
                        choices=list(MODELS.keys()) + ['all'],
                        help=f'Models to train (default: {" ".join(DEFAULT_MODELS)})')
    args = parser.parse_args()

    if args.action == 'list':
        print('Available models:')
        for k, v in MODELS.items():
            print(f'  {k:6s} -- {v["desc"]}')
        print(f'\nActions:')
        print(f'  prep   - Run data preparation only')
        print(f'  train  - Train models (use --models to select)')
        print(f'  blend  - Run ensemble blending + submission')
        print(f'  full   - Run everything')
        return

    if args.action == 'prep':
        cmd_prep()
        return

    if args.action == 'train':
        models = DEFAULT_MODELS if 'all' in args.models else args.models
        cmd_train(models)
        return

    if args.action == 'blend':
        cmd_blend()
        return

    if args.action == 'full':
        cmd_full()
        return


if __name__ == '__main__':
    main()
