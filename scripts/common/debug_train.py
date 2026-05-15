import subprocess
import os
from pathlib import Path

print("Starting debug_train.py...")
try:
    root = Path(__file__).resolve().parents[2]
    log_path = root / "results" / "logs" / "misc" / "train_debug.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(['python', '-u', str(root / 'scripts' / 'common' / 'train_model.py')], capture_output=True, text=True, timeout=600)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
    print("Finished train_model.py")
except subprocess.TimeoutExpired:
    print("train_model.py timed out")
except Exception as e:
    print(f"Error: {e}")
    root = Path(__file__).resolve().parents[2]
    log_path = root / "results" / "logs" / "misc" / "train_debug.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(str(e))
