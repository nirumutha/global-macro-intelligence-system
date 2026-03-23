# ============================================================
# GMIS WEEKLY REFRESH
# Run once per week — heavier computation
# ============================================================

import subprocess
import sys
import os
from datetime import datetime

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE_PATH, script_name)],
        capture_output=False
    )
    if result.returncode == 0:
        print(f"✅ {script_name} completed successfully")
    else:
        print(f"❌ {script_name} failed")
    return result.returncode

print(f"\n{'='*50}")
print(f"GMIS WEEKLY REFRESH")
print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
print(f"{'='*50}")

scripts = [
    '18_walkforward_backtest.py',  # ~90 seconds
    '21_signal_tracker.py --report',
]

for script in scripts:
    run_script(script)

print(f"\n{'='*50}")
print(f"WEEKLY REFRESH COMPLETE")
print(f"{'='*50}\n")

