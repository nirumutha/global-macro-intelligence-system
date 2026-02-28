# ============================================================
# DAILY REFRESH SCRIPT
# Run this every morning before opening the dashboard
# Takes about 2-3 minutes to complete
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
        [sys.executable, os.path.join(BASE_PATH, script_name)],
        capture_output=False
    )
    if result.returncode == 0:
        print(f"✅ {script_name} completed successfully")
    else:
        print(f"❌ {script_name} failed with error code {result.returncode}")
    return result.returncode

print(f"\n{'='*50}")
print(f"GMIS DAILY REFRESH")
print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
print(f"{'='*50}")

scripts = [
    '01_data_collection.py',    # Fresh prices
    '02_macro_data.py',         # Fresh macro data
    '03_database_setup.py',     # Rebuild database
    '09_sentiment_engine.py',   # Live headlines
    '12_signal_engine.py',      # Recalculate signals
]

failed = []
for script in scripts:
    code = run_script(script)
    if code != 0:
        failed.append(script)

print(f"\n{'='*50}")
print(f"REFRESH COMPLETE — {datetime.now().strftime('%H:%M')}")
if failed:
    print(f"⚠️  Failed scripts: {', '.join(failed)}")
else:
    print(f"✅ All scripts completed successfully")
    print(f"✅ Dashboard is ready with today's data")
    print(f"\nRun: streamlit run dashboard.py")
print(f"{'='*50}\n")