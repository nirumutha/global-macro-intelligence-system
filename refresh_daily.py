# ============================================================
# GMIS DAILY REFRESH SCRIPT
# Run this every morning before opening the dashboard
# Takes about 3-4 minutes to complete (FinBERT adds ~40 secs)
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
    '01_data_collection.py',     # Fresh market prices
    '02_macro_data.py',          # Fresh macro data from FRED
    '03_database_setup.py',      # Rebuild database
    '15_finbert_sentiment.py',   # FinBERT sentiment (replaces 09)
    '12_signal_engine.py',       # Recalculate signals
    '14_alert_engine.py',        # Check for alerts and notify
    '16_signal_v3.py',
    '17_historical_analog.py',
    '19_decision_engine.py',
    '20_entry_exit_engine.py',
    '21_signal_tracker.py',
    '22_model_health.py',
    '23_physical_basis.py',
    '24_economic_calendar.py',  # Upcoming high-impact events
    '25_mtf_signals.py',        # Multi-timeframe signal confirmation
    '26_institutional_flows.py',# FII/DII flows, COT, PCR
    '27_portfolio_risk.py',     # Kelly sizing, correlation, drawdown
    '28_correlation_regime.py', # Correlation shift detection & regime
    '29_external_intelligence.py', # Tiered RSS + FinBERT + theme extraction
    '30_liquidity_pulse.py',       # TGA, term premium, Fed BS, fiscal stance
    '31_yield_equity_beta.py',     # Rolling yield-equity beta, sensitivity regime
    '32_polarization_monitor.py',  # Mag-7 vs Russell 2000 breadth & concentration
    '33_geopolitical_premium.py',  # Oil & Gold risk premium vs fundamental fair value
    '34_narrative_engine.py',      # Market narrative scoring, momentum, asset impact map
    '35_dynamic_weights.py',       # Monthly IC-based weight adjustment for decision engine
    '36_execution_friction.py',    # Slippage, spread, backtest adjustment, trade timing
    '37_india_macro.py',           # India macro regime + FII integration + NIFTY adjustment
    '38_rv_ratio_overlay.py',      # Gold/Silver, Gold/SP500, Crude/Gold RV ratios & rotation signals
    '39_gift_gap_engine.py',       # Overnight NIFTY gap detection, fill-rate analysis, Breakout vs Trap
    '40_composite_liquidity.py',   # Cross-module liquidity composite score + asset-specific adjustments
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
