# ============================================================
# MODULE 1 — MACRO DATA COLLECTION
# Downloads US macroeconomic data from FRED
# ============================================================

import pandas_datareader.data as pdr
import pandas as pd
import os
from datetime import datetime

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(data_path, exist_ok=True)

START_DATE = '2010-01-01'
END_DATE   = '2024-12-31'

# ── Define FRED series to download ───────────────────────────
fred_series = {
    'US_10Y_YIELD':     'DGS10',      # US 10-Year Treasury Yield
    'US_2Y_YIELD':      'DGS2',       # US 2-Year Treasury Yield
    'US_FED_RATE':      'FEDFUNDS',   # Federal Funds Rate
    'US_CPI':           'CPIAUCSL',   # US Consumer Price Index
    'US_GDP':           'GDP',        # US GDP Quarterly
    'US_PAYROLLS':      'PAYEMS',     # Nonfarm Payrolls
    'US_PCE':           'PCE',        # Personal Consumption Expenditures
    'US_UNEMPLOYMENT':  'UNRATE',     # Unemployment Rate
}

print("Downloading macroeconomic data from FRED...\n")

for name, series_id in fred_series.items():
    print(f"Downloading {name} ({series_id})...")
    try:
        df = pdr.DataReader(series_id, 'fred', START_DATE, END_DATE)
        filepath = os.path.join(data_path, f"{name}.csv")
        df.to_csv(filepath)
        print(f"  Saved {len(df)} rows → data/{name}.csv\n")
    except Exception as e:
        print(f"  ERROR downloading {name}: {e}\n")

print("=" * 50)
print("Macro data download complete.")
print("=" * 50)