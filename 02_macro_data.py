# ============================================================
# MODULE 1 — MACRO DATA COLLECTION
# Downloads US macroeconomic data from FRED
# ============================================================

import pandas_datareader.data as pdr
import pandas as pd
import os
import time
from datetime import datetime

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(data_path, exist_ok=True)

START_DATE = '2010-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

MAX_RETRIES = 3       # Number of attempts per series
RETRY_DELAY = 10      # Seconds to wait between retries
TIMEOUT     = 60      # Seconds before giving up on a request

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

failed = []

for name, series_id in fred_series.items():
    print(f"Downloading {name} ({series_id})...")
    success = False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Pass timeout via requests session
            import requests
            session = requests.Session()
            session.request = lambda method, url, **kwargs: requests.Session.request(
                session, method, url, timeout=TIMEOUT, **kwargs
            )

            df = pdr.DataReader(series_id, 'fred', START_DATE, END_DATE)
            filepath = os.path.join(data_path, f"{name}.csv")
            df.to_csv(filepath)
            print(f"  Saved {len(df)} rows → data/{name}.csv\n")
            success = True
            break

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  ⚠️  Attempt {attempt} failed: {e}")
                print(f"  Retrying in {RETRY_DELAY}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ❌ All {MAX_RETRIES} attempts failed for {name}: {e}\n")
                failed.append(name)

print("=" * 50)
print("Macro data download complete.")
if failed:
    print(f"⚠️  Failed series: {', '.join(failed)}")
    print("   These will use previously saved CSVs in the database step.")
else:
    print("✅ All series downloaded successfully.")
print("=" * 50)