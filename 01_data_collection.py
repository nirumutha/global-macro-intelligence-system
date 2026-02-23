# ============================================================
# MODULE 1 — DATA COLLECTION
# Downloads 15 years of price data for all markets
# ============================================================

import yfinance as yf
import pandas as pd
import os

# ── Create data folder path ──────────────────────────────────
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(data_path, exist_ok=True)

# ── Define all assets to download ───────────────────────────
assets = {
    'NIFTY50':    '^NSEI',
    'SP500':      '^GSPC',
    'GOLD':       'GC=F',
    'SILVER':     'SI=F',
    'CRUDE_WTI':  'CL=F',
    'CRUDE_BRENT':'BZ=F',
    'USD_INR':    'USDINR=X',
    'DXY':        'DX-Y.NYB',
    'VIX_US':     '^VIX',
    'VIX_INDIA':  '^INDIAVIX',
}

START_DATE = '2010-01-01'
END_DATE   = '2024-12-31'

# ── Download each asset and save as CSV ─────────────────────
print("Starting data download...\n")

for name, ticker in assets.items():
    print(f"Downloading {name} ({ticker})...")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
        if df.empty:
            print(f"  WARNING: No data returned for {name}\n")
            continue
        filepath = os.path.join(data_path, f"{name}.csv")
        df.to_csv(filepath)
        print(f"  Saved {len(df)} rows → data/{name}.csv\n")
    except Exception as e:
        print(f"  ERROR downloading {name}: {e}\n")

print("=" * 50)
print("Download complete. Check your data/ folder.")
print("=" * 50)
