# ============================================================
# MODULE 1 — DATABASE SETUP (Updated for yfinance format)
# Loads all CSV files into a clean SQLite database
# ============================================================

import pandas as pd
import sqlite3
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')
DB_PATH   = os.path.join(DATA_PATH, 'macro_system.db')

conn = sqlite3.connect(DB_PATH)
print(f"Database connected at: data/macro_system.db\n")

# ── Market price files ────────────────────────────────────────
price_files = [
    'NIFTY50', 'SP500', 'GOLD', 'SILVER',
    'CRUDE_WTI', 'CRUDE_BRENT', 'USD_INR',
    'DXY', 'VIX_US', 'VIX_INDIA'
]

print("Loading market price data into database...")
for name in price_files:
    filepath = os.path.join(DATA_PATH, f"{name}.csv")
    if not os.path.exists(filepath):
        print(f"  WARNING: {name}.csv not found, skipping")
        continue
    try:
        # Read the raw file first to detect format
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()

        # If first line contains 'Ticker', skip first 2 rows
        if 'Ticker' in first_line or 'Price' in first_line:
            df = pd.read_csv(filepath, skiprows=2)
        else:
            df = pd.read_csv(filepath)

        # Rename first column to Date
        df = df.rename(columns={df.columns[0]: 'Date'})

        # Drop rows where Date is not a valid date
        df = df[pd.to_datetime(df['Date'], errors='coerce').notna()]
        df['Date'] = pd.to_datetime(df['Date']).astype(str)
        df = df.sort_values('Date').reset_index(drop=True)

        # Drop empty columns
        df = df.dropna(axis=1, how='all')

        df.to_sql(name, conn, if_exists='replace', index=False)
        print(f"  {name}: {len(df)} rows loaded ✓")
    except Exception as e:
        print(f"  ERROR loading {name}: {e}")

# ── Macro data files ──────────────────────────────────────────
macro_files = [
    'US_10Y_YIELD', 'US_2Y_YIELD', 'US_FED_RATE',
    'US_CPI', 'US_GDP', 'US_PAYROLLS',
    'US_PCE', 'US_UNEMPLOYMENT'
]

print("\nLoading macro economic data into database...")
for name in macro_files:
    filepath = os.path.join(DATA_PATH, f"{name}.csv")
    if not os.path.exists(filepath):
        print(f"  WARNING: {name}.csv not found, skipping")
        continue
    try:
        df = pd.read_csv(filepath)
        df = df.rename(columns={df.columns[0]: 'Date'})
        df['Date'] = pd.to_datetime(df['Date']).astype(str)
        df = df.sort_values('Date').reset_index(drop=True)
        df.to_sql(name, conn, if_exists='replace', index=False)
        print(f"  {name}: {len(df)} rows loaded ✓")
    except Exception as e:
        print(f"  ERROR loading {name}: {e}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("DATABASE SUMMARY")
print("=" * 50)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()
total_rows = 0
for (table,) in tables:
    cursor.execute(f"SELECT COUNT(*) FROM '{table}'")
    count = cursor.fetchone()[0]
    total_rows += count
    print(f"  {table:<22} {count:>6} rows")

print(f"\n  Total tables:  {len(tables)}")
print(f"  Total rows:    {total_rows}")
print("=" * 50)
print("\nDatabase setup complete.")
conn.close()
