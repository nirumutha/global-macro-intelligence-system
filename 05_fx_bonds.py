# ============================================================
# MODULE 3 — FX & BOND YIELD ANALYSIS
# How the Dollar, Interest Rates and Yields drive markets
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')
OUT_PATH  = os.path.join(BASE_PATH, 'outputs')
DB_PATH   = os.path.join(DATA_PATH, 'macro_system.db')
os.makedirs(OUT_PATH, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

# ── Load all data ─────────────────────────────────────────────
print("Loading data from database...")

def load_table(table, value_col=None):
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    if value_col:
        return df[[value_col]]
    return df

# Market prices
nifty    = load_table('NIFTY50')
sp500    = load_table('SP500')
gold     = load_table('GOLD')
silver   = load_table('SILVER')
dxy      = load_table('DXY')
usd_inr  = load_table('USD_INR')
vix      = load_table('VIX_US')

# Macro data
us_10y   = load_table('US_10Y_YIELD')
us_2y    = load_table('US_2Y_YIELD')
us_fed   = load_table('US_FED_RATE')
us_cpi   = load_table('US_CPI')

conn.close()

# ── Helper: get close price ───────────────────────────────────
def get_close(df):
    close_col = [c for c in df.columns if 'Close' in c or 'close' in c]
    if close_col:
        return df[close_col[0]]
    return df.iloc[:, 0]

nifty_close   = get_close(nifty)
sp500_close   = get_close(sp500)
gold_close    = get_close(gold)
silver_close  = get_close(silver)
dxy_close     = get_close(dxy)
usd_inr_close = get_close(usd_inr)

# ── Get macro series ──────────────────────────────────────────
yield_10y = us_10y.iloc[:, 0]
yield_2y  = us_2y.iloc[:, 0]
fed_rate  = us_fed.iloc[:, 0]
cpi       = us_cpi.iloc[:, 0]

# ════════════════════════════════════════════════════════════
# CHART 1 — US Yield Curve (10Y minus 2Y spread)
# ════════════════════════════════════════════════════════════
print("Creating Chart 1: US Yield Curve...")

# Align dates
yields = pd.DataFrame({
    '10Y': yield_10y,
    '2Y':  yield_2y
}).dropna()
yields['Spread'] = yields['10Y'] - yields['2Y']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle('US Yield Curve — 10Y vs 2Y Treasury Yields (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

ax1.plot(yields.index, yields['10Y'], color='#1F3864', linewidth=1.2, label='10-Year Yield')
ax1.plot(yields.index, yields['2Y'],  color='#C55A11', linewidth=1.2, label='2-Year Yield')
ax1.set_ylabel('Yield (%)', fontsize=10)
ax1.legend(fontsize=10)
ax1.set_title('US Treasury Yields', fontsize=11)

ax2.plot(yields.index, yields['Spread'], color='#2E75B6', linewidth=1.2, label='10Y - 2Y Spread')
ax2.axhline(0, color='red', linestyle='--', linewidth=1, label='Inversion level (0)')
ax2.fill_between(yields.index, yields['Spread'], 0,
                  where=yields['Spread'] >= 0, alpha=0.25, color='green', label='Normal curve')
ax2.fill_between(yields.index, yields['Spread'], 0,
                  where=yields['Spread'] < 0,  alpha=0.35, color='red',   label='Inverted (recession signal)')
ax2.set_ylabel('Spread (%)', fontsize=10)
ax2.set_title('Yield Curve Spread — When this goes negative, recession often follows', fontsize=11)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '05_yield_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/05_yield_curve.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 2 — Real Yield vs Gold Price
# ════════════════════════════════════════════════════════════
print("Creating Chart 2: Real Yield vs Gold...")

# Real yield = 10Y nominal yield minus CPI inflation (monthly, forward filled)
cpi_pct = cpi.pct_change(12) * 100  # Year-over-year CPI change
cpi_daily = cpi_pct.resample('D').interpolate()

aligned = pd.DataFrame({
    'Yield10Y': yield_10y,
    'CPI_YOY':  cpi_daily,
    'Gold':     gold_close
}).dropna()

aligned['Real_Yield'] = aligned['Yield10Y'] - aligned['CPI_YOY']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle('Real Yield vs Gold Price — The Most Reliable Gold Driver (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

ax1.plot(aligned.index, aligned['Gold'], color='#C55A11', linewidth=1.2)
ax1.set_ylabel('Gold Price (USD)', fontsize=10)
ax1.set_title('Gold Price — rises when real yields fall, falls when real yields rise', fontsize=11)
ax1.fill_between(aligned.index, aligned['Gold'].min(), aligned['Gold'],
                  alpha=0.1, color='#C55A11')

ax2.plot(aligned.index, aligned['Real_Yield'], color='#1F3864', linewidth=1.2)
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax2.fill_between(aligned.index, aligned['Real_Yield'], 0,
                  where=aligned['Real_Yield'] < 0, alpha=0.3, color='green',
                  label='Negative real yield (supports Gold)')
ax2.fill_between(aligned.index, aligned['Real_Yield'], 0,
                  where=aligned['Real_Yield'] >= 0, alpha=0.3, color='red',
                  label='Positive real yield (pressures Gold)')
ax2.set_ylabel('Real Yield (%)', fontsize=10)
ax2.set_title('US Real Yield (10Y minus CPI inflation)', fontsize=11)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '06_real_yield_vs_gold.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/06_real_yield_vs_gold.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 3 — DXY Dollar Index vs Commodities
# ════════════════════════════════════════════════════════════
print("Creating Chart 3: Dollar vs Commodities...")

comm = pd.DataFrame({
    'DXY':    dxy_close,
    'Gold':   gold_close,
    'Silver': silver_close,
}).dropna()

# Normalise to 100 at start for comparison
comm_norm = comm / comm.iloc[0] * 100

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(comm_norm.index, comm_norm['DXY'],    color='#1F3864', linewidth=1.5, label='DXY Dollar Index')
ax.plot(comm_norm.index, comm_norm['Gold'],   color='#C55A11', linewidth=1.2, label='Gold', alpha=0.9)
ax.plot(comm_norm.index, comm_norm['Silver'], color='#7030A0', linewidth=1.2, label='Silver', alpha=0.9)
ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel('Indexed to 100 at Start', fontsize=10)
ax.set_title('US Dollar (DXY) vs Gold vs Silver — Indexed to 100 (2010–2024)\nStrong Dollar typically pressures commodities',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '07_dollar_vs_commodities.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/07_dollar_vs_commodities.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 4 — USD/INR vs NIFTY (Currency impact on India)
# ════════════════════════════════════════════════════════════
print("Creating Chart 4: USD/INR vs NIFTY...")

india = pd.DataFrame({
    'NIFTY':   nifty_close,
    'USD_INR': usd_inr_close,
}).dropna()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle('India FX Story: USD/INR vs NIFTY 50 (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

ax1.plot(india.index, india['NIFTY'], color='#1F3864', linewidth=1.2)
ax1.set_ylabel('NIFTY 50', fontsize=10)
ax1.set_title('NIFTY 50 — Indian equity market performance', fontsize=11)
ax1.fill_between(india.index, india['NIFTY'].min(), india['NIFTY'],
                  alpha=0.1, color='#1F3864')

ax2.plot(india.index, india['USD_INR'], color='#C00000', linewidth=1.2)
ax2.set_ylabel('USD/INR', fontsize=10)
ax2.set_title('USD/INR — Rising = Rupee weakening = pressure on India', fontsize=11)
ax2.fill_between(india.index, india['USD_INR'].min(), india['USD_INR'],
                  alpha=0.1, color='#C00000')

# Rolling 60-day correlation
returns_india = india.pct_change().dropna()
rolling_corr = returns_india['NIFTY'].rolling(60).corr(returns_india['USD_INR'])
ax3.plot(rolling_corr.index, rolling_corr, color='#2E75B6', linewidth=1.2)
ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax3.fill_between(rolling_corr.index, rolling_corr, 0,
                  where=rolling_corr >= 0, alpha=0.3, color='orange',
                  label='Positive correlation (unusual)')
ax3.fill_between(rolling_corr.index, rolling_corr, 0,
                  where=rolling_corr < 0, alpha=0.3, color='blue',
                  label='Negative correlation (normal — Rupee weakens, NIFTY falls)')
ax3.set_ylabel('60-Day Correlation', fontsize=10)
ax3.set_title('Rolling Correlation — NIFTY vs USD/INR', fontsize=11)
ax3.set_ylim(-1, 1)
ax3.legend(fontsize=9)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '08_usdinr_vs_nifty.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/08_usdinr_vs_nifty.png ✓")

# ════════════════════════════════════════════════════════════
# PRINT KEY INSIGHTS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("KEY INSIGHTS FROM FX & BOND ANALYSIS")
print("=" * 55)

# Yield curve inversions
inversions = yields[yields['Spread'] < 0]
print(f"\nYield Curve Inversions (recession signals):")
print(f"  Days inverted: {len(inversions)} out of {len(yields)} total ({len(inversions)/len(yields)*100:.1f}%)")

# Gold vs real yield correlation
corr_gold_ry = aligned['Gold'].corr(aligned['Real_Yield'])
print(f"\nGold vs Real Yield correlation: {corr_gold_ry:.3f}")
print(f"  (Strong negative = Gold rises when real yields fall — as expected)")

# USD/INR total depreciation
start_fx = india['USD_INR'].iloc[0]
end_fx   = india['USD_INR'].iloc[-1]
depr     = (end_fx - start_fx) / start_fx * 100
print(f"\nRupee depreciation 2010–2024:")
print(f"  USD/INR: {start_fx:.1f} → {end_fx:.1f} ({depr:.1f}% depreciation)")

# NIFTY total return
start_nifty = india['NIFTY'].iloc[0]
end_nifty   = india['NIFTY'].iloc[-1]
nifty_ret   = (end_nifty - start_nifty) / start_nifty * 100
print(f"\nNIFTY 50 total return 2010–2024:")
print(f"  {start_nifty:.0f} → {end_nifty:.0f} (+{nifty_ret:.1f}%)")

print("=" * 55)
print(f"\nAll 4 charts saved to your outputs/ folder.")
