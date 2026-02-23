# ============================================================
# MODULE 5 — OPTIONS & VOLATILITY INTELLIGENCE
# VIX analysis, volatility regimes, and fear indicators
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

print("Loading data from database...")

def load_close(table):
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    close_col = [c for c in df.columns if 'Close' in c or 'close' in c]
    return df[close_col[0]] if close_col else df.iloc[:, 0]

nifty     = load_close('NIFTY50')
sp500     = load_close('SP500')
gold      = load_close('GOLD')
vix_us    = load_close('VIX_US')
vix_india = load_close('VIX_INDIA')
usd_inr   = load_close('USD_INR')

conn.close()

# ── Align data ────────────────────────────────────────────────
data = pd.DataFrame({
    'NIFTY':     nifty,
    'SP500':     sp500,
    'Gold':      gold,
    'VIX_US':    vix_us,
    'VIX_INDIA': vix_india,
    'USD_INR':   usd_inr,
}).dropna()

returns = data.pct_change().dropna()

# ── Calculate Realised Volatility ─────────────────────────────
# Realised vol = actual volatility that occurred (backward looking)
# Implied vol  = VIX = what options market expects (forward looking)
# Gap between them = volatility risk premium

rv_20_nifty = returns['NIFTY'].rolling(20).std() * np.sqrt(252) * 100
rv_20_sp500 = returns['SP500'].rolling(20).std() * np.sqrt(252) * 100

# Volatility Risk Premium: VIX minus Realised Vol
# Positive = options expensive (IV > RV) — sell volatility
# Negative = options cheap (IV < RV) — buy volatility
vrp = data['VIX_US'] - rv_20_sp500

# ── VIX Percentile Rank ───────────────────────────────────────
# Where is today's VIX relative to its history?
vix_pct_rank = data['VIX_US'].rolling(252).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
)

# ════════════════════════════════════════════════════════════
# CHART 1 — US VIX vs India VIX Comparison
# ════════════════════════════════════════════════════════════
print("Creating Chart 1: VIX US vs India VIX...")

vix_data = pd.DataFrame({
    'VIX_US':    data['VIX_US'],
    'VIX_INDIA': data['VIX_INDIA'],
}).dropna()
vix_data['Spread'] = vix_data['VIX_INDIA'] - vix_data['VIX_US']

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle('US VIX vs India VIX — Global Fear Comparison (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

ax1.plot(vix_data.index, vix_data['VIX_US'],    color='#C00000', linewidth=1.2, label='VIX US')
ax1.plot(vix_data.index, vix_data['VIX_INDIA'], color='#1F3864', linewidth=1.2, label='India VIX', alpha=0.8)
ax1.axhline(20, color='orange', linestyle='--', linewidth=0.8, label='Caution level (20)')
ax1.axhline(30, color='red',    linestyle='--', linewidth=0.8, label='Fear level (30)')
ax1.set_ylabel('VIX Level', fontsize=10)
ax1.set_title('VIX Levels — Above 20 = caution, Above 30 = fear', fontsize=11)
ax1.legend(fontsize=9)

ax2.plot(vix_data.index, vix_data['Spread'], color='#7030A0', linewidth=1.2)
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.fill_between(vix_data.index, vix_data['Spread'], 0,
                  where=vix_data['Spread'] > 0, alpha=0.3, color='red',
                  label='India VIX higher (India-specific fear)')
ax2.fill_between(vix_data.index, vix_data['Spread'], 0,
                  where=vix_data['Spread'] <= 0, alpha=0.3, color='green',
                  label='US VIX higher (global fear)')
ax2.set_ylabel('India VIX − US VIX', fontsize=10)
ax2.set_title('VIX Spread — India minus US', fontsize=11)
ax2.legend(fontsize=9)

ax3.plot(vix_pct_rank.index, vix_pct_rank, color='#2E75B6', linewidth=1)
ax3.axhline(80, color='red',   linestyle='--', linewidth=0.8, label='Extreme fear (80th pct)')
ax3.axhline(20, color='green', linestyle='--', linewidth=0.8, label='Complacency (20th pct)')
ax3.fill_between(vix_pct_rank.index, vix_pct_rank, 80,
                  where=vix_pct_rank >= 80, alpha=0.3, color='red')
ax3.fill_between(vix_pct_rank.index, vix_pct_rank, 20,
                  where=vix_pct_rank <= 20, alpha=0.3, color='green')
ax3.set_ylabel('VIX Percentile Rank', fontsize=10)
ax3.set_title('VIX Percentile Rank — where is today vs history (252-day window)', fontsize=11)
ax3.set_ylim(0, 100)
ax3.legend(fontsize=9)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '13_vix_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/13_vix_comparison.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 2 — Implied vs Realised Volatility (VRP)
# ════════════════════════════════════════════════════════════
print("Creating Chart 2: Implied vs Realised Volatility...")

vrp_data = pd.DataFrame({
    'VIX':  data['VIX_US'],
    'RV20': rv_20_sp500,
    'VRP':  vrp,
}).dropna()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle('Implied vs Realised Volatility — Volatility Risk Premium (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

ax1.plot(vrp_data.index, vrp_data['VIX'],  color='#C00000', linewidth=1.2, label='VIX (Implied Vol)')
ax1.plot(vrp_data.index, vrp_data['RV20'], color='#1F3864', linewidth=1.2, label='Realised Vol 20d', alpha=0.8)
ax1.set_ylabel('Volatility (%)', fontsize=10)
ax1.set_title('VIX (what market fears) vs Realised Volatility (what actually happened)', fontsize=11)
ax1.legend(fontsize=9)

ax2.plot(vrp_data.index, vrp_data['VRP'], color='#2E75B6', linewidth=1.2)
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.fill_between(vrp_data.index, vrp_data['VRP'], 0,
                  where=vrp_data['VRP'] > 0, alpha=0.3, color='orange',
                  label='VRP positive: options expensive (IV > RV)')
ax2.fill_between(vrp_data.index, vrp_data['VRP'], 0,
                  where=vrp_data['VRP'] <= 0, alpha=0.3, color='purple',
                  label='VRP negative: options cheap (IV < RV) — crisis signal')
ax2.set_ylabel('VRP (%)', fontsize=10)
ax2.set_title('Volatility Risk Premium — Positive = options overpriced, Negative = panic', fontsize=11)
ax2.legend(fontsize=9)

ax3.plot(data.index, data['SP500'], color='#1F3864', linewidth=1.2)
ax3.set_ylabel('S&P 500', fontsize=10)
ax3.set_title('S&P 500 — compare to VRP above', fontsize=11)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '14_implied_vs_realised_vol.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/14_implied_vs_realised_vol.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 3 — Volatility Regime Classification
# ════════════════════════════════════════════════════════════
print("Creating Chart 3: Volatility Regime Classification...")

def classify_vol_regime(vix_val):
    if vix_val < 15:
        return 1   # Complacency
    elif vix_val < 20:
        return 2   # Normal
    elif vix_val < 30:
        return 3   # Elevated
    else:
        return 4   # Crisis

vol_regime       = data['VIX_US'].apply(classify_vol_regime)
regime_labels    = {1: 'Complacency', 2: 'Normal', 3: 'Elevated', 4: 'Crisis'}
regime_colors_v  = {1: '#1E6B3C', 2: '#2E75B6', 3: '#C55A11', 4: '#C00000'}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle('Volatility Regime Classification — 4 States (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

for regime_id, color in regime_colors_v.items():
    mask = vol_regime == regime_id
    ax1.fill_between(data.index, 0, 1,
                      where=mask.values,
                      transform=ax1.get_xaxis_transform(),
                      alpha=0.3, color=color,
                      label=regime_labels[regime_id])
ax1.plot(data.index, data['VIX_US'], color='black', linewidth=0.8, alpha=0.7)
ax1.axhline(15, color='green',  linestyle=':', linewidth=0.8)
ax1.axhline(20, color='orange', linestyle=':', linewidth=0.8)
ax1.axhline(30, color='red',    linestyle=':', linewidth=0.8)
ax1.set_ylabel('VIX Level', fontsize=10)
ax1.set_title('VIX with Volatility Regime Background', fontsize=11)
ax1.legend(fontsize=9, ncol=4)

ax2.plot(data.index, data['NIFTY'], color='#1F3864', linewidth=1.2)
for regime_id, color in regime_colors_v.items():
    mask = vol_regime == regime_id
    ax2.fill_between(data.index, 0, 1,
                      where=mask.values,
                      transform=ax2.get_xaxis_transform(),
                      alpha=0.15, color=color)
ax2.set_ylabel('NIFTY 50', fontsize=10)
ax2.set_title('NIFTY 50 coloured by volatility regime', fontsize=11)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '15_volatility_regime.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/15_volatility_regime.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 4 — Volatility Clustering (NIFTY & S&P 500)
# ════════════════════════════════════════════════════════════
print("Creating Chart 4: Volatility Clustering...")

rv_nifty = returns['NIFTY'].rolling(20).std() * np.sqrt(252) * 100
rv_sp500 = returns['SP500'].rolling(20).std() * np.sqrt(252) * 100

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle('Volatility Clustering — NIFTY 50 & S&P 500 (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

ax1.bar(returns.index, returns['NIFTY'] * 100,
         color=np.where(returns['NIFTY'] >= 0, '#1E6B3C', '#C00000'),
         width=1, alpha=0.7)
ax1.set_ylabel('Daily Return (%)', fontsize=10)
ax1.set_title('NIFTY 50 Daily Returns — notice how big moves cluster together', fontsize=11)

ax2.plot(rv_nifty.index, rv_nifty,  color='#1F3864', linewidth=1.2, label='NIFTY RV 20d')
ax2.plot(rv_sp500.index, rv_sp500,  color='#C55A11', linewidth=1.2, label='S&P 500 RV 20d', alpha=0.8)
ax2.axhline(20, color='orange', linestyle='--', linewidth=0.8, label='20% vol threshold')
ax2.axhline(30, color='red',    linestyle='--', linewidth=0.8, label='30% vol threshold')
ax2.set_ylabel('Realised Volatility (%)', fontsize=10)
ax2.set_title('Realised Volatility — 20-day rolling annualised', fontsize=11)
ax2.legend(fontsize=9)

# Correlation between NIFTY and S&P500 realised vols
vol_corr = rv_nifty.rolling(60).corr(rv_sp500)
ax3.plot(vol_corr.index, vol_corr, color='#7030A0', linewidth=1.2)
ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax3.fill_between(vol_corr.index, vol_corr, 0,
                  where=vol_corr >= 0, alpha=0.3, color='purple',
                  label='Positive vol correlation (global risk-off)')
ax3.set_ylabel('Vol Correlation', fontsize=10)
ax3.set_title('Rolling 60-day Correlation of Volatility — NIFTY vs S&P 500', fontsize=11)
ax3.set_ylim(-1, 1)
ax3.legend(fontsize=9)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '16_volatility_clustering.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/16_volatility_clustering.png ✓")

# ════════════════════════════════════════════════════════════
# KEY INSIGHTS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("KEY INSIGHTS — VOLATILITY & OPTIONS ANALYSIS")
print("=" * 55)

# VIX regime breakdown
total = len(vol_regime)
for rid, label in regime_labels.items():
    count = (vol_regime == rid).sum()
    pct   = count / total * 100
    print(f"\n  {label:<15} {count:>4} days  ({pct:.1f}%)")

# VRP stats
print(f"\nVolatility Risk Premium (VRP):")
print(f"  Average VRP:          {vrp_data['VRP'].mean():.2f}%")
print(f"  VRP positive days:    {(vrp_data['VRP'] > 0).sum()} ({(vrp_data['VRP'] > 0).mean()*100:.1f}%)")
print(f"  VRP negative days:    {(vrp_data['VRP'] < 0).sum()} ({(vrp_data['VRP'] < 0).mean()*100:.1f}%)")

# Extreme VIX days
vix_80th = data['VIX_US'].quantile(0.80)
vix_20th = data['VIX_US'].quantile(0.20)
print(f"\nVIX Extremes:")
print(f"  80th percentile VIX level: {vix_80th:.1f}")
print(f"  20th percentile VIX level: {vix_20th:.1f}")
print(f"  All-time high VIX:         {data['VIX_US'].max():.1f} on {data['VIX_US'].idxmax().date()}")
print(f"  All-time low VIX:          {data['VIX_US'].min():.1f} on {data['VIX_US'].idxmin().date()}")

# VIX vs NIFTY next-day returns
aligned_vix = pd.DataFrame({
    'VIX':       data['VIX_US'],
    'NIFTY_ret': returns['NIFTY'].shift(-1)
}).dropna()
high_vix_ret = aligned_vix[aligned_vix['VIX'] > 30]['NIFTY_ret'].mean() * 100
low_vix_ret  = aligned_vix[aligned_vix['VIX'] < 15]['NIFTY_ret'].mean() * 100
print(f"\nNIFTY Next-Day Returns by VIX Level:")
print(f"  When VIX > 30 (fear):        {high_vix_ret:.3f}% avg next day")
print(f"  When VIX < 15 (complacency): {low_vix_ret:.3f}% avg next day")

print("=" * 55)
print(f"\nAll 4 charts saved to your outputs/ folder.")
print(f"Total charts built so far: 16")
