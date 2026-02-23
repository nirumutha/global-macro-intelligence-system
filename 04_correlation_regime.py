# ============================================================
# MODULE 2 — CORRELATION & MARKET REGIME ANALYSIS
# Analyses how markets move together and detects market regimes
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

# ── Load closing prices for all markets ──────────────────────
assets = {
    'NIFTY50':    'NIFTY50',
    'SP500':      'SP500',
    'Gold':       'GOLD',
    'Silver':     'SILVER',
    'Crude_WTI':  'CRUDE_WTI',
    'USD_INR':    'USD_INR',
    'DXY':        'DXY',
    'VIX_US':     'VIX_US',
}

print("Loading data from database...")
prices = {}
for label, table in assets.items():
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    # Find the Close column
    close_col = [c for c in df.columns if 'Close' in c or 'close' in c]
    if close_col:
        prices[label] = df[close_col[0]]
    else:
        prices[label] = df.iloc[:, 0]

conn.close()

# ── Combine into one DataFrame ────────────────────────────────
price_df = pd.DataFrame(prices).dropna(how='all')
price_df = price_df.ffill().dropna()
print(f"Combined price data: {len(price_df)} trading days\n")

# ── Calculate daily returns ───────────────────────────────────
returns = price_df.pct_change().dropna()

# ════════════════════════════════════════════════════════════
# CHART 1 — Full Period Correlation Heatmap
# ════════════════════════════════════════════════════════════
print("Creating Chart 1: Full Period Correlation Heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
corr = returns.corr()

im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, shrink=0.8)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(corr.columns, fontsize=11)

for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                ha='center', va='center', fontsize=10,
                color='black' if abs(corr.iloc[i, j]) < 0.7 else 'white')

ax.set_title('Cross-Asset Correlation Matrix (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '01_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/01_correlation_heatmap.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 2 — Rolling 60-Day Correlation: NIFTY vs S&P 500
# ════════════════════════════════════════════════════════════
print("Creating Chart 2: Rolling Correlation NIFTY vs S&P 500...")
rolling_corr = returns['NIFTY50'].rolling(60).corr(returns['SP500'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(price_df.index, price_df['NIFTY50'], color='#1F3864', linewidth=1, label='NIFTY 50')
ax1_twin = ax1.twinx()
ax1_twin.plot(price_df.index, price_df['SP500'], color='#C55A11', linewidth=1, alpha=0.7, label='S&P 500')
ax1.set_ylabel('NIFTY 50', color='#1F3864', fontsize=11)
ax1_twin.set_ylabel('S&P 500', color='#C55A11', fontsize=11)
ax1.set_title('NIFTY 50 vs S&P 500 — Price & Rolling 60-Day Correlation', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

ax2.plot(rolling_corr.index, rolling_corr, color='#2E75B6', linewidth=1.2)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax2.axhline(y=0.5, color='green', linestyle=':', linewidth=0.8, alpha=0.7, label='High correlation (0.5)')
ax2.axhline(y=-0.5, color='red', linestyle=':', linewidth=0.8, alpha=0.7, label='Negative correlation (-0.5)')
ax2.fill_between(rolling_corr.index, rolling_corr, 0,
                  where=rolling_corr >= 0, alpha=0.3, color='green')
ax2.fill_between(rolling_corr.index, rolling_corr, 0,
                  where=rolling_corr < 0, alpha=0.3, color='red')
ax2.set_ylabel('60-Day Rolling Correlation', fontsize=11)
ax2.set_ylim(-1, 1)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '02_rolling_correlation_nifty_sp500.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/02_rolling_correlation_nifty_sp500.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 3 — Market Regime Detection using VIX
# ════════════════════════════════════════════════════════════
print("Creating Chart 3: Market Regime Detection...")

vix = price_df['VIX_US'].copy()
sp  = price_df['SP500'].copy()
sp_ma200 = sp.rolling(200).mean()

# Define regimes based on VIX levels and trend
def classify_regime(row):
    vix_val = row['VIX']
    above_ma = row['SP500'] > row['MA200']
    if vix_val > 30:
        return 'Crisis / High Volatility'
    elif vix_val > 20 and not above_ma:
        return 'Bear Market'
    elif vix_val <= 20 and above_ma:
        return 'Bull Market'
    else:
        return 'Sideways / Uncertain'

regime_df = pd.DataFrame({
    'VIX':   vix,
    'SP500': sp,
    'MA200': sp_ma200
}).dropna()

regime_df['Regime'] = regime_df.apply(classify_regime, axis=1)

regime_colors = {
    'Bull Market':              '#1E6B3C',
    'Bear Market':              '#C00000',
    'Crisis / High Volatility': '#7030A0',
    'Sideways / Uncertain':     '#C55A11',
}

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle('Market Regime Detection — S&P 500 (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold', y=0.98)

# Panel 1: S&P 500 with regime background
for regime, color in regime_colors.items():
    mask = regime_df['Regime'] == regime
    ax1.fill_between(regime_df.index, 0, 1,
                      where=mask, transform=ax1.get_xaxis_transform(),
                      alpha=0.2, color=color, label=regime)
ax1.plot(regime_df.index, regime_df['SP500'], color='#1F3864', linewidth=1.2)
ax1.plot(regime_df.index, regime_df['MA200'], color='orange', linewidth=1,
          linestyle='--', alpha=0.8, label='200-day MA')
ax1.set_ylabel('S&P 500 Price', fontsize=10)
ax1.legend(loc='upper left', fontsize=8, ncol=3)

# Panel 2: VIX with threshold lines
ax2.plot(regime_df.index, regime_df['VIX'], color='#7030A0', linewidth=1)
ax2.axhline(20, color='orange', linestyle='--', linewidth=0.8, label='VIX 20 (caution)')
ax2.axhline(30, color='red', linestyle='--', linewidth=0.8, label='VIX 30 (fear)')
ax2.fill_between(regime_df.index, regime_df['VIX'], 20,
                  where=regime_df['VIX'] > 20, alpha=0.3, color='red')
ax2.set_ylabel('VIX (Fear Index)', fontsize=10)
ax2.legend(loc='upper right', fontsize=8)

# Panel 3: Regime over time as colour bar
regime_numeric = regime_df['Regime'].map({
    'Bull Market': 1,
    'Sideways / Uncertain': 2,
    'Bear Market': 3,
    'Crisis / High Volatility': 4
})
ax3.fill_between(regime_df.index, regime_numeric, step='mid', alpha=0.8, color='#2E75B6')
ax3.set_yticks([1, 2, 3, 4])
ax3.set_yticklabels(['Bull', 'Sideways', 'Bear', 'Crisis'], fontsize=9)
ax3.set_ylabel('Regime', fontsize=10)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '03_market_regime_detection.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/03_market_regime_detection.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 4 — Gold vs USD/INR vs NIFTY (India Macro Story)
# ════════════════════════════════════════════════════════════
print("Creating Chart 4: India Macro Story...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('India Macro Story: NIFTY 50 vs Gold vs USD/INR (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

axes[0].plot(price_df.index, price_df['NIFTY50'], color='#1F3864', linewidth=1.2)
axes[0].set_ylabel('NIFTY 50', fontsize=10)
axes[0].fill_between(price_df.index, price_df['NIFTY50'].min(),
                      price_df['NIFTY50'], alpha=0.1, color='#1F3864')

axes[1].plot(price_df.index, price_df['Gold'], color='#C55A11', linewidth=1.2)
axes[1].set_ylabel('Gold (USD)', fontsize=10)
axes[1].fill_between(price_df.index, price_df['Gold'].min(),
                      price_df['Gold'], alpha=0.1, color='#C55A11')

axes[2].plot(price_df.index, price_df['USD_INR'], color='#2E75B6', linewidth=1.2)
axes[2].set_ylabel('USD/INR', fontsize=10)
axes[2].fill_between(price_df.index, price_df['USD_INR'].min(),
                      price_df['USD_INR'], alpha=0.1, color='#2E75B6')
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '04_india_macro_story.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/04_india_macro_story.png ✓")

# ── Print regime summary ──────────────────────────────────────
print("\n" + "=" * 50)
print("REGIME SUMMARY (% of trading days)")
print("=" * 50)
regime_counts = regime_df['Regime'].value_counts()
total = len(regime_df)
for regime, count in regime_counts.items():
    pct = count / total * 100
    print(f"  {regime:<30} {pct:>6.1f}%  ({count} days)")
print("=" * 50)
print(f"\nAll 4 charts saved to your outputs/ folder.")