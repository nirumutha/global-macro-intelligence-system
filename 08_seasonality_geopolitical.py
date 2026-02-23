# ============================================================
# MODULE 6 — SEASONALITY & GEOPOLITICAL RISK
# Best/worst months to invest + risk event detection
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
silver    = load_close('SILVER')
crude     = load_close('CRUDE_WTI')
vix_us    = load_close('VIX_US')
usd_inr   = load_close('USD_INR')

conn.close()

data = pd.DataFrame({
    'NIFTY':   nifty,
    'SP500':   sp500,
    'Gold':    gold,
    'Silver':  silver,
    'Crude':   crude,
    'VIX':     vix_us,
    'USD_INR': usd_inr,
}).dropna()

returns = data.pct_change().dropna()

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

# ════════════════════════════════════════════════════════════
# CHART 1 — Monthly Seasonality Heatmap (All Assets)
# ════════════════════════════════════════════════════════════
print("Creating Chart 1: Monthly Seasonality Heatmap...")

# Calculate average monthly returns for each asset
monthly_rets = {}
for col in ['NIFTY','SP500','Gold','Silver','Crude']:
    monthly = returns[col].copy()
    monthly.index = pd.to_datetime(monthly.index)
    avg = monthly.groupby(monthly.index.month).mean() * 100
    monthly_rets[col] = avg

seasonality_df = pd.DataFrame(monthly_rets, index=range(1,13))
seasonality_df.index = MONTH_NAMES

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(seasonality_df.T, cmap='RdYlGn', aspect='auto',
               vmin=-0.5, vmax=0.5)
plt.colorbar(im, ax=ax, shrink=0.8, label='Avg Daily Return (%)')

ax.set_xticks(range(12))
ax.set_yticks(range(5))
ax.set_xticklabels(MONTH_NAMES, fontsize=11)
ax.set_yticklabels(['NIFTY','S&P 500','Gold','Silver','Crude WTI'], fontsize=11)

for i in range(5):
    for j in range(12):
        val = seasonality_df.iloc[j, i]
        ax.text(j, i, f'{val:.3f}%', ha='center', va='center',
                fontsize=9, color='black' if abs(val) < 0.3 else 'white')

ax.set_title('Monthly Seasonality Heatmap — Average Daily Return by Month (2010–2024)\n'
             'Green = historically strong month | Red = historically weak month',
             fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '17_seasonality_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/17_seasonality_heatmap.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 2 — Monthly Return Bar Charts (NIFTY & S&P 500)
# ════════════════════════════════════════════════════════════
print("Creating Chart 2: Monthly Return Bar Charts...")

nifty_monthly  = returns['NIFTY'].groupby(returns.index.month).mean() * 100
sp500_monthly  = returns['SP500'].groupby(returns.index.month).mean() * 100
gold_monthly   = returns['Gold'].groupby(returns.index.month).mean() * 100

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11))
fig.suptitle('Monthly Seasonality — Average Daily Return by Month (2010–2024)\n'
             'Global Macro Intelligence System',
             fontsize=13, fontweight='bold')

for ax, series, title, color_pos, color_neg in [
    (ax1, nifty_monthly,  'NIFTY 50',   '#1F3864', '#C00000'),
    (ax2, sp500_monthly,  'S&P 500',    '#2E75B6', '#C55A11'),
    (ax3, gold_monthly,   'Gold',       '#C55A11', '#7030A0'),
]:
    colors = [color_pos if v >= 0 else color_neg for v in series]
    bars = ax.bar(MONTH_NAMES, series, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Avg Daily Return (%)', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    for bar, val in zip(bars, series):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.002 if val >= 0 else -0.008),
                f'{val:.3f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '18_monthly_returns_bars.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/18_monthly_returns_bars.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 3 — Year-by-Year Annual Returns Comparison
# ════════════════════════════════════════════════════════════
print("Creating Chart 3: Annual Returns Comparison...")

annual = {}
for col in ['NIFTY','SP500','Gold','Crude']:
    yearly = (1 + returns[col]).resample('YE').prod() - 1
    yearly.index = yearly.index.year
    annual[col] = yearly * 100

annual_df = pd.DataFrame(annual)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Annual Returns by Asset Class (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

pairs = [('NIFTY','#1F3864'), ('SP500','#2E75B6'),
         ('Gold','#C55A11'),  ('Crude','#7030A0')]

for ax, (col, color) in zip(axes.flatten(), pairs):
    values = annual_df[col].dropna()
    bar_colors = [color if v >= 0 else '#C00000' for v in values]
    bars = ax.bar(values.index.astype(str), values,
                   color=bar_colors, alpha=0.85, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Return (%)', fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (1 if val >= 0 else -4),
                f'{val:.0f}%', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '19_annual_returns.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/19_annual_returns.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 4 — Geopolitical Risk Proxy & Black Swan Detector
# ════════════════════════════════════════════════════════════
print("Creating Chart 4: Geopolitical Risk & Black Swan Detector...")

# Build geopolitical risk proxy from market data
# VIX spikes + equity drops + gold surges = geopolitical stress
vix_zscore   = (data['VIX'] - data['VIX'].rolling(252).mean()) / \
                data['VIX'].rolling(252).std()
sp_ret_20    = returns['SP500'].rolling(20).mean() * 252
gold_ret_20  = returns['Gold'].rolling(20).mean() * 252
inr_stress   = returns['USD_INR'].rolling(20).mean() * 252

# Composite geo-risk score
geo_risk = (
    vix_zscore.clip(-3, 3) * 0.35 +
    (-sp_ret_20 * 2).clip(-3, 3) * 0.30 +
    (gold_ret_20 * 2).clip(-3, 3) * 0.20 +
    (inr_stress * 3).clip(-3, 3) * 0.15
).rolling(10).mean()

# Black Swan detector: geo_risk > 2 standard deviations
geo_mean = geo_risk.rolling(252).mean()
geo_std  = geo_risk.rolling(252).std()
black_swan_threshold = geo_mean + 2 * geo_std
black_swan_events    = geo_risk[geo_risk > black_swan_threshold]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Geopolitical Risk Proxy & Black Swan Detector (2010–2024)\n'
             'Global Macro Intelligence System',
             fontsize=13, fontweight='bold')

# Panel 1: NIFTY with black swan overlays
ax1.plot(data.index, data['NIFTY'], color='#1F3864', linewidth=1.2)
for date in black_swan_events.index:
    ax1.axvline(date, color='red', alpha=0.15, linewidth=1)
ax1.set_ylabel('NIFTY 50', fontsize=10)
ax1.set_title('NIFTY 50 — red lines = Black Swan alert dates', fontsize=11)

# Panel 2: Geo-risk score
ax2.plot(geo_risk.index, geo_risk, color='#7030A0', linewidth=1, alpha=0.7)
ax2.plot(geo_risk.index, geo_risk.rolling(30).mean(),
          color='#C00000', linewidth=1.8, label='30-day smoothed')
if len(black_swan_threshold.dropna()) > 0:
    ax2.plot(black_swan_threshold.index, black_swan_threshold,
              color='orange', linewidth=1, linestyle='--', label='Black Swan threshold (2σ)')
ax2.fill_between(geo_risk.index, geo_risk, black_swan_threshold,
                  where=geo_risk > black_swan_threshold,
                  alpha=0.4, color='red', label='Black Swan zone')
ax2.set_ylabel('Geo-Risk Score', fontsize=10)
ax2.set_title('Geopolitical Risk Proxy — spikes flag potential crisis events', fontsize=11)
ax2.legend(fontsize=9)

# Panel 3: VIX z-score
ax3.plot(vix_zscore.index, vix_zscore, color='#2E75B6', linewidth=1)
ax3.axhline(2,  color='red',   linestyle='--', linewidth=0.8, label='2σ extreme fear')
ax3.axhline(-1, color='green', linestyle='--', linewidth=0.8, label='-1σ complacency')
ax3.fill_between(vix_zscore.index, vix_zscore, 2,
                  where=vix_zscore >= 2, alpha=0.3, color='red')
ax3.set_ylabel('VIX Z-Score', fontsize=10)
ax3.set_title('VIX Z-Score — how many standard deviations above normal is fear today', fontsize=11)
ax3.legend(fontsize=9)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '20_geopolitical_risk.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/20_geopolitical_risk.png ✓")

# ════════════════════════════════════════════════════════════
# KEY INSIGHTS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("KEY INSIGHTS — SEASONALITY & GEOPOLITICAL RISK")
print("=" * 55)

print("\nBest months for NIFTY (avg daily return):")
nifty_sorted = nifty_monthly.sort_values(ascending=False)
for i, (month_num, val) in enumerate(nifty_sorted.head(3).items()):
    print(f"  #{i+1}: {MONTH_NAMES[month_num-1]:<5}  {val:+.4f}% per day")

print("\nWorst months for NIFTY (avg daily return):")
for i, (month_num, val) in enumerate(nifty_sorted.tail(3).iloc[::-1].items()):
    print(f"  #{i+1}: {MONTH_NAMES[month_num-1]:<5}  {val:+.4f}% per day")

print("\nBest months for Gold (avg daily return):")
gold_sorted = gold_monthly.sort_values(ascending=False)
for i, (month_num, val) in enumerate(gold_sorted.head(3).items()):
    print(f"  #{i+1}: {MONTH_NAMES[month_num-1]:<5}  {val:+.4f}% per day")

print(f"\nBlack Swan alerts detected: {len(black_swan_events)} days")
print(f"  ({len(black_swan_events)/len(geo_risk)*100:.1f}% of all trading days)")

print("\nAnnual return summary:")
for col in ['NIFTY','SP500','Gold','Crude']:
    best_year  = annual_df[col].idxmax()
    worst_year = annual_df[col].idxmin()
    avg_return = annual_df[col].mean()
    print(f"\n  {col}:")
    print(f"    Best year:   {best_year} ({annual_df[col][best_year]:+.1f}%)")
    print(f"    Worst year:  {worst_year} ({annual_df[col][worst_year]:+.1f}%)")
    print(f"    Avg annual:  {avg_return:+.1f}%")

print("=" * 55)
print(f"\nAll 4 charts saved to your outputs/ folder.")
print(f"Total charts built so far: 20")
