# ============================================================
# MODULE 4 — INSTITUTIONAL FLOW ANALYSIS
# Tracks smart money behaviour using market proxies
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

nifty    = load_close('NIFTY50')
sp500    = load_close('SP500')
gold     = load_close('GOLD')
vix_us   = load_close('VIX_US')
vix_india= load_close('VIX_INDIA')
usd_inr  = load_close('USD_INR')
dxy      = load_close('DXY')

conn.close()

# ── Align all series ──────────────────────────────────────────
data = pd.DataFrame({
    'NIFTY':     nifty,
    'SP500':     sp500,
    'Gold':      gold,
    'VIX_US':    vix_us,
    'VIX_INDIA': vix_india,
    'USD_INR':   usd_inr,
    'DXY':       dxy,
}).dropna()

returns = data.pct_change().dropna()

# ════════════════════════════════════════════════════════════
# BUILD SMART MONEY INDICATORS
# These proxy institutional behaviour using market data
# ════════════════════════════════════════════════════════════

# 1. Risk Appetite Index
# When institutions are confident: VIX low, equities rising, Gold stable
# Normalise each component to 0-100 scale
def normalise(series, window=252):
    roll_min = series.rolling(window).min()
    roll_max = series.rolling(window).max()
    return (series - roll_min) / (roll_max - roll_min) * 100

vix_inv      = 100 - normalise(data['VIX_US'])        # Invert VIX — low VIX = high confidence
nifty_mom    = normalise(data['NIFTY'].rolling(20).mean() / data['NIFTY'].rolling(60).mean() * 100)
sp500_mom    = normalise(data['SP500'].rolling(20).mean() / data['SP500'].rolling(60).mean() * 100)
dxy_inv      = 100 - normalise(data['DXY'])            # Weak dollar = risk on for emerging markets

risk_appetite = (vix_inv + nifty_mom + sp500_mom + dxy_inv).dropna() / 4

# 2. India Foreign Flow Proxy
# FII flows into India: strong when rupee stable, VIX low, global risk-on
vix_india_inv = 100 - normalise(data['VIX_INDIA'])
inr_stability = 100 - normalise(data['USD_INR'])       # Stable/strong rupee = FII inflows
india_flow_proxy = (vix_india_inv * 0.4 + inr_stability * 0.3 + nifty_mom * 0.3).dropna()

# 3. Smart Money vs Retail Divergence
# Smart money tends to buy fear (low VIX turning up) and sell euphoria (high VIX)
# Retail tends to do the opposite
nifty_ret_20  = returns['NIFTY'].rolling(20).mean() * 252  # Annualised momentum
vix_change_20 = data['VIX_US'].pct_change(20)

# Divergence: market rising but VIX also rising = smart money selling into strength
divergence = nifty_ret_20 - (-vix_change_20 * 100)
divergence_smooth = divergence.rolling(10).mean()

# 4. Institutional Accumulation/Distribution (price & momentum based)
# When price makes new highs but momentum slows = distribution (selling)
# When price makes new lows but momentum stabilises = accumulation (buying)
price_52w_pct = (data['NIFTY'] - data['NIFTY'].rolling(252).min()) / \
                (data['NIFTY'].rolling(252).max() - data['NIFTY'].rolling(252).min()) * 100
momentum_10d  = returns['NIFTY'].rolling(10).mean() * 252
accum_distrib = (price_52w_pct.diff(10) + momentum_10d * 10).rolling(20).mean()

# ════════════════════════════════════════════════════════════
# CHART 1 — Global Risk Appetite Index
# ════════════════════════════════════════════════════════════
print("Creating Chart 1: Global Risk Appetite Index...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle('Global Risk Appetite Index — Institutional Confidence Tracker (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

ax1.plot(data.index, data['NIFTY'], color='#1F3864', linewidth=1.2, label='NIFTY 50')
ax1.set_ylabel('NIFTY 50', fontsize=10)
ax1.legend(loc='upper left', fontsize=9)

ax2.plot(risk_appetite.index, risk_appetite, color='#2E75B6', linewidth=1, label='Risk Appetite Index')
ax2.plot(risk_appetite.index, risk_appetite.rolling(30).mean(),
          color='#C55A11', linewidth=1.8, label='30-day smoothed', alpha=0.9)
ax2.axhline(70, color='green', linestyle='--', linewidth=0.8, alpha=0.7, label='High confidence (70)')
ax2.axhline(30, color='red',   linestyle='--', linewidth=0.8, alpha=0.7, label='Fear zone (30)')
ax2.fill_between(risk_appetite.index, risk_appetite, 50,
                  where=risk_appetite >= 50, alpha=0.15, color='green')
ax2.fill_between(risk_appetite.index, risk_appetite, 50,
                  where=risk_appetite < 50,  alpha=0.15, color='red')
ax2.set_ylabel('Risk Appetite (0–100)', fontsize=10)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '09_risk_appetite_index.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/09_risk_appetite_index.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 2 — India Foreign Flow Proxy vs NIFTY
# ════════════════════════════════════════════════════════════
print("Creating Chart 2: India Foreign Flow Proxy...")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle('India Foreign Institutional Flow Proxy vs NIFTY (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

ax1.plot(data.index, data['NIFTY'], color='#1F3864', linewidth=1.2)
ax1.set_ylabel('NIFTY 50', fontsize=10)
ax1.set_title('NIFTY 50 Price', fontsize=11)

ax2.plot(india_flow_proxy.index, india_flow_proxy, color='#2E75B6', linewidth=1, alpha=0.7)
ax2.plot(india_flow_proxy.index, india_flow_proxy.rolling(20).mean(),
          color='#C55A11', linewidth=1.8)
ax2.axhline(60, color='green', linestyle='--', linewidth=0.8, alpha=0.7, label='Strong inflow signal (60)')
ax2.axhline(40, color='red',   linestyle='--', linewidth=0.8, alpha=0.7, label='Outflow signal (40)')
ax2.set_ylabel('FII Flow Proxy (0–100)', fontsize=10)
ax2.set_title('Foreign Institutional Flow Proxy — Higher = More FII Inflows', fontsize=11)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=9)

ax3.plot(data.index, data['USD_INR'], color='#C00000', linewidth=1.2)
ax3.set_ylabel('USD/INR', fontsize=10)
ax3.set_title('USD/INR — Rising = Rupee weakening = FII outflows', fontsize=11)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '10_india_flow_proxy.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/10_india_flow_proxy.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 3 — Smart Money vs Retail Divergence
# ════════════════════════════════════════════════════════════
print("Creating Chart 3: Smart Money vs Retail Divergence...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle('Smart Money vs Retail Divergence Indicator (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

ax1.plot(data.index, data['NIFTY'], color='#1F3864', linewidth=1.2)
ax1.set_ylabel('NIFTY 50', fontsize=10)
ax1.set_title('NIFTY 50 — watch for divergence signals', fontsize=11)

ax2.plot(divergence_smooth.index, divergence_smooth,
          color='#7030A0', linewidth=1.2, label='Divergence (smoothed)')
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.fill_between(divergence_smooth.index, divergence_smooth, 0,
                  where=divergence_smooth > 0, alpha=0.25, color='green',
                  label='Positive: Market & momentum aligned (institutional buying)')
ax2.fill_between(divergence_smooth.index, divergence_smooth, 0,
                  where=divergence_smooth <= 0, alpha=0.25, color='red',
                  label='Negative: Market rising but momentum fading (institutional selling)')
ax2.set_ylabel('Divergence Score', fontsize=10)
ax2.set_title('Smart Money Divergence — negative = institutions quietly selling into market strength', fontsize=11)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '11_smart_money_divergence.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/11_smart_money_divergence.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 4 — Accumulation vs Distribution Detector
# ════════════════════════════════════════════════════════════
print("Creating Chart 4: Accumulation vs Distribution...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle('Institutional Accumulation vs Distribution — NIFTY 50 (2010–2024)\nGlobal Macro Intelligence System',
             fontsize=13, fontweight='bold')

ax1.plot(data.index, data['NIFTY'], color='#1F3864', linewidth=1.2)
ax1.set_ylabel('NIFTY 50', fontsize=10)
ax1.set_title('NIFTY 50 Price', fontsize=11)

ax2.plot(accum_distrib.index, accum_distrib, color='gray', linewidth=0.8, alpha=0.5)
ax2.plot(accum_distrib.index, accum_distrib.rolling(20).mean(),
          color='#2E75B6', linewidth=1.5)
ax2.fill_between(accum_distrib.index, accum_distrib.rolling(20).mean(), 0,
                  where=accum_distrib.rolling(20).mean() > 0,
                  alpha=0.3, color='green', label='Accumulation phase (institutions buying)')
ax2.fill_between(accum_distrib.index, accum_distrib.rolling(20).mean(), 0,
                  where=accum_distrib.rolling(20).mean() <= 0,
                  alpha=0.3, color='red', label='Distribution phase (institutions selling)')
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_ylabel('Accumulation / Distribution', fontsize=10)
ax2.set_title('Blue line above zero = accumulation | Below zero = distribution', fontsize=11)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '12_accumulation_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/12_accumulation_distribution.png ✓")

# ════════════════════════════════════════════════════════════
# KEY INSIGHTS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("KEY INSIGHTS — INSTITUTIONAL FLOW ANALYSIS")
print("=" * 55)

# Risk appetite stats
avg_risk    = risk_appetite.mean()
high_conf   = (risk_appetite > 70).sum()
fear_zone   = (risk_appetite < 30).sum()
total_days  = len(risk_appetite)

print(f"\nGlobal Risk Appetite Index:")
print(f"  Average level:        {avg_risk:.1f} / 100")
print(f"  Days in high confidence (>70): {high_conf} ({high_conf/total_days*100:.1f}%)")
print(f"  Days in fear zone (<30):       {fear_zone} ({fear_zone/total_days*100:.1f}%)")

# India flow proxy
avg_flow  = india_flow_proxy.mean()
inflow_d  = (india_flow_proxy > 60).sum()
outflow_d = (india_flow_proxy < 40).sum()
total_f   = len(india_flow_proxy)

print(f"\nIndia FII Flow Proxy:")
print(f"  Average level:        {avg_flow:.1f} / 100")
print(f"  Strong inflow days:   {inflow_d} ({inflow_d/total_f*100:.1f}%)")
print(f"  Outflow signal days:  {outflow_d} ({outflow_d/total_f*100:.1f}%)")

# Correlation between flow proxy and NIFTY returns
aligned_flow = pd.DataFrame({
    'Flow':        india_flow_proxy,
    'NIFTY_ret':   returns['NIFTY']
}).dropna()
corr_flow_nifty = aligned_flow['Flow'].corr(aligned_flow['NIFTY_ret'])
print(f"\nCorrelation — FII Flow Proxy vs NIFTY daily returns: {corr_flow_nifty:.3f}")

print("=" * 55)
print(f"\nAll 4 charts saved to your outputs/ folder.")
print(f"Total charts built so far: 12")
