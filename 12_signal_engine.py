# ============================================================
# MODULE 10 — SIGNAL GENERATION ENGINE (Improved v2)
# Fixes signal lag by adding RSI counter-trend component
# Combines trend-following MA signals WITH mean reversion RSI
# Output: Long / Short / Neutral for each asset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')
OUT_PATH  = os.path.join(BASE_PATH, 'outputs')
DB_PATH   = os.path.join(DATA_PATH, 'macro_system.db')
os.makedirs(OUT_PATH, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

print("Loading all data from database...")

def load_close(table):
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    close_col = [c for c in df.columns if 'Close' in c or 'close' in c]
    return df[close_col[0]].dropna() if close_col else df.iloc[:, 0].dropna()

def load_macro(table):
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    return df.iloc[:, 0].dropna()

# ── Market prices ─────────────────────────────────────────────
nifty     = load_close('NIFTY50')
sp500     = load_close('SP500')
gold      = load_close('GOLD')
silver    = load_close('SILVER')
crude     = load_close('CRUDE_WTI')
vix_us    = load_close('VIX_US')
vix_india = load_close('VIX_INDIA')
usd_inr   = load_close('USD_INR')
dxy       = load_close('DXY')

# ── Macro data ────────────────────────────────────────────────
cpi          = load_macro('US_CPI')
fed_rate     = load_macro('US_FED_RATE')
unemployment = load_macro('US_UNEMPLOYMENT')
gdp          = load_macro('US_GDP')
payrolls     = load_macro('US_PAYROLLS')
yield_10y    = load_macro('US_10Y_YIELD')
yield_2y     = load_macro('US_2Y_YIELD')

# ── Sentiment ─────────────────────────────────────────────────
try:
    sentiment_db    = pd.read_sql("SELECT * FROM SENTIMENT_DAILY", conn)
    sentiment_score = sentiment_db['score'].mean()
except:
    sentiment_score = 0.0

conn.close()
print(f"Data loaded. Building signal components...\n")

# ════════════════════════════════════════════════════════════
# SIGNAL COMPONENTS
# Each component returns a score between -1 and +1
# -1 = strong bearish | 0 = neutral | +1 = strong bullish
# ════════════════════════════════════════════════════════════

def normalise_signal(series, window=252):
    """Normalise a series to -1 to +1 range using rolling window."""
    roll_min = series.rolling(window, min_periods=60).min()
    roll_max = series.rolling(window, min_periods=60).max()
    norm     = (series - roll_min) / (roll_max - roll_min + 1e-10)
    return (norm * 2 - 1).clip(-1, 1)

# ── Component 1: Price Momentum (Trend-Following) ─────────────
def momentum_signal(price, fast=20, slow=60, trend=200):
    """
    Triple moving average momentum signal.
    Confirms the direction of the trend.
    +1 = all MAs aligned bullishly
    -1 = all MAs aligned bearishly
    """
    ma_fast  = price.rolling(fast).mean()
    ma_slow  = price.rolling(slow).mean()
    ma_trend = price.rolling(trend).mean()

    score  = pd.Series(0.0, index=price.index)
    score += np.where(price > ma_fast,   0.33, -0.33)
    score += np.where(ma_fast > ma_slow, 0.33, -0.33)
    score += np.where(ma_slow > ma_trend,0.34, -0.34)
    return pd.Series(score, index=price.index).clip(-1, 1)

# ── Component 2: RSI Counter-Trend (NEW — fixes signal lag) ───
def rsi_signal(price, period=14, overbought=70, oversold=30):
    """
    RSI-based mean reversion counter-trend component.

    This is the KEY FIX for signal lag:
    - When RSI > 70 (overbought): reduce Long score → market stretched
    - When RSI < 30 (oversold):  reduce Short score → market oversold
    - When RSI is neutral (40-60): no adjustment

    This prevents the system from going Long after a big rally
    or Short after a big drop — which is when most losses occur.

    Combined with momentum:
    - Strong trend UP + RSI neutral   = confident Long
    - Strong trend UP + RSI overbought= cautious, reduce position
    - Downtrend + RSI oversold        = reduce Short, possible bounce
    """
    delta  = price.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs  = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Convert RSI to -1 to +1 counter-trend score
    # Overbought (RSI>70) → negative score (reduce longs)
    # Oversold  (RSI<30) → positive score (reduce shorts)
    # Neutral   (40-60)  → zero (no adjustment)
    score = pd.Series(0.0, index=price.index)

    # Strongly overbought (RSI > 80) → strong negative
    score = np.where(rsi > 80, -1.0,
            # Overbought (RSI 70-80) → moderate negative
            np.where(rsi > 70, -0.5,
            # Mildly overbought (RSI 60-70) → small negative
            np.where(rsi > 60, -0.1,
            # Neutral zone — no signal
            np.where(rsi >= 40,  0.0,
            # Mildly oversold (RSI 30-40) → small positive
            np.where(rsi >= 30,  0.1,
            # Oversold (RSI 20-30) → moderate positive
            np.where(rsi >= 20,  0.5,
            # Strongly oversold (RSI < 20) → strong positive
                                  1.0))))))

    return pd.Series(score, index=price.index), rsi

# ── Component 3: Rate of Change Momentum (short-term) ─────────
def roc_signal(price, period=10):
    """
    Rate of Change — short-term price momentum.
    Faster than MA crossover, less lag.
    Helps catch momentum before MAs confirm it.
    """
    roc = price.pct_change(period) * 100
    return normalise_signal(roc).clip(-1, 1)

# ── Component 4: Volatility Filter ───────────────────────────
def volatility_signal(vix_series, crisis=30, caution=20):
    """VIX-based risk filter."""
    score = np.where(vix_series < caution,  0.5,
            np.where(vix_series < crisis,   0.0, -1.0))
    return pd.Series(score, index=vix_series.index)

# ── Component 5: Macro Regime Signal ─────────────────────────
def macro_regime_signal(cpi_series, gdp_series, asset_type='equity'):
    """Macro regime classification signal."""
    cpi_yoy = cpi_series.pct_change(12) * 100
    gdp_qoq = gdp_series.pct_change(1)  * 100

    cpi_daily = cpi_yoy.resample('D').interpolate().reindex(
        pd.date_range(cpi_yoy.index.min(), cpi_yoy.index.max(), freq='D')
    ).ffill()
    gdp_daily = gdp_qoq.resample('D').interpolate().reindex(
        pd.date_range(gdp_qoq.index.min(), gdp_qoq.index.max(), freq='D')
    ).ffill()

    scores = []
    dates  = []
    for date in pd.date_range(
        max(cpi_daily.index.min(), gdp_daily.index.min()),
        min(cpi_daily.index.max(), gdp_daily.index.max()),
        freq='B'
    ):
        try:
            inf = cpi_daily.loc[date]
            gr  = gdp_daily.loc[date]
        except:
            continue

        high_inf = inf > 3.0
        good_gr  = gr  > 0

        if asset_type == 'equity':
            if   not high_inf and good_gr:  score =  1.0
            elif high_inf and good_gr:      score = -0.3
            elif high_inf and not good_gr:  score = -1.0
            else:                           score = -0.5
        elif asset_type == 'gold':
            if   not high_inf and good_gr:  score =  0.0
            elif high_inf and good_gr:      score =  0.7
            elif high_inf and not good_gr:  score =  1.0
            else:                           score =  0.8
        elif asset_type == 'crude':
            if   not high_inf and good_gr:  score =  0.5
            elif high_inf and good_gr:      score =  0.3
            elif high_inf and not good_gr:  score = -0.5
            else:                           score = -1.0
        else:
            score = 0.0

        scores.append(score)
        dates.append(date)

    return pd.Series(scores, index=dates)

# ── Component 6: Yield Curve Signal ──────────────────────────
def yield_curve_signal(yield_10y, yield_2y, asset_type='equity'):
    """Yield curve shape signal."""
    spread = yield_10y - yield_2y
    spread = spread.reindex(
        pd.date_range(spread.index.min(), spread.index.max(), freq='B')
    ).ffill()

    if asset_type == 'equity':
        score = np.where(spread > 1.0,    1.0,
                np.where(spread > 0.0,    0.3,
                np.where(spread > -0.5,  -0.5, -1.0)))
    elif asset_type == 'gold':
        score = np.where(spread < 0,      0.8,
                np.where(spread < 0.5,    0.3,  0.0))
    else:
        score = np.where(spread > 0,      0.3, -0.3)

    return pd.Series(score, index=spread.index)

# ── Component 7: Currency Signal ─────────────────────────────
def currency_signal(usd_inr_series, dxy_series, asset_type='NIFTY'):
    """FX signal."""
    inr_mom = usd_inr_series.pct_change(20)
    dxy_mom = dxy_series.pct_change(20)

    if asset_type == 'NIFTY':
        return -normalise_signal(inr_mom)
    elif asset_type in ['Gold', 'Silver']:
        return -normalise_signal(dxy_mom)
    elif asset_type == 'Crude':
        return -normalise_signal(dxy_mom) * 0.5
    else:
        return pd.Series(0.0, index=usd_inr_series.index)

# ── Component 8: Sentiment Signal ────────────────────────────
def sentiment_signal_score(score, asset_type='equity'):
    """Live sentiment score to signal."""
    if asset_type in ['equity', 'NIFTY', 'SP500']:
        return float(np.clip(score * 2, -1, 1))
    elif asset_type in ['Gold', 'Silver']:
        return float(np.clip(-score * 1.5, -1, 1))
    else:
        return float(np.clip(score, -1, 1))

# ════════════════════════════════════════════════════════════
# COMPOSITE SIGNAL BUILDER
# ════════════════════════════════════════════════════════════

def build_signal(price, components_weights):
    """
    Build composite signal from components and weights.
    Returns daily composite score clipped to -1 to +1.
    """
    composite = pd.Series(0.0, index=price.index)
    total_w   = 0
    for series, weight in components_weights:
        aligned    = series.reindex(price.index).ffill().bfill()
        composite += aligned * weight
        total_w   += weight
    return (composite / total_w).clip(-1, 1)

def score_to_signal(score):
    """Convert numeric score to Long/Short/Neutral."""
    if   score >= 0.15: return 'Long',    '#1E6B3C'
    elif score <= -0.15: return 'Short',  '#C00000'
    else:                return 'Neutral','#C55A11'

# ════════════════════════════════════════════════════════════
# PRE-COMPUTE ALL SHARED COMPONENTS
# ════════════════════════════════════════════════════════════
print("Computing signal components...\n")

# Macro
macro_eq  = macro_regime_signal(cpi, gdp, 'equity')
macro_gld = macro_regime_signal(cpi, gdp, 'gold')
macro_crd = macro_regime_signal(cpi, gdp, 'crude')

# Yield curve
yc_eq  = yield_curve_signal(yield_10y, yield_2y, 'equity')
yc_gld = yield_curve_signal(yield_10y, yield_2y, 'gold')
yc_crd = yield_curve_signal(yield_10y, yield_2y, 'other')

# Volatility
vix_eq  = volatility_signal(vix_us, crisis=30, caution=20)
vix_gld = -volatility_signal(vix_us, crisis=30, caution=20) * 0.5

# ════════════════════════════════════════════════════════════
# BUILD IMPROVED SIGNALS
#
# Key change from v1:
# We now split the momentum weight between:
#   - MA momentum (trend confirmation, slow)
#   - RSI counter-trend (mean reversion, fast)
#   - ROC short-term momentum (faster trend, less lag)
#
# This three-part momentum system catches trends earlier
# and exits before they reverse — fixing the signal lag.
# ════════════════════════════════════════════════════════════

print("Building improved composite signals...\n")

# ── NIFTY ─────────────────────────────────────────────────────
nifty_ma   = momentum_signal(nifty)
nifty_rsi, nifty_rsi_raw = rsi_signal(nifty)
nifty_roc  = roc_signal(nifty, period=10)
nifty_fx   = currency_signal(usd_inr, dxy, 'NIFTY')
nifty_sent = sentiment_signal_score(sentiment_score, 'NIFTY')

nifty_signal = build_signal(nifty, [
    (nifty_ma,   0.20),   # Trend-following MA (reduced from 0.30)
    (nifty_rsi,  0.15),   # RSI counter-trend (NEW)
    (nifty_roc,  0.10),   # Short-term ROC (NEW — less lag)
    (vix_eq,     0.20),   # Volatility filter
    (macro_eq,   0.20),   # Macro regime
    (yc_eq,      0.10),   # Yield curve
    (nifty_fx,   0.05),   # Currency
])
nifty_signal = (nifty_signal + nifty_sent * 0.05).clip(-1, 1)

# ── S&P 500 ───────────────────────────────────────────────────
sp500_ma  = momentum_signal(sp500)
sp500_rsi, sp500_rsi_raw = rsi_signal(sp500)
sp500_roc = roc_signal(sp500, period=10)

sp500_signal = build_signal(sp500, [
    (sp500_ma,   0.20),
    (sp500_rsi,  0.15),   # RSI counter-trend (NEW)
    (sp500_roc,  0.10),   # Short-term ROC (NEW)
    (vix_eq,     0.25),
    (macro_eq,   0.20),
    (yc_eq,      0.10),
])
sp500_signal = (sp500_signal + nifty_sent * 0.05).clip(-1, 1)

# ── GOLD ──────────────────────────────────────────────────────
gold_ma  = momentum_signal(gold)
gold_rsi, gold_rsi_raw = rsi_signal(gold)
gold_roc = roc_signal(gold, period=10)
gold_fx  = currency_signal(usd_inr, dxy, 'Gold')
gold_sent= sentiment_signal_score(sentiment_score, 'Gold')

gold_signal = build_signal(gold, [
    (gold_ma,    0.20),
    (gold_rsi,   0.15),   # RSI counter-trend (NEW)
    (gold_roc,   0.10),   # Short-term ROC (NEW)
    (vix_gld,    0.15),
    (macro_gld,  0.25),
    (yc_gld,     0.10),
    (gold_fx,    0.05),
])
gold_signal = (gold_signal + gold_sent * 0.05).clip(-1, 1)

# ── SILVER ────────────────────────────────────────────────────
silver_ma  = momentum_signal(silver)
silver_rsi, silver_rsi_raw = rsi_signal(silver)
silver_roc = roc_signal(silver, period=10)
silver_fx  = currency_signal(usd_inr, dxy, 'Silver')

silver_signal = build_signal(silver, [
    (silver_ma,  0.20),
    (silver_rsi, 0.15),   # RSI counter-trend (NEW)
    (silver_roc, 0.10),   # Short-term ROC (NEW)
    (vix_gld,    0.15),
    (macro_gld,  0.25),
    (yc_gld,     0.10),
    (silver_fx,  0.05),
])

# ── CRUDE ─────────────────────────────────────────────────────
crude_ma  = momentum_signal(crude)
crude_rsi, crude_rsi_raw = rsi_signal(crude)
crude_roc = roc_signal(crude, period=10)
crude_fx  = currency_signal(usd_inr, dxy, 'Crude')

crude_signal = build_signal(crude, [
    (crude_ma,   0.20),
    (crude_rsi,  0.15),   # RSI counter-trend (NEW)
    (crude_roc,  0.10),   # Short-term ROC (NEW)
    (vix_eq,     0.15),
    (macro_crd,  0.25),
    (yc_crd,     0.10),
    (crude_fx,   0.05),
])

# ════════════════════════════════════════════════════════════
# SAVE SIGNALS TO DATABASE
# ════════════════════════════════════════════════════════════
signals_df = pd.DataFrame({
    'NIFTY_score':  nifty_signal,
    'SP500_score':  sp500_signal,
    'Gold_score':   gold_signal,
    'Silver_score': silver_signal,
    'Crude_score':  crude_signal,
}).dropna()

for col in ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']:
    signals_df[f'{col}_signal'] = signals_df[f'{col}_score'].apply(
        lambda x: score_to_signal(x)[0]
    )

conn = sqlite3.connect(DB_PATH)
signals_df.to_sql('SIGNALS', conn, if_exists='replace', index=True)
conn.close()
print(f"Signals saved to database. Total signal dates: {len(signals_df)}\n")

# ════════════════════════════════════════════════════════════
# CHART 1 — RSI Component Visualisation (new — shows the fix)
# ════════════════════════════════════════════════════════════
print("Creating Chart 1: RSI Counter-Trend Component...")

fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig.suptitle('RSI Counter-Trend Fix — Prevents Buying Overbought & Selling Oversold\n'
             'Global Macro Intelligence System v2',
             fontsize=13, fontweight='bold')

assets_rsi = [
    ('S&P 500',  sp500,  sp500_rsi_raw,  sp500_signal,  '#2E75B6'),
    ('NIFTY 50', nifty,  nifty_rsi_raw,  nifty_signal,  '#1F3864'),
    ('Gold',     gold,   gold_rsi_raw,   gold_signal,   '#C55A11'),
    ('Crude',    crude,  crude_rsi_raw,  crude_signal,  '#1E6B3C'),
]

for idx, (name, price, rsi_raw, signal, color) in enumerate(assets_rsi):
    ax = axes[idx]
    ax2 = ax.twinx()

    # Price
    ax.plot(price.index, price, color=color, linewidth=1, alpha=0.7, label=name)
    ax.set_ylabel(name, fontsize=9, color=color)

    # RSI
    ax2.plot(rsi_raw.index, rsi_raw, color='gray', linewidth=0.8, alpha=0.6)
    ax2.axhline(70, color='red',   linestyle='--', linewidth=0.8,
                 label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', linewidth=0.8,
                 label='Oversold (30)')
    ax2.fill_between(rsi_raw.index, rsi_raw, 70,
                      where=rsi_raw >= 70, alpha=0.2, color='red',
                      label='Signal reduced (overbought)')
    ax2.fill_between(rsi_raw.index, rsi_raw, 30,
                      where=rsi_raw <= 30, alpha=0.2, color='green',
                      label='Signal boosted (oversold)')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI', fontsize=8, color='gray')
    ax2.legend(loc='upper right', fontsize=7)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '32_rsi_counter_trend.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/32_rsi_counter_trend.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 2 — Signal History for All Assets
# ════════════════════════════════════════════════════════════
print("Creating Chart 2: Signal History...")

fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
fig.suptitle('Signal Engine v2 — Daily Signals for All Assets (2010–2024)\n'
             'Global Macro Intelligence System',
             fontsize=13, fontweight='bold')

assets_plot = [
    ('NIFTY 50',  nifty,  nifty_signal,  '#1F3864'),
    ('S&P 500',   sp500,  sp500_signal,  '#2E75B6'),
    ('Gold',      gold,   gold_signal,   '#C55A11'),
    ('Silver',    silver, silver_signal, '#7030A0'),
    ('Crude WTI', crude,  crude_signal,  '#1E6B3C'),
]

for idx, (name, price, signal, color) in enumerate(assets_plot):
    ax  = axes[idx]
    ax2 = ax.twinx()

    aligned_sig = signal.reindex(price.index).ffill()

    ax.fill_between(price.index, 0, 1,
                     where=aligned_sig >= 0.15,
                     transform=ax.get_xaxis_transform(),
                     alpha=0.15, color='green', label='Long')
    ax.fill_between(price.index, 0, 1,
                     where=aligned_sig <= -0.15,
                     transform=ax.get_xaxis_transform(),
                     alpha=0.15, color='red', label='Short')

    ax.plot(price.index, price, color=color, linewidth=1, alpha=0.9)
    ax.set_ylabel(name, fontsize=9, color=color)

    ax2.plot(aligned_sig.index, aligned_sig,
              color='gray', linewidth=0.6, alpha=0.5)
    ax2.axhline( 0.15, color='green', linestyle=':', linewidth=0.6)
    ax2.axhline(-0.15, color='red',   linestyle=':', linewidth=0.6)
    ax2.axhline( 0,    color='black', linestyle='--', linewidth=0.4)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_ylabel('Score', fontsize=7, color='gray')

    if idx == 0:
        ax.legend(loc='upper left', fontsize=7)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '33_signal_history.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/33_signal_history.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 3 — Signal Backtest (Forward Returns by Signal)
# ════════════════════════════════════════════════════════════
print("Creating Chart 3: Signal Backtest...")

def backtest_signal(price, signal, forward_days=20):
    aligned = signal.reindex(price.index).ffill()
    fwd_ret = price.pct_change(forward_days).shift(-forward_days) * 100
    results = pd.DataFrame({
        'signal':  aligned.apply(lambda x: score_to_signal(x)[0]),
        'fwd_ret': fwd_ret,
    }).dropna()
    return results.groupby('signal')['fwd_ret'].agg(['mean','count','std'])

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Signal Engine v2 Backtest — Forward Returns by Signal (2010–2024)\n'
             '20-day forward return when signal is Long vs Short vs Neutral\n'
             'v2 improvements: RSI counter-trend + ROC short-term momentum',
             fontsize=12, fontweight='bold')
axes = axes.flatten()

bt_assets = [
    ('NIFTY 50', nifty,  nifty_signal,  '#1F3864'),
    ('S&P 500',  sp500,  sp500_signal,  '#2E75B6'),
    ('Gold',     gold,   gold_signal,   '#C55A11'),
    ('Silver',   silver, silver_signal, '#7030A0'),
    ('Crude',    crude,  crude_signal,  '#1E6B3C'),
]

all_bt_results = {}

for idx, (name, price, signal, color) in enumerate(bt_assets):
    bt = backtest_signal(price, signal, forward_days=20)
    all_bt_results[name] = bt

    ax = axes[idx]
    if not bt.empty:
        signal_order = [s for s in ['Long','Neutral','Short'] if s in bt.index]
        means  = bt.loc[signal_order, 'mean']
        bar_colors = ['#1E6B3C' if s == 'Long' else
                       '#C00000' if s == 'Short' else '#C55A11'
                       for s in signal_order]

        bars = ax.bar(signal_order, means, color=bar_colors, alpha=0.85)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('Avg 20-day Return (%)', fontsize=9)

        # Check if Long > Short (quality indicator)
        long_ret  = bt.loc['Long',  'mean'] if 'Long'  in bt.index else 0
        short_ret = bt.loc['Short', 'mean'] if 'Short' in bt.index else 0
        quality   = '✅' if long_ret > short_ret else '⚠️'

        ax.set_title(f'{name} {quality}', fontsize=11, fontweight='bold')

        for bar, val, cnt in zip(bars, means,
                                  bt.loc[signal_order, 'count']):
            ax.text(bar.get_x() + bar.get_width()/2,
                     val + 0.1 if val >= 0 else val - 0.3,
                     f'{val:+.2f}%\n(n={cnt:.0f})',
                     ha='center', va='bottom', fontsize=9)

axes[-1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '34_signal_backtest_v2.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/34_signal_backtest_v2.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 4 — Current Signal Dashboard
# ════════════════════════════════════════════════════════════
print("Creating Chart 4: Current Signal Dashboard...")

latest_scores = {
    'NIFTY 50':  nifty_signal.dropna().iloc[-1],
    'S&P 500':   sp500_signal.dropna().iloc[-1],
    'Gold':      gold_signal.dropna().iloc[-1],
    'Silver':    silver_signal.dropna().iloc[-1],
    'Crude WTI': crude_signal.dropna().iloc[-1],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Current Signal Dashboard v2 — {signals_df.index[-1].strftime("%d %B %Y")}\n'
             f'Global Macro Intelligence System',
             fontsize=13, fontweight='bold')

names  = list(latest_scores.keys())
scores = list(latest_scores.values())
colors = ['#1E6B3C' if s >= 0.15 else '#C00000' if s <= -0.15
           else '#C55A11' for s in scores]

bars = ax1.barh(names, scores, color=colors, alpha=0.85, height=0.5)
ax1.axvline(0,     color='black', linewidth=1)
ax1.axvline( 0.15, color='green', linestyle='--', linewidth=0.8,
              label='Long threshold (+0.15)')
ax1.axvline(-0.15, color='red',   linestyle='--', linewidth=0.8,
              label='Short threshold (-0.15)')
ax1.set_xlim(-1, 1)
ax1.set_xlabel('Signal Score (-1 to +1)', fontsize=10)
ax1.set_title('Current Signal Scores — v2 (with RSI fix)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)

for bar, score in zip(bars, scores):
    label, _ = score_to_signal(score)
    ax1.text(score + 0.03 if score >= 0 else score - 0.03,
              bar.get_y() + bar.get_height()/2,
              f'{label} ({score:+.2f})',
              ha='left' if score >= 0 else 'right',
              va='center', fontsize=10, fontweight='bold')

signal_labels = [score_to_signal(s)[0] for s in scores]
counts = pd.Series(signal_labels).value_counts()
pie_colors = {'Long':'#1E6B3C','Short':'#C00000','Neutral':'#C55A11'}
ax2.pie(counts, labels=counts.index,
         colors=[pie_colors.get(l, 'gray') for l in counts.index],
         autopct='%1.0f%%', startangle=90,
         textprops={'fontsize': 12})
ax2.set_title('Signal Distribution Today', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '35_current_signals_v2.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/35_current_signals_v2.png ✓")

# ════════════════════════════════════════════════════════════
# KEY INSIGHTS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("KEY INSIGHTS — SIGNAL ENGINE v2 (with RSI fix)")
print("=" * 60)

print(f"\n📡 CURRENT SIGNALS ({signals_df.index[-1].strftime('%d %B %Y')}):")
print(f"  {'Asset':<15} {'Score':>8}  {'Signal':<10} {'Conviction'}")
print("  " + "-" * 52)
for name, score in latest_scores.items():
    label, _ = score_to_signal(score)
    conviction = 'HIGH' if abs(score) > 0.5 else \
                  'MEDIUM' if abs(score) > 0.3 else 'LOW'
    arrow = '▲' if label=='Long' else '▼' if label=='Short' else '→'
    print(f"  {name:<15} {score:>+8.3f}  {arrow} {label:<8} {conviction}")

print(f"\n📊 BACKTEST RESULTS — v1 vs v2 comparison:")
print(f"  {'Asset':<12} {'Long Avg':>10} {'Short Avg':>10} "
      f"{'Edge':>8} {'Quality'}")
print("  " + "-" * 55)
for name, bt in all_bt_results.items():
    long_ret  = bt.loc['Long',  'mean'] if 'Long'  in bt.index else 0
    short_ret = bt.loc['Short', 'mean'] if 'Short' in bt.index else 0
    edge      = long_ret - short_ret
    quality   = '✅ GOOD' if long_ret > short_ret else '⚠️ NEEDS WORK'
    print(f"  {name:<12} {long_ret:>+10.2f}% {short_ret:>+10.2f}% "
          f"{edge:>+8.2f}% {quality}")

print(f"\n🔧 WHAT THE RSI FIX DOES:")
print(f"  When RSI > 80 → score reduced by -1.0 (stops chasing overbought rallies)")
print(f"  When RSI > 70 → score reduced by -0.5 (caution on stretched markets)")
print(f"  When RSI < 30 → score boosted by +0.5  (reduces shorts near bottoms)")
print(f"  When RSI < 20 → score boosted by +1.0  (strong oversold bounce signal)")

print(f"\n📈 CURRENT RSI READINGS:")
rsi_readings = {
    'NIFTY 50':  nifty_rsi_raw.dropna().iloc[-1],
    'S&P 500':   sp500_rsi_raw.dropna().iloc[-1],
    'Gold':      gold_rsi_raw.dropna().iloc[-1],
    'Silver':    silver_rsi_raw.dropna().iloc[-1],
    'Crude':     crude_rsi_raw.dropna().iloc[-1],
}
for name, rsi_val in rsi_readings.items():
    status = 'OVERBOUGHT ⚠️' if rsi_val > 70 else \
              'OVERSOLD ✅' if rsi_val < 30 else 'NEUTRAL'
    print(f"  {name:<12} RSI: {rsi_val:.1f}  → {status}")

print("=" * 60)
print(f"\nAll 4 charts saved to your outputs/ folder.")
print(f"Total charts built so far: 35")
print(f"\nSignals saved to database table: SIGNALS")
print(f"Ready for backtesting in Week 13.")

# ════════════════════════════════════════════════════════════
# DIAGNOSTIC — Bull Market Bias Check
# The correct evaluation in a bull market is not
# Long vs Short absolute returns but Long vs market average
# ════════════════════════════════════════════════════════════

print(f"\n🔍 BULL MARKET BIAS DIAGNOSTIC:")
print(f"  In a 15-year bull market, ALL 20-day forward returns")
print(f"  tend to be positive — even during 'Short' signals.")
print(f"  The correct metric is: does Long BEAT the market average?\n")

print(f"  {'Asset':<12} {'Mkt Avg':>10} {'Long Avg':>10} "
      f"{'Short Avg':>10} {'Long Edge vs Mkt'}")
print("  " + "-" * 62)

for name, bt in all_bt_results.items():
    price = {
        'NIFTY 50': nifty, 'S&P 500': sp500,
        'Gold': gold, 'Silver': silver, 'Crude': crude
    }[name]

    # Market average 20-day forward return
    mkt_avg = price.pct_change(20).shift(-20).dropna().mean() * 100

    long_ret  = bt.loc['Long',  'mean'] if 'Long'  in bt.index else 0
    short_ret = bt.loc['Short', 'mean'] if 'Short' in bt.index else 0
    edge_vs_mkt = long_ret - mkt_avg
    quality = '✅ BEATS MARKET' if edge_vs_mkt > 0 else '⚠️ LAGS MARKET'

    print(f"  {name:<12} {mkt_avg:>+10.2f}% {long_ret:>+10.2f}% "
          f"{short_ret:>+10.2f}% {edge_vs_mkt:>+8.2f}%  {quality}")

print(f"\n  KEY INSIGHT: If Long Avg > Market Avg, the signal")
print(f"  is correctly overweighting the best periods.")
print(f"  If Short Avg > Market Avg, it confirms bull market")
print(f"  bias — shorts during dips that quickly recovered.")
