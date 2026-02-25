# ============================================================
# MODULE 11 — FULL BACKTESTING SUITE
# Tests signal engine against 15 years of real data
# Calculates: Sharpe ratio, max drawdown, win rate,
# CAGR, and compares against buy-and-hold benchmark
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

print("Loading prices and signals from database...")

def load_close(table):
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    close_col = [c for c in df.columns if 'Close' in c or 'close' in c]
    return df[close_col[0]].dropna() if close_col else df.iloc[:, 0].dropna()

# ── Load prices ───────────────────────────────────────────────
nifty  = load_close('NIFTY50')
sp500  = load_close('SP500')
gold   = load_close('GOLD')
silver = load_close('SILVER')
crude  = load_close('CRUDE_WTI')

# ── Load signals ──────────────────────────────────────────────
try:
    signals = pd.read_sql("SELECT * FROM SIGNALS", conn)
    signals['Date'] = pd.to_datetime(signals['Date'])
    signals = signals.set_index('Date').sort_index()
    print(f"Signals loaded: {len(signals)} dates")
except Exception as e:
    print(f"Error loading signals: {e}")
    signals = pd.DataFrame()

conn.close()

# ════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ════════════════════════════════════════════════════════════

def run_backtest(price, signal_scores, name,
                 transaction_cost=0.001,
                 signal_threshold=0.15):
    """
    Full backtesting engine.

    Rules:
    - Signal score >= +threshold → Long (invested 100%)
    - Signal score <= -threshold → Short (cash, or -100% if shorting)
    - Between thresholds         → Neutral (cash, 0% invested)

    Transaction cost: 0.1% per trade (realistic for ETFs/futures)
    No leverage used.
    Shorting = go to cash (conservative assumption)

    Returns full performance metrics and equity curve.
    """
    # Align signal to price index
    sig = signal_scores.reindex(price.index).ffill().bfill()
    daily_ret = price.pct_change().fillna(0)

    # Position: 1 = Long, 0 = Neutral/Short (cash)
    position = pd.Series(0.0, index=price.index)
    position[sig >= signal_threshold]  =  1.0   # Long
    position[sig <= -signal_threshold] =  0.0   # Cash (conservative)

    # Detect position changes for transaction costs
    position_change = position.diff().abs().fillna(0)
    trades          = (position_change > 0).sum()

    # Strategy daily returns
    strat_ret = position.shift(1) * daily_ret
    strat_ret -= position_change * transaction_cost  # Apply costs

    # Buy-and-hold returns
    bnh_ret = daily_ret.copy()

    # Equity curves (start at 100)
    strat_equity = (1 + strat_ret).cumprod() * 100
    bnh_equity   = (1 + bnh_ret).cumprod()   * 100

    # ── Performance Metrics ───────────────────────────────────
    trading_days = 252

    def calc_metrics(returns, equity):
        total_ret   = equity.iloc[-1] / equity.iloc[0] - 1
        n_years     = len(returns) / trading_days
        if n_years > 0 and equity.iloc[-1] > 0:
            cagr = (equity.iloc[-1] / 100.0) ** (1/n_years) - 1
        else:
            cagr = 0.0


        sharpe      = (returns.mean() / (returns.std() + 1e-10)) * \
                       np.sqrt(trading_days)

        # Max drawdown
        roll_max    = equity.cummax()
        drawdown    = (equity - roll_max) / roll_max
        max_dd      = drawdown.min()

        # Calmar ratio = CAGR / abs(Max Drawdown)
        calmar      = cagr / abs(max_dd + 1e-10)

        # Win rate (% of days with positive return)
        win_rate    = (returns > 0).sum() / (returns != 0).sum()

        # Sortino ratio (penalises downside volatility only)
        downside    = returns[returns < 0].std() + 1e-10
        sortino     = (returns.mean() / downside) * np.sqrt(trading_days)

        # Volatility (annualised)
        volatility  = returns.std() * np.sqrt(trading_days)

        return {
            'Total Return':  total_ret  * 100,
            'CAGR':          cagr       * 100,
            'Sharpe':        sharpe,
            'Sortino':       sortino,
            'Calmar':        calmar,
            'Max Drawdown':  max_dd     * 100,
            'Volatility':    volatility * 100,
            'Win Rate':      win_rate   * 100,
        }

    strat_metrics = calc_metrics(strat_ret,  strat_equity)
    bnh_metrics   = calc_metrics(bnh_ret,    bnh_equity)

    strat_metrics['Trades'] = trades
    strat_metrics['Avg Days in Market'] = position.mean() * 100

    return {
        'name':          name,
        'strat_equity':  strat_equity,
        'bnh_equity':    bnh_equity,
        'strat_ret':     strat_ret,
        'bnh_ret':       bnh_ret,
        'position':      position,
        'strat_metrics': strat_metrics,
        'bnh_metrics':   bnh_metrics,
        'drawdown':      (strat_equity - strat_equity.cummax()) /
                          strat_equity.cummax() * 100,
    }

# ── Run backtests for all assets ──────────────────────────────
print("\nRunning backtests...\n")

backtest_configs = [
    ('NIFTY 50', nifty,  'NIFTY_score'),
    ('S&P 500',  sp500,  'SP500_score'),
    ('Gold',     gold,   'Gold_score'),
    ('Silver',   silver, 'Silver_score'),
    ('Crude',    crude,  'Crude_score'),
]

COLORS = {
    'NIFTY 50': '#1F3864',
    'S&P 500':  '#2E75B6',
    'Gold':     '#C55A11',
    'Silver':   '#7030A0',
    'Crude':    '#1E6B3C',
}

results = {}
for name, price, score_col in backtest_configs:
    if not signals.empty and score_col in signals.columns:
        score_series = signals[score_col]
    else:
        # If signals not available, use zero scores
        score_series = pd.Series(0.0, index=price.index)

    bt = run_backtest(price, score_series, name)
    results[name] = bt

    sm = bt['strat_metrics']
    bm = bt['bnh_metrics']
    print(f"  {name}:")
    print(f"    Strategy: CAGR={sm['CAGR']:+.1f}%  "
          f"Sharpe={sm['Sharpe']:.2f}  "
          f"MaxDD={sm['Max Drawdown']:.1f}%  "
          f"WinRate={sm['Win Rate']:.1f}%")
    print(f"    B&H:      CAGR={bm['CAGR']:+.1f}%  "
          f"Sharpe={bm['Sharpe']:.2f}  "
          f"MaxDD={bm['Max Drawdown']:.1f}%")
    print()

# ════════════════════════════════════════════════════════════
# CHART 1 — Equity Curves (Strategy vs Buy-and-Hold)
# ════════════════════════════════════════════════════════════
print("Creating Chart 1: Equity Curves...")

fig, axes = plt.subplots(3, 2, figsize=(15, 14))
fig.suptitle('Backtest — Strategy vs Buy-and-Hold Equity Curves (2010–2024)\n'
             'Starting value: 100 | Transaction cost: 0.1% per trade\n'
             'Global Macro Intelligence System',
             fontsize=12, fontweight='bold')
axes = axes.flatten()

for idx, (name, bt) in enumerate(results.items()):
    ax = axes[idx]
    color = COLORS[name]

    ax.plot(bt['strat_equity'].index,
             bt['strat_equity'].values,
             color=color, linewidth=1.8,
             label=f"Signal Strategy")
    ax.plot(bt['bnh_equity'].index,
             bt['bnh_equity'].values,
             color='gray', linewidth=1.2,
             linestyle='--', alpha=0.7,
             label=f"Buy & Hold")

    sm = bt['strat_metrics']
    bm = bt['bnh_metrics']

    # Shade outperformance
    strat_aligned = bt['strat_equity']
    bnh_aligned   = bt['bnh_equity'].reindex(strat_aligned.index)
    ax.fill_between(strat_aligned.index,
                     strat_aligned, bnh_aligned,
                     where=strat_aligned >= bnh_aligned,
                     alpha=0.1, color='green',
                     label='Strategy ahead')
    ax.fill_between(strat_aligned.index,
                     strat_aligned, bnh_aligned,
                     where=strat_aligned < bnh_aligned,
                     alpha=0.1, color='red',
                     label='B&H ahead')

    beat = sm['CAGR'] > bm['CAGR']
    title_color = '#1E6B3C' if beat else '#C00000'
    ax.set_title(
        f"{name} — Strategy: {sm['CAGR']:+.1f}% CAGR | "
        f"B&H: {bm['CAGR']:+.1f}% CAGR",
        fontsize=10, fontweight='bold', color=title_color
    )
    ax.set_ylabel('Portfolio Value (Base 100)', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=30, labelsize=8)

axes[-1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '36_equity_curves.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/36_equity_curves.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 2 — Drawdown Analysis
# ════════════════════════════════════════════════════════════
print("Creating Chart 2: Drawdown Analysis...")

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Maximum Drawdown Analysis — Strategy vs Buy-and-Hold\n'
             'Global Macro Intelligence System',
             fontsize=12, fontweight='bold')
axes = axes.flatten()

for idx, (name, bt) in enumerate(results.items()):
    ax = axes[idx]

    # Strategy drawdown
    strat_dd = bt['drawdown']

    # B&H drawdown
    bnh_eq  = bt['bnh_equity']
    bnh_dd  = (bnh_eq - bnh_eq.cummax()) / bnh_eq.cummax() * 100

    ax.fill_between(strat_dd.index, strat_dd, 0,
                     alpha=0.5, color=COLORS[name],
                     label=f"Strategy DD")
    ax.plot(bnh_dd.index, bnh_dd,
             color='gray', linewidth=0.8,
             linestyle='--', alpha=0.7,
             label=f"B&H DD")

    sm = bt['strat_metrics']
    bm = bt['bnh_metrics']

    ax.set_title(
        f"{name} — Strategy MaxDD: {sm['Max Drawdown']:.1f}% | "
        f"B&H MaxDD: {bm['Max Drawdown']:.1f}%",
        fontsize=10, fontweight='bold'
    )
    ax.set_ylabel('Drawdown (%)', fontsize=9)
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=30, labelsize=8)

axes[-1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '37_drawdown_analysis.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/37_drawdown_analysis.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 3 — Performance Metrics Comparison Table
# ════════════════════════════════════════════════════════════
print("Creating Chart 3: Performance Metrics Table...")

fig, ax = plt.subplots(figsize=(15, 8))
ax.axis('off')
fig.suptitle('Full Performance Metrics — Signal Strategy vs Buy-and-Hold\n'
             'Global Macro Intelligence System | 2010–2024',
             fontsize=13, fontweight='bold')

# Build table
headers = ['Asset', 'CAGR\nStrat', 'CAGR\nB&H', 'Sharpe\nStrat',
           'Sharpe\nB&H', 'MaxDD\nStrat', 'MaxDD\nB&H',
           'Sortino\nStrat', 'Win\nRate', 'Trades', '% In\nMarket']

rows = []
for name, bt in results.items():
    sm = bt['strat_metrics']
    bm = bt['bnh_metrics']
    rows.append([
        name,
        f"{sm['CAGR']:+.1f}%",
        f"{bm['CAGR']:+.1f}%",
        f"{sm['Sharpe']:.2f}",
        f"{bm['Sharpe']:.2f}",
        f"{sm['Max Drawdown']:.1f}%",
        f"{bm['Max Drawdown']:.1f}%",
        f"{sm['Sortino']:.2f}",
        f"{sm['Win Rate']:.1f}%",
        f"{sm['Trades']:.0f}",
        f"{sm['Avg Days in Market']:.1f}%",
    ])

table = ax.table(
    cellText=rows,
    colLabels=headers,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(10)

# Style header
for j in range(len(headers)):
    table[0, j].set_facecolor('#1F3864')
    table[0, j].set_text_props(color='white', fontweight='bold')
    table[0, j].set_height(0.15)

# Style rows — green if strategy beats B&H CAGR
for i, (name, bt) in enumerate(results.items(), 1):
    sm = bt['strat_metrics']
    bm = bt['bnh_metrics']
    beats = sm['CAGR'] > bm['CAGR']
    row_color = '#D5E8D4' if beats else '#FFCCCC'
    for j in range(len(headers)):
        table[i, j].set_facecolor(row_color)
        table[i, j].set_height(0.12)
        if j == 0:
            table[i, j].set_text_props(fontweight='bold')

# Highlight best Sharpe ratio cells
sharpe_vals = [results[n]['strat_metrics']['Sharpe'] for n in results]
best_sharpe_idx = np.argmax(sharpe_vals) + 1
table[best_sharpe_idx, 3].set_facecolor('#C6E0B4')
table[best_sharpe_idx, 3].set_text_props(fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '38_performance_table.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/38_performance_table.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 4 — Rolling Sharpe Ratio
# ════════════════════════════════════════════════════════════
print("Creating Chart 4: Rolling Sharpe Ratio...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Rolling 252-Day Sharpe Ratio — Strategy vs Buy-and-Hold\n'
             'Measures risk-adjusted returns over time\n'
             'Global Macro Intelligence System',
             fontsize=12, fontweight='bold')
axes = axes.flatten()

plot_assets = ['NIFTY 50', 'S&P 500', 'Gold', 'Crude']

for idx, name in enumerate(plot_assets):
    bt  = results[name]
    ax  = axes[idx]

    # Rolling Sharpe
    strat_roll_sharpe = (
        bt['strat_ret'].rolling(252).mean() /
        (bt['strat_ret'].rolling(252).std() + 1e-10)
    ) * np.sqrt(252)

    bnh_roll_sharpe = (
        bt['bnh_ret'].rolling(252).mean() /
        (bt['bnh_ret'].rolling(252).std() + 1e-10)
    ) * np.sqrt(252)

    ax.plot(strat_roll_sharpe.index,
             strat_roll_sharpe,
             color=COLORS[name], linewidth=1.5,
             label='Strategy Sharpe')
    ax.plot(bnh_roll_sharpe.index,
             bnh_roll_sharpe,
             color='gray', linewidth=1,
             linestyle='--', alpha=0.7,
             label='B&H Sharpe')
    ax.axhline(0,   color='black', linestyle='--', linewidth=0.8)
    ax.axhline(1.0, color='green', linestyle=':',  linewidth=0.8,
                label='Sharpe = 1.0 (good)')
    ax.axhline(0.5, color='orange',linestyle=':',  linewidth=0.8,
                label='Sharpe = 0.5 (acceptable)')

    sm = bt['strat_metrics']
    ax.set_title(f"{name} — Overall Sharpe: {sm['Sharpe']:.2f}",
                  fontsize=11, fontweight='bold')
    ax.set_ylabel('Rolling Sharpe Ratio', fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylim(-2, 3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '39_rolling_sharpe.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/39_rolling_sharpe.png ✓")

# ════════════════════════════════════════════════════════════
# PORTFOLIO BACKTEST — Combine all 5 signals equally
# ════════════════════════════════════════════════════════════
print("\nRunning combined portfolio backtest...")

# Equal weight portfolio of all 5 strategies
portfolio_ret = pd.DataFrame({
    name: bt['strat_ret'] for name, bt in results.items()
}).dropna()

portfolio_ret['Portfolio'] = portfolio_ret.mean(axis=1)
portfolio_equity = (1 + portfolio_ret['Portfolio']).cumprod() * 100

# B&H equal weight
bnh_portfolio = pd.DataFrame({
    name: bt['bnh_ret'] for name, bt in results.items()
}).dropna()
bnh_portfolio['Portfolio'] = bnh_portfolio.mean(axis=1)
bnh_equity = (1 + bnh_portfolio['Portfolio']).cumprod() * 100

# Portfolio metrics
port_ret   = portfolio_ret['Portfolio']
port_cagr  = ((portfolio_equity.iloc[-1]/100) **
               (252/len(port_ret)) - 1) * 100
port_sharpe= (port_ret.mean() / (port_ret.std() + 1e-10)) * np.sqrt(252)
port_dd    = ((portfolio_equity - portfolio_equity.cummax()) /
               portfolio_equity.cummax() * 100).min()
port_sortino= (port_ret.mean() /
               (port_ret[port_ret < 0].std() + 1e-10)) * np.sqrt(252)

bnh_ret_s  = bnh_portfolio['Portfolio']
bnh_cagr   = ((bnh_equity.iloc[-1]/100) **
               (252/len(bnh_ret_s)) - 1) * 100
bnh_sharpe = (bnh_ret_s.mean() /
               (bnh_ret_s.std() + 1e-10)) * np.sqrt(252)
bnh_dd     = ((bnh_equity - bnh_equity.cummax()) /
               bnh_equity.cummax() * 100).min()

# ════════════════════════════════════════════════════════════
# CHART 5 — Combined Portfolio Performance
# ════════════════════════════════════════════════════════════
print("Creating Chart 5: Combined Portfolio...")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Combined Equal-Weight Portfolio — All 5 Signal Strategies\n'
             'Global Macro Intelligence System | 2010–2024',
             fontsize=13, fontweight='bold')

ax1.plot(portfolio_equity.index, portfolio_equity,
          color='#1F3864', linewidth=2,
          label=f'Signal Portfolio (CAGR: {port_cagr:+.1f}%)')
ax1.plot(bnh_equity.index, bnh_equity,
          color='gray', linewidth=1.2, linestyle='--',
          label=f'B&H Portfolio (CAGR: {bnh_cagr:+.1f}%)')
# Align indices before comparing
bnh_aligned = bnh_equity.reindex(portfolio_equity.index).ffill()
ax1.fill_between(portfolio_equity.index,
                  portfolio_equity, bnh_aligned,
                  where=portfolio_equity.values >= bnh_aligned.values,
                  alpha=0.1, color='green')
ax1.fill_between(portfolio_equity.index,
                  portfolio_equity, bnh_aligned,
                  where=portfolio_equity.values < bnh_aligned.values,
                  alpha=0.1, color='red')

ax1.set_ylabel('Portfolio Value (Base 100)', fontsize=10)
ax1.set_title(f'Equity Curve | Sharpe: {port_sharpe:.2f} vs '
               f'B&H Sharpe: {bnh_sharpe:.2f}',
               fontsize=11)
ax1.legend(fontsize=9)

# Drawdown
port_drawdown = (portfolio_equity - portfolio_equity.cummax()) / \
                 portfolio_equity.cummax() * 100
ax2.fill_between(port_drawdown.index, port_drawdown, 0,
                  alpha=0.6, color='#C00000',
                  label=f'Max Drawdown: {port_dd:.1f}%')
ax2.set_ylabel('Drawdown (%)', fontsize=10)
ax2.set_title('Portfolio Drawdown', fontsize=11)
ax2.legend(fontsize=9)

# Monthly returns heatmap data as bar chart
monthly_port = portfolio_ret['Portfolio'].resample('ME').sum() * 100
colors_bar   = ['#1E6B3C' if r >= 0 else '#C00000'
                  for r in monthly_port]
ax3.bar(monthly_port.index, monthly_port,
         color=colors_bar, alpha=0.8, width=20)
ax3.axhline(0, color='black', linewidth=0.8)
ax3.set_ylabel('Monthly Return (%)', fontsize=10)
ax3.set_title('Monthly Returns', fontsize=11)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '40_portfolio_backtest.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/40_portfolio_backtest.png ✓")

# ════════════════════════════════════════════════════════════
# KEY INSIGHTS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("KEY INSIGHTS — FULL BACKTEST RESULTS")
print("=" * 65)

print(f"\n{'Asset':<12} {'Strat CAGR':>12} {'B&H CAGR':>10} "
      f"{'Strat Sharpe':>14} {'B&H Sharpe':>12} "
      f"{'Strat MaxDD':>12} {'B&H MaxDD':>10} {'Result'}")
print("  " + "-" * 90)

beat_count = 0
for name, bt in results.items():
    sm   = bt['strat_metrics']
    bm   = bt['bnh_metrics']
    beat = sm['CAGR'] > bm['CAGR']
    if beat:
        beat_count += 1
    verdict = '✅ BEATS B&H' if beat else '⚠️ LAGS B&H'
    print(f"  {name:<12} {sm['CAGR']:>+11.1f}% {bm['CAGR']:>+9.1f}% "
          f"{sm['Sharpe']:>14.2f} {bm['Sharpe']:>12.2f} "
          f"{sm['Max Drawdown']:>11.1f}% {bm['Max Drawdown']:>9.1f}% "
          f"  {verdict}")

print(f"\n  Signal strategy beats B&H in "
      f"{beat_count}/{len(results)} assets")

print(f"\n📊 COMBINED PORTFOLIO RESULTS:")
print(f"  Strategy CAGR:     {port_cagr:+.2f}%")
print(f"  B&H CAGR:          {bnh_cagr:+.2f}%")
print(f"  Strategy Sharpe:   {port_sharpe:.3f}")
print(f"  B&H Sharpe:        {bnh_sharpe:.3f}")
print(f"  Strategy Max DD:   {port_dd:.2f}%")
print(f"  Strategy Sortino:  {port_sortino:.3f}")

print(f"\n📈 SHARPE RATIO INTERPRETATION:")
print(f"  < 0:    Losing money on risk-adjusted basis")
print(f"  0–0.5:  Below average")
print(f"  0.5–1:  Acceptable — most active funds sit here")
print(f"  1–2:    Good — top quartile hedge fund territory")
print(f"  > 2:    Excellent — likely overfitting if in-sample")

print(f"\n🏆 WHAT TO TELL INTERVIEWERS:")
print(f"  'I built a multi-asset signal engine combining momentum,")
print(f"  mean reversion (RSI), macro regime, yield curve, FX,")
print(f"  and sentiment signals. Backtested on 15 years of real")
print(f"  data with realistic transaction costs of 0.1% per trade.")
print(f"  The combined portfolio achieved a Sharpe ratio of")
print(f"  {port_sharpe:.2f} vs {bnh_sharpe:.2f} for buy-and-hold,")
print(f"  with a maximum drawdown of {port_dd:.1f}%.'")

print("=" * 65)
print(f"\nAll 5 charts saved to your outputs/ folder.")
print(f"Total charts built so far: 40")
