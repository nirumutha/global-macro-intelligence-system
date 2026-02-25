# ============================================================
# MODULE 8 — MACRO SCORECARD & ECONOMIC SURPRISE INDEX
# Classifies the macro regime and measures how data
# surprises vs expectations
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import sqlite3
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')
OUT_PATH  = os.path.join(BASE_PATH, 'outputs')
DB_PATH   = os.path.join(DATA_PATH, 'macro_system.db')
os.makedirs(OUT_PATH, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

print("Loading macro data from database...")

def load_macro(table):
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    return df.iloc[:, 0].dropna()

def load_close(table):
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    close_col = [c for c in df.columns if 'Close' in c or 'close' in c]
    return df[close_col[0]].dropna() if close_col else df.iloc[:, 0].dropna()

# ── Load all macro series ─────────────────────────────────────
cpi          = load_macro('US_CPI')
fed_rate     = load_macro('US_FED_RATE')
unemployment = load_macro('US_UNEMPLOYMENT')
gdp          = load_macro('US_GDP')
payrolls     = load_macro('US_PAYROLLS')
pce          = load_macro('US_PCE')
yield_10y    = load_macro('US_10Y_YIELD')
yield_2y     = load_macro('US_2Y_YIELD')

# ── Load market prices ────────────────────────────────────────
sp500  = load_close('SP500')
nifty  = load_close('NIFTY50')
gold   = load_close('GOLD')
vix    = load_close('VIX_US')

conn.close()

# ── Calculate derived macro indicators ───────────────────────
# CPI year-over-year inflation
cpi_yoy = cpi.pct_change(12) * 100

# GDP quarter-over-quarter growth
gdp_qoq = gdp.pct_change(1) * 100

# Payroll month-over-month change (thousands)
payroll_mom = payrolls.diff(1)

# Real Fed Funds Rate = Fed Rate minus CPI inflation
# Align to monthly
fed_monthly = fed_rate.resample('ME').last()
cpi_monthly = cpi_yoy.resample('ME').last()
real_rate   = (fed_monthly - cpi_monthly).dropna()

# Yield spread
yields = pd.DataFrame({'10Y': yield_10y, '2Y': yield_2y}).dropna()
spread = yields['10Y'] - yields['2Y']

# ════════════════════════════════════════════════════════════
# MACRO REGIME CLASSIFICATION
# Four regimes based on growth + inflation quadrant
# Goldilocks:  growth good, inflation low  → best for equities
# Overheating: growth good, inflation high → Fed hikes, mixed
# Stagflation: growth bad,  inflation high → worst for equities
# Recession:   growth bad,  inflation low  → bonds/gold rally
# ════════════════════════════════════════════════════════════

def classify_macro_regime(inflation, growth, unemployment_rate):
    """
    Classify macro regime based on inflation and growth.
    Returns regime name and colour.
    """
    high_inflation = inflation > 3.0    # Above 3% = elevated
    good_growth    = growth > 0         # Positive quarterly GDP
    high_unemp     = unemployment_rate > 5.0

    if not high_inflation and good_growth:
        return 'Goldilocks', '#1E6B3C'      # Green — best for equities
    elif high_inflation and good_growth:
        return 'Overheating', '#C55A11'     # Orange — Fed tightens
    elif high_inflation and not good_growth:
        return 'Stagflation', '#C00000'     # Red — worst scenario
    else:
        return 'Recession/Slowdown', '#7030A0'  # Purple — contraction

# Build monthly regime classification
# Align all monthly series
monthly_data = pd.DataFrame({
    'CPI_YOY':  cpi_yoy.resample('ME').last(),
    'GDP_QOQ':  gdp_qoq.resample('ME').last().ffill(),
    'UNEMP':    unemployment.resample('ME').last(),
}).dropna()

regimes = []
for date, row in monthly_data.iterrows():
    label, color = classify_macro_regime(
        row['CPI_YOY'], row['GDP_QOQ'], row['UNEMP']
    )
    regimes.append({'Date': date, 'Regime': label, 'Color': color})

regime_df = pd.DataFrame(regimes).set_index('Date')

# ════════════════════════════════════════════════════════════
# ECONOMIC SURPRISE INDEX
# Measures whether economic data is beating or missing
# consensus expectations
# We proxy consensus using trailing 12-month rolling average
# Surprise = actual minus expected (rolling mean)
# ════════════════════════════════════════════════════════════

def surprise_index(series, window=12):
    """
    Economic Surprise Index.
    Positive = data beating expectations (rolling mean proxy)
    Negative = data missing expectations
    """
    expected = series.rolling(window).mean()
    surprise = series - expected
    # Normalise to z-score for comparability
    surprise_z = (surprise - surprise.rolling(window).mean()) / \
                  surprise.rolling(window).std()
    return surprise_z.clip(-3, 3)

# Calculate surprise index for each indicator
surp_payrolls = surprise_index(payroll_mom)
surp_cpi      = surprise_index(cpi_yoy)
surp_unemp    = -surprise_index(unemployment)  # Invert: lower unemployment = positive surprise
surp_gdp      = surprise_index(gdp_qoq)

# Composite US Economic Surprise Index
# Weight: payrolls 35%, GDP 30%, unemployment 20%, CPI 15%
surp_composite = pd.DataFrame({
    'Payrolls': surp_payrolls,
    'GDP':      surp_gdp,
    'Unemp':    surp_unemp,
    'CPI':      surp_cpi,
}).dropna()

surp_composite['ESI'] = (
    surp_composite['Payrolls'] * 0.35 +
    surp_composite['GDP']      * 0.30 +
    surp_composite['Unemp']    * 0.20 +
    surp_composite['CPI']      * 0.15
)
surp_composite['ESI_smooth'] = surp_composite['ESI'].rolling(3).mean()

# ════════════════════════════════════════════════════════════
# MACRO SCORECARD — CURRENT STATE
# ════════════════════════════════════════════════════════════

def get_score(value, thresholds, labels):
    """Map a value to a score and label."""
    for i, t in enumerate(thresholds):
        if value <= t:
            return i, labels[i]
    return len(labels)-1, labels[-1]

# Get latest values
latest_cpi   = cpi_yoy.dropna().iloc[-1]
latest_gdp   = gdp_qoq.dropna().iloc[-1]
latest_unemp = unemployment.dropna().iloc[-1]
latest_fed   = fed_rate.dropna().iloc[-1]
latest_real  = real_rate.dropna().iloc[-1]
latest_spread= spread.dropna().iloc[-1]
latest_esi   = surp_composite['ESI_smooth'].dropna().iloc[-1]
latest_pay   = payroll_mom.dropna().iloc[-1]

scorecard = {
    'CPI Inflation':      {'value': f'{latest_cpi:.2f}%',   'signal': '🟢 Low'     if latest_cpi < 2.5  else '🟡 Elevated' if latest_cpi < 4 else '🔴 High'},
    'GDP Growth':         {'value': f'{latest_gdp:.2f}%',   'signal': '🟢 Strong'  if latest_gdp > 2    else '🟡 Moderate' if latest_gdp > 0 else '🔴 Contraction'},
    'Unemployment':       {'value': f'{latest_unemp:.1f}%', 'signal': '🟢 Low'     if latest_unemp < 4  else '🟡 Normal'   if latest_unemp < 5.5 else '🔴 High'},
    'Fed Funds Rate':     {'value': f'{latest_fed:.2f}%',   'signal': '🟢 Low'     if latest_fed < 2    else '🟡 Neutral'  if latest_fed < 4 else '🔴 Restrictive'},
    'Real Interest Rate': {'value': f'{latest_real:.2f}%',  'signal': '🟢 Positive' if latest_real > 0  else '🔴 Negative (gold bullish)'},
    'Yield Curve':        {'value': f'{latest_spread:.2f}%','signal': '🟢 Normal'  if latest_spread > 0.5 else '🟡 Flat'  if latest_spread > 0 else '🔴 Inverted'},
    'Surprise Index':     {'value': f'{latest_esi:.2f}',    'signal': '🟢 Beating' if latest_esi > 0.3  else '🟡 In-line' if latest_esi > -0.3 else '🔴 Missing'},
    'Nonfarm Payrolls':   {'value': f'{latest_pay:,.0f}K',  'signal': '🟢 Strong'  if latest_pay > 200  else '🟡 Moderate' if latest_pay > 100 else '🔴 Weak'},
}

# ════════════════════════════════════════════════════════════
# CHART 1 — Macro Regime Timeline
# ════════════════════════════════════════════════════════════
print("Creating Chart 1: Macro Regime Timeline...")

sp500_monthly = sp500.resample('ME').last()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Macro Regime Classification — Growth vs Inflation Quadrant (2010–2024)\n'
             'Global Macro Intelligence System',
             fontsize=13, fontweight='bold')

# Panel 1: S&P 500 with regime background
regime_color_map = {
    'Goldilocks':        '#1E6B3C',
    'Overheating':       '#C55A11',
    'Stagflation':       '#C00000',
    'Recession/Slowdown':'#7030A0',
}

for regime, color in regime_color_map.items():
    mask = regime_df['Regime'] == regime
    ax1.fill_between(regime_df.index, 0, 1,
                      where=mask.values,
                      transform=ax1.get_xaxis_transform(),
                      alpha=0.25, color=color, label=regime)

ax1.plot(sp500_monthly.index, sp500_monthly, color='#1F3864',
          linewidth=1.5, label='S&P 500')
ax1.set_ylabel('S&P 500', fontsize=10)
ax1.set_title('S&P 500 coloured by Macro Regime', fontsize=11)
ax1.legend(loc='upper left', fontsize=8, ncol=5)

# Panel 2: CPI Inflation
ax2.plot(cpi_yoy.index, cpi_yoy, color='#C00000', linewidth=1.2, label='CPI YoY%')
ax2.axhline(2,   color='green',  linestyle='--', linewidth=0.8, label='Target (2%)')
ax2.axhline(3,   color='orange', linestyle='--', linewidth=0.8, label='Elevated (3%)')
ax2.axhline(5,   color='red',    linestyle='--', linewidth=0.8, label='High (5%)')
ax2.fill_between(cpi_yoy.index, cpi_yoy, 2,
                  where=cpi_yoy > 2, alpha=0.15, color='red')
ax2.set_ylabel('CPI YoY %', fontsize=10)
ax2.set_title('US CPI Inflation — above 3% = overheating / stagflation risk', fontsize=11)
ax2.legend(fontsize=9, loc='upper left')

# Panel 3: Real Fed Funds Rate
ax3.plot(real_rate.index, real_rate, color='#2E75B6', linewidth=1.2,
          label='Real Fed Rate (Fed Rate - CPI)')
ax3.axhline(0,    color='black',  linestyle='--', linewidth=0.8)
ax3.axhline(2,    color='red',    linestyle=':',  linewidth=0.8,
             label='Restrictive territory (2%)')
ax3.fill_between(real_rate.index, real_rate, 0,
                  where=real_rate < 0, alpha=0.2, color='green',
                  label='Negative real rate (gold bullish)')
ax3.fill_between(real_rate.index, real_rate, 0,
                  where=real_rate >= 0, alpha=0.2, color='red',
                  label='Positive real rate (gold bearish)')
ax3.set_ylabel('Real Rate %', fontsize=10)
ax3.set_title('Real Fed Funds Rate — negative = financial repression = gold and equity bullish', fontsize=11)
ax3.legend(fontsize=9)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '24_macro_regime.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/24_macro_regime.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 2 — Economic Surprise Index
# ════════════════════════════════════════════════════════════
print("Creating Chart 2: Economic Surprise Index...")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle('US Economic Surprise Index — Is Data Beating Expectations? (2010–2024)\n'
             'Global Macro Intelligence System',
             fontsize=13, fontweight='bold')

# ESI
ax1.plot(surp_composite.index, surp_composite['ESI_smooth'],
          color='#2E75B6', linewidth=1.5, label='ESI (3m smoothed)')
ax1.axhline(0,    color='black',  linestyle='--', linewidth=0.8)
ax1.axhline(0.3,  color='green',  linestyle=':', linewidth=0.8,
             label='Beating expectations (+0.3)')
ax1.axhline(-0.3, color='red',    linestyle=':', linewidth=0.8,
             label='Missing expectations (-0.3)')
ax1.fill_between(surp_composite.index, surp_composite['ESI_smooth'], 0,
                  where=surp_composite['ESI_smooth'] >= 0,
                  alpha=0.25, color='green')
ax1.fill_between(surp_composite.index, surp_composite['ESI_smooth'], 0,
                  where=surp_composite['ESI_smooth'] < 0,
                  alpha=0.25, color='red')
ax1.set_ylabel('Surprise Index (Z-score)', fontsize=10)
ax1.set_title('Composite Economic Surprise Index — positive = data better than expected', fontsize=11)
ax1.legend(fontsize=9)

# Payrolls surprise
ax2.bar(surp_payrolls.index, surp_payrolls,
         color=np.where(surp_payrolls >= 0, '#1E6B3C', '#C00000'),
         alpha=0.7, width=20, label='Payrolls Surprise')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_ylabel('Surprise (Z-score)', fontsize=10)
ax2.set_title('Nonfarm Payrolls Surprise — green = beat, red = miss', fontsize=11)

# S&P 500 vs ESI
sp_ret_annual = sp500.pct_change(252) * 100
sp_monthly_ret = sp_ret_annual.resample('ME').last()

esi_sp_aligned = pd.DataFrame({
    'ESI': surp_composite['ESI_smooth'],
    'SP500_ret': sp_monthly_ret,
}).dropna()

ax3.plot(esi_sp_aligned.index, esi_sp_aligned['SP500_ret'],
          color='#1F3864', linewidth=1.2, label='S&P 500 1Y Return %')
ax3_twin = ax3.twinx()
ax3_twin.plot(esi_sp_aligned.index, esi_sp_aligned['ESI'],
               color='#C55A11', linewidth=1.2, linestyle='--',
               alpha=0.8, label='ESI')
ax3.set_ylabel('S&P 500 1Y Return %', fontsize=10, color='#1F3864')
ax3_twin.set_ylabel('ESI', fontsize=10, color='#C55A11')
ax3.set_title('S&P 500 Annual Return vs Economic Surprise — ESI leads market moves', fontsize=11)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '25_surprise_index.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/25_surprise_index.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 3 — Macro Scorecard Visual
# ════════════════════════════════════════════════════════════
print("Creating Chart 3: Macro Scorecard...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
fig.suptitle(
    f'US Macro Scorecard — Current State\n'
    f'Global Macro Intelligence System | Data as of {cpi_yoy.dropna().index[-1].strftime("%b %Y")}',
    fontsize=14, fontweight='bold', y=0.98
)

# Table data
table_data = [[k, v['value'], v['signal']] for k, v in scorecard.items()]
col_labels  = ['Indicator', 'Latest Value', 'Signal']

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(12)

# Style header
for j in range(3):
    table[0, j].set_facecolor('#1F3864')
    table[0, j].set_text_props(color='white', fontweight='bold')
    table[0, j].set_height(0.12)

# Style rows
for i in range(1, len(table_data) + 1):
    signal_text = table_data[i-1][2]
    if '🟢' in signal_text:
        row_color = '#D5E8D4'
    elif '🔴' in signal_text:
        row_color = '#FFCCCC'
    else:
        row_color = '#FFF3CD'

    for j in range(3):
        table[i, j].set_facecolor(row_color)
        table[i, j].set_height(0.10)
        if j == 0:
            table[i, j].set_text_props(fontweight='bold')

table.auto_set_column_width([0, 1, 2])
plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '26_macro_scorecard.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/26_macro_scorecard.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 4 — Regime vs Asset Performance
# ════════════════════════════════════════════════════════════
print("Creating Chart 4: Regime vs Asset Performance...")

# Calculate average monthly returns per asset in each regime
sp500_m  = sp500.resample('ME').last().pct_change() * 100
gold_m   = gold.resample('ME').last().pct_change() * 100
nifty_m  = nifty.resample('ME').last().pct_change() * 100

perf_df = pd.DataFrame({
    'SP500':  sp500_m,
    'Gold':   gold_m,
    'NIFTY':  nifty_m,
    'Regime': regime_df['Regime'],
}).dropna()

regime_perf = perf_df.groupby('Regime')[['SP500', 'Gold', 'NIFTY']].mean()

fig, ax = plt.subplots(figsize=(12, 7))
x     = np.arange(len(regime_perf))
width = 0.25

bars1 = ax.bar(x - width, regime_perf['SP500'], width,
                label='S&P 500', color='#2E75B6', alpha=0.85)
bars2 = ax.bar(x,          regime_perf['Gold'],  width,
                label='Gold',    color='#C55A11', alpha=0.85)
bars3 = ax.bar(x + width,  regime_perf['NIFTY'], width,
                label='NIFTY',   color='#1F3864', alpha=0.85)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(regime_perf.index, fontsize=11)
ax.set_ylabel('Average Monthly Return (%)', fontsize=11)
ax.set_title('Asset Performance by Macro Regime — Which assets win in each environment?\n'
             'Global Macro Intelligence System',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                h + 0.05 if h >= 0 else h - 0.15,
                f'{h:.2f}%', ha='center', va='bottom', fontsize=9)

# Colour regime labels
regime_colors_ax = {
    'Goldilocks':         '#1E6B3C',
    'Overheating':        '#C55A11',
    'Stagflation':        '#C00000',
    'Recession/Slowdown': '#7030A0',
}
for tick, label in zip(ax.get_xticklabels(), regime_perf.index):
    tick.set_color(regime_colors_ax.get(label, 'black'))
    tick.set_fontweight('bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '27_regime_asset_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/27_regime_asset_performance.png ✓")

# ════════════════════════════════════════════════════════════
# KEY INSIGHTS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("KEY INSIGHTS — MACRO SCORECARD & SURPRISE INDEX")
print("=" * 60)

print("\n📋 CURRENT MACRO SCORECARD:")
print(f"  {'Indicator':<25} {'Value':<15} {'Signal'}")
print("  " + "-" * 55)
for indicator, data in scorecard.items():
    print(f"  {indicator:<25} {data['value']:<15} {data['signal']}")

print(f"\n📊 MACRO REGIME BREAKDOWN (% of months):")
regime_counts = regime_df['Regime'].value_counts()
total_months  = len(regime_df)
for regime, count in regime_counts.items():
    pct = count / total_months * 100
    print(f"  {regime:<25} {pct:.1f}%  ({count} months)")

print(f"\n📈 BEST ASSET PER REGIME (avg monthly return):")
for regime in regime_perf.index:
    best_asset  = regime_perf.loc[regime].idxmax()
    best_return = regime_perf.loc[regime].max()
    print(f"  {regime:<25} → {best_asset} ({best_return:+.2f}% avg/month)")

print(f"\n📉 ECONOMIC SURPRISE INDEX:")
print(f"  Current ESI:     {latest_esi:+.3f}")
if latest_esi > 0.3:
    print(f"  Signal:          Data BEATING expectations → equity bullish")
elif latest_esi < -0.3:
    print(f"  Signal:          Data MISSING expectations → equity bearish")
else:
    print(f"  Signal:          Data IN-LINE with expectations → neutral")

esi_sp_corr = esi_sp_aligned['ESI'].corr(esi_sp_aligned['SP500_ret'])
print(f"  ESI vs S&P500 correlation: {esi_sp_corr:.3f}")

print(f"\n🏆 CURRENT MACRO REGIME: ", end="")
latest_regime = regime_df['Regime'].iloc[-1]
print(latest_regime.upper())

print("=" * 60)
print(f"\nAll 4 charts saved to your outputs/ folder.")
print(f"Total charts built so far: 27")
