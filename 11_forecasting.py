# ============================================================
# MODULE 9 — PRICE FORECASTING
# Prophet (trend + seasonality) + ARIMA (short-term direction)
# Generates 90-day forecasts for all major assets
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

print("Loading price data from database...")

def load_close(table):
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    close_col = [c for c in df.columns if 'Close' in c or 'close' in c]
    return df[close_col[0]].dropna() if close_col else df.iloc[:, 0].dropna()

nifty  = load_close('NIFTY50')
sp500  = load_close('SP500')
gold   = load_close('GOLD')
silver = load_close('SILVER')
crude  = load_close('CRUDE_WTI')
conn.close()

assets = {
    'NIFTY50': nifty,
    'SP500':   sp500,
    'Gold':    gold,
    'Silver':  silver,
    'Crude':   crude,
}

FORECAST_DAYS = 90
COLORS = {
    'NIFTY50': '#1F3864',
    'SP500':   '#2E75B6',
    'Gold':    '#C55A11',
    'Silver':  '#7030A0',
    'Crude':   '#1E6B3C',
}

# ════════════════════════════════════════════════════════════
# PROPHET FORECASTING
# Handles trend, seasonality, and holiday effects
# Works best for longer-term forecasts (30-365 days)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("PROPHET FORECASTING — 90-day price forecasts")
print("=" * 55)

try:
    from prophet import Prophet

    prophet_results = {}

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'Prophet 90-Day Price Forecasts — All Assets\n'
                 f'Global Macro Intelligence System',
                 fontsize=13, fontweight='bold')
    axes = axes.flatten()

    for idx, (name, series) in enumerate(assets.items()):
        print(f"\nFitting Prophet model for {name}...")
        try:
            # Prophet requires columns named 'ds' and 'y'
            df_prophet = pd.DataFrame({
                'ds': series.index,
                'y':  series.values
            }).dropna()

            # Use last 3 years for faster fitting
            df_prophet = df_prophet.tail(756)

            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=0.80,
            )
            model.fit(df_prophet)

            # Create future dates
            future = model.make_future_dataframe(
                periods=FORECAST_DAYS, freq='B'
            )
            forecast = model.predict(future)

            prophet_results[name] = {
                'forecast': forecast,
                'actual':   df_prophet,
            }

            # Plot
            ax = axes[idx]
            # Historical
            ax.plot(pd.to_datetime(df_prophet['ds']),
                    df_prophet['y'],
                    color=COLORS[name], linewidth=1.2,
                    label='Actual', alpha=0.8)

            # Forecast
            future_only = forecast[forecast['ds'] > df_prophet['ds'].max()]
            ax.plot(future_only['ds'], future_only['yhat'],
                    color='red', linewidth=1.5,
                    linestyle='--', label='Forecast')
            ax.fill_between(future_only['ds'],
                             future_only['yhat_lower'],
                             future_only['yhat_upper'],
                             alpha=0.2, color='red',
                             label='80% confidence')

            # Trend line
            ax.plot(forecast['ds'], forecast['trend'],
                    color='gray', linewidth=0.8,
                    linestyle=':', alpha=0.6, label='Trend')

            last_actual  = df_prophet['y'].iloc[-1]
            last_forecast= future_only['yhat'].iloc[-1]
            pct_change   = (last_forecast - last_actual) / last_actual * 100

            ax.set_title(f'{name} — 90-day forecast: {pct_change:+.1f}%',
                          fontsize=11, fontweight='bold')
            ax.set_ylabel('Price', fontsize=9)
            ax.tick_params(axis='x', rotation=30, labelsize=8)
            ax.legend(fontsize=7)

            print(f"  {name}: Current={last_actual:.1f} | "
                  f"90d forecast={last_forecast:.1f} | "
                  f"Change={pct_change:+.1f}%")

        except Exception as e:
            print(f"  ERROR for {name}: {e}")
            axes[idx].set_title(f'{name} — Error', fontsize=11)
            axes[idx].axis('off')

    # Hide last empty subplot
    if len(assets) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, '28_prophet_forecasts.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved → outputs/28_prophet_forecasts.png ✓")

except Exception as e:
    print(f"Prophet error: {e}")
    prophet_results = {}

# ════════════════════════════════════════════════════════════
# ARIMA FORECASTING
# Best for short-term directional forecasts (5-30 days)
# Auto-selects best parameters using AIC
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("ARIMA FORECASTING — 30-day directional forecasts")
print("=" * 55)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

arima_results = {}

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('ARIMA 30-Day Directional Forecasts — All Assets\n'
             'Global Macro Intelligence System',
             fontsize=13, fontweight='bold')
axes = axes.flatten()

ARIMA_DAYS = 30

for idx, (name, series) in enumerate(assets.items()):
    print(f"\nFitting ARIMA model for {name}...")
    try:
        # Use returns for ARIMA (more stationary than prices)
        returns = series.pct_change().dropna() * 100
        returns_fit = returns.tail(504)  # 2 years

        # Test for stationarity
        adf_result = adfuller(returns_fit)
        is_stationary = adf_result[1] < 0.05

        # Fit ARIMA(2,0,2) on returns — standard choice for financial returns
        model = ARIMA(returns_fit, order=(2, 0, 2))
        fitted = model.fit()

        # Forecast
        forecast_obj  = fitted.forecast(steps=ARIMA_DAYS)
        forecast_conf = fitted.get_forecast(steps=ARIMA_DAYS).conf_int(alpha=0.2)

        # Convert return forecasts back to price levels
        last_price    = series.iloc[-1]
        cumret        = (1 + forecast_obj / 100).cumprod()
        price_forecast= last_price * cumret

        conf_lower = last_price * (1 + forecast_conf.iloc[:, 0] / 100).cumprod()
        conf_upper = last_price * (1 + forecast_conf.iloc[:, 1] / 100).cumprod()

        # Create future date index
        last_date    = series.index[-1]
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=ARIMA_DAYS
        )

        arima_results[name] = {
            'price_forecast': price_forecast,
            'future_dates':   future_dates,
            'last_price':     last_price,
            'direction':      'UP' if price_forecast.iloc[-1] > last_price else 'DOWN',
            'pct_change':     (price_forecast.iloc[-1] - last_price) / last_price * 100,
        }

        # Plot
        ax = axes[idx]
        history = series.tail(120)
        ax.plot(history.index, history.values,
                color=COLORS[name], linewidth=1.3,
                label='Historical (120d)', alpha=0.9)
        ax.plot(future_dates, price_forecast.values,
                color='red', linewidth=1.8,
                linestyle='--', label='ARIMA Forecast')
        ax.fill_between(future_dates,
                         conf_lower.values,
                         conf_upper.values,
                         alpha=0.2, color='red', label='80% CI')
        ax.axvline(last_date, color='gray',
                    linestyle=':', linewidth=1)

        pct = arima_results[name]['pct_change']
        direction_arrow = '▲' if pct > 0 else '▼'
        ax.set_title(f'{name} — 30d: {direction_arrow} {pct:+.2f}%',
                      fontsize=11, fontweight='bold',
                      color='#1E6B3C' if pct > 0 else '#C00000')
        ax.set_ylabel('Price', fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.legend(fontsize=7)

        print(f"  {name}: {last_price:.1f} → {price_forecast.iloc[-1]:.1f} "
              f"({pct:+.2f}%) — {arima_results[name]['direction']}")

    except Exception as e:
        print(f"  ERROR for {name}: {e}")
        axes[idx].set_title(f'{name} — Error', fontsize=11)
        axes[idx].axis('off')

if len(assets) < len(axes):
    axes[-1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '29_arima_forecasts.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved → outputs/29_arima_forecasts.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 3 — Forecast Comparison Dashboard
# ════════════════════════════════════════════════════════════
print("\nCreating Chart 3: Forecast Comparison Dashboard...")

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')
fig.suptitle('Forecast Summary Dashboard — Prophet (90d) vs ARIMA (30d)\n'
             'Global Macro Intelligence System',
             fontsize=13, fontweight='bold')

# Build summary table
table_rows = []
for name in assets.keys():
    # Prophet
    p_chg = 'N/A'
    p_dir = 'N/A'
    if name in prophet_results:
        try:
            fc   = prophet_results[name]['forecast']
            act  = prophet_results[name]['actual']
            last = act['y'].iloc[-1]
            fut  = fc[fc['ds'] > act['ds'].max()]['yhat'].iloc[-1]
            p_chg= f'{(fut-last)/last*100:+.1f}%'
            p_dir= '▲ UP' if fut > last else '▼ DOWN'
        except:
            pass

    # ARIMA
    a_chg = 'N/A'
    a_dir = 'N/A'
    if name in arima_results:
        a_chg = f"{arima_results[name]['pct_change']:+.2f}%"
        a_dir = f"{'▲' if arima_results[name]['direction']=='UP' else '▼'} {arima_results[name]['direction']}"

    # Consensus
    if p_dir != 'N/A' and a_dir != 'N/A':
        p_up = 'UP' in p_dir
        a_up = 'UP' in a_dir
        consensus = '✅ AGREE' if p_up == a_up else '⚠️ SPLIT'
    else:
        consensus = 'N/A'

    table_rows.append([name, p_chg, p_dir, a_chg, a_dir, consensus])

col_labels = ['Asset', 'Prophet 90d', 'Direction', 'ARIMA 30d', 'Direction', 'Consensus']
table = ax.table(
    cellText=table_rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0, 0.1, 1, 0.85]
)
table.auto_set_font_size(False)
table.set_fontsize(12)

for j in range(6):
    table[0, j].set_facecolor('#1F3864')
    table[0, j].set_text_props(color='white', fontweight='bold')
    table[0, j].set_height(0.14)

for i in range(1, len(table_rows) + 1):
    consensus = table_rows[i-1][5]
    row_color = '#D5E8D4' if '✅' in consensus else \
                '#FFF3CD' if '⚠️' in consensus else '#F4F4F4'
    for j in range(6):
        table[i, j].set_facecolor(row_color)
        table[i, j].set_height(0.12)
        if j == 0:
            table[i, j].set_text_props(fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '30_forecast_summary.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/30_forecast_summary.png ✓")

# ════════════════════════════════════════════════════════════
# CHART 4 — Rolling Forecast Accuracy (Backtesting ARIMA)
# ════════════════════════════════════════════════════════════
print("Creating Chart 4: ARIMA Backtest Accuracy...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ARIMA Forecast Direction Accuracy — Walk-Forward Backtest\n'
             'Global Macro Intelligence System',
             fontsize=13, fontweight='bold')
axes = axes.flatten()

backtest_assets = ['NIFTY50', 'SP500', 'Gold', 'Crude']

for idx, name in enumerate(backtest_assets):
    series = assets[name]
    returns = series.pct_change().dropna() * 100

    correct = []
    window  = 252  # 1 year training
    step    = 21   # Test monthly

    print(f"  Backtesting ARIMA for {name}...")

    for start in range(window, len(returns) - step, step):
        try:
            train = returns.iloc[start - window:start]
            actual_ret = returns.iloc[start:start + step].sum()

            model   = ARIMA(train, order=(2, 0, 2))
            fitted  = model.fit()
            fc      = fitted.forecast(steps=step)
            pred_ret= fc.sum()

            pred_up   = pred_ret > 0
            actual_up = actual_ret > 0
            correct.append(1 if pred_up == actual_up else 0)
        except:
            correct.append(0)

    if correct:
        accuracy       = np.mean(correct) * 100
        rolling_acc    = pd.Series(correct).rolling(12).mean() * 100
        dates_backtest = series.index[
            window + step:window + step + len(correct) * step:step
        ][:len(rolling_acc)]

        ax = axes[idx]
        if len(dates_backtest) == len(rolling_acc):
            ax.plot(dates_backtest, rolling_acc,
                    color=COLORS[name], linewidth=1.5)
            ax.fill_between(dates_backtest, rolling_acc, 50,
                             where=rolling_acc >= 50,
                             alpha=0.2, color='green',
                             label='Better than random')
            ax.fill_between(dates_backtest, rolling_acc, 50,
                             where=rolling_acc < 50,
                             alpha=0.2, color='red',
                             label='Worse than random')
        ax.axhline(50, color='black', linestyle='--',
                    linewidth=0.8, label='Random (50%)')
        ax.axhline(accuracy, color=COLORS[name], linestyle=':',
                    linewidth=1, label=f'Overall: {accuracy:.1f}%')
        ax.set_ylim(20, 80)
        ax.set_ylabel('Direction Accuracy %', fontsize=9)
        ax.set_title(f'{name} — Overall: {accuracy:.1f}% accuracy',
                      fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        print(f"    {name}: {accuracy:.1f}% direction accuracy")

plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, '31_arima_backtest.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → outputs/31_arima_backtest.png ✓")

# ════════════════════════════════════════════════════════════
# KEY INSIGHTS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("KEY INSIGHTS — FORECASTING MODELS")
print("=" * 60)

print("\n📈 ARIMA 30-DAY FORECASTS:")
for name, result in arima_results.items():
    arrow = '▲' if result['direction'] == 'UP' else '▼'
    print(f"  {name:<12} {result['last_price']:>10.1f}  →  "
          f"{arrow} {result['pct_change']:+.2f}%  ({result['direction']})")

print("\n🤝 MODEL CONSENSUS:")
for row in table_rows:
    print(f"  {row[0]:<12} Prophet: {row[1]:<8} ARIMA: {row[3]:<8} {row[5]}")

print("\n📊 IMPORTANT DISCLAIMER:")
print("  These are statistical models based on historical patterns.")
print("  They do NOT incorporate fundamental news or macro shifts.")
print("  Use as ONE input alongside your other 8 modules — not alone.")
print("  Direction accuracy of ~55-60% is considered good for financial models.")

print("=" * 60)
print(f"\nAll 4 charts saved to your outputs/ folder.")
print(f"Total charts built so far: 31")
