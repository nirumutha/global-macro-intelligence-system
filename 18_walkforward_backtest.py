# ============================================================
# GMIS 2.0 — MODULE 18 — WALK-FORWARD BACKTEST
# Tests strategy on data it never saw during development
# Institutional-grade validation framework
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')

ASSETS           = ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']
TRANSACTION_COST = 0.001   # 0.1% per trade
SIGNAL_THRESHOLD = 0.15
TRAIN_YEARS      = 3       # minimum training window
TEST_YEARS       = 1       # test on 1 year at a time

# ═════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ═════════════════════════════════════════════════════════════

def load_data():
    """Load prices and signals from database."""
    conn = sqlite3.connect(DB_PATH)

    prices = {}
    for asset, table in [
        ('NIFTY',  'NIFTY50'),
        ('SP500',  'SP500'),
        ('Gold',   'GOLD'),
        ('Silver', 'SILVER'),
        ('Crude',  'CRUDE_WTI'),
    ]:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        close_col = [c for c in df.columns
                     if 'Close' in c or 'close' in c]
        col = close_col[0] if close_col else df.columns[0]
        prices[asset] = df[col].dropna()

    signals = pd.read_sql("SELECT * FROM SIGNALS", conn)
    signals['Date'] = pd.to_datetime(signals['Date'])
    signals = signals.set_index('Date').sort_index()

    conn.close()
    return prices, signals

# ═════════════════════════════════════════════════════════════
# SECTION 2 — SINGLE PERIOD BACKTEST
# ═════════════════════════════════════════════════════════════

def run_single_backtest(price, signal_scores,
                         start_date, end_date):
    """
    Run backtest for a single time period.
    Returns strategy and buy-and-hold metrics.
    """
    # Filter to period
    mask_p = (price.index >= start_date) & \
             (price.index <= end_date)
    mask_s = (signal_scores.index >= start_date) & \
             (signal_scores.index <= end_date)

    p = price[mask_p]
    s = signal_scores[mask_s]

    if len(p) < 20 or len(s) < 20:
        return None

    # Align signals to price dates
    sig = s.reindex(p.index).ffill().bfill()

    # Generate positions
    daily_ret = p.pct_change().fillna(0)
    position  = pd.Series(0.0, index=p.index)
    position[sig >= SIGNAL_THRESHOLD]  = 1.0
    position[sig <= -SIGNAL_THRESHOLD] = 0.0

    # Apply transaction costs
    pos_change = position.diff().abs().fillna(0)
    strat_ret  = position.shift(1) * daily_ret
    strat_ret -= pos_change * TRANSACTION_COST

    # Equity curves
    strat_eq = (1 + strat_ret).cumprod() * 100
    bnh_eq   = (1 + daily_ret).cumprod() * 100

    # Metrics
    def calc_metrics(ret, eq):
        n_years = len(ret) / 252
        if n_years <= 0:
            return {}
        total   = eq.iloc[-1] / 100
        cagr    = (total ** (1/n_years) - 1) * 100 \
                   if total > 0 else -100
        sharpe  = (ret.mean() / (ret.std() + 1e-10)) \
                   * np.sqrt(252)
        roll_max = eq.cummax()
        maxdd    = ((eq - roll_max) / roll_max).min() * 100
        sortino  = (ret.mean() /
                    (ret[ret < 0].std() + 1e-10)) * np.sqrt(252)
        win_rate = (ret > 0).sum() / \
                   max((ret != 0).sum(), 1) * 100
        return {
            'cagr':     round(cagr,    2),
            'sharpe':   round(sharpe,  3),
            'maxdd':    round(maxdd,   2),
            'sortino':  round(sortino, 3),
            'win_rate': round(win_rate,1),
        }

    return {
        'strat':   calc_metrics(strat_ret, strat_eq),
        'bnh':     calc_metrics(daily_ret, bnh_eq),
        'strat_eq': strat_eq,
        'bnh_eq':   bnh_eq,
    }

# ═════════════════════════════════════════════════════════════
# SECTION 3 — WALK-FORWARD ENGINE
# ═════════════════════════════════════════════════════════════

def run_walkforward(prices, signals, asset):
    """
    Run walk-forward backtest for one asset.

    Structure:
    Train: 2010-2012 → Test: 2013
    Train: 2010-2013 → Test: 2014
    Train: 2010-2014 → Test: 2015
    ... and so on until 2025
    """
    score_col = f'{asset}_score'
    if score_col not in signals.columns:
        return []

    price         = prices[asset]
    signal_scores = signals[score_col]

    # Get year range
    start_year = price.index.year.min() + TRAIN_YEARS
    end_year   = price.index.year.max() - 1

    results = []

    for test_year in range(start_year, end_year + 1):
        # Training period — everything before test year
        train_start = pd.Timestamp(f'{price.index.year.min()}-01-01')
        train_end   = pd.Timestamp(f'{test_year - 1}-12-31')

        # Test period — the test year only
        test_start  = pd.Timestamp(f'{test_year}-01-01')
        test_end    = pd.Timestamp(f'{test_year}-12-31')

        # Run backtest on test period only
        result = run_single_backtest(
            price, signal_scores, test_start, test_end
        )

        if result and result['strat']:
            results.append({
                'year':          test_year,
                'train_end':     train_end.year,
                'strat_cagr':    result['strat']['cagr'],
                'strat_sharpe':  result['strat']['sharpe'],
                'strat_maxdd':   result['strat']['maxdd'],
                'strat_sortino': result['strat']['sortino'],
                'strat_winrate': result['strat']['win_rate'],
                'bnh_cagr':      result['bnh']['cagr'],
                'bnh_sharpe':    result['bnh']['sharpe'],
                'bnh_maxdd':     result['bnh']['maxdd'],
            })

    return results

# ═════════════════════════════════════════════════════════════
# SECTION 4 — PORTFOLIO WALK-FORWARD
# ═════════════════════════════════════════════════════════════

def run_portfolio_walkforward(prices, signals):
    """
    Run walk-forward on equal-weight portfolio of all assets.
    """
    start_year = 2013
    end_year   = prices['SP500'].index.year.max() - 1

    portfolio_results = []

    for test_year in range(start_year, end_year + 1):
        test_start = pd.Timestamp(f'{test_year}-01-01')
        test_end   = pd.Timestamp(f'{test_year}-12-31')

        year_strat_rets = []
        year_bnh_rets   = []

        for asset in ASSETS:
            score_col = f'{asset}_score'
            if score_col not in signals.columns:
                continue

            price  = prices[asset]
            scores = signals[score_col]

            # Filter to test year
            mask_p = (price.index >= test_start) & \
                     (price.index <= test_end)
            mask_s = (scores.index >= test_start) & \
                     (scores.index <= test_end)

            p = price[mask_p]
            s = scores[mask_s]

            if len(p) < 20:
                continue

            sig       = s.reindex(p.index).ffill().bfill()
            daily_ret = p.pct_change().fillna(0)
            position  = pd.Series(0.0, index=p.index)
            position[sig >= SIGNAL_THRESHOLD]  = 1.0
            position[sig <= -SIGNAL_THRESHOLD] = 0.0

            pos_change = position.diff().abs().fillna(0)
            strat_ret  = position.shift(1) * daily_ret
            strat_ret -= pos_change * TRANSACTION_COST

            year_strat_rets.append(strat_ret)
            year_bnh_rets.append(daily_ret)

        if not year_strat_rets:
            continue

        # Equal weight portfolio
        port_ret  = pd.concat(year_strat_rets, axis=1) \
                      .dropna().mean(axis=1)
        bnh_ret   = pd.concat(year_bnh_rets, axis=1) \
                      .dropna().mean(axis=1)

        if len(port_ret) < 20:
            continue

        port_eq   = (1 + port_ret).cumprod() * 100
        bnh_eq    = (1 + bnh_ret).cumprod()  * 100

        # Metrics
        n_years   = len(port_ret) / 252
        port_cagr = ((port_eq.iloc[-1]/100)**(1/n_years)-1)*100 \
                     if n_years > 0 else 0
        port_sharpe = (port_ret.mean() /
                       (port_ret.std() + 1e-10)) * np.sqrt(252)
        port_dd   = ((port_eq - port_eq.cummax()) /
                      port_eq.cummax() * 100).min()
        bnh_cagr  = ((bnh_eq.iloc[-1]/100)**(1/n_years)-1)*100 \
                     if n_years > 0 else 0
        bnh_sharpe= (bnh_ret.mean() /
                     (bnh_ret.std() + 1e-10)) * np.sqrt(252)

        portfolio_results.append({
            'year':         test_year,
            'port_cagr':    round(port_cagr,   2),
            'port_sharpe':  round(port_sharpe, 3),
            'port_maxdd':   round(port_dd,     2),
            'bnh_cagr':     round(bnh_cagr,    2),
            'bnh_sharpe':   round(bnh_sharpe,  3),
            'beat_bnh':     port_sharpe > bnh_sharpe,
        })

    return portfolio_results

# ═════════════════════════════════════════════════════════════
# SECTION 5 — SAVE RESULTS
# ═════════════════════════════════════════════════════════════

def save_results(all_asset_results, portfolio_results):
    """Save walk-forward results to database."""
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime('%Y-%m-%d')

    try:
        # Asset results
        all_rows = []
        for asset, results in all_asset_results.items():
            for r in results:
                all_rows.append({'asset': asset, **r})

        if all_rows:
            df = pd.DataFrame(all_rows)
            df['run_date'] = today
            df.to_sql('WALKFORWARD_RESULTS', conn,
                      if_exists='replace', index=False)

        # Portfolio results
        if portfolio_results:
            df_port = pd.DataFrame(portfolio_results)
            df_port['run_date'] = today
            df_port.to_sql('WALKFORWARD_PORTFOLIO', conn,
                           if_exists='replace', index=False)

        conn.commit()
        print(f"\n  ✅ Walk-forward results saved to database")

    except Exception as e:
        print(f"\n  ❌ Save failed: {e}")
    finally:
        conn.close()

# ═════════════════════════════════════════════════════════════
# SECTION 6 — PRINT RESULTS
# ═════════════════════════════════════════════════════════════

def print_results(all_asset_results, portfolio_results):
    """Print walk-forward results clearly."""

    print("\n" + "="*70)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("Out-of-sample performance — strategy never saw test data")
    print("="*70)

    # Per-asset summary
    for asset, results in all_asset_results.items():
        if not results:
            continue

        df = pd.DataFrame(results)

        avg_sharpe     = df['strat_sharpe'].mean()
        avg_bnh_sharpe = df['bnh_sharpe'].mean()
        avg_maxdd      = df['strat_maxdd'].mean()
        avg_bnh_maxdd  = df['bnh_maxdd'].mean()
        years_positive = (df['strat_sharpe'] > 0).sum()
        years_beat     = (df['strat_sharpe'] >
                          df['bnh_sharpe']).sum()
        total_years    = len(df)

        print(f"\n{'─'*70}")
        print(f"  {asset}")
        print(f"{'─'*70}")
        print(f"  {'Year':<6} {'Strat CAGR':>10} "
              f"{'Strat Sharpe':>13} {'B&H Sharpe':>11} "
              f"{'MaxDD':>8} {'Result'}")
        print(f"  {'-'*65}")

        for _, row in df.iterrows():
            beat   = '✅' if row['strat_sharpe'] > \
                              row['bnh_sharpe'] else '⚠️'
            print(f"  {int(row['year']):<6} "
                  f"{row['strat_cagr']:>+9.1f}%  "
                  f"{row['strat_sharpe']:>12.3f}  "
                  f"{row['bnh_sharpe']:>10.3f}  "
                  f"{row['strat_maxdd']:>7.1f}%  "
                  f"{beat}")

        print(f"\n  Summary:")
        print(f"    Avg Sharpe (strategy):  {avg_sharpe:.3f}")
        print(f"    Avg Sharpe (B&H):       {avg_bnh_sharpe:.3f}")
        print(f"    Avg Max Drawdown:        {avg_maxdd:.1f}%  "
              f"(vs {avg_bnh_maxdd:.1f}% B&H)")
        print(f"    Years with pos Sharpe:  "
              f"{years_positive}/{total_years}")
        print(f"    Years beating B&H:      "
              f"{years_beat}/{total_years}")

    # Portfolio summary
    if portfolio_results:
        print(f"\n{'='*70}")
        print("  COMBINED EQUAL-WEIGHT PORTFOLIO")
        print(f"{'='*70}")

        df_port = pd.DataFrame(portfolio_results)

        print(f"\n  {'Year':<6} {'Port CAGR':>10} "
              f"{'Port Sharpe':>12} {'B&H Sharpe':>11} "
              f"{'MaxDD':>8} {'Result'}")
        print(f"  {'-'*60}")

        for _, row in df_port.iterrows():
            beat = '✅' if row['beat_bnh'] else '⚠️'
            print(f"  {int(row['year']):<6} "
                  f"{row['port_cagr']:>+9.1f}%  "
                  f"{row['port_sharpe']:>11.3f}  "
                  f"{row['bnh_sharpe']:>10.3f}  "
                  f"{row['port_maxdd']:>7.1f}%  "
                  f"{beat}")

        avg_port_sharpe = df_port['port_sharpe'].mean()
        avg_bnh_sharpe  = df_port['bnh_sharpe'].mean()
        years_beat      = df_port['beat_bnh'].sum()
        total_years     = len(df_port)
        consistency     = years_beat / total_years * 100

        print(f"\n  Portfolio Summary:")
        print(f"    Avg out-of-sample Sharpe: {avg_port_sharpe:.3f}")
        print(f"    Avg B&H Sharpe:           {avg_bnh_sharpe:.3f}")
        print(f"    Years beating B&H:        "
              f"{years_beat}/{total_years} ({consistency:.0f}%)")

        if avg_port_sharpe > 0.4:
            verdict = "✅ ROBUST — strategy holds up out-of-sample"
        elif avg_port_sharpe > 0.2:
            verdict = "⚠️  ACCEPTABLE — some signal but not strong"
        else:
            verdict = "❌ WEAK — limited out-of-sample evidence"

        print(f"\n  Verdict: {verdict}")

    print("\n" + "="*70)
    print("Note: Each year tested on data the model never saw.")
    print("This is the gold standard for strategy validation.")
    print("="*70)

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_walkforward_backtest():
    print("\n" + "="*70)
    print("GMIS MODULE 18 — WALK-FORWARD BACKTEST")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*70)

    print("\nLoading data...")
    prices, signals = load_data()
    print(f"  Loaded {len(signals)} signal rows")

    print("\nRunning walk-forward tests...")
    all_asset_results = {}

    for asset in ASSETS:
        print(f"  Testing {asset}...")
        results = run_walkforward(prices, signals, asset)
        all_asset_results[asset] = results
        print(f"    {len(results)} years tested")

    print("\nRunning portfolio walk-forward...")
    portfolio_results = run_portfolio_walkforward(
        prices, signals
    )
    print(f"  {len(portfolio_results)} years tested")

    print("\nSaving results...")
    save_results(all_asset_results, portfolio_results)

    print_results(all_asset_results, portfolio_results)

    return all_asset_results, portfolio_results

if __name__ == "__main__":
    run_walkforward_backtest()
