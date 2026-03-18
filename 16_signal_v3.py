# ============================================================
# GMIS 2.0 — MODULE 16 — SIGNAL ENGINE V3
# Adds confidence scoring and stability layer
# Saves to SIGNALS_V3 table — existing SIGNALS untouched
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')

ASSETS     = ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']
THRESHOLD  = 0.15
STABILITY_PERIODS = 3  # signal must hold N days to be official

# ═════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ═════════════════════════════════════════════════════════════

def load_signals():
    """Load existing signal scores from database."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM SIGNALS", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    return df

def load_prices():
    """Load price series for adaptive weight calculation."""
    conn   = sqlite3.connect(DB_PATH)
    prices = {}
    for asset, table in [
        ('NIFTY', 'NIFTY50'),
        ('SP500', 'SP500'),
        ('Gold',  'GOLD'),
        ('Silver','SILVER'),
        ('Crude', 'CRUDE_WTI'),
    ]:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        close_col = [c for c in df.columns
                     if 'Close' in c or 'close' in c]
        col = close_col[0] if close_col else df.columns[0]
        prices[asset] = df[col].dropna()
    conn.close()
    return prices

# ═════════════════════════════════════════════════════════════
# SECTION 2 — CONFIDENCE SCORING
# ═════════════════════════════════════════════════════════════

def get_confidence(score):
    """
    Convert numeric score to confidence level.
    HIGH   = abs(score) >= 0.40
    MEDIUM = abs(score) >= 0.25
    LOW    = abs(score) >= 0.15 (signal threshold)
    NONE   = below threshold (Neutral)
    """
    abs_score = abs(score)
    if abs_score >= 0.40:
        return 'HIGH'
    elif abs_score >= 0.25:
        return 'MEDIUM'
    elif abs_score >= 0.15:
        return 'LOW'
    else:
        return 'NONE'

def get_signal(score):
    """Convert score to signal direction."""
    if score >= THRESHOLD:
        return 'Long'
    elif score <= -THRESHOLD:
        return 'Short'
    else:
        return 'Neutral'

# ═════════════════════════════════════════════════════════════
# SECTION 3 — STABILITY LAYER
# ═════════════════════════════════════════════════════════════

def apply_stability_layer(signal_series, min_periods=STABILITY_PERIODS):
    """
    A signal only officially changes after holding
    for min_periods consecutive days.

    Example with min_periods=3:
    Day 1: Long  → stable = Long  (existing)
    Day 2: Short → stable = Long  (not confirmed yet)
    Day 3: Short → stable = Long  (not confirmed yet)
    Day 4: Short → stable = Short (3 days confirmed — now official)

    This prevents daily flip-flopping.
    """
    signals = signal_series.values
    stable  = signals.copy()

    for i in range(min_periods, len(signals)):
        window = signals[i - min_periods + 1: i + 1]

        # Check if all periods in window are the same
        if len(set(window)) == 1:
            stable[i] = window[0]  # confirmed change
        else:
            stable[i] = stable[i - 1]  # hold previous stable signal

    return pd.Series(stable, index=signal_series.index)

# ═════════════════════════════════════════════════════════════
# SECTION 4 — ADAPTIVE WEIGHTS
# ═════════════════════════════════════════════════════════════

def calculate_signal_performance(signals_df, prices, 
                                  asset, lookback=60):
    """
    Measure how well the signal predicted price direction
    over the last N days.

    Returns hit rate — % of days where signal direction
    matched next-day price movement.
    """
    try:
        score_col  = f'{asset}_score'
        price_s    = prices[asset]

        # Align dates
        common = signals_df.index.intersection(price_s.index)
        if len(common) < lookback:
            return 0.5  # not enough data — assume 50%

        scores = signals_df.loc[common, score_col].tail(lookback)
        prices_aligned = price_s.loc[common].tail(lookback)

        # Next day return
        next_ret = prices_aligned.pct_change().shift(-1)

        # Signal direction: +1 for Long, -1 for Short, 0 for Neutral
        sig_direction = np.where(scores >= THRESHOLD,  1,
                        np.where(scores <= -THRESHOLD, -1, 0))

        # Hit: signal direction matches return direction
        ret_direction = np.sign(next_ret)

        # Only count days where we had a signal (not Neutral)
        active = sig_direction != 0
        if active.sum() == 0:
            return 0.5

        hits = (sig_direction[active] == ret_direction.values[active])
        return hits.mean()

    except Exception as e:
        return 0.5  # default on error

def get_adaptive_weights(signals_df, prices, lookback=60):
    """
    Calculate performance-based weight adjustments.
    
    Base weights stay the same.
    This produces a performance score per asset
    showing which signals are currently working best.
    """
    performance = {}
    for asset in ASSETS:
        hit_rate = calculate_signal_performance(
            signals_df, prices, asset, lookback
        )
        performance[asset] = hit_rate

    return performance

# ═════════════════════════════════════════════════════════════
# SECTION 5 — BUILD SIGNALS V3
# ═════════════════════════════════════════════════════════════

def build_signals_v3(signals_df, prices):
    """
    Build the enhanced signal dataframe with:
    - Raw signal (same as before)
    - Confidence level (HIGH/MEDIUM/LOW/NONE)
    - Stable signal (after stability filter)
    - Performance score (recent hit rate)
    """
    v3 = pd.DataFrame(index=signals_df.index)

    # Get adaptive performance scores
    print("\n  Calculating signal performance...")
    performance = get_adaptive_weights(signals_df, prices)

    for asset in ASSETS:
        score_col = f'{asset}_score'
        scores    = signals_df[score_col]

        # Raw signal
        raw_signals = scores.apply(get_signal)

        # Confidence
        v3[f'{asset}_score']      = scores
        v3[f'{asset}_signal']     = raw_signals
        v3[f'{asset}_confidence'] = scores.apply(get_confidence)

        # Stable signal
        v3[f'{asset}_stable']     = apply_stability_layer(raw_signals)

        # Recent performance
        perf = performance.get(asset, 0.5)
        v3[f'{asset}_hit_rate']   = round(perf, 3)

        print(f"  {asset:<8} | "
              f"Signal: {raw_signals.iloc[-1]:<8} | "
              f"Confidence: {scores.apply(get_confidence).iloc[-1]:<6} | "
              f"Stable: {apply_stability_layer(raw_signals).iloc[-1]:<8} | "
              f"Hit rate: {perf:.1%}")

    return v3

# ═════════════════════════════════════════════════════════════
# SECTION 6 — SAVE TO DATABASE
# ═════════════════════════════════════════════════════════════

def save_signals_v3(v3_df):
    """Save to SIGNALS_V3 table."""
    conn = sqlite3.connect(DB_PATH)
    try:
        v3_save = v3_df.reset_index()
        v3_save.to_sql('SIGNALS_V3', conn,
                       if_exists='replace', index=False)
        conn.commit()
        print(f"\n  ✅ {len(v3_df)} rows saved to SIGNALS_V3 table")
    except Exception as e:
        print(f"\n  ❌ Save failed: {e}")
    finally:
        conn.close()

# ═════════════════════════════════════════════════════════════
# SECTION 7 — PRINT CURRENT SIGNALS
# ═════════════════════════════════════════════════════════════

def print_current_signals(v3_df):
    """Print today's enhanced signals."""
    latest = v3_df.iloc[-1]
    date   = v3_df.index[-1].strftime('%d %B %Y')

    print("\n" + "="*65)
    print(f"CURRENT SIGNALS V3 — {date}")
    print("="*65)
    print(f"{'Asset':<8} {'Score':>7} {'Signal':<8} "
          f"{'Confidence':<10} {'Stable':<8} {'Hit Rate'}")
    print("-"*65)

    for asset in ASSETS:
        score  = latest[f'{asset}_score']
        signal = latest[f'{asset}_signal']
        conf   = latest[f'{asset}_confidence']
        stable = latest[f'{asset}_stable']
        hit    = latest[f'{asset}_hit_rate']

        # Signal emoji
        emoji = '🟢' if signal == 'Long'  else \
                '🔴' if signal == 'Short' else '🟡'

        # Stable emoji
        s_emoji = '✅' if stable == signal else '⏳'

        print(f"{asset:<8} {score:>+7.3f} "
              f"{emoji} {signal:<6} "
              f"{conf:<10} "
              f"{s_emoji} {stable:<6} "
              f"{hit:.1%}")

    print("="*65)

    # Highlight actionable signals
    print("\n📡 ACTIONABLE SIGNALS (Stable + Confidence ≥ MEDIUM):")
    actionable = False
    for asset in ASSETS:
        signal = latest[f'{asset}_signal']
        conf   = latest[f'{asset}_confidence']
        stable = latest[f'{asset}_stable']

        is_actionable = (
            signal != 'Neutral' and
            stable == signal and
            conf in ['HIGH', 'MEDIUM']
        )

        if is_actionable:
            emoji = '🟢' if signal == 'Long' else '🔴'
            print(f"  {emoji} {asset}: {signal} "
                  f"| Confidence: {conf} "
                  f"| Score: {latest[f'{asset}_score']:+.3f}")
            actionable = True

    if not actionable:
        print("  No high-confidence stable signals today — stay patient")

    print("="*65)

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_signal_v3():
    print("\n" + "="*65)
    print("GMIS MODULE 16 — SIGNAL ENGINE V3")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    # Load data
    print("\nLoading signals and prices...")
    signals_df = load_signals()
    prices     = load_prices()
    print(f"  Signals loaded: {len(signals_df)} rows")
    print(f"  Price data loaded for {len(prices)} assets")

    # Build v3
    print("\nBuilding Signal V3...")
    v3_df = build_signals_v3(signals_df, prices)

    # Save
    print("\nSaving to database...")
    save_signals_v3(v3_df)

    # Print current signals
    print_current_signals(v3_df)

    return v3_df

if __name__ == "__main__":
    run_signal_v3()