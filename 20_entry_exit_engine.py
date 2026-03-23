# ============================================================
# GMIS 2.0 — MODULE 20 — ENTRY/EXIT ENGINE
# Calculates entry zones, targets, stop levels, R/R ratio
# Based on: ATR, technical levels, historical analog behavior
# Delivers actionable price levels via Telegram
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import os
import json
import asyncio
import telegram
from datetime import datetime
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID   = int(os.getenv('TELEGRAM_CHAT_ID'))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')
DATA_PATH = os.path.join(BASE_PATH, 'data')

# ── Asset config ──────────────────────────────────────────────
ASSET_CONFIG = {
    'NIFTY':  {'file': 'NIFTY50.csv',   'currency': '₹',
               'atr_period': 14},
    'SP500':  {'file': 'SP500.csv',     'currency': '$',
               'atr_period': 14},
    'Gold':   {'file': 'GOLD.csv',      'currency': '$',
               'atr_period': 14},
    'Silver': {'file': 'SILVER.csv',    'currency': '$',
               'atr_period': 14},
    'Crude':  {'file': 'CRUDE_WTI.csv', 'currency': '$',
               'atr_period': 14},
}

# ═════════════════════════════════════════════════════════════
# SECTION 1 — LOAD PRICE DATA
# ═════════════════════════════════════════════════════════════

def load_price_data(asset):
    """Load OHLCV data from CSV for an asset."""
    config   = ASSET_CONFIG[asset]
    filepath = os.path.join(DATA_PATH, config['file'])

    df = pd.read_csv(filepath)

    # Rename columns — CSV uses 'Price' for date
    df = df.rename(columns={'Price': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').dropna(subset=['Close'])
    df = df.set_index('Date')

    return df

# ═════════════════════════════════════════════════════════════
# SECTION 2 — ATR CALCULATION
# ═════════════════════════════════════════════════════════════

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR).
    ATR = average of true ranges over N days.
    True range = max of:
      - High minus Low
      - abs(High minus previous Close)
      - abs(Low minus previous Close)
    """
    high  = df['High']
    low   = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr        = true_range.rolling(period).mean()

    return float(atr.iloc[-1])

def calculate_volatility(df, period=20):
    """Calculate annualised volatility from daily returns."""
    returns    = df['Close'].pct_change().dropna()
    daily_std  = returns.tail(period).std()
    annual_vol = daily_std * np.sqrt(252)
    return daily_std, annual_vol

# ═════════════════════════════════════════════════════════════
# SECTION 3 — TECHNICAL LEVELS
# ═════════════════════════════════════════════════════════════

def find_technical_levels(df):
    """
    Find key support and resistance levels.
    Uses swing highs/lows and moving averages.
    """
    close    = df['Close'].dropna()
    current  = float(close.iloc[-1])

    # Moving averages
    ma20  = float(close.rolling(20).mean().iloc[-1])
    ma60  = float(close.rolling(60).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])

    # Recent highs and lows
    high_20  = float(close.tail(20).max())
    low_20   = float(close.tail(20).min())
    high_60  = float(close.tail(60).max())
    low_60   = float(close.tail(60).min())
    high_252 = float(close.tail(252).max())
    low_252  = float(close.tail(252).min())

    # Swing highs and lows (5-bar pivot)
    prices = close.values
    swing_highs = []
    swing_lows  = []

    for i in range(5, len(prices) - 5):
        if prices[i] == max(prices[i-5:i+5]):
            swing_highs.append(float(prices[i]))
        if prices[i] == min(prices[i-5:i+5]):
            swing_lows.append(float(prices[i]))

    # Nearest resistance above current price
    resistance_levels = sorted(
        [l for l in swing_highs if l > current]
    )
    nearest_resistance = resistance_levels[0] \
        if resistance_levels else high_252

    # Nearest support below current price
    support_levels = sorted(
        [l for l in swing_lows if l < current],
        reverse=True
    )
    nearest_support = support_levels[0] \
        if support_levels else low_252

    return {
        'current':            current,
        'ma20':               ma20,
        'ma60':               ma60,
        'ma200':              ma200,
        'high_20':            high_20,
        'low_20':             low_20,
        'high_60':            high_60,
        'low_60':             low_60,
        'nearest_resistance': nearest_resistance,
        'nearest_support':    nearest_support,
    }

# ═════════════════════════════════════════════════════════════
# SECTION 4 — LOAD ANALOG OUTCOMES
# ═════════════════════════════════════════════════════════════

def load_analog_outcomes(asset):
    """Load historical analog forward return data."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT * FROM ANALOG_OUTCOMES WHERE asset = ?",
            conn, params=(asset,)
        )
        conn.close()
        return df
    except:
        conn.close()
        return pd.DataFrame()

# ═════════════════════════════════════════════════════════════
# SECTION 5 — CALCULATE ENTRY/EXIT ZONES
# ═════════════════════════════════════════════════════════════

def calculate_levels(asset, bias, df, atr,
                      tech, analog_df, confidence):
    """
    Calculate entry zone, target zone, and stop level.

    Entry zone: current price ± 2×ATR
    Target zone: based on analog median return
    Stop level: 3×ATR from entry (or nearest support/resistance)

    Zone width is adjusted based on confidence:
    HIGH confidence = tighter zones
    MEDIUM confidence = standard zones
    LOW confidence = wider zones
    """
    current = tech['current']

    # Confidence-based width multiplier
    width_mult = {
        'HIGH':   0.85,
        'MEDIUM': 1.00,
        'LOW':    1.20,
        'NONE':   1.40,
    }.get(confidence, 1.0)

    # Analog-based forward returns
    analog_30 = analog_df[
        analog_df['forward_days'] == 30] \
        if not analog_df.empty else pd.DataFrame()

    if not analog_30.empty:
        median_ret = float(analog_30.iloc[0]['median_return'])
        p25_ret    = float(analog_30.iloc[0]['p25_return'])
        p75_ret    = float(analog_30.iloc[0]['p75_return'])
        prob_pos   = float(analog_30.iloc[0]['prob_positive'])
    else:
        median_ret = 0.0
        p25_ret    = -2.0
        p75_ret    = +2.0
        prob_pos   = 50.0

    if bias == 'LONG':
        # ── Entry zone ────────────────────────────────────────
        # Lower: MA20 or nearest support (whichever is higher)
        # Upper: current price + 2×ATR (do not chase beyond)
        entry_low  = max(
            tech['ma20'],
            tech['nearest_support']
        ) * width_mult
        entry_high = current + (atr * 2 * width_mult)

        # If price is already below MA20, adjust entry
        if current < tech['ma20']:
            entry_low  = current - (atr * width_mult)
            entry_high = current + (atr * 2 * width_mult)

        # ── Target zone ───────────────────────────────────────
        # Based on analog median and P75 returns
        target_low  = current * (1 + max(median_ret, 1.0) / 100)
        target_high = current * (1 + max(p75_ret,    2.0) / 100)

        # Cap target below nearest resistance
        target_high = min(
            target_high,
            tech['nearest_resistance'] * 1.02
        )

        # ── Stop level ────────────────────────────────────────
        # 3×ATR below entry midpoint OR nearest support - 1×ATR
        entry_mid   = (entry_low + entry_high) / 2
        stop_atr    = entry_mid - (atr * 3 * width_mult)
        stop_support= tech['nearest_support'] - atr
        stop_level  = max(stop_atr, stop_support)

    elif bias == 'SHORT':
        # ── Entry zone ────────────────────────────────────────
        entry_high = min(
            tech['ma20'],
            tech['nearest_resistance']
        ) / width_mult
        entry_low  = current - (atr * 2 * width_mult)

        # ── Target zone ───────────────────────────────────────
        target_high = current * (1 + min(median_ret, -1.0) / 100)
        target_low  = current * (1 + min(p25_ret,   -2.0) / 100)

        # Cap target above nearest support
        target_low = max(
            target_low,
            tech['nearest_support'] * 0.98
        )

        # ── Stop level ────────────────────────────────────────
        entry_mid    = (entry_low + entry_high) / 2
        stop_atr     = entry_mid + (atr * 3 * width_mult)
        stop_resist  = tech['nearest_resistance'] + atr
        stop_level   = min(stop_atr, stop_resist)

    else:
        return None

    # ── Risk/Reward ratio ─────────────────────────────────────
    entry_mid  = (entry_low + entry_high) / 2
    target_mid = (target_low + target_high) / 2

    if bias == 'LONG':
        risk   = abs(entry_mid - stop_level)
        reward = abs(target_mid - entry_mid)
    else:
        risk   = abs(stop_level - entry_mid)
        reward = abs(entry_mid - target_mid)

    rr_ratio = round(reward / risk, 1) if risk > 0 else 0

    # ── Round to sensible precision ───────────────────────────
    precision = 0 if current > 1000 else 2

    return {
        'asset':       asset,
        'bias':        bias,
        'current':     round(current, precision),
        'entry_low':   round(entry_low,   precision),
        'entry_high':  round(entry_high,  precision),
        'target_low':  round(target_low,  precision),
        'target_high': round(target_high, precision),
        'stop_level':  round(stop_level,  precision),
        'rr_ratio':    rr_ratio,
        'atr':         round(atr, precision),
        'prob_pos':    prob_pos,
        'analog_median': median_ret,
        'confidence':  confidence,
        'entry_basis': f"MA20 ({tech['ma20']:.0f}) + "
                        f"support ({tech['nearest_support']:.0f})",
        'target_basis': f"Analog median {median_ret:+.1f}% → "
                         f"P75 {p75_ret:+.1f}%",
        'stop_basis':  f"3×ATR ({atr:.0f}×3) below entry",
    }

# ═════════════════════════════════════════════════════════════
# SECTION 6 — LOAD DECISIONS
# ═════════════════════════════════════════════════════════════

def load_latest_decisions():
    """Load today's decisions from database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT * FROM DECISIONS ORDER BY date DESC",
            conn
        )
        conn.close()
        if df.empty:
            return {}
        latest = df.groupby('asset').first().reset_index()
        decisions = {}
        for _, row in latest.iterrows():
            decisions[row['asset']] = {
                'bias':       row['bias'],
                'confidence': row['confidence'],
                'combined':   row['combined'],
                'agreement':  row['agreement'],
            }
        return decisions
    except Exception as e:
        conn.close()
        print(f"  ⚠️ Decision load failed: {e}")
        return {}

# ═════════════════════════════════════════════════════════════
# SECTION 7 — SAVE LEVELS
# ═════════════════════════════════════════════════════════════

def save_levels(all_levels):
    """Save entry/exit levels to database."""
    conn  = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime('%Y-%m-%d')

    rows = []
    for asset, levels in all_levels.items():
        if levels is None:
            continue
        rows.append({
            'date':          today,
            'asset':         asset,
            'bias':          levels['bias'],
            'current':       levels['current'],
            'entry_low':     levels['entry_low'],
            'entry_high':    levels['entry_high'],
            'target_low':    levels['target_low'],
            'target_high':   levels['target_high'],
            'stop_level':    levels['stop_level'],
            'rr_ratio':      levels['rr_ratio'],
            'atr':           levels['atr'],
            'prob_pos':      levels['prob_pos'],
            'confidence':    levels['confidence'],
        })

    if not rows:
        conn.close()
        return

    try:
        df = pd.DataFrame(rows)
        conn.execute(
            "DELETE FROM ENTRY_EXIT WHERE date = ?", (today,)
        )
        df.to_sql('ENTRY_EXIT', conn,
                  if_exists='append', index=False)
        conn.commit()
        print(f"  ✅ Levels saved to ENTRY_EXIT table")
    except:
        try:
            df = pd.DataFrame(rows)
            df.to_sql('ENTRY_EXIT', conn,
                      if_exists='replace', index=False)
            conn.commit()
            print(f"  ✅ ENTRY_EXIT table created and saved")
        except Exception as e:
            print(f"  ❌ Save failed: {e}")
    finally:
        conn.close()

# ═════════════════════════════════════════════════════════════
# SECTION 8 — PRINT LEVELS
# ═════════════════════════════════════════════════════════════

def print_levels(all_levels, decisions):
    """Print entry/exit levels clearly."""
    print("\n" + "="*65)
    print("ENTRY/EXIT LEVELS — ALL ASSETS")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    for asset in ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']:
        levels = all_levels.get(asset)
        d      = decisions.get(asset, {})
        bias   = d.get('bias', 'NO TRADE')

        if levels is None or bias == 'NO TRADE':
            print(f"\n⬜ {asset} — NO TRADE")
            print(f"   Decision engine says wait.")
            continue

        emoji = '🟢' if bias == 'LONG' else '🔴'
        curr  = levels['currency'] if 'currency' \
                in levels else ''

        print(f"\n{emoji} {asset} — {bias}")
        print(f"   Current Price:  "
              f"{curr}{levels['current']:,.0f}")
        print(f"   Entry Zone:     "
              f"{curr}{levels['entry_low']:,.0f} — "
              f"{curr}{levels['entry_high']:,.0f}")
        print(f"   Target Zone:    "
              f"{curr}{levels['target_low']:,.0f} — "
              f"{curr}{levels['target_high']:,.0f}")
        print(f"   Stop Level:     "
              f"{curr}{levels['stop_level']:,.0f}")
        print(f"   Risk/Reward:    {levels['rr_ratio']}:1")
        print(f"   ATR (14d):      "
              f"{curr}{levels['atr']:,.0f}")
        print(f"   Analog Prob:    "
              f"{levels['prob_pos']:.0f}% positive")
        print(f"   Confidence:     {levels['confidence']}")
        print(f"   Entry basis:    {levels['entry_basis']}")
        print(f"   Target basis:   {levels['target_basis']}")
        print(f"   Stop basis:     {levels['stop_basis']}")

    print("\n" + "="*65)

# ═════════════════════════════════════════════════════════════
# SECTION 9 — TELEGRAM DELIVERY
# ═════════════════════════════════════════════════════════════

async def send_telegram(message):
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID,
            text=message,
            parse_mode='HTML'
        )
        print("  ✅ Telegram sent")
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")

def build_telegram_message(all_levels, decisions):
    """Build Telegram message with levels."""
    date = datetime.now().strftime('%d %b %Y %H:%M')
    lines = [
        f"📐 <b>GMIS ENTRY/EXIT LEVELS</b>",
        f"{date}",
        f"{'─' * 30}",
        ""
    ]

    has_trade = False
    for asset in ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']:
        levels = all_levels.get(asset)
        d      = decisions.get(asset, {})
        bias   = d.get('bias', 'NO TRADE')

        if levels is None or bias == 'NO TRADE':
            lines.append(f"⬜ <b>{asset}</b>: NO TRADE")
            continue

        has_trade = True
        emoji = '🟢' if bias == 'LONG' else '🔴'
        conf  = d.get('confidence', '')

        lines.append(f"{emoji} <b>{asset} — {bias}</b> "
                     f"| {conf}")
        lines.append(
            f"  Entry:  {levels['entry_low']:,.0f} — "
            f"{levels['entry_high']:,.0f}"
        )
        lines.append(
            f"  Target: {levels['target_low']:,.0f} — "
            f"{levels['target_high']:,.0f}"
        )
        lines.append(
            f"  Stop:   {levels['stop_level']:,.0f}"
        )
        lines.append(
            f"  R/R: {levels['rr_ratio']}:1 | "
            f"Prob: {levels['prob_pos']:.0f}%"
        )
        lines.append("")

    if not has_trade:
        lines.append("")
        lines.append("⬜ <b>No trades today — all assets NO TRADE</b>")
        lines.append("Patience. Wait for better alignment.")

    lines.append("<i>GMIS Entry/Exit Engine</i>")
    return "\n".join(lines)

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_entry_exit_engine(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 20 — ENTRY/EXIT ENGINE")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    # Load decisions
    print("\nLoading decisions...")
    decisions = load_latest_decisions()
    if not decisions:
        print("  ⚠️ No decisions found. "
              "Run 19_decision_engine.py first.")
        return

    print(f"  Loaded decisions for "
          f"{len(decisions)} assets")

    # Calculate levels for each asset
    print("\nCalculating entry/exit levels...")
    all_levels = {}

    asset_map = {
        'NIFTY':  'NIFTY',
        'SP500':  'SP500',
        'Gold':   'Gold',
        'Silver': 'Silver',
        'Crude':  'Crude',
    }

    for asset, decision_key in asset_map.items():
        d    = decisions.get(decision_key, {})
        bias = d.get('bias', 'NO TRADE')
        conf = d.get('confidence', 'NONE')

        if bias == 'NO TRADE':
            all_levels[asset] = None
            print(f"  {asset}: NO TRADE — skipping levels")
            continue

        try:
            # Load price data
            df  = load_price_data(asset)
            atr = calculate_atr(df)
            tech = find_technical_levels(df)
            analog_df = load_analog_outcomes(
                decision_key
            )

            # Calculate levels
            levels = calculate_levels(
                asset, bias, df, atr,
                tech, analog_df, conf
            )

            # Add currency symbol
            if levels:
                currency = ASSET_CONFIG[asset]['currency']
                levels['currency'] = currency

            all_levels[asset] = levels
            if levels:
                print(f"  {asset}: {bias} | "
                      f"Entry {tech['current']:,.0f} | "
                      f"ATR {atr:,.0f} | "
                      f"R/R {levels['rr_ratio']}:1")

        except Exception as e:
            print(f"  ❌ {asset} failed: {e}")
            all_levels[asset] = None

    # Save to database
    print("\nSaving levels...")
    save_levels(all_levels)

    # Print full report
    print_levels(all_levels, decisions)

    # Send Telegram
    if send_telegram_flag and BOT_TOKEN:
        print("\nSending Telegram...")
        msg = build_telegram_message(all_levels, decisions)
        asyncio.run(send_telegram(msg))

    return all_levels

if __name__ == "__main__":
    import sys
    no_telegram = '--no-telegram' in sys.argv
    run_entry_exit_engine(
        send_telegram_flag=not no_telegram
    )
