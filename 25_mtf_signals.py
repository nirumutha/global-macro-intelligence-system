# ============================================================
# GMIS 2.0 — MODULE 25 — MULTI-TIMEFRAME SIGNAL CONFIRMATION
# Calculates Daily / Weekly / Monthly signals for each asset
# and produces a CONFIRMATION STATUS showing whether all
# timeframes agree before committing to a trade.
#
# Rules:
#   THREE_WAY_CONFIRMED  — all 3 TFs agree  → highest conviction
#   TWO_WAY_CONFIRMED    — 2 of 3 agree     → medium conviction
#   CONFLICTED           — daily vs monthly disagree → suppress
#   NEUTRAL              — no clear signal on any TF
#
# Key rule enforced:
#   A daily SHORT against a monthly UPTREND is always
#   flagged CONFLICTED (and never traded).
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import asyncio
import telegram
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID   = int(os.getenv('TELEGRAM_CHAT_ID', '0'))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')
DATA_PATH = os.path.join(BASE_PATH, 'data')

# ── Asset config ──────────────────────────────────────────────
ASSET_CONFIG = {
    'NIFTY':  {'file': 'NIFTY50.csv',   'currency': '₹'},
    'SP500':  {'file': 'SP500.csv',     'currency': '$'},
    'Gold':   {'file': 'GOLD.csv',      'currency': '$'},
    'Silver': {'file': 'SILVER.csv',    'currency': '$'},
    'Crude':  {'file': 'CRUDE_WTI.csv', 'currency': '$'},
}

# ── MA & RSI periods per timeframe ────────────────────────────
# Resampled candles, so periods are in bars of that timeframe
TF_PARAMS = {
    'daily':   {'fast': 20, 'slow': 60, 'trend': 200, 'rsi': 14,
                'resample': None,  'label': 'Daily'},
    'weekly':  {'fast': 10, 'slow': 26, 'trend':  52, 'rsi': 10,
                'resample': 'W',   'label': 'Weekly'},
    'monthly': {'fast':  3, 'slow':  6, 'trend':  12, 'rsi':  6,
                'resample': 'ME',  'label': 'Monthly'},
}

# Score thresholds for Long / Short / Neutral
LONG_THRESH  =  0.20
SHORT_THRESH = -0.20

# ═════════════════════════════════════════════════════════════
# SECTION 1 — LOAD PRICE DATA
# ═════════════════════════════════════════════════════════════

def load_price_data(asset):
    """Load OHLCV data from CSV (same pattern as Module 20)."""
    filepath = os.path.join(DATA_PATH, ASSET_CONFIG[asset]['file'])
    df       = pd.read_csv(filepath)

    # Drop header-repeat rows like 'Ticker' / 'Date'
    df = df[~df['Price'].astype(str).str.match(r'^[A-Za-z]', na=False)]
    df = df.rename(columns={'Price': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('Date').dropna(subset=['Close'])
    df = df.set_index('Date')
    return df

# ═════════════════════════════════════════════════════════════
# SECTION 2 — TECHNICAL INDICATORS
# ═════════════════════════════════════════════════════════════

def calc_rsi(series, period=14):
    """Wilder RSI."""
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_l  = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs     = avg_g / (avg_l + 1e-10)
    return 100 - 100 / (1 + rs)


def calc_signal_score(close, params):
    """
    Compute a [-1, +1] signal score from triple-MA momentum + RSI.

    Triple-MA (weight 0.75):
      +0.25 each if: price > fast_MA, fast_MA > slow_MA, slow_MA > trend_MA

    RSI (weight 0.25):
      RSI > 60  → +0.25  (bullish momentum)
      RSI < 40  → -0.25  (bearish momentum)
      Otherwise →  0
    """
    fast  = params['fast']
    slow  = params['slow']
    trend = params['trend']
    rsi_p = params['rsi']

    min_bars = trend + 5   # need enough bars for trend MA

    if len(close) < min_bars:
        return None, None, None, None

    ma_fast  = close.rolling(fast,  min_periods=fast).mean()
    ma_slow  = close.rolling(slow,  min_periods=slow).mean()
    ma_trend = close.rolling(trend, min_periods=trend).mean()
    rsi      = calc_rsi(close, rsi_p)

    # Latest values
    p  = float(close.iloc[-1])
    mf = float(ma_fast.iloc[-1])
    ms = float(ma_slow.iloc[-1])
    mt = float(ma_trend.iloc[-1])
    r  = float(rsi.iloc[-1])

    if any(np.isnan([p, mf, ms, mt, r])):
        return None, None, None, None

    # Score
    score  = 0.0
    score += 0.25 if p  > mf else -0.25
    score += 0.25 if mf > ms else -0.25
    score += 0.25 if ms > mt else -0.25

    if r > 60:
        score += 0.25
    elif r < 40:
        score -= 0.25

    score = max(-1.0, min(1.0, score))

    # Direction
    if score >= LONG_THRESH:
        direction = 'LONG'
    elif score <= SHORT_THRESH:
        direction = 'SHORT'
    else:
        direction = 'NEUTRAL'

    return score, direction, r, mt


def resample_ohlcv(df, rule):
    """Resample daily OHLCV to weekly or monthly bars."""
    agg = {
        'Open':  'first',
        'High':  'max',
        'Low':   'min',
        'Close': 'last',
    }
    if 'Volume' in df.columns:
        agg['Volume'] = 'sum'

    # Only keep columns that exist
    agg = {k: v for k, v in agg.items() if k in df.columns}
    resampled = df.resample(rule).agg(agg).dropna(subset=['Close'])
    return resampled

# ═════════════════════════════════════════════════════════════
# SECTION 3 — COMPUTE ALL TIMEFRAME SIGNALS FOR ONE ASSET
# ═════════════════════════════════════════════════════════════

def compute_mtf_signals(asset):
    """
    Compute Daily, Weekly, Monthly signals for one asset.
    Returns dict with per-TF results and confirmation status.
    """
    df = load_price_data(asset)

    tf_results = {}

    for tf_name, params in TF_PARAMS.items():
        rule = params['resample']

        if rule is None:
            close = df['Close'].dropna()
        else:
            try:
                resampled = resample_ohlcv(df, rule)
                close     = resampled['Close'].dropna()
            except Exception as e:
                print(f"    ⚠️  {asset} {tf_name} resample failed: {e}")
                tf_results[tf_name] = {
                    'direction': 'NEUTRAL', 'score': 0.0,
                    'rsi': 50.0, 'trend_ma': None, 'bars': 0,
                }
                continue

        score, direction, rsi_val, trend_ma = \
            calc_signal_score(close, params)

        if direction is None:
            print(f"    ⚠️  {asset} {tf_name}: not enough bars "
                  f"({len(close)} < {params['trend']+5})")
            direction = 'NEUTRAL'
            score     = 0.0
            rsi_val   = 50.0
            trend_ma  = None

        tf_results[tf_name] = {
            'direction': direction,
            'score':     round(score, 3) if score else 0.0,
            'rsi':       round(rsi_val, 1) if rsi_val else 50.0,
            'trend_ma':  round(trend_ma, 2) if trend_ma else None,
            'bars':      len(close),
            'price':     float(close.iloc[-1]) if len(close) else 0.0,
        }

    # ── Derive confirmation status ─────────────────────────────
    d_dir = tf_results['daily']['direction']
    w_dir = tf_results['weekly']['direction']
    m_dir = tf_results['monthly']['direction']

    # Key rule: daily SHORT vs monthly UPTREND → CONFLICTED
    if d_dir == 'SHORT' and m_dir == 'LONG':
        status = 'CONFLICTED'
        note   = 'Daily Short against Monthly Uptrend — suppressed'

    # Inverse: daily LONG vs monthly DOWNTREND → CONFLICTED
    elif d_dir == 'LONG' and m_dir == 'SHORT':
        status = 'CONFLICTED'
        note   = 'Daily Long against Monthly Downtrend — suppressed'

    # All three agree
    elif d_dir == w_dir == m_dir and d_dir != 'NEUTRAL':
        status = 'THREE_WAY_CONFIRMED'
        note   = f'All 3 timeframes: {d_dir}'

    # At least 2 agree (and the agreed signal is not NEUTRAL)
    else:
        directions = [d_dir, w_dir, m_dir]
        for sig in ['LONG', 'SHORT']:
            if directions.count(sig) >= 2:
                status = 'TWO_WAY_CONFIRMED'
                note   = f'{directions.count(sig)}/3 timeframes: {sig}'
                break
        else:
            # All NEUTRAL or mixed without clear majority
            if all(d == 'NEUTRAL' for d in directions):
                status = 'NEUTRAL'
                note   = 'No signal on any timeframe'
            else:
                status = 'CONFLICTED'
                note   = (f'Mixed: D={d_dir} W={w_dir} M={m_dir}')

    # Overall trade direction (only when confirmed)
    if status in ('THREE_WAY_CONFIRMED', 'TWO_WAY_CONFIRMED'):
        counts = {'LONG': 0, 'SHORT': 0, 'NEUTRAL': 0}
        for d in [d_dir, w_dir, m_dir]:
            counts[d] += 1
        overall = 'LONG' if counts['LONG'] > counts['SHORT'] \
                  else 'SHORT'
    else:
        overall = 'NO TRADE'

    return {
        'asset':      asset,
        'daily':      tf_results['daily'],
        'weekly':     tf_results['weekly'],
        'monthly':    tf_results['monthly'],
        'status':     status,
        'note':       note,
        'overall':    overall,
        'price':      tf_results['daily']['price'],
        'currency':   ASSET_CONFIG[asset]['currency'],
    }

# ═════════════════════════════════════════════════════════════
# SECTION 4 — SAVE TO DATABASE
# ═════════════════════════════════════════════════════════════

def save_mtf_signals(all_results):
    """Save MTF signal results to MTF_SIGNALS table."""
    conn  = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime('%Y-%m-%d')

    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS MTF_SIGNALS (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT NOT NULL,
                asset           TEXT NOT NULL,
                daily_dir       TEXT,
                daily_score     REAL,
                daily_rsi       REAL,
                weekly_dir      TEXT,
                weekly_score    REAL,
                weekly_rsi      REAL,
                monthly_dir     TEXT,
                monthly_score   REAL,
                monthly_rsi     REAL,
                status          TEXT,
                overall         TEXT,
                note            TEXT,
                price           REAL,
                UNIQUE(date, asset)
            )
        ''')
        conn.commit()

        for asset, r in all_results.items():
            conn.execute('''
                INSERT OR REPLACE INTO MTF_SIGNALS
                (date, asset,
                 daily_dir, daily_score, daily_rsi,
                 weekly_dir, weekly_score, weekly_rsi,
                 monthly_dir, monthly_score, monthly_rsi,
                 status, overall, note, price)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                today, asset,
                r['daily']['direction'],
                r['daily']['score'],
                r['daily']['rsi'],
                r['weekly']['direction'],
                r['weekly']['score'],
                r['weekly']['rsi'],
                r['monthly']['direction'],
                r['monthly']['score'],
                r['monthly']['rsi'],
                r['status'],
                r['overall'],
                r['note'],
                r['price'],
            ))

        conn.commit()
        print(f"  ✅ MTF signals saved ({len(all_results)} assets)")

    except Exception as e:
        print(f"  ❌ Save failed: {e}")
    finally:
        conn.close()

# ═════════════════════════════════════════════════════════════
# SECTION 5 — PRINT REPORT
# ═════════════════════════════════════════════════════════════

# Status display helpers
STATUS_EMOJI = {
    'THREE_WAY_CONFIRMED': '✅✅✅',
    'TWO_WAY_CONFIRMED':   '✅✅⬜',
    'CONFLICTED':          '⚠️ CONF',
    'NEUTRAL':             '⬜⬜⬜',
}

DIR_EMOJI = {
    'LONG':    '🟢',
    'SHORT':   '🔴',
    'NEUTRAL': '⬜',
}


def _dir_str(direction, score):
    e = DIR_EMOJI.get(direction, '⬜')
    return f"{e} {direction:<7} ({score:+.2f})"


def print_mtf_report(all_results):
    print("\n" + "="*75)
    print("MULTI-TIMEFRAME SIGNAL CONFIRMATION")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*75)

    print(f"\n{'Asset':<8} {'Daily':<22} {'Weekly':<22} "
          f"{'Monthly':<22} {'Status'}")
    print("-"*90)

    for asset in ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']:
        r = all_results.get(asset)
        if r is None:
            print(f"{asset:<8} ERROR")
            continue

        d_str = _dir_str(r['daily']['direction'],   r['daily']['score'])
        w_str = _dir_str(r['weekly']['direction'],  r['weekly']['score'])
        m_str = _dir_str(r['monthly']['direction'], r['monthly']['score'])
        s_str = STATUS_EMOJI.get(r['status'], r['status'])

        print(f"{asset:<8} {d_str:<22} {w_str:<22} {m_str:<22} "
              f"{s_str}  {r['overall']}")

    # ── Detailed section ──────────────────────────────────────
    print("\n" + "="*75)
    print("DETAIL")
    print("="*75)

    for asset in ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']:
        r = all_results.get(asset)
        if r is None:
            continue

        status = r['status']
        if status == 'THREE_WAY_CONFIRMED':
            s_emoji = '✅'
        elif status == 'TWO_WAY_CONFIRMED':
            s_emoji = '🟡'
        elif status == 'CONFLICTED':
            s_emoji = '⚠️'
        else:
            s_emoji = '⬜'

        ccy = r['currency']
        print(f"\n{s_emoji} {asset}  —  "
              f"{r['overall']}  |  {r['status']}")
        print(f"   Price: {ccy}{r['price']:,.2f}")
        print(f"   Note:  {r['note']}")

        for tf_name in ['daily', 'weekly', 'monthly']:
            tf   = r[tf_name]
            label = TF_PARAMS[tf_name]['label']
            e    = DIR_EMOJI.get(tf['direction'], '⬜')
            print(f"   {label:<8} {e} {tf['direction']:<7} "
                  f"score={tf['score']:+.2f}  "
                  f"RSI={tf['rsi']:.0f}  "
                  f"bars={tf['bars']}")

    # ── Summary counts ────────────────────────────────────────
    print("\n" + "="*75)
    three = sum(1 for r in all_results.values()
                if r['status'] == 'THREE_WAY_CONFIRMED')
    two   = sum(1 for r in all_results.values()
                if r['status'] == 'TWO_WAY_CONFIRMED')
    conf  = sum(1 for r in all_results.values()
                if r['status'] == 'CONFLICTED')
    neut  = sum(1 for r in all_results.values()
                if r['status'] == 'NEUTRAL')

    print(f"  ✅✅✅ THREE_WAY: {three}  "
          f"✅✅⬜ TWO_WAY: {two}  "
          f"⚠️  CONFLICTED: {conf}  "
          f"⬜ NEUTRAL: {neut}")

    tradeable = [
        a for a, r in all_results.items()
        if r['status'] in ('THREE_WAY_CONFIRMED',
                           'TWO_WAY_CONFIRMED')
        and r['overall'] != 'NO TRADE'
    ]
    if tradeable:
        print(f"\n  TRADEABLE: {', '.join(tradeable)}")
    else:
        print(f"\n  TRADEABLE: None — wait for alignment")

    print("="*75)

# ═════════════════════════════════════════════════════════════
# SECTION 6 — TELEGRAM
# ═════════════════════════════════════════════════════════════

async def send_telegram(message):
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID, text=message, parse_mode='HTML'
        )
        print("  ✅ Telegram sent")
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")


def build_telegram_message(all_results):
    date  = datetime.now().strftime('%d %b %Y %H:%M')
    lines = [
        f"📊 <b>GMIS MTF SIGNAL CONFIRMATION</b>",
        f"{date}",
        f"{'─' * 30}",
        "",
    ]

    # THREE_WAY_CONFIRMED first
    three_way = [
        (a, r) for a, r in all_results.items()
        if r['status'] == 'THREE_WAY_CONFIRMED'
    ]
    if three_way:
        lines.append(f"✅ <b>THREE-WAY CONFIRMED</b>")
        for asset, r in three_way:
            e = DIR_EMOJI.get(r['overall'], '⬜')
            lines.append(
                f"  {e} <b>{asset}</b> — {r['overall']}"
            )
            for tf_name in ['daily', 'weekly', 'monthly']:
                tf    = r[tf_name]
                label = TF_PARAMS[tf_name]['label'][:1]
                lines.append(
                    f"    {label}: {tf['direction']} "
                    f"(score {tf['score']:+.2f}, "
                    f"RSI {tf['rsi']:.0f})"
                )
            lines.append("")

    # TWO_WAY
    two_way = [
        (a, r) for a, r in all_results.items()
        if r['status'] == 'TWO_WAY_CONFIRMED'
    ]
    if two_way:
        lines.append(f"🟡 <b>TWO-WAY CONFIRMED</b>")
        for asset, r in two_way:
            e = DIR_EMOJI.get(r['overall'], '⬜')
            lines.append(
                f"  {e} <b>{asset}</b> — {r['overall']}"
            )
            lines.append(f"    {r['note']}")
            lines.append("")

    # Conflicted
    conflicted = [
        (a, r) for a, r in all_results.items()
        if r['status'] == 'CONFLICTED'
    ]
    if conflicted:
        lines.append(f"⚠️ <b>CONFLICTED — DO NOT TRADE</b>")
        for asset, r in conflicted:
            lines.append(f"  ⚠️ <b>{asset}</b>: {r['note']}")
        lines.append("")

    lines.append("<i>GMIS Multi-Timeframe Engine</i>")
    return "\n".join(lines)

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_mtf_signals(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 25 — MULTI-TIMEFRAME SIGNAL CONFIRMATION")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    all_results = {}

    for asset in ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']:
        print(f"\n  Processing {asset}...")
        try:
            result = compute_mtf_signals(asset)
            all_results[asset] = result
            print(f"    Daily={result['daily']['direction']} "
                  f"Weekly={result['weekly']['direction']} "
                  f"Monthly={result['monthly']['direction']} "
                  f"→ {result['status']}")
        except Exception as e:
            print(f"  ❌ {asset} failed: {e}")
            all_results[asset] = None

    print("\nSaving to database...")
    save_mtf_signals({k: v for k, v in all_results.items()
                      if v is not None})

    print_mtf_report(all_results)

    if send_telegram_flag and BOT_TOKEN:
        # Send only if there is something notable
        has_confirmed = any(
            r is not None and
            r['status'] in ('THREE_WAY_CONFIRMED',
                            'TWO_WAY_CONFIRMED')
            for r in all_results.values()
        )
        has_conflict = any(
            r is not None and r['status'] == 'CONFLICTED'
            for r in all_results.values()
        )

        if '--force-send' in sys.argv:
            has_confirmed = True

        if has_confirmed or has_conflict:
            print("\nSending MTF summary to Telegram...")
            msg = build_telegram_message(all_results)
            asyncio.run(send_telegram(msg))
        else:
            print("\n  All neutral — no Telegram alert")
    elif not send_telegram_flag:
        print("\n  Telegram skipped (--no-telegram)")

    return all_results


if __name__ == "__main__":
    no_telegram = '--no-telegram' in sys.argv
    run_mtf_signals(send_telegram_flag=not no_telegram)
