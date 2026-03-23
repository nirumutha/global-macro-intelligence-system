# ============================================================
# GMIS 2.0 — MODULE 21 — SIGNAL PERFORMANCE TRACKER
# Records every signal at time of generation
# Auto-tracks outcome at +10, +30, +60 days
# Builds live accountability — did the system work?
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import asyncio
import telegram
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID   = int(os.getenv('TELEGRAM_CHAT_ID'))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')
DATA_PATH = os.path.join(BASE_PATH, 'data')

ASSETS    = ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']
HORIZONS  = [10, 30, 60]

# ── Asset to CSV mapping ──────────────────────────────────────
ASSET_FILES = {
    'NIFTY':  'NIFTY50.csv',
    'SP500':  'SP500.csv',
    'Gold':   'GOLD.csv',
    'Silver': 'SILVER.csv',
    'Crude':  'CRUDE_WTI.csv',
}

# ═════════════════════════════════════════════════════════════
# SECTION 1 — DATABASE SETUP
# ═════════════════════════════════════════════════════════════

def setup_tracker_tables():
    """Create signal tracking tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)

    # Signal log — one row per signal generated
    conn.execute('''
        CREATE TABLE IF NOT EXISTS SIGNAL_LOG (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_date   TEXT,
            asset         TEXT,
            bias          TEXT,
            confidence    TEXT,
            combined      REAL,
            agreement     REAL,
            price_at_signal REAL,
            entry_low     REAL,
            entry_high    REAL,
            target_low    REAL,
            target_high   REAL,
            stop_level    REAL,
            rr_ratio      REAL,
            status        TEXT DEFAULT 'OPEN'
        )
    ''')

    # Outcome tracking — one row per horizon per signal
    conn.execute('''
        CREATE TABLE IF NOT EXISTS SIGNAL_OUTCOMES (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id       INTEGER,
            signal_date     TEXT,
            asset           TEXT,
            bias            TEXT,
            horizon_days    INTEGER,
            outcome_date    TEXT,
            price_at_signal REAL,
            price_at_outcome REAL,
            return_pct      REAL,
            hit_target      INTEGER,
            hit_stop        INTEGER,
            outcome_label   TEXT,
            FOREIGN KEY (signal_id) REFERENCES SIGNAL_LOG(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("  ✅ Tracker tables ready")

# ═════════════════════════════════════════════════════════════
# SECTION 2 — LOAD PRICE DATA
# ═════════════════════════════════════════════════════════════

def load_price_series(asset):
    """Load clean price series from CSV."""
    filepath = os.path.join(DATA_PATH, ASSET_FILES[asset])
    df = pd.read_csv(filepath)

    # Drop non-date rows
    df = df[~df['Price'].astype(str).str.match(
        r'^[A-Za-z]', na=False)]
    df = df.rename(columns={'Price': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    df = df.sort_values('Date').set_index('Date')

    return df['Close']

def get_price_on_date(price_series, target_date):
    """Get the closest available price on or after a date."""
    if isinstance(target_date, str):
        target_date = pd.Timestamp(target_date)

    future = price_series[price_series.index >= target_date]
    if future.empty:
        return None
    return float(future.iloc[0])

def get_price_n_days_later(price_series, start_date, n_days):
    """Get price approximately n trading days after start."""
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)

    future = price_series[price_series.index > start_date]
    if len(future) < n_days:
        return None
    return float(future.iloc[n_days - 1])

# ═════════════════════════════════════════════════════════════
# SECTION 3 — RECORD NEW SIGNALS
# ═════════════════════════════════════════════════════════════

def record_new_signals():
    """
    Check today's decisions and entry/exit levels.
    Record any new LONG or SHORT signals that haven't
    been logged yet.
    """
    conn  = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime('%Y-%m-%d')
    new_signals = 0

    try:
        # Load today's decisions
        decisions = pd.read_sql(
            "SELECT * FROM DECISIONS WHERE date = ?",
            conn, params=(today,)
        )

        # Load today's entry/exit levels
        try:
            levels = pd.read_sql(
                "SELECT * FROM ENTRY_EXIT WHERE date = ?",
                conn, params=(today,)
            )
        except:
            levels = pd.DataFrame()

        for _, dec in decisions.iterrows():
            asset = dec['asset']
            bias  = dec['bias']

            # Only log LONG or SHORT — not NO TRADE
            if bias == 'NO TRADE':
                continue

            # Check if already logged today
            existing = pd.read_sql(
                """SELECT id FROM SIGNAL_LOG
                   WHERE signal_date = ? AND asset = ?
                   AND bias = ?""",
                conn, params=(today, asset, bias)
            )

            if not existing.empty:
                continue  # Already logged

            # Get price at signal
            try:
                price_series = load_price_series(asset)
                price_now    = get_price_on_date(
                    price_series, today)
            except:
                price_now = None

            # Get entry/exit levels if available
            asset_levels = levels[
                levels['asset'] == asset
            ] if not levels.empty else pd.DataFrame()

            entry_low   = float(asset_levels.iloc[0]['entry_low'])   if not asset_levels.empty else None
            entry_high  = float(asset_levels.iloc[0]['entry_high'])  if not asset_levels.empty else None
            target_low  = float(asset_levels.iloc[0]['target_low'])  if not asset_levels.empty else None
            target_high = float(asset_levels.iloc[0]['target_high']) if not asset_levels.empty else None
            stop_level  = float(asset_levels.iloc[0]['stop_level'])  if not asset_levels.empty else None
            rr_ratio    = float(asset_levels.iloc[0]['rr_ratio'])    if not asset_levels.empty else None

            # Insert signal log
            conn.execute('''
                INSERT INTO SIGNAL_LOG
                (signal_date, asset, bias, confidence,
                 combined, agreement, price_at_signal,
                 entry_low, entry_high, target_low,
                 target_high, stop_level, rr_ratio, status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                today, asset, bias,
                dec['confidence'],
                float(dec['combined']),
                float(dec['agreement']),
                price_now,
                entry_low, entry_high,
                target_low, target_high,
                stop_level, rr_ratio,
                'OPEN'
            ))

            new_signals += 1
            print(f"  📝 Logged: {asset} {bias} "
                  f"@ {price_now:.0f}" if price_now
                  else f"  📝 Logged: {asset} {bias}")

        conn.commit()

    except Exception as e:
        print(f"  ❌ Signal logging failed: {e}")
    finally:
        conn.close()

    return new_signals

# ═════════════════════════════════════════════════════════════
# SECTION 4 — UPDATE OUTCOMES
# ═════════════════════════════════════════════════════════════

def update_outcomes():
    """
    For all open signals, check if outcome dates
    have been reached and calculate returns.
    """
    conn    = sqlite3.connect(DB_PATH)
    today   = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))
    updated = 0

    try:
        # Load all open signals
        open_signals = pd.read_sql(
            "SELECT * FROM SIGNAL_LOG WHERE status = 'OPEN'",
            conn
        )

        if open_signals.empty:
            print("  No open signals to update")
            conn.close()
            return 0

        print(f"  Checking {len(open_signals)} open signals...")

        for _, sig in open_signals.iterrows():
            signal_id   = int(sig['id'])
            asset       = sig['asset']
            bias        = sig['bias']
            signal_date = pd.Timestamp(sig['signal_date'])
            price_entry = sig['price_at_signal']
            stop_level  = sig['stop_level']
            target_high = sig['target_high']
            target_low  = sig['target_low']

            if price_entry is None:
                continue

            try:
                price_series = load_price_series(asset)
            except:
                continue

            all_horizons_done = True

            for horizon in HORIZONS:
                # Check if outcome already recorded
                existing = pd.read_sql(
                    """SELECT id FROM SIGNAL_OUTCOMES
                       WHERE signal_id = ?
                       AND horizon_days = ?""",
                    conn,
                    params=(signal_id, horizon)
                )
                if not existing.empty:
                    continue

                # Check if horizon date has passed
                horizon_date = signal_date + \
                               pd.Timedelta(days=horizon * 1.4)

                if today < horizon_date:
                    all_horizons_done = False
                    continue

                # Get outcome price
                price_outcome = get_price_n_days_later(
                    price_series, signal_date, horizon
                )

                if price_outcome is None:
                    all_horizons_done = False
                    continue

                # Calculate return
                ret_pct = ((price_outcome - price_entry)
                           / price_entry * 100)

                if bias == 'SHORT':
                    ret_pct = -ret_pct

                # Did it hit target or stop?
                hit_target = 0
                hit_stop   = 0

                if bias == 'LONG':
                    if (target_high and
                            price_outcome >= target_high):
                        hit_target = 1
                    if (stop_level and
                            price_outcome <= stop_level):
                        hit_stop = 1
                elif bias == 'SHORT':
                    if (target_low and
                            price_outcome <= target_low):
                        hit_target = 1
                    if (stop_level and
                            price_outcome >= stop_level):
                        hit_stop = 1

                # Outcome label
                if hit_target:
                    label = 'TARGET HIT ✅'
                elif hit_stop:
                    label = 'STOP HIT ❌'
                elif ret_pct > 2:
                    label = 'WINNING ✅'
                elif ret_pct < -2:
                    label = 'LOSING ❌'
                else:
                    label = 'FLAT ➡️'

                outcome_date = (
                    price_series[
                        price_series.index > signal_date
                    ].index[horizon - 1]
                    if len(price_series[
                        price_series.index > signal_date
                    ]) >= horizon
                    else None
                )

                conn.execute('''
                    INSERT INTO SIGNAL_OUTCOMES
                    (signal_id, signal_date, asset, bias,
                     horizon_days, outcome_date,
                     price_at_signal, price_at_outcome,
                     return_pct, hit_target, hit_stop,
                     outcome_label)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ''', (
                    signal_id,
                    str(signal_date.date()),
                    asset, bias, horizon,
                    str(outcome_date.date())
                    if outcome_date else None,
                    price_entry, price_outcome,
                    round(ret_pct, 2),
                    hit_target, hit_stop, label
                ))

                updated += 1
                print(f"  📊 {asset} {bias} "
                      f"+{horizon}d: {ret_pct:+.1f}% "
                      f"→ {label}")

            # Mark signal as CLOSED if all horizons done
            if all_horizons_done:
                conn.execute(
                    "UPDATE SIGNAL_LOG SET status='CLOSED' "
                    "WHERE id=?", (signal_id,)
                )

        conn.commit()

    except Exception as e:
        print(f"  ❌ Outcome update failed: {e}")
    finally:
        conn.close()

    return updated

# ═════════════════════════════════════════════════════════════
# SECTION 5 — PERFORMANCE SUMMARY
# ═════════════════════════════════════════════════════════════

def calculate_performance_summary():
    """Calculate overall performance statistics."""
    conn = sqlite3.connect(DB_PATH)

    try:
        outcomes = pd.read_sql(
            "SELECT * FROM SIGNAL_OUTCOMES", conn)
        signals  = pd.read_sql(
            "SELECT * FROM SIGNAL_LOG", conn)
        conn.close()
    except:
        conn.close()
        return None

    if outcomes.empty:
        return None

    summary = {}

    # Overall stats
    total       = len(outcomes)
    wins        = (outcomes['return_pct'] > 0).sum()
    losses      = (outcomes['return_pct'] < 0).sum()
    win_rate    = wins / total * 100 if total > 0 else 0
    avg_win     = outcomes[
        outcomes['return_pct'] > 0]['return_pct'].mean()
    avg_loss    = outcomes[
        outcomes['return_pct'] < 0]['return_pct'].mean()
    expectancy  = (win_rate/100 * avg_win +
                   (1 - win_rate/100) * avg_loss) \
                  if total > 0 else 0

    summary['overall'] = {
        'total_signals':  len(signals),
        'total_outcomes': total,
        'win_rate':       round(win_rate, 1),
        'avg_win':        round(avg_win, 2)
        if not np.isnan(avg_win) else 0,
        'avg_loss':       round(avg_loss, 2)
        if not np.isnan(avg_loss) else 0,
        'expectancy':     round(expectancy, 2),
    }

    # Per asset
    summary['by_asset'] = {}
    for asset in ASSETS:
        asset_out = outcomes[outcomes['asset'] == asset]
        if asset_out.empty:
            continue
        a_wins    = (asset_out['return_pct'] > 0).sum()
        a_total   = len(asset_out)
        a_wr      = a_wins / a_total * 100
        a_avg_ret = asset_out['return_pct'].mean()
        summary['by_asset'][asset] = {
            'outcomes':   a_total,
            'win_rate':   round(a_wr, 1),
            'avg_return': round(a_avg_ret, 2),
        }

    # Per horizon
    summary['by_horizon'] = {}
    for h in HORIZONS:
        h_out  = outcomes[outcomes['horizon_days'] == h]
        if h_out.empty:
            continue
        h_wins = (h_out['return_pct'] > 0).sum()
        h_total= len(h_out)
        h_wr   = h_wins / h_total * 100
        h_avg  = h_out['return_pct'].mean()
        summary['by_horizon'][h] = {
            'outcomes':   h_total,
            'win_rate':   round(h_wr, 1),
            'avg_return': round(h_avg, 2),
        }

    return summary

# ═════════════════════════════════════════════════════════════
# SECTION 6 — PRINT SUMMARY
# ═════════════════════════════════════════════════════════════

def print_summary(summary):
    """Print performance summary."""
    print("\n" + "="*60)
    print("SIGNAL PERFORMANCE TRACKER — LIVE RESULTS")
    print("="*60)

    if not summary:
        print("\n  No outcomes recorded yet.")
        print("  Signals need 10+ trading days before")
        print("  first outcomes appear.")
        print("  Keep running refresh_daily.py daily.")
        print("="*60)
        return

    ov = summary['overall']
    print(f"\n📊 Overall Performance:")
    print(f"  Total signals logged:  {ov['total_signals']}")
    print(f"  Outcomes recorded:     {ov['total_outcomes']}")
    print(f"  Win rate:              {ov['win_rate']:.1f}%")
    print(f"  Average win:           +{ov['avg_win']:.2f}%")
    print(f"  Average loss:          {ov['avg_loss']:.2f}%")
    print(f"  Expectancy per signal: {ov['expectancy']:+.2f}%")

    if ov['expectancy'] > 0:
        verdict = "✅ System generating positive expectancy"
    elif ov['expectancy'] > -0.5:
        verdict = "⚠️ Marginally negative — monitor closely"
    else:
        verdict = "❌ Negative expectancy — review signals"

    print(f"\n  Verdict: {verdict}")

    if summary.get('by_asset'):
        print(f"\n📈 Performance by Asset:")
        for asset, stats in summary['by_asset'].items():
            print(f"  {asset:<8} Win: {stats['win_rate']:>5.1f}% | "
                  f"Avg return: {stats['avg_return']:>+6.2f}% | "
                  f"Outcomes: {stats['outcomes']}")

    if summary.get('by_horizon'):
        print(f"\n⏱  Performance by Horizon:")
        for h, stats in summary['by_horizon'].items():
            print(f"  +{h:>2}d    Win: {stats['win_rate']:>5.1f}% | "
                  f"Avg return: {stats['avg_return']:>+6.2f}% | "
                  f"Outcomes: {stats['outcomes']}")

    print("="*60)

# ═════════════════════════════════════════════════════════════
# SECTION 7 — TELEGRAM WEEKLY REPORT
# ═════════════════════════════════════════════════════════════

async def send_telegram(message):
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID,
            text=message,
            parse_mode='HTML'
        )
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")

def send_weekly_report(summary):
    """Send weekly performance report to Telegram."""
    if not summary:
        return

    ov   = summary['overall']
    date = datetime.now().strftime('%d %b %Y')

    lines = [
        f"📊 <b>GMIS SIGNAL PERFORMANCE REPORT</b>",
        f"{date}",
        f"{'─' * 30}",
        f"",
        f"<b>Overall Stats:</b>",
        f"  Win rate: {ov['win_rate']:.1f}%",
        f"  Avg win:  +{ov['avg_win']:.2f}%",
        f"  Avg loss: {ov['avg_loss']:.2f}%",
        f"  Expectancy: {ov['expectancy']:+.2f}% per signal",
        f"  Total outcomes: {ov['total_outcomes']}",
        f"",
    ]

    if summary.get('by_asset'):
        lines.append("<b>By Asset:</b>")
        for asset, stats in summary['by_asset'].items():
            emoji = '✅' if stats['avg_return'] > 0 else '❌'
            lines.append(
                f"  {emoji} {asset}: {stats['win_rate']:.0f}% "
                f"win | {stats['avg_return']:+.1f}% avg"
            )
        lines.append("")

    verdict = "✅ Positive expectancy" \
        if ov['expectancy'] > 0 \
        else "⚠️ Monitor — negative expectancy"
    lines.append(f"<b>Verdict:</b> {verdict}")
    lines.append("")
    lines.append("<i>GMIS Signal Tracker</i>")

    msg = "\n".join(lines)
    asyncio.run(send_telegram(msg))
    print("  ✅ Weekly report sent to Telegram")

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_signal_tracker(send_report=False):
    print("\n" + "="*60)
    print("GMIS MODULE 21 — SIGNAL PERFORMANCE TRACKER")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*60)

    # Setup tables
    print("\nSetting up tracker tables...")
    setup_tracker_tables()

    # Record any new signals from today
    print("\nRecording new signals...")
    new = record_new_signals()
    print(f"  {new} new signal(s) logged today")

    # Update outcomes for open signals
    print("\nUpdating outcomes for open signals...")
    updated = update_outcomes()
    print(f"  {updated} outcome(s) updated")

    # Calculate and print summary
    print("\nCalculating performance summary...")
    summary = calculate_performance_summary()
    print_summary(summary)

    # Send weekly report if requested
    if send_report and summary and BOT_TOKEN:
        print("\nSending weekly report to Telegram...")
        send_weekly_report(summary)

    return summary

if __name__ == "__main__":
    import sys
    send_report = '--report' in sys.argv
    run_signal_tracker(send_report=send_report)
