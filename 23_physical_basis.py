# ============================================================
# GMIS 2.0 — MODULE 23 — PHYSICAL BASIS MONITOR
# Detects whether price moves are backed by real physical
# demand or just paper/sentiment trading
# Backwardation = physical scarcity = confirms Long signals
# Contango = paper trading = warns against Long signals
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import os
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

# ── Futures contract config ───────────────────────────────────
# Front month and second month tickers
FUTURES_CONFIG = {
    'Crude': {
        'front':       'CL=F',
        'second':      'CLK26.NYM',
        'name':        'Crude Oil WTI',
        'unit':        '$/barrel',
        'normal_basis': -1.0,  # typical contango in $/barrel
    },
    'Gold': {
        'front':       'GC=F',
        'second':      'GCM26.CMX',
        'name':        'Gold',
        'unit':        '$/oz',
        'normal_basis': -8.0,  # typical contango (storage+financing)
    },
    'Silver': {
        'front':       'SI=F',
        'second':      'SIN26.CMX',
        'name':        'Silver',
        'unit':        '$/oz',
        'normal_basis': -0.10,
    },
}

# ── Basis thresholds ──────────────────────────────────────────
# Basis = Front Month Price - Second Month Price
# Positive = Backwardation (physical scarcity)
# Negative = Contango (oversupply/paper market)

# How far above/below normal basis before flagging
BACKWARDATION_THRESHOLD = 0.5   # % above normal = strong physical
CONTANGO_THRESHOLD      = -0.5  # % below normal = paper market

# ═════════════════════════════════════════════════════════════
# SECTION 1 — FETCH FUTURES DATA
# ═════════════════════════════════════════════════════════════

def fetch_futures_prices(ticker, period='10d'):
    """Fetch recent futures prices from yfinance."""
    try:
        df = yf.download(
            ticker, period=period,
            interval='1d', progress=False
        )
        if df.empty:
            return None
        close = df['Close'].dropna()
        if close.empty:
            return None
        return float(close.iloc[-1])
    except Exception as e:
        return None

def fetch_futures_history(ticker, period='60d'):
    """Fetch futures price history."""
    try:
        df = yf.download(
            ticker, period=period,
            interval='1d', progress=False
        )
        if df.empty:
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        close = df['Close'].dropna()
        if close.empty:
            return None

        close.index = pd.to_datetime(close.index)
        return close

    except Exception as e:
        return None

# ═════════════════════════════════════════════════════════════
# SECTION 2 — CALCULATE BASIS
# ═════════════════════════════════════════════════════════════

def calculate_basis(front_price, second_price):
    """
    Basis = Front Month - Second Month

    Positive (Backwardation):
      Physical buyers paying premium for immediate delivery
      → Real scarcity → Confirms Long signal
      Example: Crude front $98 vs second $96 → +$2 basis

    Negative (Contango):
      No urgency to buy now → Adequate supply/storage
      → Paper market → Warns against Long signal
      Example: Gold front $4,463 vs second $4,491 → -$28 basis
    """
    if front_price is None or second_price is None:
        return None
    return front_price - second_price

def classify_basis(basis, normal_basis, asset):
    """
    Classify basis relative to normal levels.
    Normal basis for Gold is around -$8 (storage costs).
    Anything significantly below normal = excess contango.
    Anything above zero = backwardation = bullish physical.
    """
    if basis is None:
        return 'UNKNOWN', 'gray'

    # For Gold and Silver, adjust for normal carrying costs
    adjusted_basis = basis - normal_basis

    if basis > 0:
        # Backwardation — physically tight
        return 'BACKWARDATION', 'bullish'
    elif adjusted_basis > BACKWARDATION_THRESHOLD:
        # Above normal — mild physical support
        return 'MILD_BACKWARDATION', 'mildly_bullish'
    elif adjusted_basis < CONTANGO_THRESHOLD * 5:
        # Deep contango — paper market, warn against longs
        return 'DEEP_CONTANGO', 'bearish'
    elif adjusted_basis < CONTANGO_THRESHOLD:
        # Normal to mild contango
        return 'CONTANGO', 'neutral'
    else:
        return 'NORMAL', 'neutral'

def get_basis_signal_adjustment(basis_status):
    """
    Return signal adjustment based on basis.
    This modifies the Decision Engine confidence.
    Backwardation → confirms Long, warns Short
    Deep contango → warns Long, confirms Short
    """
    adjustments = {
        'BACKWARDATION':      {'long': +0.15, 'short': -0.15},
        'MILD_BACKWARDATION': {'long': +0.05, 'short': -0.05},
        'NORMAL':             {'long':  0.00, 'short':  0.00},
        'CONTANGO':           {'long': -0.05, 'short': +0.05},
        'DEEP_CONTANGO':      {'long': -0.15, 'short': +0.15},
        'UNKNOWN':            {'long':  0.00, 'short':  0.00},
    }
    return adjustments.get(basis_status,
                           {'long': 0.00, 'short': 0.00})

# ═════════════════════════════════════════════════════════════
# SECTION 3 — HISTORICAL BASIS ANALYSIS
# ═════════════════════════════════════════════════════════════

def calculate_historical_basis(asset):
    """
    Calculate rolling 30-day basis history.
    Shows trend in basis — is contango deepening or shrinking?
    """
    config = FUTURES_CONFIG[asset]

    front_hist  = fetch_futures_history(
        config['front'],  period='60d')
    second_hist = fetch_futures_history(
        config['second'], period='60d')

    if front_hist is None or second_hist is None:
        return None

    # Align dates
    combined = pd.DataFrame({
        'front':  front_hist,
        'second': second_hist,
    }).dropna()

    if combined.empty:
        return None

    combined['basis'] = combined['front'] - combined['second']

    return combined['basis']

def get_basis_trend(basis_history):
    """
    Determine if basis is improving (moving toward
    backwardation) or deteriorating (deepening contango).
    """
    if basis_history is None or len(basis_history) < 5:
        return 'UNKNOWN'

    recent = basis_history.tail(5).mean()
    older  = basis_history.iloc[-10:-5].mean() \
             if len(basis_history) >= 10 else None

    if older is None:
        return 'UNKNOWN'

    change = recent - older

    if change > 0.5:
        return 'IMPROVING'    # moving toward backwardation
    elif change < -0.5:
        return 'DETERIORATING'  # deepening contango
    else:
        return 'STABLE'

# ═════════════════════════════════════════════════════════════
# SECTION 4 — MAIN ANALYSIS
# ═════════════════════════════════════════════════════════════

def analyse_all_assets():
    """Run basis analysis for all commodity assets."""
    results = {}

    for asset, config in FUTURES_CONFIG.items():
        print(f"  Fetching {asset} futures data...")

        front_price  = fetch_futures_prices(config['front'])
        second_price = fetch_futures_prices(config['second'])

        if front_price is None:
            print(f"    ⚠️ {asset} front month data unavailable")
            results[asset] = None
            continue

        if second_price is None:
            print(f"    ⚠️ {asset} second month data unavailable")
            results[asset] = None
            continue

        basis        = calculate_basis(front_price,
                                        second_price)
        status, tone = classify_basis(
            basis, config['normal_basis'], asset)
        adjustment   = get_basis_signal_adjustment(status)
        basis_hist   = calculate_historical_basis(asset)
        trend        = get_basis_trend(basis_hist)

        # Basis percentile vs recent history
        if basis_hist is not None and len(basis_hist) > 5:
            basis_pct = float(
                (basis_hist <= basis).mean() * 100)
        else:
            basis_pct = 50.0

        results[asset] = {
            'asset':          asset,
            'front_price':    round(front_price, 2),
            'second_price':   round(second_price, 2),
            'basis':          round(basis, 2),
            'normal_basis':   config['normal_basis'],
            'status':         status,
            'tone':           tone,
            'trend':          trend,
            'basis_pct':      round(basis_pct, 1),
            'long_adj':       adjustment['long'],
            'short_adj':      adjustment['short'],
            'unit':           config['unit'],
        }

        print(f"    {asset}: Front {front_price:.1f} | "
              f"Second {second_price:.1f} | "
              f"Basis {basis:+.2f} → {status} ({trend})")

    return results

# ═════════════════════════════════════════════════════════════
# SECTION 5 — SAVE TO DATABASE
# ═════════════════════════════════════════════════════════════

def save_basis_data(results):
    """Save basis analysis to database."""
    conn  = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime('%Y-%m-%d')
    rows  = []

    for asset, r in results.items():
        if r is None:
            continue
        rows.append({
            'date':         today,
            'asset':        asset,
            'front_price':  r['front_price'],
            'second_price': r['second_price'],
            'basis':        r['basis'],
            'status':       r['status'],
            'trend':        r['trend'],
            'basis_pct':    r['basis_pct'],
            'long_adj':     r['long_adj'],
            'short_adj':    r['short_adj'],
        })

    if not rows:
        conn.close()
        return

    try:
        df = pd.DataFrame(rows)
        conn.execute(
            "DELETE FROM PHYSICAL_BASIS WHERE date = ?",
            (today,)
        )
        df.to_sql('PHYSICAL_BASIS', conn,
                  if_exists='append', index=False)
        conn.commit()
        print(f"  ✅ Basis data saved")
    except:
        try:
            df = pd.DataFrame(rows)
            df.to_sql('PHYSICAL_BASIS', conn,
                      if_exists='replace', index=False)
            conn.commit()
            print(f"  ✅ PHYSICAL_BASIS table created")
        except Exception as e:
            print(f"  ❌ Save failed: {e}")
    finally:
        conn.close()

# ═════════════════════════════════════════════════════════════
# SECTION 6 — PRINT RESULTS
# ═════════════════════════════════════════════════════════════

def print_basis_report(results):
    """Print basis analysis clearly."""
    print("\n" + "="*65)
    print("PHYSICAL BASIS MONITOR — COMMODITY MARKET STRUCTURE")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    print(f"\n{'Asset':<8} {'Front':>8} {'Second':>8} "
          f"{'Basis':>8} {'Status':<20} {'Trend':<12} "
          f"{'Signal Impact'}")
    print("-"*80)

    for asset in ['Crude', 'Gold', 'Silver']:
        r = results.get(asset)
        if r is None:
            print(f"{asset:<8} Data unavailable")
            continue

        # Status emoji
        if r['status'] == 'BACKWARDATION':
            s_emoji = '🟢'
        elif r['status'] == 'MILD_BACKWARDATION':
            s_emoji = '🟡'
        elif r['status'] == 'DEEP_CONTANGO':
            s_emoji = '🔴'
        else:
            s_emoji = '⬜'

        # Trend emoji
        t_emoji = '↑' if r['trend'] == 'IMPROVING' \
             else '↓' if r['trend'] == 'DETERIORATING' \
             else '→'

        # Signal impact
        if r['long_adj'] > 0:
            impact = f"LONG +{r['long_adj']:.2f} ✅"
        elif r['long_adj'] < 0:
            impact = f"LONG {r['long_adj']:.2f} ⚠️"
        else:
            impact = "Neutral"

        print(f"{asset:<8} "
              f"{r['front_price']:>8.1f} "
              f"{r['second_price']:>8.1f} "
              f"{r['basis']:>+8.2f} "
              f"{s_emoji} {r['status']:<18} "
              f"{t_emoji} {r['trend']:<10} "
              f"{impact}")

    # Interpretation
    print("\n" + "="*65)
    print("INTERPRETATION:")
    print("="*65)

    for asset in ['Crude', 'Gold', 'Silver']:
        r = results.get(asset)
        if r is None:
            continue

        print(f"\n{asset}:")

        if r['status'] == 'BACKWARDATION':
            print(f"  🟢 Physical scarcity confirmed")
            print(f"     Front month premium: "
                  f"{r['basis']:+.2f} {r['unit']}")
            print(f"     Real buyers paying up for "
                  f"immediate delivery")
            print(f"     Long signals backed by "
                  f"fundamentals — HIGH conviction")

        elif r['status'] == 'MILD_BACKWARDATION':
            print(f"  🟡 Mild physical support")
            print(f"     Basis slightly above normal: "
                  f"{r['basis']:+.2f} {r['unit']}")
            print(f"     Some physical demand "
                  f"but not dominant")

        elif r['status'] == 'DEEP_CONTANGO':
            print(f"  🔴 Paper market — physical "
                  f"supply adequate")
            print(f"     Basis: {r['basis']:+.2f} "
                  f"{r['unit']} "
                  f"(normal: {r['normal_basis']:+.1f})")
            print(f"     Price move is sentiment-driven "
                  f"not physical")
            print(f"     Long signals at HIGH RISK "
                  f"of reversal")
            print(f"     Recommendation: Avoid new longs "
                  f"or reduce size")

        elif r['status'] == 'CONTANGO':
            print(f"  ⬜ Normal contango "
                  f"(storage + financing costs)")
            print(f"     Basis: {r['basis']:+.2f} "
                  f"{r['unit']}")
            print(f"     No physical scarcity signal")

        if r['trend'] == 'DETERIORATING':
            print(f"  ⚠️  Trend: Contango deepening — "
                  f"physical demand weakening")
        elif r['trend'] == 'IMPROVING':
            print(f"  ✅ Trend: Moving toward "
                  f"backwardation — demand strengthening")

    print("\n" + "="*65)
    print("HOW THIS AFFECTS YOUR SIGNALS:")
    print("="*65)
    for asset in ['Crude', 'Gold', 'Silver']:
        r = results.get(asset)
        if r is None:
            continue
        if r['long_adj'] != 0:
            direction = "BOOSTS" if r['long_adj'] > 0 \
                else "REDUCES"
            print(f"  {asset}: Basis {direction} Long "
                  f"signal confidence by "
                  f"{abs(r['long_adj']):.0%}")

# ═════════════════════════════════════════════════════════════
# SECTION 7 — TELEGRAM
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

def build_telegram_message(results):
    """Build basis summary for Telegram."""
    date  = datetime.now().strftime('%d %b %Y %H:%M')
    lines = [
        f"🛢️ <b>GMIS PHYSICAL BASIS MONITOR</b>",
        f"{date}",
        f"{'─' * 30}",
        f"",
    ]

    for asset in ['Crude', 'Gold', 'Silver']:
        r = results.get(asset)
        if r is None:
            continue

        if r['status'] == 'BACKWARDATION':
            emoji = '🟢'
        elif r['status'] == 'DEEP_CONTANGO':
            emoji = '🔴'
        else:
            emoji = '⬜'

        lines.append(
            f"{emoji} <b>{asset}</b>: {r['status']}"
        )
        lines.append(
            f"  Basis: {r['basis']:+.2f} {r['unit']} "
            f"| Trend: {r['trend']}"
        )

        if r['status'] == 'DEEP_CONTANGO':
            lines.append(
                f"  ⚠️ Paper market — avoid new longs"
            )
        elif r['status'] == 'BACKWARDATION':
            lines.append(
                f"  ✅ Physical scarcity — longs confirmed"
            )

        if r['long_adj'] != 0:
            direction = "+" if r['long_adj'] > 0 else ""
            lines.append(
                f"  Signal adj: Long "
                f"{direction}{r['long_adj']:.0%}"
            )
        lines.append("")

    lines.append("<i>GMIS Physical Basis Monitor</i>")
    return "\n".join(lines)

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_physical_basis(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 23 — PHYSICAL BASIS MONITOR")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    print("\nFetching futures basis data...")
    results = analyse_all_assets()

    print("\nSaving to database...")
    save_basis_data(results)

    print_basis_report(results)

    if send_telegram_flag and BOT_TOKEN:
        # Only send if any deep contango or backwardation
        notable = any(
            r is not None and
            r['status'] in ['BACKWARDATION',
                            'DEEP_CONTANGO']
            for r in results.values()
        )

        import sys
        if '--force-send' in sys.argv:
            notable = True

        if notable:
            print("\nSending basis alert to Telegram...")
            msg = build_telegram_message(results)
            asyncio.run(send_telegram(msg))
        else:
            print("\n  Normal basis — no Telegram alert")

    return results

if __name__ == "__main__":
    import sys
    no_telegram = '--no-telegram' in sys.argv
    run_physical_basis(
        send_telegram_flag=not no_telegram
    )
