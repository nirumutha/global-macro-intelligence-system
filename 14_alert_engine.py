# ============================================================
# GMIS 2.0 — MODULE 14 — ALERT ENGINE
# Telegram-based intelligent alerting system
# Triggers only on meaningful market changes
# ============================================================

import asyncio
import telegram
import sqlite3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────
load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID   = int(os.getenv('TELEGRAM_CHAT_ID'))

BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_PATH, 'data', 'macro_system.db')
STATE_FILE = os.path.join(BASE_PATH, 'data', 'alert_state.json')

# ── Thresholds ────────────────────────────────────────────────
VIX_CRISIS_LEVEL  = 30
VIX_CAUTION_LEVEL = 20
SENTIMENT_EXTREME = 0.50
SIGNAL_THRESHOLD  = 0.15

# ═════════════════════════════════════════════════════════════
# SECTION 1 — SEND MESSAGE
# ═════════════════════════════════════════════════════════════

async def send_telegram(message):
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID,
            text=message,
            parse_mode='HTML'
        )
        print(f"  ✅ Alert sent: {message[:60]}...")
    except Exception as e:
        print(f"  ❌ Failed to send alert: {e}")

def send_alert(message):
    asyncio.run(send_telegram(message))

# ═════════════════════════════════════════════════════════════
# SECTION 2 — LOAD CURRENT STATE FROM DATABASE
# ═════════════════════════════════════════════════════════════

def load_current_state():
    conn  = sqlite3.connect(DB_PATH)
    state = {}

    try:
        for asset, table in [
            ('NIFTY',  'NIFTY50'),
            ('SP500',  'SP500'),
            ('Gold',   'GOLD'),
            ('Silver', 'SILVER'),
            ('Crude',  'CRUDE_WTI'),
            ('VIX',    'VIX_US'),
        ]:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            close_col = [c for c in df.columns
                         if 'Close' in c or 'close' in c]
            col = close_col[0] if close_col else df.columns[1]
            state[f'{asset}_price'] = float(df[col].iloc[-1])
            state[f'{asset}_date']  = str(df['Date'].iloc[-1].date())

        sig_df = pd.read_sql("SELECT * FROM SIGNALS", conn)
        sig_df['Date'] = pd.to_datetime(sig_df['Date'])
        sig_df = sig_df.sort_values('Date')
        latest_sig = sig_df.iloc[-1]

        for asset, col in [
            ('NIFTY',  'NIFTY_score'),
            ('SP500',  'SP500_score'),
            ('Gold',   'Gold_score'),
            ('Silver', 'Silver_score'),
            ('Crude',  'Crude_score'),
        ]:
            score = float(latest_sig[col]) if col in latest_sig else 0.0
            if score >= SIGNAL_THRESHOLD:
                signal = 'Long'
            elif score <= -SIGNAL_THRESHOLD:
                signal = 'Short'
            else:
                signal = 'Neutral'
            state[f'{asset}_signal'] = signal
            state[f'{asset}_score']  = score

        sent_df = pd.read_sql("SELECT * FROM SENTIMENT_DAILY", conn)
        state['sentiment_score'] = float(sent_df['score'].mean())

        yields  = pd.read_sql(
            "SELECT * FROM US_10Y_YIELD ORDER BY Date DESC LIMIT 1", conn)
        yields2 = pd.read_sql(
            "SELECT * FROM US_2Y_YIELD ORDER BY Date DESC LIMIT 1",  conn)
        state['yield_spread'] = float(yields.iloc[0,1]) - float(yields2.iloc[0,1])

    except Exception as e:
        print(f"  ⚠️ Error loading state: {e}")
    finally:
        conn.close()

    vix = state.get('VIX_price', 20)
    if vix > VIX_CRISIS_LEVEL:
        state['regime'] = 'Crisis'
    elif vix > VIX_CAUTION_LEVEL:
        state['regime'] = 'Sideways'
    else:
        state['regime'] = 'Bull Market'

    return state

# ═════════════════════════════════════════════════════════════
# SECTION 3 — LOAD AND SAVE PREVIOUS STATE
# ═════════════════════════════════════════════════════════════

def load_previous_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return None

def save_current_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    print("  ✅ State saved for next comparison")

# ═════════════════════════════════════════════════════════════
# SECTION 4 — ALERT CHECKS
# ═════════════════════════════════════════════════════════════

def check_signal_changes(current, previous):
    alerts = []
    for asset in ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']:
        curr_sig  = current.get(f'{asset}_signal', 'Neutral')
        prev_sig  = previous.get(f'{asset}_signal', 'Neutral')
        curr_score= current.get(f'{asset}_score', 0)

        if curr_sig != prev_sig:
            emoji = '🟢' if curr_sig == 'Long' else \
                    '🔴' if curr_sig == 'Short' else '🟡'
            abs_score = abs(curr_score)
            confidence = 'HIGH'   if abs_score >= 0.40 else \
                         'MEDIUM' if abs_score >= 0.25 else 'LOW'
            alerts.append(
                f"{emoji} <b>SIGNAL CHANGE — {asset}</b>\n"
                f"{prev_sig} → <b>{curr_sig}</b>\n"
                f"Score: {curr_score:+.3f} | Confidence: {confidence}\n"
                f"Date: {current.get('SP500_date', 'today')}"
            )
    return alerts

def check_regime_shift(current, previous):
    alerts = []
    curr_regime = current.get('regime', 'Sideways')
    prev_regime = previous.get('regime', 'Sideways')
    curr_vix    = current.get('VIX_price', 20)

    if curr_regime != prev_regime:
        emoji = '🚨' if curr_regime == 'Crisis'     else \
                '🔴' if curr_regime == 'Bear Market' else \
                '🟢' if curr_regime == 'Bull Market' else '⚠️'
        alerts.append(
            f"{emoji} <b>REGIME SHIFT</b>\n"
            f"{prev_regime} → <b>{curr_regime}</b>\n"
            f"VIX: {curr_vix:.1f}\n"
            f"Action: Review all positions"
        )
    return alerts

def check_vix_spike(current, previous):
    alerts = []
    curr_vix = current.get('VIX_price', 20)
    prev_vix = previous.get('VIX_price', 20)

    if curr_vix > VIX_CRISIS_LEVEL and prev_vix <= VIX_CRISIS_LEVEL:
        alerts.append(
            f"🚨 <b>VIX CRISIS THRESHOLD BREACHED</b>\n"
            f"VIX: {prev_vix:.1f} → <b>{curr_vix:.1f}</b>\n"
            f"Action: Consider reducing exposure immediately"
        )
    elif curr_vix > VIX_CAUTION_LEVEL and prev_vix <= VIX_CAUTION_LEVEL:
        alerts.append(
            f"⚠️ <b>VIX CAUTION LEVEL REACHED</b>\n"
            f"VIX: {prev_vix:.1f} → <b>{curr_vix:.1f}</b>\n"
            f"Action: Tighten stops, avoid new longs"
        )
    elif curr_vix < VIX_CAUTION_LEVEL and prev_vix >= VIX_CAUTION_LEVEL:
        alerts.append(
            f"✅ <b>VIX RETURNING TO CALM</b>\n"
            f"VIX: {prev_vix:.1f} → <b>{curr_vix:.1f}</b>\n"
            f"Action: Risk appetite recovering"
        )
    return alerts

def check_sentiment_extreme(current, previous):
    alerts = []
    curr_sent = current.get('sentiment_score', 0)
    prev_sent = previous.get('sentiment_score', 0)

    if curr_sent < -SENTIMENT_EXTREME and prev_sent >= -SENTIMENT_EXTREME:
        alerts.append(
            f"📰 <b>EXTREME NEGATIVE SENTIMENT</b>\n"
            f"Score: <b>{curr_sent:+.3f}</b> (was {prev_sent:+.3f})\n"
            f"Watch: NIFTY, SP500 for downside pressure"
        )
    elif curr_sent > SENTIMENT_EXTREME and prev_sent <= SENTIMENT_EXTREME:
        alerts.append(
            f"📰 <b>EXTREME POSITIVE SENTIMENT</b>\n"
            f"Score: <b>{curr_sent:+.3f}</b> (was {prev_sent:+.3f})\n"
            f"Watch: Equities for upside momentum"
        )
    return alerts

def check_yield_curve(current, previous):
    alerts = []
    curr_spread = current.get('yield_spread', 0.5)
    prev_spread = previous.get('yield_spread', 0.5)

    if curr_spread < 0 and prev_spread >= 0:
        alerts.append(
            f"⚠️ <b>YIELD CURVE INVERTED</b>\n"
            f"Spread: {prev_spread:+.2f}% → <b>{curr_spread:+.2f}%</b>\n"
            f"Historically precedes recession by 12-18 months"
        )
    elif curr_spread >= 0 and prev_spread < 0:
        alerts.append(
            f"✅ <b>YIELD CURVE NORMALISED</b>\n"
            f"Spread: {prev_spread:+.2f}% → <b>{curr_spread:+.2f}%</b>"
        )
    return alerts

# ═════════════════════════════════════════════════════════════
# SECTION 5 — DAILY SUMMARY
# ═════════════════════════════════════════════════════════════

def generate_daily_summary(current):
    vix    = current.get('VIX_price', 0)
    regime = current.get('regime', 'Unknown')
    sent   = current.get('sentiment_score', 0)
    spread = current.get('yield_spread', 0)

    signal_lines = []
    for asset in ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']:
        sig   = current.get(f'{asset}_signal', 'Neutral')
        score = current.get(f'{asset}_score', 0)
        emoji = '🟢' if sig == 'Long' else \
                '🔴' if sig == 'Short' else '🟡'
        signal_lines.append(
            f"  {emoji} {asset}: {sig} ({score:+.2f})"
        )

    sent_label = 'Positive' if sent > 0.05 else \
                 'Negative' if sent < -0.05 else 'Neutral'

    return (
        f"📊 <b>GMIS DAILY SUMMARY</b>\n"
        f"{datetime.now().strftime('%d %b %Y — %H:%M IST')}\n"
        f"{'─' * 30}\n\n"
        f"<b>Market Regime:</b> {regime}\n"
        f"<b>VIX:</b> {vix:.1f}\n"
        f"<b>Yield Curve:</b> {spread:+.2f}%\n"
        f"<b>Sentiment:</b> {sent_label} ({sent:+.3f})\n\n"
        f"<b>Active Signals:</b>\n"
        f"{chr(10).join(signal_lines)}\n\n"
        f"<b>Prices:</b>\n"
        f"  NIFTY: {current.get('NIFTY_price', 0):,.0f}\n"
        f"  SP500: {current.get('SP500_price', 0):,.0f}\n"
        f"  Gold:  ${current.get('Gold_price', 0):,.0f}\n"
        f"  Crude: ${current.get('Crude_price', 0):,.1f}\n\n"
        f"<i>GMIS Intelligence System</i>"
    )

# ═════════════════════════════════════════════════════════════
# SECTION 6 — MAIN RUNNER
# ═════════════════════════════════════════════════════════════

def run_alert_check(send_summary=False):
    print("\n" + "="*50)
    print("GMIS ALERT ENGINE")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*50)

    print("\nLoading current market state...")
    current  = load_current_state()
    previous = load_previous_state()

    if previous is None:
        print("  First run — sending daily summary and saving baseline...")
        send_alert(generate_daily_summary(current))
        save_current_state(current)
        return

    print("\nChecking alert conditions...")
    all_alerts = []
    all_alerts += check_signal_changes(current, previous)
    all_alerts += check_regime_shift(current, previous)
    all_alerts += check_vix_spike(current, previous)
    all_alerts += check_sentiment_extreme(current, previous)
    all_alerts += check_yield_curve(current, previous)

    if all_alerts:
        print(f"\n  {len(all_alerts)} alert(s) triggered:")
        for alert in all_alerts:
            send_alert(alert)
    else:
        print("\n  No significant changes — no alerts sent")

    if send_summary:
        print("\n  Sending daily summary...")
        send_alert(generate_daily_summary(current))

    print("\nSaving state...")
    save_current_state(current)
    print("\n" + "="*50)
    print("ALERT CHECK COMPLETE")
    print("="*50)

if __name__ == "__main__":
    import sys
    send_summary = '--summary' in sys.argv
    run_alert_check(send_summary=send_summary)
