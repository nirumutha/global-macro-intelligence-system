# ============================================================
# GMIS MODULE 36 — EXECUTION FRICTION MODEL
# Estimates realistic slippage, spread costs, and timing
# quality for each asset under current market conditions.
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import asyncio
import os
import sys
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────────────
BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_PATH, 'data', 'macro_system.db')
NO_TELEGRAM = '--no-telegram' in sys.argv

ASSETS = ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']

PRICE_TABLES = {
    'NIFTY':  'NIFTY50',
    'SP500':  'SP500',
    'Gold':   'GOLD',
    'Silver': 'SILVER',
    'Crude':  'CRUDE_WTI',
}

# ── Slippage parameters (Component 1) ────────────────────────
# (base_pct, high_vix_pct)  VIX threshold = 25
SLIPPAGE = {
    'NIFTY':  (0.0005, 0.0010),   # 0.05% / 0.10%
    'SP500':  (0.0003, 0.0008),   # 0.03% / 0.08%
    'Gold':   (0.0004, 0.0009),   # 0.04% / 0.09%
    'Silver': (0.0008, 0.0015),   # 0.08% / 0.15%
    'Crude':  (0.0005, 0.0012),   # 0.05% / 0.12%
}
VIX_HIGH_THRESHOLD = 25.0
EVENT_SLIPPAGE_FACTOR = 0.50    # +50% of base if event in next 24h

# ── Spread parameters (Component 2) ─────────────────────────
SPREAD_NORMAL   = 0.0010   # 0.10%
SPREAD_ELEVATED = 0.0015   # 0.15% (VIX 20-30)
SPREAD_CRISIS   = 0.0025   # 0.25% (VIX > 30)
SPREAD_EVENT    = 0.0020   # 0.20% (event day)
VIX_ELEVATED_THRESHOLD = 20.0
VIX_CRISIS_THRESHOLD   = 30.0

# ── Historical backtest baseline (Component 3) ───────────────
HISTORICAL_FLAT_FRICTION = 0.001   # 0.1% — used in Module 13/18

# ── Timing quality thresholds (Component 4) ──────────────────
FRICTION_THRESHOLDS = {
    'GOOD':    0.0010,   # total friction < 0.10%
    'AVERAGE': 0.0020,   # 0.10% – 0.20%
    'POOR':    0.0030,   # 0.20% – 0.30%
    # AVOID: VIX>30 AND event tomorrow
}


# ── Telegram ────────────────────────────────────────────────
async def send_telegram(message: str):
    if NO_TELEGRAM:
        return
    try:
        from telegram import Bot
        token   = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if not token or not chat_id:
            return
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=message,
                               parse_mode='HTML')
    except Exception as e:
        print(f"  ⚠️  Telegram error: {e}")


def get_conn():
    return sqlite3.connect(DB_PATH)


def setup_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS EXECUTION_FRICTION (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            date                TEXT,
            asset               TEXT,
            vix_us              REAL,
            vix_india           REAL,
            vix_regime          TEXT,
            event_today         INTEGER,
            event_tomorrow      INTEGER,
            event_label         TEXT,
            slippage_pct        REAL,
            spread_pct          REAL,
            total_friction_pct  REAL,
            flat_friction_pct   REAL,
            friction_vs_flat    REAL,
            timing_quality      TEXT,
            timing_note         TEXT
        )
    """)
    conn.commit()


# ──────────────────────────────────────────────────────────────
# Market condition loaders
# ──────────────────────────────────────────────────────────────
def get_latest_vix(conn) -> tuple[float, float]:
    """Return (vix_us, vix_india) — most recent close."""
    vix_us = np.nan
    try:
        df = pd.read_sql(
            "SELECT * FROM VIX_US ORDER BY Date DESC LIMIT 1", conn)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        vix_us = float(df['Close'].iloc[0])
    except Exception as e:
        print(f"  ⚠️  VIX_US load: {e}")

    vix_india = np.nan
    try:
        df = pd.read_sql(
            "SELECT * FROM VIX_INDIA ORDER BY Date DESC LIMIT 1", conn)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        vix_india = float(df['Close'].iloc[0])
    except Exception as e:
        print(f"  ⚠️  VIX_INDIA load: {e}")

    return vix_us, vix_india


def get_upcoming_events(conn, today: datetime) -> tuple[bool, bool, str]:
    """
    Returns (event_today, event_tomorrow, label).
    Checks ECONOMIC_CALENDAR for HIGH-impact events.
    """
    today_str    = today.strftime('%Y-%m-%d')
    tomorrow_str = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        df = pd.read_sql(
            "SELECT event, date, country FROM ECONOMIC_CALENDAR "
            "WHERE impact = 'HIGH'",
            conn)
        today_events    = df[df['date'] == today_str]
        tomorrow_events = df[df['date'] == tomorrow_str]

        labels = []
        if not today_events.empty:
            labels += [f"{r['event']} ({r['country']})"
                       for _, r in today_events.head(2).iterrows()]
        if not tomorrow_events.empty:
            labels += [f"{r['event']} ({r['country']}) [tmrw]"
                       for _, r in tomorrow_events.head(2).iterrows()]

        return (
            not today_events.empty,
            not tomorrow_events.empty,
            '; '.join(labels) if labels else '',
        )
    except Exception as e:
        print(f"  ⚠️  Calendar load: {e}")
        return False, False, ''


# ──────────────────────────────────────────────────────────────
# COMPONENT 1 — Slippage estimation
# ──────────────────────────────────────────────────────────────
def estimate_slippage(asset: str, vix_us: float,
                      event_in_24h: bool) -> float:
    base, high = SLIPPAGE[asset]
    slip = high if (not np.isnan(vix_us) and vix_us > VIX_HIGH_THRESHOLD) \
           else base
    if event_in_24h:
        slip += base * EVENT_SLIPPAGE_FACTOR
    return slip


# ──────────────────────────────────────────────────────────────
# COMPONENT 2 — Spread widening
# ──────────────────────────────────────────────────────────────
def estimate_spread(vix_us: float, event_today: bool,
                    event_tomorrow: bool) -> float:
    if np.isnan(vix_us):
        return SPREAD_NORMAL

    # Base from VIX regime
    if vix_us > VIX_CRISIS_THRESHOLD:
        spread = SPREAD_CRISIS
    elif vix_us > VIX_ELEVATED_THRESHOLD:
        spread = SPREAD_ELEVATED
    else:
        spread = SPREAD_NORMAL

    # Override upward if event today or tomorrow
    if event_today or event_tomorrow:
        spread = max(spread, SPREAD_EVENT)

    return spread


def vix_regime_label(vix_us: float) -> str:
    if np.isnan(vix_us):
        return 'UNKNOWN'
    if vix_us > VIX_CRISIS_THRESHOLD:
        return 'CRISIS'
    if vix_us > VIX_HIGH_THRESHOLD:
        return 'HIGH'
    if vix_us > VIX_ELEVATED_THRESHOLD:
        return 'ELEVATED'
    return 'NORMAL'


# ──────────────────────────────────────────────────────────────
# COMPONENT 3 — Backtest reality adjustment
# ──────────────────────────────────────────────────────────────
def load_price_close(conn, table: str) -> pd.Series:
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Close']).sort_values('Date')
    return df.set_index('Date')['Close']


def load_signal_positions(conn, asset: str) -> pd.Series:
    """Map SIGNALS_V3 signal column to +1 / 0 / -1 position."""
    sig_map = {'Long': 1, 'Short': -1, 'Neutral': 0}
    try:
        df = pd.read_sql(
            f"SELECT Date, {asset}_signal FROM SIGNALS_V3", conn,
            parse_dates=['Date'])
        df = df.sort_values('Date').set_index('Date')
        return df[f'{asset}_signal'].map(sig_map).fillna(0)
    except Exception as e:
        print(f"  ⚠️  SIGNALS_V3 load for {asset}: {e}")
        return pd.Series(dtype=float)


def run_backtest_with_friction(positions: pd.Series,
                                prices: pd.Series,
                                friction_pct: float) -> float:
    """
    Computes annualised Sharpe of a long/short strategy.
    friction_pct is charged on each TRADE (position change).
    """
    idx = positions.index.intersection(prices.index)
    if len(idx) < 60:
        return np.nan
    pos = positions.loc[idx]
    px  = prices.loc[idx]
    ret = px.pct_change()

    # Strategy return = position × next-day return
    strat = (pos.shift(1) * ret).dropna()

    # Identify trades (position changes)
    trades     = (pos.diff().abs() > 0.5).astype(float)
    trade_cost = (trades * friction_pct).loc[strat.index]
    net_strat  = strat - trade_cost

    if net_strat.std() < 1e-8:
        return np.nan
    return float(net_strat.mean() / net_strat.std() * np.sqrt(252))


def backtest_comparison(conn,
                         current_friction: dict) -> dict:
    """
    Returns { asset: {flat_sharpe, realistic_sharpe, trades_pa, drag} }
    """
    results = {}
    for asset in ASSETS:
        prices    = load_price_close(conn, PRICE_TABLES[asset])
        positions = load_signal_positions(conn, asset)
        if positions.empty:
            results[asset] = None
            continue

        flat_sharpe = run_backtest_with_friction(
            positions, prices, HISTORICAL_FLAT_FRICTION)
        real_sharpe = run_backtest_with_friction(
            positions, prices, current_friction[asset])

        # Estimate trades per annum
        idx = positions.index.intersection(prices.index)
        trades_total = (positions.loc[idx].diff().abs() > 0.5).sum()
        n_years      = max((idx[-1] - idx[0]).days / 365.25, 1)
        trades_pa    = trades_total / n_years

        extra_drag_pa = trades_pa * (
            current_friction[asset] - HISTORICAL_FLAT_FRICTION)

        results[asset] = {
            'flat_sharpe':    flat_sharpe,
            'real_sharpe':    real_sharpe,
            'trades_pa':      trades_pa,
            'extra_drag_bps': extra_drag_pa * 10_000,
        }
    return results


# ──────────────────────────────────────────────────────────────
# COMPONENT 4 — Timing quality
# ──────────────────────────────────────────────────────────────
def assess_timing(asset: str, total_friction: float,
                  vix_us: float, event_today: bool,
                  event_tomorrow: bool) -> tuple[str, str]:
    # AVOID: worst-case — high VIX AND imminent event
    if (not np.isnan(vix_us)
            and vix_us > VIX_CRISIS_THRESHOLD
            and (event_today or event_tomorrow)):
        return ('AVOID',
                'Extreme friction: VIX>30 + event imminent — '
                'do not enter new positions')

    if total_friction <= FRICTION_THRESHOLDS['GOOD']:
        return ('GOOD_TIME',
                'Friction below average — good time to execute')

    if total_friction <= FRICTION_THRESHOLDS['AVERAGE']:
        return ('AVERAGE_TIME',
                'Normal friction — proceed as planned')

    if total_friction <= FRICTION_THRESHOLDS['POOR']:
        return ('POOR_TIME',
                'Elevated friction — consider waiting for lower-VIX window')

    return ('POOR_TIME',
            'High friction environment — prefer limit orders, '
            'avoid market-on-open entries')


# ──────────────────────────────────────────────────────────────
# Save to DB
# ──────────────────────────────────────────────────────────────
def save_results(conn, today_str: str, rows: list):
    cur = conn.cursor()
    cur.execute("DELETE FROM EXECUTION_FRICTION WHERE date = ?",
                (today_str,))
    cur.executemany("""
        INSERT INTO EXECUTION_FRICTION
          (date, asset, vix_us, vix_india, vix_regime,
           event_today, event_tomorrow, event_label,
           slippage_pct, spread_pct, total_friction_pct,
           flat_friction_pct, friction_vs_flat,
           timing_quality, timing_note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    print(f"  ✅ Saved {len(rows)} friction records")


# ──────────────────────────────────────────────────────────────
# Print report
# ──────────────────────────────────────────────────────────────
def print_report(today_str: str, vix_us: float, vix_india: float,
                 event_today: bool, event_tomorrow: bool, event_label: str,
                 friction_rows: list, bt_results: dict):

    print(f"\n{'='*70}")
    print(f"EXECUTION FRICTION MODEL — DAILY REPORT")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*70}")

    # Market conditions header
    vix_r = vix_regime_label(vix_us)
    ev_str = f"YES — {event_label}" if (event_today or event_tomorrow) else "None"
    print(f"\n  VIX (US):     {vix_us:.2f}  [{vix_r}]")
    print(f"  VIX (India):  {vix_india:.2f}")
    print(f"  Events 24-48h: {ev_str}")

    # ── Component 1+2 — Friction per asset ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 1+2 — SLIPPAGE & SPREAD BY ASSET")
    print(f"{'─'*70}")
    print(f"  {'Asset':<8} {'Slip%':>6} {'Sprd%':>6} {'Total%':>7} "
          f"{'vs Flat':>8} {'Regime':>12}  Timing")
    print(f"  {'-'*68}")

    # Build lookup from rows
    row_map = {r[1]: r for r in friction_rows}  # r[1] = asset

    for asset in ASSETS:
        r   = row_map[asset]
        slp = r[8]
        spd = r[9]
        tot = r[10]
        vfl = r[12]
        tq  = r[13]
        regime = r[4]   # vix_regime
        vfl_str = f"{vfl*10000:+.1f}bps"
        tq_icon = {'GOOD_TIME':'🟢','AVERAGE_TIME':'🟡',
                   'POOR_TIME':'🟠','AVOID':'🔴'}.get(tq, ' ')
        print(f"  {asset:<8} {slp*100:>5.3f}% {spd*100:>5.3f}% "
              f"{tot*100:>6.3f}% {vfl_str:>8}  {regime:>12}  "
              f"{tq_icon} {tq}")

    print(f"\n  Flat baseline: {HISTORICAL_FLAT_FRICTION*100:.2f}% per trade "
          f"(Module 13/18 assumption)")

    # ── Component 3 — Backtest comparison ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 3 — BACKTEST REALITY ADJUSTMENT")
    print(f"{'─'*70}")
    print(f"  {'Asset':<8} {'FlatFriction':>14} {'RealisticFriction':>18} "
          f"{'ΔSharpe':>8} {'Trades/yr':>10} {'ExtraDrag':>10}")
    print(f"  {'-'*68}")

    for asset in ASSETS:
        bt = bt_results.get(asset)
        r  = row_map[asset]
        tot = r[10]
        if bt is None:
            print(f"  {asset:<8}  {'N/A':>14}  {'N/A':>18}  {'N/A':>8}")
            continue
        fs  = bt['flat_sharpe']
        rs  = bt['real_sharpe']
        dsh = rs - fs if (not np.isnan(rs) and not np.isnan(fs)) else np.nan
        tpa = bt['trades_pa']
        drg = bt['extra_drag_bps']
        fs_s  = f"{fs:.3f}" if not np.isnan(fs) else "N/A"
        rs_s  = f"{rs:.3f}" if not np.isnan(rs) else "N/A"
        dsh_s = f"{dsh:+.3f}" if not np.isnan(dsh) else "N/A"
        drg_s = f"{drg:+.1f}bps" if not np.isnan(drg) else "N/A"
        print(f"  {asset:<8} Sharpe {fs_s:>7} (0.10% flat) → "
              f"{rs_s:>7} ({tot*100:.3f}% real)  "
              f"{dsh_s:>7}  {tpa:>6.1f}/yr  {drg_s:>10}")

    # ── Component 4 — Overall timing call ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 4 — TRADE TIMING RECOMMENDATION")
    print(f"{'─'*70}")

    # Aggregate — use worst single-asset timing
    priority = {'AVOID': 4, 'POOR_TIME': 3, 'AVERAGE_TIME': 2,
                 'GOOD_TIME': 1}
    worst_tq   = max([r[13] for r in friction_rows],
                     key=lambda x: priority.get(x, 0))
    worst_note = next(r[14] for r in friction_rows if r[13] == worst_tq)

    icon = {'GOOD_TIME':'🟢','AVERAGE_TIME':'🟡',
            'POOR_TIME':'🟠','AVOID':'🔴'}.get(worst_tq,'')
    print(f"\n  {icon} OVERALL: {worst_tq}")
    print(f"  {worst_note}")

    print(f"\n  Per-asset timing:")
    for asset in ASSETS:
        r  = row_map[asset]
        tq = r[13]
        tn = r[14]
        ic = {'GOOD_TIME':'🟢','AVERAGE_TIME':'🟡',
              'POOR_TIME':'🟠','AVOID':'🔴'}.get(tq,'')
        print(f"    {asset:<8} {ic} {tq:<14}  {tn}")

    print(f"\n{'='*70}\n")

    return worst_tq


# ──────────────────────────────────────────────────────────────
# Telegram message (AVOID only)
# ──────────────────────────────────────────────────────────────
def build_telegram_message(today_str: str, vix_us: float,
                            event_label: str,
                            friction_rows: list) -> str:
    row_map = {r[1]: r for r in friction_rows}
    lines = [
        "🔴 <b>GMIS MODULE 36 — EXECUTION ALERT</b>",
        f"📅 {today_str}",
        "",
        "⚠️ <b>AVOID condition triggered</b>",
        f"  VIX: {vix_us:.2f} | Events: {event_label}",
        "",
        "<b>Do not enter new positions today.</b>",
        "Extreme friction: very high VIX + imminent event.",
        "",
        "<b>Friction by asset:</b>",
    ]
    for asset in ASSETS:
        r   = row_map[asset]
        tot = r[10]
        tq  = r[13]
        ic  = {'GOOD_TIME':'🟢','AVERAGE_TIME':'🟡',
               'POOR_TIME':'🟠','AVOID':'🔴'}.get(tq,'')
        lines.append(f"  {ic} {asset}: {tot*100:.3f}% total friction")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    today     = datetime.now()
    today_str = today.strftime('%Y-%m-%d')

    print(f"\n{'='*65}")
    print(f"GMIS MODULE 36 — EXECUTION FRICTION MODEL")
    print(f"{today.strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*65}")

    conn = get_conn()
    setup_table(conn)

    # ── Market conditions ──
    print("\nLoading market conditions...")
    vix_us, vix_india = get_latest_vix(conn)
    event_today, event_tomorrow, event_label = get_upcoming_events(conn, today)
    event_in_24h = event_today or event_tomorrow
    vix_regime   = vix_regime_label(vix_us)

    print(f"  VIX US:    {vix_us:.2f}  [{vix_regime}]")
    print(f"  VIX India: {vix_india:.2f}")
    print(f"  Events:    today={event_today}, tomorrow={event_tomorrow}")
    if event_label:
        print(f"  Labels:    {event_label}")

    # ── Component 1+2 — Per-asset friction ──
    print("\nComponent 1+2 — Slippage + spread per asset...")
    current_friction = {}
    friction_rows    = []

    for asset in ASSETS:
        slip   = estimate_slippage(asset, vix_us, event_in_24h)
        spread = estimate_spread(vix_us, event_today, event_tomorrow)
        total  = slip + spread
        vs_flat = total - HISTORICAL_FLAT_FRICTION
        current_friction[asset] = total
        tq, tn = assess_timing(asset, total, vix_us,
                                event_today, event_tomorrow)
        friction_rows.append((
            today_str,
            asset,
            float(vix_us)    if not np.isnan(vix_us)    else None,
            float(vix_india) if not np.isnan(vix_india) else None,
            vix_regime,
            int(event_today),
            int(event_tomorrow),
            event_label,
            slip,
            spread,
            total,
            HISTORICAL_FLAT_FRICTION,
            vs_flat,
            tq,
            tn,
        ))

    # ── Component 3 — Backtest comparison ──
    print("\nComponent 3 — Backtest comparison (flat vs realistic friction)...")
    bt_results = backtest_comparison(conn, current_friction)

    # ── Save ──
    print("\nSaving results...")
    save_results(conn, today_str, friction_rows)

    # ── Report ──
    worst_tq = print_report(
        today_str, vix_us, vix_india,
        event_today, event_tomorrow, event_label,
        friction_rows, bt_results)

    # ── Telegram — AVOID only ──
    if worst_tq == 'AVOID' and not NO_TELEGRAM:
        msg = build_telegram_message(
            today_str, vix_us, event_label, friction_rows)
        asyncio.run(send_telegram(msg))
        print("  📱 Telegram AVOID alert sent")
    elif NO_TELEGRAM:
        print("  Telegram skipped (--no-telegram)")
    else:
        print(f"  Telegram not triggered (timing={worst_tq})")

    conn.close()


if __name__ == '__main__':
    main()
