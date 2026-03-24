# ============================================================
# GMIS MODULE 38 — RELATIVE VALUE RATIO OVERLAY
# Gold/Silver, Gold/SP500, Crude/Gold ratios — percentile
# positioning and rotation signals.
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import asyncio
import os
import sys
from datetime import datetime

# ── Config ───────────────────────────────────────────────────
BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_PATH, 'data', 'macro_system.db')
NO_TELEGRAM = '--no-telegram' in sys.argv

# Rolling window for percentile (5 years ≈ 1260 trading days)
PERCENTILE_WINDOW = 1260

# Extreme thresholds
EXTREME_HIGH = 90   # trigger Telegram
EXTREME_LOW  = 10
SIGNAL_HIGH  = 80   # directional signal
SIGNAL_LOW   = 20

# Historical reference averages (for context)
GS_HIST_AVG  = 70.0    # Gold/Silver long-run average
GSP_HIST_AVG = 0.50    # Gold/SP500 approximate mid-range
CG_HIST_AVG  = 0.065   # Crude/Gold (~15-20 barrels per oz)


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
        CREATE TABLE IF NOT EXISTS RV_RATIOS (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            date                TEXT,
            -- Gold/Silver
            gs_ratio            REAL,
            gs_pct_5y           REAL,
            gs_pct_full         REAL,
            gs_signal           TEXT,
            gs_rotation         TEXT,
            -- Gold/SP500
            gsp_ratio           REAL,
            gsp_pct_5y          REAL,
            gsp_pct_full        REAL,
            gsp_signal          TEXT,
            gsp_rotation        TEXT,
            -- Crude/Gold
            cg_ratio            REAL,
            cg_pct_5y           REAL,
            cg_pct_full         REAL,
            cg_signal           TEXT,
            cg_rotation         TEXT,
            -- Summary
            extremes_count      INTEGER,
            extremes_flagged    TEXT,
            rotation_summary    TEXT
        )
    """)
    conn.commit()


# ──────────────────────────────────────────────────────────────
# Price loader
# ──────────────────────────────────────────────────────────────
def load_close(conn, table: str) -> pd.Series:
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date']  = pd.to_datetime(df['Date'])
    df = (df.dropna(subset=['Close'])
            .query('Close > 0')
            .sort_values('Date')
            .set_index('Date'))
    return df['Close']


# ──────────────────────────────────────────────────────────────
# Ratio helpers
# ──────────────────────────────────────────────────────────────
def build_ratio_series(num: pd.Series, den: pd.Series) -> pd.Series:
    """Align on common dates and compute ratio."""
    idx    = num.index.intersection(den.index)
    ratio  = num.loc[idx] / den.loc[idx]
    return ratio.dropna()


def rolling_percentile(series: pd.Series,
                        window: int,
                        current_val: float) -> float:
    """
    Percentile of current_val within the trailing `window` observations.
    Returns value in [0, 100].
    """
    hist = series.tail(window).dropna()
    if len(hist) < 30:
        return np.nan
    return float(pd.Series(hist).rank(pct=True).iloc[-1] * 100)


def full_history_percentile(series: pd.Series,
                             current_val: float) -> float:
    hist = series.dropna()
    if len(hist) < 30:
        return np.nan
    return float(pd.Series(hist).rank(pct=True).iloc[-1] * 100)


# ──────────────────────────────────────────────────────────────
# Signal + rotation language
# ──────────────────────────────────────────────────────────────
def classify_percentile(pct: float) -> str:
    if np.isnan(pct):
        return 'UNKNOWN'
    if pct >= EXTREME_HIGH:
        return 'EXTREME_HIGH'
    if pct >= SIGNAL_HIGH:
        return 'HIGH'
    if pct <= EXTREME_LOW:
        return 'EXTREME_LOW'
    if pct <= SIGNAL_LOW:
        return 'LOW'
    return 'NEUTRAL'


# ── Gold/Silver ──────────────────────────────────────────────
def gs_rotation_signal(ratio: float, pct: float) -> tuple[str, str]:
    """(signal, rotation_text)"""
    label = classify_percentile(pct)
    pct_s = f"{pct:.0f}th" if not np.isnan(pct) else "N/A"

    if label in ('EXTREME_HIGH', 'HIGH'):
        return (label,
                f"Gold/Silver ratio {ratio:.1f} at {pct_s} percentile — "
                f"Silver historically cheap vs Gold → "
                f"Silver Long more attractive; consider rotating Gold → Silver")

    if label in ('EXTREME_LOW', 'LOW'):
        return (label,
                f"Gold/Silver ratio {ratio:.1f} at {pct_s} percentile — "
                f"Silver expensive vs Gold → "
                f"Prefer Gold over Silver; Silver premium may compress")

    return ('NEUTRAL',
            f"Gold/Silver ratio {ratio:.1f} at {pct_s} percentile — "
            f"within historical norm (avg ~{GS_HIST_AVG:.0f}), no rotation signal")


# ── Gold/SP500 ───────────────────────────────────────────────
def gsp_rotation_signal(ratio: float, pct: float) -> tuple[str, str]:
    label = classify_percentile(pct)
    pct_s = f"{pct:.0f}th" if not np.isnan(pct) else "N/A"
    barrels = f"{1/ratio:.1f}" if ratio > 0 else "N/A"

    if label in ('EXTREME_HIGH', 'HIGH'):
        return (label,
                f"Gold/SP500 ratio {ratio:.4f} at {pct_s} percentile — "
                f"Equities cheap vs Gold (mean-reversion signal) → "
                f"Consider rotating from Gold into equities")

    if label in ('EXTREME_LOW', 'LOW'):
        return (label,
                f"Gold/SP500 ratio {ratio:.4f} at {pct_s} percentile — "
                f"Gold cheap vs equities (Buffett-style RV) → "
                f"Consider rotating from equities into Gold")

    return ('NEUTRAL',
            f"Gold/SP500 ratio {ratio:.4f} at {pct_s} percentile — "
            f"balanced relative valuation, no rotation signal")


# ── Crude/Gold ───────────────────────────────────────────────
def cg_rotation_signal(ratio: float, pct: float) -> tuple[str, str]:
    label  = classify_percentile(pct)
    pct_s  = f"{pct:.0f}th" if not np.isnan(pct) else "N/A"
    barrels = f"{1/ratio:.1f}" if ratio > 0 else "N/A"

    if label in ('EXTREME_HIGH', 'HIGH'):
        return (label,
                f"Crude/Gold ratio {ratio:.4f} at {pct_s} percentile "
                f"({barrels} barrels/oz) — "
                f"Oil expensive vs Gold; supply-shock premium in Crude likely inflated → "
                f"Crude downside risk, Gold relatively cheaper")

    if label in ('EXTREME_LOW', 'LOW'):
        return (label,
                f"Crude/Gold ratio {ratio:.4f} at {pct_s} percentile "
                f"({barrels} barrels/oz) — "
                f"Gold expensive vs Oil; geopolitical premium in Gold may be inflated → "
                f"Gold downside risk on premium compression")

    return ('NEUTRAL',
            f"Crude/Gold ratio {ratio:.4f} at {pct_s} percentile "
            f"({barrels} barrels/oz) — "
            f"within historical norm (avg ~{CG_HIST_AVG:.3f}), no premium signal")


# ──────────────────────────────────────────────────────────────
# Extreme crossing check (for Telegram)
# ──────────────────────────────────────────────────────────────
def get_prev_signals(conn) -> dict:
    """Load previous day's signals for crossing detection."""
    try:
        df = pd.read_sql(
            "SELECT gs_signal, gsp_signal, cg_signal "
            "FROM RV_RATIOS ORDER BY date DESC LIMIT 1",
            conn)
        if not df.empty:
            return {
                'gs':  df['gs_signal'].iloc[0],
                'gsp': df['gsp_signal'].iloc[0],
                'cg':  df['cg_signal'].iloc[0],
            }
    except Exception:
        pass
    return {'gs': None, 'gsp': None, 'cg': None}


def detect_extreme_crossings(prev: dict,
                               curr: dict) -> list[tuple[str, str, str]]:
    """
    Returns list of (ratio_name, prev_signal, curr_signal) for any
    new EXTREME crossings (entering EXTREME_HIGH or EXTREME_LOW).
    """
    extremes = {'EXTREME_HIGH', 'EXTREME_LOW'}
    crossings = []
    for key, name in [('gs', 'Gold/Silver'),
                       ('gsp', 'Gold/SP500'),
                       ('cg', 'Crude/Gold')]:
        p = prev.get(key)
        c = curr.get(key)
        if c in extremes and (p is None or p not in extremes):
            crossings.append((name, p or 'UNKNOWN', c))
    return crossings


# ──────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────
def save_results(conn, today_str: str, data: dict):
    cur = conn.cursor()
    cur.execute("DELETE FROM RV_RATIOS WHERE date = ?", (today_str,))
    cur.execute("""
        INSERT INTO RV_RATIOS
          (date,
           gs_ratio, gs_pct_5y, gs_pct_full, gs_signal, gs_rotation,
           gsp_ratio, gsp_pct_5y, gsp_pct_full, gsp_signal, gsp_rotation,
           cg_ratio, cg_pct_5y, cg_pct_full, cg_signal, cg_rotation,
           extremes_count, extremes_flagged, rotation_summary)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        today_str,
        data['gs_ratio'],   data['gs_pct_5y'],   data['gs_pct_full'],
        data['gs_signal'],  data['gs_rotation'],
        data['gsp_ratio'],  data['gsp_pct_5y'],  data['gsp_pct_full'],
        data['gsp_signal'], data['gsp_rotation'],
        data['cg_ratio'],   data['cg_pct_5y'],   data['cg_pct_full'],
        data['cg_signal'],  data['cg_rotation'],
        data['extremes_count'],
        data['extremes_flagged'],
        data['rotation_summary'],
    ))
    conn.commit()
    print(f"  ✅ Saved RV ratios to RV_RATIOS")


# ──────────────────────────────────────────────────────────────
# Print report
# ──────────────────────────────────────────────────────────────
def pct_bar(pct: float, width: int = 20) -> str:
    """ASCII percentile bar with marker at current position."""
    if np.isnan(pct):
        return '[' + '?' * width + ']'
    pos   = int(round(pct / 100 * width))
    pos   = max(0, min(width, pos))
    bar   = '░' * pos + '█' + '░' * (width - pos)
    return f"[{bar}]"


def signal_icon(signal: str) -> str:
    return {
        'EXTREME_HIGH': '🔴',
        'HIGH':         '🟠',
        'NEUTRAL':      '🟡',
        'LOW':          '🟠',
        'EXTREME_LOW':  '🔴',
        'UNKNOWN':      '⚪',
    }.get(signal, '⚪')


def print_report(today_str: str, data: dict,
                 gold_price: float, silver_price: float,
                 sp500_price: float, crude_price: float):

    print(f"\n{'='*70}")
    print(f"RELATIVE VALUE RATIO OVERLAY — DAILY REPORT")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*70}")

    # Spot prices
    print(f"\n  Spot prices (latest available):")
    print(f"    Gold:   ${gold_price:>8,.2f}/oz")
    print(f"    Silver: ${silver_price:>8,.2f}/oz")
    print(f"    SP500:  {sp500_price:>10,.2f}")
    print(f"    Crude:  ${crude_price:>8,.2f}/bbl")

    # ── Component 1 — Gold/Silver ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 1 — GOLD / SILVER RATIO")
    print(f"{'─'*70}")
    gs   = data['gs_ratio']
    gsp5 = data['gs_pct_5y']
    gsf  = data['gs_pct_full']
    print(f"\n  Current ratio:    {gs:.2f}   (hist avg ~{GS_HIST_AVG:.0f})")
    print(f"  5-yr percentile:  {gsp5:.1f}th  {pct_bar(gsp5)}")
    print(f"  Full-hist pctile: {gsf:.1f}th")
    print(f"  Signal:  {signal_icon(data['gs_signal'])} {data['gs_signal']}")
    print(f"\n  {data['gs_rotation']}")

    # ── Component 2 — Gold/SP500 ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 2 — GOLD / SP500 RATIO  (Buffett-style)")
    print(f"{'─'*70}")
    gsp   = data['gsp_ratio']
    gspp5 = data['gsp_pct_5y']
    gspf  = data['gsp_pct_full']
    sp_per_oz = f"{1/gsp:.1f}" if gsp > 0 else "N/A"
    print(f"\n  Current ratio:    {gsp:.4f}   (~{sp_per_oz} S&P pts per $1 of Gold)")
    print(f"  5-yr percentile:  {gspp5:.1f}th  {pct_bar(gspp5)}")
    print(f"  Full-hist pctile: {gspf:.1f}th")
    print(f"  Signal:  {signal_icon(data['gsp_signal'])} {data['gsp_signal']}")
    print(f"\n  {data['gsp_rotation']}")

    # ── Component 3 — Crude/Gold ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 3 — CRUDE / GOLD RATIO")
    print(f"{'─'*70}")
    cg   = data['cg_ratio']
    cgp5 = data['cg_pct_5y']
    cgf  = data['cg_pct_full']
    barrels = f"{1/cg:.1f}" if cg > 0 else "N/A"
    print(f"\n  Current ratio:    {cg:.4f}   ({barrels} barrels of oil per oz Gold)")
    print(f"  5-yr percentile:  {cgp5:.1f}th  {pct_bar(cgp5)}")
    print(f"  Full-hist pctile: {cgf:.1f}th")
    print(f"  Signal:  {signal_icon(data['cg_signal'])} {data['cg_signal']}")
    print(f"\n  {data['cg_rotation']}")

    # ── Component 4 — Rotation summary ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 4 — ROTATION SIGNALS SUMMARY")
    print(f"{'─'*70}")

    all_signals = [
        ('Gold/Silver', data['gs_signal'],  data['gs_pct_5y']),
        ('Gold/SP500',  data['gsp_signal'], data['gsp_pct_5y']),
        ('Crude/Gold',  data['cg_signal'],  data['cg_pct_5y']),
    ]
    print(f"\n  {'Ratio':<14} {'5yr Pct':>8}  {'Signal':<16}  Bar")
    print(f"  {'-'*60}")
    for name, sig, pct in all_signals:
        icon = signal_icon(sig)
        pct_s = f"{pct:.1f}th" if not np.isnan(pct) else "  N/A"
        print(f"  {name:<14} {pct_s:>8}  {icon} {sig:<14}  {pct_bar(pct, 16)}")

    if data['extremes_count'] > 0:
        print(f"\n  ⚠️  EXTREME READINGS ({data['extremes_count']}):")
        for line in data['extremes_flagged'].split('|'):
            if line.strip():
                print(f"    {line.strip()}")
    else:
        print(f"\n  ✅ No extreme percentile readings (all ratios within 10th–90th)")

    # Actionable rotation calls
    print(f"\n  Rotation calls:")
    for rot in [data['gs_rotation'], data['gsp_rotation'], data['cg_rotation']]:
        bullet = '  ⚡' if '→' in rot else '  •'
        print(f"{bullet} {rot}")

    print(f"\n{'='*70}\n")


# ──────────────────────────────────────────────────────────────
# Telegram message
# ──────────────────────────────────────────────────────────────
def build_telegram_message(today_str: str,
                            data: dict,
                            crossings: list) -> str:
    lines = [
        "⚖️ <b>GMIS MODULE 38 — EXTREME RV SIGNAL</b>",
        f"📅 {today_str}",
        "",
        f"⚠️ <b>{len(crossings)} ratio(s) at extreme percentile</b>",
        "",
    ]
    for name, prev_sig, curr_sig in crossings:
        direction = "EXTREME HIGH" if curr_sig == 'EXTREME_HIGH' else "EXTREME LOW"
        lines.append(f"  🔴 <b>{name}</b>: {prev_sig} → {curr_sig}")
    lines += [
        "",
        "<b>Current ratios:</b>",
        f"  Gold/Silver: {data['gs_ratio']:.2f}  ({data['gs_pct_5y']:.0f}th pct 5yr)",
        f"  Gold/SP500:  {data['gsp_ratio']:.4f}  ({data['gsp_pct_5y']:.0f}th pct 5yr)",
        f"  Crude/Gold:  {data['cg_ratio']:.4f}  ({data['cg_pct_5y']:.0f}th pct 5yr)",
        "",
        "<b>Rotation signals:</b>",
        f"  {data['gs_rotation'][:80]}...",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*65}")
    print(f"GMIS MODULE 38 — RV RATIO OVERLAY")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*65}")

    conn = get_conn()
    setup_table(conn)

    # ── Load prices ───────────────────────────────────────────
    print("\nLoading price series...")
    gold   = load_close(conn, 'GOLD')
    silver = load_close(conn, 'SILVER')
    sp500  = load_close(conn, 'SP500')
    crude  = load_close(conn, 'CRUDE_WTI')

    gold_last   = float(gold.iloc[-1])
    silver_last = float(silver.iloc[-1])
    sp500_last  = float(sp500.iloc[-1])
    crude_last  = float(crude.iloc[-1])

    print(f"  Gold: ${gold_last:,.2f}  Silver: ${silver_last:,.2f}  "
          f"SP500: {sp500_last:,.2f}  Crude: ${crude_last:,.2f}")

    # ── Component 1 — Gold/Silver ─────────────────────────────
    print("\nComponent 1 — Gold/Silver ratio...")
    gs_series = build_ratio_series(gold, silver)
    gs_ratio  = float(gs_series.iloc[-1])
    gs_pct_5y = rolling_percentile(gs_series, PERCENTILE_WINDOW, gs_ratio)
    gs_pct_f  = full_history_percentile(gs_series, gs_ratio)
    gs_signal, gs_rotation = gs_rotation_signal(gs_ratio, gs_pct_5y)
    print(f"  Ratio: {gs_ratio:.2f}  5yr-pct: {gs_pct_5y:.1f}th  "
          f"full-pct: {gs_pct_f:.1f}th  → {gs_signal}")

    # ── Component 2 — Gold/SP500 ──────────────────────────────
    print("\nComponent 2 — Gold/SP500 ratio...")
    gsp_series = build_ratio_series(gold, sp500)
    gsp_ratio  = float(gsp_series.iloc[-1])
    gsp_pct_5y = rolling_percentile(gsp_series, PERCENTILE_WINDOW, gsp_ratio)
    gsp_pct_f  = full_history_percentile(gsp_series, gsp_ratio)
    gsp_signal, gsp_rotation = gsp_rotation_signal(gsp_ratio, gsp_pct_5y)
    print(f"  Ratio: {gsp_ratio:.4f}  5yr-pct: {gsp_pct_5y:.1f}th  "
          f"full-pct: {gsp_pct_f:.1f}th  → {gsp_signal}")

    # ── Component 3 — Crude/Gold ──────────────────────────────
    print("\nComponent 3 — Crude/Gold ratio...")
    cg_series = build_ratio_series(crude, gold)
    cg_ratio  = float(cg_series.iloc[-1])
    cg_pct_5y = rolling_percentile(cg_series, PERCENTILE_WINDOW, cg_ratio)
    cg_pct_f  = full_history_percentile(cg_series, cg_ratio)
    cg_signal, cg_rotation = cg_rotation_signal(cg_ratio, cg_pct_5y)
    print(f"  Ratio: {cg_ratio:.4f}  5yr-pct: {cg_pct_5y:.1f}th  "
          f"full-pct: {cg_pct_f:.1f}th  → {cg_signal}")

    # ── Component 4 — Rotation summary ───────────────────────
    curr_signals = {
        'gs':  gs_signal,
        'gsp': gsp_signal,
        'cg':  cg_signal,
    }
    extremes = [
        s for s in [gs_signal, gsp_signal, cg_signal]
        if s in ('EXTREME_HIGH', 'EXTREME_LOW')
    ]
    extremes_flagged = ' | '.join(
        f"{name}: {sig}"
        for name, sig in [
            ('Gold/Silver', gs_signal),
            ('Gold/SP500',  gsp_signal),
            ('Crude/Gold',  cg_signal),
        ]
        if sig in ('EXTREME_HIGH', 'EXTREME_LOW')
    )
    rotation_summary = ' | '.join(filter(None, [
        f"GS:{gs_signal}", f"GSP:{gsp_signal}", f"CG:{cg_signal}"
    ]))

    data = {
        'gs_ratio':    gs_ratio,   'gs_pct_5y':  gs_pct_5y,
        'gs_pct_full': gs_pct_f,   'gs_signal':  gs_signal,
        'gs_rotation': gs_rotation,
        'gsp_ratio':   gsp_ratio,  'gsp_pct_5y': gsp_pct_5y,
        'gsp_pct_full':gsp_pct_f,  'gsp_signal': gsp_signal,
        'gsp_rotation':gsp_rotation,
        'cg_ratio':    cg_ratio,   'cg_pct_5y':  cg_pct_5y,
        'cg_pct_full': cg_pct_f,   'cg_signal':  cg_signal,
        'cg_rotation': cg_rotation,
        'extremes_count':   len(extremes),
        'extremes_flagged': extremes_flagged,
        'rotation_summary': rotation_summary,
    }

    # ── Extreme crossing detection ────────────────────────────
    prev_signals = get_prev_signals(conn)
    crossings    = detect_extreme_crossings(prev_signals, curr_signals)

    # ── Save ──────────────────────────────────────────────────
    print("\nSaving results...")
    save_results(conn, today_str, data)

    # ── Report ────────────────────────────────────────────────
    print_report(today_str, data,
                 gold_last, silver_last, sp500_last, crude_last)

    # ── Telegram — extreme crossings only ────────────────────
    if crossings and not NO_TELEGRAM:
        msg = build_telegram_message(today_str, data, crossings)
        asyncio.run(send_telegram(msg))
        print(f"  📱 Telegram sent ({len(crossings)} extreme crossing(s))")
    elif NO_TELEGRAM:
        print("  Telegram skipped (--no-telegram)")
    else:
        print(f"  No new extreme crossings — Telegram not sent")

    conn.close()


if __name__ == '__main__':
    main()
