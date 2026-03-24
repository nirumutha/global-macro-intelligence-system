# ============================================================
# GMIS MODULE 39 — GIFT-GAP ENGINE
# Overnight gap detection, historical fill-rate analysis,
# gap classification (Breakout vs Trap) and morning alert.
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import asyncio
import os
import sys
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ── Config ───────────────────────────────────────────────────
BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_PATH, 'data', 'macro_system.db')
NO_TELEGRAM = '--no-telegram' in sys.argv

# Gap size buckets (abs %)
GAP_BUCKETS = {
    'SMALL':   (0.0,  0.3),
    'MEDIUM':  (0.3,  0.8),
    'LARGE':   (0.8,  1.5),
    'EXTREME': (1.5, 99.9),
}
FILL_RATE_TRAP_THRESHOLD = 0.65   # >65% fill rate → lean TRAP

# VIX thresholds
VIX_ELEVATED = 20.0
VIX_HIGH     = 25.0

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
        CREATE TABLE IF NOT EXISTS GIFT_GAPS (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            date                TEXT,
            prev_close          REAL,
            today_open          REAL,
            gap_points          REAL,
            gap_pct             REAL,
            gap_direction       TEXT,
            gap_size_class      TEXT,
            hist_fill_rate      REAL,
            hist_avg_cont_pct   REAL,
            mtf_trend           TEXT,
            sp500_direction     TEXT,
            vix_us              REAL,
            vix_india           REAL,
            vix_regime          TEXT,
            fii_net             REAL,
            fii_direction       TEXT,
            breakout_score      INTEGER,
            trap_score          INTEGER,
            classification      TEXT,
            recommendation      TEXT,
            wait_level          REAL
        )
    """)
    conn.commit()


# ──────────────────────────────────────────────────────────────
# Price loaders
# ──────────────────────────────────────────────────────────────
def load_nifty_db(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM NIFTY50 ORDER BY Date", conn)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date']  = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Open', 'Close']).query('Close > 0')
    return df.reset_index(drop=True)


def fetch_live_nifty() -> pd.DataFrame | None:
    """Fetch latest 5 days from yfinance to get today's open."""
    if not YF_AVAILABLE:
        return None
    try:
        raw = yf.download('^NSEI', period='5d', interval='1d', progress=False)
        if raw.empty:
            return None
        raw = raw.reset_index()
        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] if c[1] == '' else c[0]
                           for c in raw.columns]
        raw.columns = [str(c).strip() for c in raw.columns]
        # Rename Date/Datetime
        if 'Datetime' in raw.columns:
            raw = raw.rename(columns={'Datetime': 'Date'})
        raw['Date'] = pd.to_datetime(raw['Date']).dt.normalize()
        return raw[['Date', 'Open', 'High', 'Low', 'Close']].dropna()
    except Exception as e:
        print(f"  ⚠️  yfinance NSEI fetch: {e}")
        return None


def fetch_sp500_direction() -> str:
    """Return direction of SP500 last close vs prior close."""
    if not YF_AVAILABLE:
        return 'UNKNOWN'
    try:
        raw = yf.download('^GSPC', period='5d', interval='1d', progress=False)
        if len(raw) < 2:
            return 'UNKNOWN'
        closes = raw['Close'].dropna()
        delta  = float(closes.iloc[-1].item()) - float(closes.iloc[-2].item())
        return 'UP' if delta > 0 else 'DOWN'
    except Exception:
        return 'UNKNOWN'


# ──────────────────────────────────────────────────────────────
# COMPONENT 1 — Gap detection
# ──────────────────────────────────────────────────────────────
def detect_today_gap(nifty_db: pd.DataFrame,
                      live_df: pd.DataFrame | None,
                      today_str: str) -> dict | None:
    """
    Returns gap dict or None if market hasn't opened / no data.
    Priority: yfinance live data > DB data
    """
    today = pd.Timestamp(today_str)

    # Combine DB + live, deduplicate, sort
    if live_df is not None and not live_df.empty:
        live_df = live_df.rename(
            columns={c: c.title() for c in live_df.columns})
        live_df['Date'] = pd.to_datetime(live_df['Date'])
        combined = pd.concat(
            [nifty_db[['Date','Open','High','Low','Close']],
             live_df[['Date','Open','High','Low','Close']]],
            ignore_index=True)
        combined = (combined
                    .sort_values('Date')
                    .drop_duplicates('Date', keep='last')
                    .dropna(subset=['Open','Close'])
                    .query('Close > 0')
                    .reset_index(drop=True))
    else:
        combined = nifty_db[['Date','Open','High','Low','Close']].copy()

    # Find today's row
    today_rows = combined[combined['Date'] == today]
    if today_rows.empty:
        # Try yesterday as latest
        latest = combined.iloc[-1]
        prev   = combined.iloc[-2] if len(combined) >= 2 else None
        if prev is None:
            return None
        # Use most recent available day
        today_row = latest
        prev_row  = prev
    else:
        idx       = today_rows.index[0]
        today_row = combined.loc[idx]
        if idx == 0:
            return None
        prev_row  = combined.loc[idx - 1]

    prev_close  = float(prev_row['Close'])
    today_open  = float(today_row['Open'])
    today_date  = today_row['Date'].strftime('%Y-%m-%d')

    gap_pts = today_open - prev_close
    gap_pct = gap_pts / prev_close * 100

    return {
        'date':        today_date,
        'prev_close':  prev_close,
        'today_open':  today_open,
        'gap_points':  gap_pts,
        'gap_pct':     gap_pct,
        'direction':   'UP' if gap_pct >= 0 else 'DOWN',
        'today_high':  float(today_row['High']),
        'today_low':   float(today_row['Low']),
        'today_close': float(today_row['Close']),
    }


def classify_gap_size(gap_pct_abs: float) -> str:
    for name, (lo, hi) in GAP_BUCKETS.items():
        if lo <= gap_pct_abs < hi:
            return name
    return 'EXTREME'


# ──────────────────────────────────────────────────────────────
# COMPONENT 2 — Historical gap fill analysis
# ──────────────────────────────────────────────────────────────
def compute_gap_fill_stats(nifty_db: pd.DataFrame) -> dict:
    """
    For each gap bucket: fill rate, avg continuation when no fill.
    Gap fill = today's price range touched previous close.
    """
    df = nifty_db.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['gap_pts']    = df['Open'] - df['prev_close']
    df['gap_pct']    = df['gap_pts'] / df['prev_close'] * 100
    df['gap_abs']    = df['gap_pct'].abs()
    df['direction']  = np.where(df['gap_pct'] >= 0, 'UP', 'DOWN')

    # Gap fill: gap-up fills if Low <= prev_close; gap-down fills if High >= prev_close
    df['filled'] = np.where(
        df['direction'] == 'UP',
        df['Low']  <= df['prev_close'],
        df['High'] >= df['prev_close']
    )
    # Continuation (when NOT filled): return from open to close
    df['cont_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    # Align direction: UP gap continuation = positive return good
    df['directed_cont'] = np.where(
        df['direction'] == 'UP',
        df['cont_pct'],
        -df['cont_pct']
    )

    df = df.dropna(subset=['gap_pct', 'prev_close'])
    df = df[df['gap_abs'] > 0.05]   # skip near-zero days

    stats = {}
    for bucket, (lo, hi) in GAP_BUCKETS.items():
        sub = df[(df['gap_abs'] >= lo) & (df['gap_abs'] < hi)]
        if len(sub) < 5:
            stats[bucket] = {
                'n': 0, 'fill_rate': np.nan, 'avg_cont': np.nan}
            continue
        fill_rate = float(sub['filled'].mean())
        no_fill   = sub[~sub['filled']]
        avg_cont  = float(no_fill['directed_cont'].mean()) \
                    if len(no_fill) > 0 else 0.0
        stats[bucket] = {
            'n':         len(sub),
            'fill_rate': fill_rate,
            'avg_cont':  avg_cont,   # avg continuation when gap does NOT fill
        }
    return stats


# ──────────────────────────────────────────────────────────────
# Condition loaders (for Component 3)
# ──────────────────────────────────────────────────────────────
def get_mtf_nifty(conn) -> str:
    try:
        df = pd.read_sql(
            "SELECT overall FROM MTF_SIGNALS "
            "WHERE asset='NIFTY' ORDER BY date DESC LIMIT 1",
            conn)
        return str(df.iloc[0, 0]) if not df.empty else 'UNKNOWN'
    except Exception:
        return 'UNKNOWN'


def get_vix(conn) -> tuple[float, float]:
    """Return (vix_us, vix_india)."""
    vix_us = np.nan
    try:
        df = pd.read_sql(
            "SELECT * FROM VIX_US ORDER BY Date DESC LIMIT 1", conn)
        df.columns = ['Date','Open','High','Low','Close','Volume']
        vix_us = float(df['Close'].iloc[0])
    except Exception:
        pass
    vix_india = np.nan
    try:
        df = pd.read_sql(
            "SELECT * FROM VIX_INDIA ORDER BY Date DESC LIMIT 1", conn)
        df.columns = ['Date','Open','High','Low','Close','Volume']
        vix_india = float(df['Close'].iloc[0])
    except Exception:
        pass
    return vix_us, vix_india


def get_fii(conn) -> tuple[float, str]:
    try:
        df = pd.read_sql(
            "SELECT fii_net, signal FROM FII_DII_FLOWS "
            "ORDER BY date DESC LIMIT 1",
            conn)
        if df.empty:
            return np.nan, 'UNKNOWN'
        return float(df['fii_net'].iloc[0]), str(df['signal'].iloc[0])
    except Exception:
        return np.nan, 'UNKNOWN'


def vix_regime(vix_us: float, vix_india: float) -> str:
    ref = vix_india if not np.isnan(vix_india) else vix_us
    if np.isnan(ref):
        return 'UNKNOWN'
    if ref > VIX_HIGH:
        return 'HIGH'
    if ref > VIX_ELEVATED:
        return 'ELEVATED'
    return 'NORMAL'


# ──────────────────────────────────────────────────────────────
# COMPONENT 3 — Gap classification
# ──────────────────────────────────────────────────────────────
def score_gap(gap: dict, gap_class: str,
               fill_rate: float,
               mtf_trend: str,
               sp500_dir: str,
               vix_us: float, vix_india: float,
               fii_net: float, fii_signal: str) -> tuple[str, int, int, str]:
    """
    Returns (classification, breakout_score, trap_score, reasoning).
    Each factor scores +1 for its side.
    """
    direction    = gap['direction']
    v_regime     = vix_regime(vix_us, vix_india)
    fii_dir      = ('BUY' if fii_net > 0
                    else ('SELL' if fii_net < 0 else 'NEUTRAL'))
    breakout = 0
    trap     = 0
    reasons  = []

    # ── Factor 1: MTF trend alignment ──
    if mtf_trend in ('LONG', 'SHORT'):
        aligned = ((direction == 'UP'   and mtf_trend == 'LONG') or
                   (direction == 'DOWN' and mtf_trend == 'SHORT'))
        if aligned:
            breakout += 1
            reasons.append(f"MTF trend {mtf_trend} aligns with gap direction ✓")
        else:
            trap += 1
            reasons.append(f"MTF trend {mtf_trend} AGAINST gap direction ✗")
    else:
        reasons.append(f"MTF trend {mtf_trend} — no clear alignment")

    # ── Factor 2: SP500 direction ──
    if sp500_dir != 'UNKNOWN':
        aligned_sp = ((direction == 'UP'   and sp500_dir == 'UP') or
                      (direction == 'DOWN' and sp500_dir == 'DOWN'))
        if aligned_sp:
            breakout += 1
            reasons.append(f"SP500 {sp500_dir} overnight confirms gap ✓")
        else:
            trap += 1
            reasons.append(f"SP500 {sp500_dir} overnight contradicts gap ✗")

    # ── Factor 3: VIX ──
    if v_regime == 'NORMAL':
        breakout += 1
        reasons.append("VIX normal — risk appetite supportive ✓")
    elif v_regime == 'HIGH':
        trap += 1
        reasons.append(f"VIX HIGH ({vix_india:.1f}/{vix_us:.1f}) — elevated fear ✗")
    else:
        reasons.append(f"VIX elevated — neutral")

    # ── Factor 4: FII ──
    if fii_dir == 'BUY' and direction == 'UP':
        breakout += 1
        reasons.append("FII buying supports gap up ✓")
    elif fii_dir == 'SELL' and direction == 'DOWN':
        breakout += 1
        reasons.append("FII selling supports gap down ✓")
    elif fii_dir == 'SELL' and direction == 'UP':
        trap += 1
        reasons.append("FII selling opposes gap up ✗")
    elif fii_dir == 'BUY' and direction == 'DOWN':
        trap += 1
        reasons.append("FII buying opposes gap down ✗")
    else:
        reasons.append("FII neutral — no signal")

    # ── Factor 5: Historical fill rate ──
    if not np.isnan(fill_rate):
        if fill_rate > FILL_RATE_TRAP_THRESHOLD:
            trap += 1
            reasons.append(
                f"Historical fill rate {fill_rate:.0%} > "
                f"{FILL_RATE_TRAP_THRESHOLD:.0%} → lean trap ✗")
        else:
            breakout += 1
            reasons.append(
                f"Historical fill rate {fill_rate:.0%} ≤ "
                f"{FILL_RATE_TRAP_THRESHOLD:.0%} → gap tends to continue ✓")

    # ── Classification ──
    if breakout > trap + 1:
        classification = 'BREAKOUT'
    elif trap > breakout + 1:
        classification = 'TRAP'
    elif trap == breakout:
        # Tie-break: high fill rate + large gap size → TRAP
        classification = ('TRAP'
                          if (not np.isnan(fill_rate)
                              and fill_rate > 0.55
                              and gap_class in ('LARGE', 'EXTREME'))
                          else 'UNCERTAIN')
    else:
        classification = 'UNCERTAIN'

    return classification, breakout, trap, ' | '.join(reasons)


# ──────────────────────────────────────────────────────────────
# COMPONENT 4 — Recommendation
# ──────────────────────────────────────────────────────────────
def build_recommendation(gap: dict, gap_class: str,
                          classification: str,
                          fill_rate: float,
                          avg_cont: float,
                          mtf_trend: str) -> tuple[str, float]:
    """Returns (recommendation_text, wait_level)."""
    direction  = gap['direction']
    gap_pct    = gap['gap_pct']
    prev_close = gap['prev_close']
    today_open = gap['today_open']

    # Wait level: previous close ± 0.1%
    tolerance  = prev_close * 0.001
    wait_level = prev_close

    if classification == 'BREAKOUT':
        if direction == 'UP':
            entry   = today_open * 1.002   # enter 0.2% above open dip
            rec = (f"BREAKOUT signal — gap up supported by trend/flows. "
                   f"Consider buying dips toward {today_open:.0f}–{today_open*0.998:.0f}. "
                   f"Avoid chasing > {today_open*1.005:.0f}.")
        else:
            rec = (f"BREAKOUT signal — gap down supported by trend/flows. "
                   f"Consider shorting bounces toward {today_open:.0f}–{today_open*1.002:.0f}.")
    elif classification == 'TRAP':
        fr_s = f"{fill_rate:.0%}" if not np.isnan(fill_rate) else "N/A"
        if direction == 'UP':
            rec = (f"TRAP signal — do not chase the gap open. "
                   f"Historical fill rate {fr_s} for {gap_class} gaps. "
                   f"Wait for price to return to {wait_level:.0f} "
                   f"(yesterday close ±0.1%) before considering long entry.")
        else:
            rec = (f"TRAP signal — do not chase gap down short. "
                   f"Historical fill rate {fr_s} for {gap_class} gaps. "
                   f"Wait for bounce to fill toward {wait_level:.0f} "
                   f"before considering short entry.")
    else:
        fr_s = f"{fill_rate:.0%}" if not np.isnan(fill_rate) else "N/A"
        rec = (f"UNCERTAIN — mixed signals. "
               f"Historical fill rate {fr_s}. "
               f"Use wait level {wait_level:.0f} as decision anchor. "
               f"Confirm with price action before entry.")

    return rec, wait_level


# ──────────────────────────────────────────────────────────────
# Telegram message
# ──────────────────────────────────────────────────────────────
def build_telegram_message(gap: dict, gap_class: str,
                            classification: str,
                            fill_rate: float,
                            recommendation: str,
                            wait_level: float,
                            reasons: str) -> str:
    icons = {'BREAKOUT':'🟢', 'TRAP':'🔴', 'UNCERTAIN':'🟡'}
    icon  = icons.get(classification, '⚪')
    dir_arrow = '▲' if gap['direction'] == 'UP' else '▼'
    fr_s = f"{fill_rate:.0%}" if not np.isnan(fill_rate) else "N/A"

    lines = [
        f"📊 <b>GMIS MODULE 39 — NIFTY GAP ALERT</b>",
        f"📅 {gap['date']}  |  9:15am IST",
        "",
        f"{dir_arrow} NIFTY Gap: {gap['gap_pct']:+.2f}% "
        f"({gap['gap_points']:+.0f} pts)  [{gap_class}]",
        f"Open: {gap['today_open']:,.0f}  |  Prev Close: {gap['prev_close']:,.0f}",
        "",
        f"Historical fill rate ({gap_class}): <b>{fr_s}</b>",
        f"{icon} Classification: <b>{classification}</b>",
        "",
        f"📋 {recommendation}",
        f"⏳ Wait level: <b>{wait_level:,.0f}</b>",
        "",
        f"Factors: {reasons}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Print report
# ──────────────────────────────────────────────────────────────
def print_report(gap: dict, gap_class: str,
                 fill_stats: dict,
                 classification: str,
                 breakout_score: int,
                 trap_score: int,
                 reasons: str,
                 recommendation: str,
                 wait_level: float,
                 mtf_trend: str,
                 sp500_dir: str,
                 vix_us: float, vix_india: float,
                 fii_net: float, fii_signal: str):

    today_str = gap['date']
    dir_arrow = '▲' if gap['direction'] == 'UP' else '▼'
    icons = {'BREAKOUT':'🟢', 'TRAP':'🔴', 'UNCERTAIN':'🟡'}
    icon  = icons.get(classification, '⚪')

    print(f"\n{'='*70}")
    print(f"GIFT-GAP ENGINE — MORNING REPORT")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*70}")

    # ── Component 1 ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 1 — TODAY'S GAP ({today_str})")
    print(f"{'─'*70}")
    print(f"\n  {dir_arrow} Gap:       {gap['gap_pct']:+.3f}%  "
          f"({gap['gap_points']:+.1f} pts)  [{gap_class}]")
    print(f"  Prev close: {gap['prev_close']:>10,.2f}")
    print(f"  Today open: {gap['today_open']:>10,.2f}")
    if gap['today_close'] > 0:
        print(f"  Today close:{gap['today_close']:>10,.2f}  "
              f"(open→close: {(gap['today_close']-gap['today_open'])/gap['today_open']*100:+.2f}%)")
        filled = (gap['direction'] == 'UP' and
                  gap['today_low'] <= gap['prev_close']) or \
                 (gap['direction'] == 'DOWN' and
                  gap['today_high'] >= gap['prev_close'])
        print(f"  Gap filled: {'✅ YES' if filled else '❌ NOT YET'}")

    # ── Component 2 ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 2 — HISTORICAL GAP FILL RATES")
    print(f"{'─'*70}")
    print(f"\n  {'Bucket':<10} {'N':>5} {'Fill Rate':>10} "
          f"{'Avg Cont (no fill)':>20}")
    print(f"  {'-'*50}")
    for bucket in ['SMALL', 'MEDIUM', 'LARGE', 'EXTREME']:
        s  = fill_stats.get(bucket, {})
        n  = s.get('n', 0)
        fr = s.get('fill_rate', np.nan)
        ac = s.get('avg_cont', np.nan)
        marker = '  ◄ TODAY' if bucket == gap_class else ''
        fr_s = f"{fr:.1%}" if not np.isnan(fr) else "N/A"
        ac_s = f"{ac:+.2f}%" if not np.isnan(ac) else "N/A"
        print(f"  {bucket:<10} {n:>5}  {fr_s:>10}  {ac_s:>18}{marker}")

    # ── Component 3 ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 3 — GAP CLASSIFICATION")
    print(f"{'─'*70}")
    print(f"\n  MTF trend:    {mtf_trend}")
    print(f"  SP500 prev:   {sp500_dir}")
    vix_i_s = f"{vix_india:.1f}" if not np.isnan(vix_india) else "N/A"
    vix_u_s = f"{vix_us:.1f}"   if not np.isnan(vix_us)    else "N/A"
    print(f"  VIX India:    {vix_i_s}  VIX US: {vix_u_s}  "
          f"[{vix_regime(vix_us, vix_india)}]")
    fii_s = f"₹{fii_net:,.0f} Cr" if not np.isnan(fii_net) else "N/A"
    print(f"  FII net:      {fii_s}  — {fii_signal}")
    print(f"\n  Scoring:")
    for r in reasons.split(' | '):
        marker = '  ✓' if '✓' in r else ('  ✗' if '✗' in r else '   ')
        print(f"   {r}")
    print(f"\n  Breakout score: {breakout_score}  |  Trap score: {trap_score}")
    print(f"\n  {icon} Classification: {classification}")

    # ── Component 4 ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 4 — MORNING ALERT & RECOMMENDATION")
    print(f"{'─'*70}")
    print(f"\n  {recommendation}")
    print(f"\n  Wait level: {wait_level:,.2f}  (prev close ±0.1%)")
    print(f"  Entry zone: {wait_level*0.999:,.2f} – {wait_level*1.001:,.2f}")

    print(f"\n{'='*70}\n")


# ──────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────
def save_gap(conn, gap: dict, gap_class: str,
             fill_rate: float, avg_cont: float,
             mtf_trend: str, sp500_dir: str,
             vix_us: float, vix_india: float,
             fii_net: float, fii_signal: str,
             breakout_score: int, trap_score: int,
             classification: str, recommendation: str,
             wait_level: float):
    cur = conn.cursor()
    date_s = gap['date']
    cur.execute("DELETE FROM GIFT_GAPS WHERE date = ?", (date_s,))
    cur.execute("""
        INSERT INTO GIFT_GAPS
          (date, prev_close, today_open, gap_points, gap_pct,
           gap_direction, gap_size_class,
           hist_fill_rate, hist_avg_cont_pct,
           mtf_trend, sp500_direction,
           vix_us, vix_india, vix_regime,
           fii_net, fii_direction,
           breakout_score, trap_score,
           classification, recommendation, wait_level)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        date_s,
        gap['prev_close'], gap['today_open'],
        gap['gap_points'], gap['gap_pct'],
        gap['direction'],  gap_class,
        fill_rate if not np.isnan(fill_rate) else None,
        avg_cont  if not np.isnan(avg_cont)  else None,
        mtf_trend, sp500_dir,
        vix_us    if not np.isnan(vix_us)    else None,
        vix_india if not np.isnan(vix_india) else None,
        vix_regime(vix_us, vix_india),
        fii_net   if not np.isnan(fii_net)   else None,
        'BUY' if fii_net > 0 else ('SELL' if fii_net < 0 else 'NEUTRAL')
        if not np.isnan(fii_net) else 'UNKNOWN',
        breakout_score, trap_score,
        classification, recommendation, wait_level,
    ))
    conn.commit()
    print(f"  ✅ Saved gap record to GIFT_GAPS")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*65}")
    print(f"GMIS MODULE 39 — GIFT-GAP ENGINE")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*65}")

    conn = get_conn()
    setup_table(conn)

    # ── Load data ─────────────────────────────────────────────
    print("\nLoading NIFTY price data...")
    nifty_db = load_nifty_db(conn)
    print(f"  DB: {len(nifty_db)} rows  "
          f"({nifty_db.Date.iloc[0].date()} → {nifty_db.Date.iloc[-1].date()})")

    print("  Fetching live NIFTY from yfinance...")
    live_df  = fetch_live_nifty()
    if live_df is not None:
        print(f"  Live: {len(live_df)} rows  latest: {live_df.Date.iloc[-1].date()}")
    else:
        print("  Live: unavailable")

    # ── Component 1 — Detect gap ─────────────────────────────
    print(f"\nComponent 1 — Gap detection ({today_str})...")
    gap = detect_today_gap(nifty_db, live_df, today_str)

    if gap is None:
        print("  ⚠️  No gap data available (market may be closed or data missing)")
        conn.close()
        return

    gap_class = classify_gap_size(abs(gap['gap_pct']))
    print(f"  Gap: {gap['gap_pct']:+.3f}%  ({gap['gap_points']:+.1f} pts)  "
          f"[{gap_class}]  prev_close={gap['prev_close']:.2f}  open={gap['today_open']:.2f}")

    # ── Component 2 — Historical fill stats ──────────────────
    print("\nComponent 2 — Historical gap fill analysis (2010–present)...")
    fill_stats = compute_gap_fill_stats(nifty_db)
    for bucket, s in fill_stats.items():
        if s['n'] > 0:
            print(f"  {bucket:<10}: n={s['n']:4d}  "
                  f"fill={s['fill_rate']:.1%}  "
                  f"avg_cont={s['avg_cont']:+.2f}%")

    # Live fill rate for today's gap class
    fs        = fill_stats.get(gap_class, {})
    fill_rate = fs.get('fill_rate', np.nan)
    avg_cont  = fs.get('avg_cont',  np.nan)

    # ── Load market conditions ────────────────────────────────
    print("\nLoading market conditions...")
    mtf_trend = get_mtf_nifty(conn)
    vix_us, vix_india = get_vix(conn)
    fii_net, fii_signal = get_fii(conn)
    sp500_dir = fetch_sp500_direction()
    print(f"  MTF: {mtf_trend}  SP500: {sp500_dir}  "
          f"VIX_IN: {vix_india:.1f}  VIX_US: {vix_us:.1f}  "
          f"FII: ₹{fii_net:,.0f}")

    # ── Component 3 — Classify ───────────────────────────────
    print("\nComponent 3 — Gap classification...")
    classification, breakout_score, trap_score, reasons = score_gap(
        gap, gap_class, fill_rate,
        mtf_trend, sp500_dir,
        vix_us, vix_india,
        fii_net, fii_signal)
    print(f"  Breakout={breakout_score}  Trap={trap_score}  → {classification}")

    # ── Component 4 — Recommendation ─────────────────────────
    recommendation, wait_level = build_recommendation(
        gap, gap_class, classification,
        fill_rate, avg_cont, mtf_trend)

    # ── Save ──────────────────────────────────────────────────
    print("\nSaving gap record...")
    save_gap(conn, gap, gap_class,
             fill_rate, avg_cont,
             mtf_trend, sp500_dir,
             vix_us, vix_india,
             fii_net, fii_signal,
             breakout_score, trap_score,
             classification, recommendation, wait_level)

    # ── Report ────────────────────────────────────────────────
    print_report(gap, gap_class, fill_stats,
                 classification, breakout_score, trap_score, reasons,
                 recommendation, wait_level,
                 mtf_trend, sp500_dir,
                 vix_us, vix_india,
                 fii_net, fii_signal)

    # ── Telegram ─────────────────────────────────────────────
    if not NO_TELEGRAM:
        msg = build_telegram_message(
            gap, gap_class, classification,
            fill_rate, recommendation, wait_level, reasons)
        asyncio.run(send_telegram(msg))
        print("  📱 Telegram morning alert sent")
    else:
        print("  Telegram skipped (--no-telegram)")

    conn.close()


if __name__ == '__main__':
    main()
