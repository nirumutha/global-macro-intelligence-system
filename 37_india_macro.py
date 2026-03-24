# ============================================================
# GMIS MODULE 37 — INDIA MACRO LAYER
# India-specific macro regime classification and NIFTY
# signal adjustment using FRED data + existing DB tables.
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import asyncio
import os
import sys
from datetime import datetime, timedelta

try:
    import pandas_datareader as pdr
    PDR_AVAILABLE = True
except ImportError:
    PDR_AVAILABLE = False

# ── Config ───────────────────────────────────────────────────
BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_PATH, 'data', 'macro_system.db')
NO_TELEGRAM = '--no-telegram' in sys.argv

# FRED series
FRED_CPI   = 'INDCPIALLMINMEI'   # India CPI index (monthly)
FRED_GDP   = 'INDGDPRQPSMEI'     # India GDP QoQ% SA (quarterly)
FRED_RATE  = 'INDIRLTLT01STM'    # India long-term interest rate (monthly)

FETCH_START = '2015-01-01'

# ── Regime thresholds ────────────────────────────────────────
GDP_STRONG   = 6.0   # % QoQ SA (annualised)
GDP_WEAK     = 5.0
CPI_HIGH     = 6.0   # % YoY
CPI_OK       = 5.0

# ── NIFTY signal adjustments ─────────────────────────────────
ADJUSTMENTS = {
    'INDIA_GOLDILOCKS_FII_BUY':   +0.15,
    'INDIA_GOLDILOCKS_FII_SELL':  +0.05,
    'INDIA_GOLDILOCKS_FII_NEUTRAL':+0.10,
    'INDIA_OVERHEATING':           0.00,
    'INDIA_SLOWDOWN':             -0.05,
    'INDIA_STAGFLATION':          -0.20,
}

# ── INR parameters ───────────────────────────────────────────
INR_VOL_WINDOW      = 20
INR_MOM_SHORT       = 20
INR_MOM_LONG        = 60
INR_TREND_THRESHOLD = 0.005   # 0.5% move counts as trend


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
        CREATE TABLE IF NOT EXISTS INDIA_MACRO (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            date                TEXT,
            cpi_yoy             REAL,
            gdp_growth          REAL,
            interest_rate       REAL,
            real_rate           REAL,
            inr_spot            REAL,
            inr_mom_20d         REAL,
            inr_mom_60d         REAL,
            inr_vol_20d         REAL,
            inr_trend           TEXT,
            fii_net             REAL,
            fii_signal          TEXT,
            macro_regime        TEXT,
            fii_macro_signal    TEXT,
            nifty_adjustment    REAL,
            prev_regime         TEXT,
            regime_changed      INTEGER,
            notes               TEXT
        )
    """)
    conn.commit()


# ──────────────────────────────────────────────────────────────
# COMPONENT 1 — India macro data fetch
# ──────────────────────────────────────────────────────────────
def fetch_fred_series(series_id: str, start: str) -> pd.Series:
    if not PDR_AVAILABLE:
        raise ImportError("pandas_datareader not available")
    df = pdr.get_data_fred(series_id, start=start, end=datetime.now())
    return df[series_id].dropna()


def load_india_macro_data() -> dict:
    """
    Returns dict with latest values for CPI YoY, GDP growth,
    interest rate, and derived real rate.
    """
    result = {
        'cpi_yoy':       np.nan,
        'cpi_date':      None,
        'gdp_growth':    np.nan,
        'gdp_date':      None,
        'interest_rate': np.nan,
        'rate_date':     None,
        'real_rate':     np.nan,
    }

    # CPI YoY — compute from index
    try:
        cpi_idx = fetch_fred_series(FRED_CPI, FETCH_START)
        cpi_yoy = (cpi_idx.pct_change(12) * 100).dropna()
        result['cpi_yoy']  = float(cpi_yoy.iloc[-1])
        result['cpi_date'] = cpi_yoy.index[-1].strftime('%Y-%m')
        print(f"  CPI YoY:  {result['cpi_yoy']:.2f}%  "
              f"[as of {result['cpi_date']}]")
    except Exception as e:
        print(f"  ⚠️  CPI fetch failed: {e}")

    # GDP growth (QoQ% SA, last available quarter)
    try:
        gdp = fetch_fred_series(FRED_GDP, FETCH_START)
        result['gdp_growth'] = float(gdp.iloc[-1])
        result['gdp_date']   = gdp.index[-1].strftime('%Y-Q') + \
                               str((gdp.index[-1].month - 1) // 3 + 1)
        print(f"  GDP:      {result['gdp_growth']:.2f}%  "
              f"[{result['gdp_date']}]")
    except Exception as e:
        print(f"  ⚠️  GDP fetch failed: {e}")

    # Interest rate (long-term, proxy for monetary stance)
    try:
        rate = fetch_fred_series(FRED_RATE, FETCH_START)
        result['interest_rate'] = float(rate.iloc[-1])
        result['rate_date']     = rate.index[-1].strftime('%Y-%m')
        print(f"  Rate:     {result['interest_rate']:.3f}%  "
              f"[{result['rate_date']}]")
    except Exception as e:
        print(f"  ⚠️  Rate fetch failed: {e}")

    # Real rate
    if not np.isnan(result['interest_rate']) \
            and not np.isnan(result['cpi_yoy']):
        result['real_rate'] = (result['interest_rate']
                                - result['cpi_yoy'])
        print(f"  Real rate:{result['real_rate']:+.3f}%  "
              f"(rate − CPI)")

    return result


def load_inr_data(conn) -> dict:
    """INR spot, momentum, and volatility from USD_INR table."""
    result = {
        'inr_spot':    np.nan,
        'inr_mom_20d': np.nan,
        'inr_mom_60d': np.nan,
        'inr_vol_20d': np.nan,
        'inr_trend':   'UNKNOWN',
    }
    try:
        df = pd.read_sql("SELECT * FROM USD_INR ORDER BY Date", conn)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date']  = pd.to_datetime(df['Date'])
        df = df.dropna(subset=['Close']).sort_values('Date')
        close = df.set_index('Date')['Close']

        result['inr_spot']    = float(close.iloc[-1])
        result['inr_vol_20d'] = float(
            close.pct_change().rolling(INR_VOL_WINDOW).std().iloc[-1]
            * np.sqrt(252) * 100)   # annualised %

        # Momentum: current vs N-day-ago (positive = INR weakening vs USD)
        if len(close) > INR_MOM_SHORT:
            result['inr_mom_20d'] = float(
                (close.iloc[-1] / close.iloc[-INR_MOM_SHORT] - 1) * 100)
        if len(close) > INR_MOM_LONG:
            result['inr_mom_60d'] = float(
                (close.iloc[-1] / close.iloc[-INR_MOM_LONG] - 1) * 100)

        # Trend label
        m20 = result['inr_mom_20d']
        if not np.isnan(m20):
            if m20 > INR_TREND_THRESHOLD * 100:
                result['inr_trend'] = 'WEAKENING'   # USD/INR rising
            elif m20 < -INR_TREND_THRESHOLD * 100:
                result['inr_trend'] = 'STRENGTHENING'
            else:
                result['inr_trend'] = 'STABLE'

        print(f"  INR spot: {result['inr_spot']:.2f}  "
              f"mom20d={result['inr_mom_20d']:+.2f}%  "
              f"mom60d={result['inr_mom_60d']:+.2f}%  "
              f"vol={result['inr_vol_20d']:.2f}%  "
              f"[{result['inr_trend']}]")
    except Exception as e:
        print(f"  ⚠️  INR load failed: {e}")
    return result


def load_fii_data(conn) -> dict:
    """Latest FII net flow from FII_DII_FLOWS table."""
    result = {
        'fii_net':    np.nan,
        'fii_signal': 'UNKNOWN',
        'fii_date':   None,
    }
    try:
        df = pd.read_sql(
            "SELECT * FROM FII_DII_FLOWS ORDER BY date DESC LIMIT 1",
            conn)
        if df.empty:
            print("  ⚠️  No FII data in DB")
            return result
        row = df.iloc[0]
        result['fii_net']    = float(row['fii_net'])
        result['fii_signal'] = str(row['signal'])
        result['fii_date']   = str(row['date'])
        print(f"  FII net: ₹{result['fii_net']:,.0f} Cr  "
              f"[{result['fii_date']}]  signal={result['fii_signal']}")
    except Exception as e:
        print(f"  ⚠️  FII load failed: {e}")
    return result


# ──────────────────────────────────────────────────────────────
# COMPONENT 2 — India macro regime classification
# ──────────────────────────────────────────────────────────────
def classify_regime(cpi_yoy: float, gdp_growth: float) -> tuple[str, str]:
    """
    Returns (regime, description).
    Falls back to UNKNOWN if either series is unavailable.
    """
    if np.isnan(cpi_yoy) or np.isnan(gdp_growth):
        return 'UNKNOWN', 'Insufficient data for regime classification'

    if gdp_growth >= GDP_STRONG and cpi_yoy < CPI_OK:
        return ('INDIA_GOLDILOCKS',
                f'Strong growth ({gdp_growth:.1f}%) + controlled '
                f'inflation ({cpi_yoy:.1f}%) — most favourable')

    if gdp_growth >= GDP_STRONG and cpi_yoy >= CPI_HIGH:
        return ('INDIA_OVERHEATING',
                f'Strong growth ({gdp_growth:.1f}%) but high CPI '
                f'({cpi_yoy:.1f}%) — RBI likely hawkish')

    if gdp_growth >= GDP_STRONG and CPI_OK <= cpi_yoy < CPI_HIGH:
        return ('INDIA_OVERHEATING',
                f'Strong growth ({gdp_growth:.1f}%) with rising CPI '
                f'({cpi_yoy:.1f}%) — watch RBI')

    if gdp_growth < GDP_WEAK and cpi_yoy >= CPI_HIGH:
        return ('INDIA_STAGFLATION',
                f'Slowing growth ({gdp_growth:.1f}%) + elevated CPI '
                f'({cpi_yoy:.1f}%) — most dangerous')

    if gdp_growth < GDP_WEAK and cpi_yoy < CPI_OK:
        return ('INDIA_SLOWDOWN',
                f'Weak growth ({gdp_growth:.1f}%) with controlled CPI '
                f'({cpi_yoy:.1f}%) — RBI likely to cut')

    # GDP 5-6%, CPI moderate — borderline
    if cpi_yoy >= CPI_HIGH:
        return ('INDIA_OVERHEATING',
                f'Moderate growth ({gdp_growth:.1f}%) but high CPI '
                f'({cpi_yoy:.1f}%)')
    return ('INDIA_SLOWDOWN',
            f'Moderate growth ({gdp_growth:.1f}%) with moderate CPI '
            f'({cpi_yoy:.1f}%)')


# ──────────────────────────────────────────────────────────────
# COMPONENT 3 — FII + macro integration
# ──────────────────────────────────────────────────────────────
def fii_direction(fii_net: float, fii_signal: str) -> str:
    """Classify FII flow as BUY / SELL / NEUTRAL."""
    if np.isnan(fii_net):
        return 'NEUTRAL'
    if fii_net > 0:
        return 'BUY'
    if fii_net < 0:
        return 'SELL'
    return 'NEUTRAL'


def build_combined_signal(regime: str,
                           fii_dir: str) -> tuple[str, str]:
    """Returns (signal_key, description)."""
    if regime == 'INDIA_GOLDILOCKS':
        if fii_dir == 'BUY':
            return ('INDIA_GOLDILOCKS_FII_BUY',
                    'STRONG BULLISH — Goldilocks + FII buying')
        if fii_dir == 'SELL':
            return ('INDIA_GOLDILOCKS_FII_SELL',
                    'MILDLY BULLISH — Goldilocks but FII selling')
        return ('INDIA_GOLDILOCKS_FII_NEUTRAL',
                'BULLISH — Goldilocks + neutral FII')

    if regime == 'INDIA_STAGFLATION':
        return ('INDIA_STAGFLATION',
                'STRONG BEARISH — Stagflation environment')

    if regime == 'INDIA_OVERHEATING':
        return ('INDIA_OVERHEATING',
                'NEUTRAL — Overheating: growth ok but RBI risk')

    if regime == 'INDIA_SLOWDOWN':
        return ('INDIA_SLOWDOWN',
                'MILDLY BEARISH — Slowdown, await recovery signals')

    return ('INDIA_OVERHEATING',   # fallback neutral
            f'NEUTRAL — Regime {regime} with FII {fii_dir}')


# ──────────────────────────────────────────────────────────────
# COMPONENT 4 — NIFTY adjustment
# ──────────────────────────────────────────────────────────────
def get_nifty_adjustment(signal_key: str) -> float:
    return ADJUSTMENTS.get(signal_key, 0.0)


def get_prev_regime(conn) -> str | None:
    try:
        df = pd.read_sql(
            "SELECT macro_regime FROM INDIA_MACRO "
            "ORDER BY date DESC LIMIT 1",
            conn)
        if not df.empty:
            return str(df.iloc[0, 0])
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────
def save_results(conn, today_str: str, row: dict):
    cur = conn.cursor()
    cur.execute("DELETE FROM INDIA_MACRO WHERE date = ?", (today_str,))
    cur.execute("""
        INSERT INTO INDIA_MACRO
          (date, cpi_yoy, gdp_growth, interest_rate, real_rate,
           inr_spot, inr_mom_20d, inr_mom_60d, inr_vol_20d,
           inr_trend, fii_net, fii_signal,
           macro_regime, fii_macro_signal, nifty_adjustment,
           prev_regime, regime_changed, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        today_str,
        row['cpi_yoy']       if not np.isnan(row.get('cpi_yoy', np.nan)) else None,
        row['gdp_growth']    if not np.isnan(row.get('gdp_growth', np.nan)) else None,
        row['interest_rate'] if not np.isnan(row.get('interest_rate', np.nan)) else None,
        row['real_rate']     if not np.isnan(row.get('real_rate', np.nan)) else None,
        row['inr_spot']      if not np.isnan(row.get('inr_spot', np.nan)) else None,
        row['inr_mom_20d']   if not np.isnan(row.get('inr_mom_20d', np.nan)) else None,
        row['inr_mom_60d']   if not np.isnan(row.get('inr_mom_60d', np.nan)) else None,
        row['inr_vol_20d']   if not np.isnan(row.get('inr_vol_20d', np.nan)) else None,
        row['inr_trend'],
        row['fii_net']       if not np.isnan(row.get('fii_net', np.nan)) else None,
        row['fii_signal'],
        row['macro_regime'],
        row['combined_signal'],
        row['nifty_adj'],
        row['prev_regime'],
        int(row['regime_changed']),
        row['notes'],
    ))
    conn.commit()
    print(f"  ✅ Saved India macro record to INDIA_MACRO")


# ──────────────────────────────────────────────────────────────
# Print report
# ──────────────────────────────────────────────────────────────
def print_report(today_str: str, row: dict):
    regime_icons = {
        'INDIA_GOLDILOCKS':   '🟢',
        'INDIA_OVERHEATING':  '🟡',
        'INDIA_SLOWDOWN':     '🟠',
        'INDIA_STAGFLATION':  '🔴',
        'UNKNOWN':            '⚪',
    }
    print(f"\n{'='*65}")
    print(f"INDIA MACRO LAYER — DAILY REPORT")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*65}")

    # ── Component 1 ──
    print(f"\n{'─'*65}")
    print(f"COMPONENT 1 — INDIA MACRO INDICATORS")
    print(f"{'─'*65}")

    def fmt(v, suffix='', na='N/A'):
        return f"{v:.2f}{suffix}" if not np.isnan(v) else na

    cpi   = row.get('cpi_yoy', np.nan)
    gdp   = row.get('gdp_growth', np.nan)
    rate  = row.get('interest_rate', np.nan)
    rrate = row.get('real_rate', np.nan)

    print(f"  India CPI (YoY):        {fmt(cpi, '%'):>10}  "
          f"[target: <4%]  {'🟢' if cpi<4 else ('🟡' if cpi<6 else '🔴') if not np.isnan(cpi) else ''}")
    print(f"  India GDP (QoQ% SA):    {fmt(gdp, '%'):>10}  "
          f"[strong: >6%]  {'🟢' if gdp>6 else ('🟡' if gdp>5 else '🔴') if not np.isnan(gdp) else ''}")
    print(f"  India LT Rate:          {fmt(rate, '%'):>10}  "
          f"[proxy for monetary stance]")
    print(f"  India Real Rate:        {fmt(rrate, '%'):>10}  "
          f"({'positive' if not np.isnan(rrate) and rrate>0 else 'negative'})")

    print(f"\n  INR/USD:  {fmt(row.get('inr_spot', np.nan)):>8}  "
          f"20d mom: {fmt(row.get('inr_mom_20d', np.nan), '%'):>7}  "
          f"60d mom: {fmt(row.get('inr_mom_60d', np.nan), '%'):>7}  "
          f"vol: {fmt(row.get('inr_vol_20d', np.nan), '%'):>6}  "
          f"[{row.get('inr_trend', 'UNKNOWN')}]")

    print(f"\n  FII net flow: "
          f"{'₹{:,.0f} Cr'.format(row['fii_net']) if not np.isnan(row.get('fii_net', np.nan)) else 'N/A':>15}  "
          f"[{row.get('fii_date', 'N/A')}]")
    print(f"  FII signal:  {row.get('fii_signal', 'N/A')}")

    # ── Component 2 ──
    print(f"\n{'─'*65}")
    print(f"COMPONENT 2 — INDIA MACRO REGIME")
    print(f"{'─'*65}")
    regime = row['macro_regime']
    icon   = regime_icons.get(regime, '⚪')
    print(f"\n  {icon}  {regime}")
    print(f"     {row['regime_notes']}")
    if row['regime_changed']:
        print(f"\n  ⚡ REGIME CHANGE: {row['prev_regime']} → {regime}")

    # ── Component 3 ──
    print(f"\n{'─'*65}")
    print(f"COMPONENT 3 — FII + MACRO COMBINED SIGNAL")
    print(f"{'─'*65}")
    print(f"\n  Signal key:  {row['combined_signal_key']}")
    print(f"  Description: {row['combined_signal']}")

    # ── Component 4 ──
    print(f"\n{'─'*65}")
    print(f"COMPONENT 4 — NIFTY ADJUSTMENT")
    print(f"{'─'*65}")
    adj  = row['nifty_adj']
    adj_icon = '▲' if adj > 0 else ('▼' if adj < 0 else '=')
    adj_desc = (f"{adj:+.2f} to NIFTY combined score")
    print(f"\n  {adj_icon} India adjustment: {adj_desc}")
    print(f"  Reasoning: {row['combined_signal']}")

    # Adjustment table
    print(f"\n  Adjustment reference table:")
    adj_ref = [
        ('INDIA_GOLDILOCKS + FII buying',   +0.15, regime=='INDIA_GOLDILOCKS' and row.get('fii_dir')=='BUY'),
        ('INDIA_GOLDILOCKS + FII selling',  +0.05, regime=='INDIA_GOLDILOCKS' and row.get('fii_dir')=='SELL'),
        ('INDIA_GOLDILOCKS + FII neutral',  +0.10, regime=='INDIA_GOLDILOCKS' and row.get('fii_dir')=='NEUTRAL'),
        ('INDIA_OVERHEATING',                0.00, regime=='INDIA_OVERHEATING'),
        ('INDIA_SLOWDOWN',                  -0.05, regime=='INDIA_SLOWDOWN'),
        ('INDIA_STAGFLATION',               -0.20, regime=='INDIA_STAGFLATION'),
    ]
    for label, val, active in adj_ref:
        marker = '◄ CURRENT' if active else ''
        print(f"    {label:<42} {val:+.2f}  {marker}")

    print(f"\n{'='*65}\n")


# ──────────────────────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────────────────────
def build_telegram_message(today_str: str, row: dict) -> str:
    icons = {
        'INDIA_GOLDILOCKS':  '🟢',
        'INDIA_OVERHEATING': '🟡',
        'INDIA_SLOWDOWN':    '🟠',
        'INDIA_STAGFLATION': '🔴',
    }
    regime = row['macro_regime']
    icon   = icons.get(regime, '⚪')
    lines = [
        f"🇮🇳 <b>GMIS MODULE 37 — INDIA MACRO REGIME CHANGE</b>",
        f"📅 {today_str}",
        "",
        f"⚡ Regime: {row['prev_regime']} → <b>{regime}</b>",
        f"{icon} {row['regime_notes']}",
        "",
        f"CPI: {row['cpi_yoy']:.2f}%  |  GDP: {row['gdp_growth']:.2f}%  |  "
        f"Real rate: {row['real_rate']:+.2f}%",
        f"INR: {row['inr_spot']:.2f}  [{row['inr_trend']}]",
        "",
        f"<b>NIFTY adjustment: {row['nifty_adj']:+.2f}</b>",
        f"{row['combined_signal']}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*65}")
    print(f"GMIS MODULE 37 — INDIA MACRO LAYER")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*65}")

    conn = get_conn()
    setup_table(conn)

    # ── Component 1 — Fetch data ──────────────────────────────
    print("\nComponent 1 — Fetching India macro indicators...")

    macro = load_india_macro_data()
    inr   = load_inr_data(conn)
    fii   = load_fii_data(conn)

    # ── Component 2 — Classify regime ────────────────────────
    print("\nComponent 2 — Classifying India macro regime...")
    regime, regime_notes = classify_regime(
        macro['cpi_yoy'], macro['gdp_growth'])
    prev_regime    = get_prev_regime(conn)
    regime_changed = (prev_regime is not None
                      and prev_regime != regime
                      and prev_regime != 'UNKNOWN')
    print(f"  Regime: {regime}  (prev: {prev_regime or 'none'})"
          + ("  ⚡ CHANGED" if regime_changed else ""))

    # ── Component 3 — Combined signal ────────────────────────
    print("\nComponent 3 — Building FII + macro combined signal...")
    fii_dir               = fii_direction(fii['fii_net'], fii['fii_signal'])
    combined_key, combined_desc = build_combined_signal(regime, fii_dir)
    print(f"  FII direction: {fii_dir}")
    print(f"  Combined key:  {combined_key}")
    print(f"  Description:   {combined_desc}")

    # ── Component 4 — NIFTY adjustment ───────────────────────
    nifty_adj = get_nifty_adjustment(combined_key)
    print(f"\nComponent 4 — NIFTY adjustment: {nifty_adj:+.2f}")

    # ── Assemble full row ─────────────────────────────────────
    row = {
        **macro,
        **inr,
        **fii,
        'macro_regime':       regime,
        'regime_notes':       regime_notes,
        'prev_regime':        prev_regime,
        'regime_changed':     regime_changed,
        'fii_dir':            fii_dir,
        'combined_signal_key':combined_key,
        'combined_signal':    combined_desc,
        'nifty_adj':          nifty_adj,
        'notes':              regime_notes,
    }

    # ── Save ──────────────────────────────────────────────────
    print("\nSaving results...")
    save_results(conn, today_str, row)

    # ── Report ────────────────────────────────────────────────
    print_report(today_str, row)

    # ── Telegram — regime changes only ───────────────────────
    if regime_changed and not NO_TELEGRAM:
        msg = build_telegram_message(today_str, row)
        asyncio.run(send_telegram(msg))
        print("  📱 Telegram regime-change alert sent")
    elif NO_TELEGRAM:
        print("  Telegram skipped (--no-telegram)")
    else:
        print("  No regime change — Telegram not triggered")

    conn.close()


if __name__ == '__main__':
    main()
