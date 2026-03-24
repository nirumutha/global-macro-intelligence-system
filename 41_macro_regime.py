# ============================================================
# GMIS MODULE 41 — MACRO REGIME UPGRADE
# Four-quadrant regime classification for US and India,
# regime transition detection, and signal adjustments.
# Replaces simplified macro_layer in Module 19/05.
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import asyncio
import os
import sys
from datetime import datetime, timedelta

# ── Config ───────────────────────────────────────────────────
BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_PATH, 'data', 'macro_system.db')
NO_TELEGRAM = '--no-telegram' in sys.argv

# ── US regime thresholds ─────────────────────────────────────
US_GDP_TREND       = 2.0    # % QoQ annualised — "above trend"
US_GDP_RECESSION   = 1.0    # below = recession territory
US_GDP_STRONG      = 3.0    # clearly above trend
US_CPI_TARGET      = 3.0    # % YoY — above = hot
US_CPI_LOW         = 2.5    # below = cooling
US_CPI_RECESSION   = 2.0    # true recession disinflation

# ── India regime thresholds ──────────────────────────────────
IN_GDP_TREND       = 6.0
IN_GDP_RECESSION   = 4.0
IN_GDP_STRONG      = 7.0
IN_CPI_TARGET      = 5.0
IN_CPI_LOW         = 4.0

# ── Signal adjustments by regime ────────────────────────────
US_REGIME_ADJ = {
    'GOLDILOCKS':  {'SP500':+0.10, 'Gold':-0.10, 'Silver':-0.05, 'Crude': 0.00},
    'OVERHEATING': {'SP500':-0.05, 'Gold':+0.10, 'Silver':+0.05, 'Crude':+0.10},
    'STAGFLATION': {'SP500':-0.15, 'Gold':+0.15, 'Silver':+0.05, 'Crude': 0.00},
    'RECESSION':   {'SP500':-0.15, 'Gold':+0.10, 'Silver': 0.00, 'Crude':-0.15},
    'RECOVERY':    {'SP500':+0.10, 'Gold': 0.00, 'Silver':+0.05, 'Crude':+0.05},
    'UNKNOWN':     {'SP500': 0.00, 'Gold': 0.00, 'Silver': 0.00, 'Crude': 0.00},
}
INDIA_REGIME_ADJ = {
    'INDIA_GOLDILOCKS':  {'NIFTY':+0.10},
    'INDIA_OVERHEATING': {'NIFTY': 0.00},
    'INDIA_STAGFLATION': {'NIFTY':-0.15},
    'INDIA_SLOWDOWN':    {'NIFTY':-0.05},
    'INDIA_RECOVERY':    {'NIFTY':+0.10},
    'UNKNOWN':           {'NIFTY': 0.00},
}
# Regime display names for India → generic labels
INDIA_TO_GENERIC = {
    'INDIA_GOLDILOCKS':  'GOLDILOCKS',
    'INDIA_OVERHEATING': 'OVERHEATING',
    'INDIA_STAGFLATION': 'STAGFLATION',
    'INDIA_SLOWDOWN':    'RECESSION',
    'INDIA_RECOVERY':    'RECOVERY',
    'UNKNOWN':           'UNKNOWN',
}

REGIME_COLORS = {
    'GOLDILOCKS':  '🟢',
    'OVERHEATING': '🟡',
    'STAGFLATION': '🔴',
    'RECESSION':   '🟠',
    'RECOVERY':    '🟢',
    'UNKNOWN':     '⚪',
}


# ── Telegram ─────────────────────────────────────────────────
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
        CREATE TABLE IF NOT EXISTS MACRO_REGIME (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            date                TEXT UNIQUE,
            -- US data
            us_cpi_yoy          REAL,
            us_gdp_qoq_ann      REAL,
            us_gdp_yoy          REAL,
            us_fed_rate         REAL,
            yield_spread_10y2y  REAL,
            -- India data
            india_cpi_yoy       REAL,
            india_gdp_growth    REAL,
            -- Regimes
            us_regime           TEXT,
            india_regime        TEXT,
            us_prev_regime      TEXT,
            india_prev_regime   TEXT,
            us_regime_changed   INTEGER,
            india_regime_changed INTEGER,
            divergence          INTEGER,
            divergence_note     TEXT,
            transition_flag     TEXT,
            -- Signal adjustments
            adj_sp500           REAL,
            adj_gold            REAL,
            adj_silver          REAL,
            adj_crude           REAL,
            adj_nifty           REAL,
            -- Metadata
            us_regime_desc      TEXT,
            india_regime_desc   TEXT
        )
    """)
    conn.commit()


# ──────────────────────────────────────────────────────────────
# COMPONENT 1 — Data loading
# ──────────────────────────────────────────────────────────────
def load_us_cpi(conn) -> pd.Series:
    """Returns a Series of US CPI YoY % indexed by date."""
    df = pd.read_sql("SELECT * FROM US_CPI ORDER BY Date", conn)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['CPIAUCSL']).sort_values('Date')
    yoy = df.set_index('Date')['CPIAUCSL'].pct_change(12, fill_method=None) * 100
    return yoy.dropna()


def load_us_gdp(conn) -> pd.DataFrame:
    """Returns DataFrame with GDP QoQ annualised and YoY."""
    df = pd.read_sql("SELECT * FROM US_GDP ORDER BY Date", conn)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['GDP']).sort_values('Date').set_index('Date')
    df['qoq_ann'] = df['GDP'].pct_change(fill_method=None) * 400
    df['yoy']     = df['GDP'].pct_change(4, fill_method=None) * 100
    return df.dropna(subset=['qoq_ann'])


def load_india_macro(conn) -> dict:
    try:
        df = pd.read_sql(
            "SELECT * FROM INDIA_MACRO ORDER BY date DESC LIMIT 1",
            conn)
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            'cpi_yoy':     float(row['cpi_yoy'])    if row['cpi_yoy']    else np.nan,
            'gdp_growth':  float(row['gdp_growth'])  if row['gdp_growth'] else np.nan,
            'regime_m37':  str(row['macro_regime']),
            'date':        str(row['date']),
        }
    except Exception as e:
        print(f"  ⚠️  INDIA_MACRO load: {e}")
        return {}


def load_fed_rate(conn) -> float:
    try:
        df = pd.read_sql(
            "SELECT FEDFUNDS FROM US_FED_RATE ORDER BY Date DESC LIMIT 1",
            conn)
        return float(df.iloc[0, 0]) if not df.empty else np.nan
    except Exception:
        return np.nan


def load_yield_spread(conn) -> float:
    try:
        y10 = pd.read_sql(
            "SELECT DGS10 FROM US_10Y_YIELD ORDER BY Date DESC LIMIT 1",
            conn)
        y2  = pd.read_sql(
            "SELECT DGS2 FROM US_2Y_YIELD ORDER BY Date DESC LIMIT 1",
            conn)
        return float(y10.iloc[0, 0]) - float(y2.iloc[0, 0])
    except Exception:
        return np.nan


# ──────────────────────────────────────────────────────────────
# COMPONENT 1 — Regime classification
# ──────────────────────────────────────────────────────────────
def classify_us_regime(cpi_yoy: float,
                        gdp_qoq: float,
                        gdp_yoy: float,
                        yield_spread: float,
                        gdp_hist: pd.Series) -> tuple[str, str]:
    """
    Returns (regime, description).
    Uses QoQ annualised as primary GDP indicator.
    Falls back to YoY if QoQ unavailable.
    """
    if np.isnan(cpi_yoy) or np.isnan(gdp_qoq):
        return 'UNKNOWN', 'Insufficient data'

    gdp = gdp_qoq if not np.isnan(gdp_qoq) else gdp_yoy

    # Check RECOVERY: GDP was below trend but now accelerating
    if len(gdp_hist) >= 2:
        prev_gdp = float(gdp_hist.iloc[-2])
        if (prev_gdp < US_GDP_TREND
                and gdp >= US_GDP_TREND
                and gdp > prev_gdp + 1.0):  # meaningful acceleration
            return ('RECOVERY',
                    f'GDP recovering ({prev_gdp:.1f}% → {gdp:.1f}% QoQ ann) '
                    f'with CPI {cpi_yoy:.1f}%')

    # GOLDILOCKS: strong growth + tame inflation
    if gdp > US_GDP_TREND and cpi_yoy < US_CPI_TARGET:
        return ('GOLDILOCKS',
                f'GDP {gdp:.1f}% QoQ ann > {US_GDP_TREND}% trend, '
                f'CPI {cpi_yoy:.1f}% < {US_CPI_TARGET}% target')

    # OVERHEATING: strong growth + hot inflation
    if gdp > US_GDP_TREND and cpi_yoy >= US_CPI_TARGET:
        return ('OVERHEATING',
                f'GDP {gdp:.1f}% QoQ ann above trend, '
                f'CPI {cpi_yoy:.1f}% above {US_CPI_TARGET}% target — '
                f'Fed likely hawkish')

    # STAGFLATION: weak growth + hot inflation
    if gdp < US_GDP_TREND and cpi_yoy >= US_CPI_TARGET:
        return ('STAGFLATION',
                f'GDP slowing ({gdp:.1f}% QoQ ann) with CPI still '
                f'{cpi_yoy:.1f}% — most dangerous macro regime')

    # RECESSION: weak growth + tame inflation
    if gdp < US_GDP_RECESSION and cpi_yoy < US_CPI_RECESSION:
        return ('RECESSION',
                f'GDP {gdp:.1f}% QoQ ann, CPI {cpi_yoy:.1f}% — '
                f'deflationary recession territory')

    if gdp < US_GDP_TREND and cpi_yoy < US_CPI_TARGET:
        return ('RECESSION',
                f'Below-trend growth ({gdp:.1f}%) with controlled '
                f'CPI ({cpi_yoy:.1f}%) — slowing')

    return ('UNKNOWN', f'Borderline: GDP {gdp:.1f}%, CPI {cpi_yoy:.1f}%')


def classify_india_regime(cpi_yoy: float,
                           gdp_growth: float,
                           regime_m37: str) -> tuple[str, str, str]:
    """
    Returns (regime, generic_label, description).
    Uses Module 37's regime as primary source, validates against thresholds.
    """
    # Trust Module 37 classification if available
    if regime_m37 and regime_m37 != 'UNKNOWN':
        generic = INDIA_TO_GENERIC.get(regime_m37, 'UNKNOWN')
        desc    = (f"Module 37: {regime_m37} "
                   f"[GDP {gdp_growth:.1f}%, CPI {cpi_yoy:.1f}%]"
                   if not np.isnan(gdp_growth) and not np.isnan(cpi_yoy)
                   else f"Module 37: {regime_m37}")
        return regime_m37, generic, desc

    # Fallback classification from thresholds
    if np.isnan(cpi_yoy) or np.isnan(gdp_growth):
        return 'UNKNOWN', 'UNKNOWN', 'Insufficient data'

    if gdp_growth >= IN_GDP_TREND and cpi_yoy < IN_CPI_TARGET:
        return ('INDIA_GOLDILOCKS', 'GOLDILOCKS',
                f'GDP {gdp_growth:.1f}% + CPI {cpi_yoy:.1f}% — favourable')
    if gdp_growth >= IN_GDP_TREND and cpi_yoy >= IN_CPI_TARGET:
        return ('INDIA_OVERHEATING', 'OVERHEATING',
                f'GDP {gdp_growth:.1f}% + CPI {cpi_yoy:.1f}% — RBI hawkish risk')
    if gdp_growth < IN_GDP_TREND and cpi_yoy >= IN_CPI_TARGET:
        return ('INDIA_STAGFLATION', 'STAGFLATION',
                f'GDP {gdp_growth:.1f}% + CPI {cpi_yoy:.1f}% — stagflation')
    return ('INDIA_SLOWDOWN', 'RECESSION',
            f'GDP {gdp_growth:.1f}% below trend, CPI controlled')


# ──────────────────────────────────────────────────────────────
# COMPONENT 2 — Divergence detection
# ──────────────────────────────────────────────────────────────
def detect_divergence(us_regime: str,
                       india_regime_generic: str) -> tuple[bool, str]:
    """Detect and describe US vs India macro divergence."""
    us_g    = us_regime
    in_g    = india_regime_generic

    if us_g == in_g:
        return False, f'US and India both in {us_g} — aligned'

    diverge_note = f'US={us_g} vs India={in_g} — '

    combos = {
        ('GOLDILOCKS',  'STAGFLATION'): 'India lagging US boom; watch for FII repatriation pressure on NIFTY',
        ('STAGFLATION', 'GOLDILOCKS'):  'US stress not felt in India yet; India decoupling favourable near-term',
        ('RECESSION',   'GOLDILOCKS'):  'India domestic demand resilient vs US recession; NIFTY may outperform',
        ('GOLDILOCKS',  'RECESSION'):   'India slowdown while US booms; FII likely to rotate out of India',
        ('OVERHEATING', 'RECESSION'):   'US overheating while India slows; Crude/Gold positive, NIFTY negative',
        ('RECESSION',   'STAGFLATION'): 'Both stressed but differently — Gold most favoured, equities negative',
        ('STAGFLATION', 'RECESSION'):   'Global stagflation risk; Gold strongly bullish, equities negative',
        ('OVERHEATING', 'STAGFLATION'): 'Divergent stress — NIFTY bearish, US commodities bullish',
        ('RECOVERY',    'STAGFLATION'): 'US recovering while India stagflates — NIFTY underperform',
        ('STAGFLATION', 'RECOVERY'):    'India recovering while US stagflates — NIFTY may outperform SP500',
    }
    note = combos.get((us_g, in_g),
                      f'Cross-regime divergence — use asset-specific adjustments')
    return True, diverge_note + note


# ──────────────────────────────────────────────────────────────
# COMPONENT 3 — Regime history and transition detection
# ──────────────────────────────────────────────────────────────
def build_historical_regimes(cpi_series: pd.Series,
                              gdp_df: pd.DataFrame,
                              n_months: int = 6) -> list[dict]:
    """
    Reconstruct last n_months of US regime from DB data.
    Uses quarterly GDP (repeated for each month in quarter) + monthly CPI.
    """
    history = []
    end   = cpi_series.index[-1]
    start = end - pd.DateOffset(months=n_months)
    cpi_window = cpi_series[cpi_series.index >= start]

    for date, cpi_val in cpi_window.items():
        # Find most recent GDP quarter for this date
        gdp_row = gdp_df[gdp_df.index <= date]
        if gdp_row.empty:
            continue
        gdp_val  = float(gdp_row['qoq_ann'].iloc[-1])
        gdp_hist = gdp_df['qoq_ann'][gdp_df.index <= date]
        regime, _ = classify_us_regime(
            float(cpi_val), gdp_val, gdp_val, np.nan, gdp_hist)
        history.append({
            'date':       date.strftime('%Y-%m'),
            'cpi_yoy':    float(cpi_val),
            'gdp_qoq':    gdp_val,
            'us_regime':  regime,
        })
    return history


def load_db_regime_history(conn, n: int = 6) -> list[str]:
    """Load recent US regimes from MACRO_REGIME table."""
    try:
        df = pd.read_sql(
            f"SELECT us_regime FROM MACRO_REGIME ORDER BY date DESC LIMIT {n}",
            conn)
        return df['us_regime'].tolist()[::-1]  # oldest first
    except Exception:
        return []


def detect_transition(regime_history: list[str],
                       current_regime: str,
                       gdp_trend: pd.Series) -> str | None:
    """
    Returns transition flag string or None.
    Checks for meaningful regime shifts in last 2-3 months.
    """
    if len(regime_history) < 2:
        return None

    recent = regime_history[-2:]   # last 2 periods

    # POTENTIAL_RECOVERY: was stagflation/recession, GDP now rising
    if all(r in ('STAGFLATION', 'RECESSION') for r in recent):
        if current_regime in ('GOLDILOCKS', 'RECOVERY', 'OVERHEATING'):
            return 'REGIME_IMPROVING: exiting stagnant phase → watch for sustained recovery'
        if len(gdp_trend) >= 2 and float(gdp_trend.iloc[-1]) > float(gdp_trend.iloc[-2]):
            return 'POTENTIAL_RECOVERY: GDP accelerating from distress — early transition signal'

    # DETERIORATION: was goldilocks, now heading south
    if all(r == 'GOLDILOCKS' for r in recent):
        if current_regime in ('OVERHEATING', 'STAGFLATION', 'RECESSION'):
            return f'REGIME_DETERIORATING: Goldilocks ending → entering {current_regime}'

    # STAGFLATION_RISK: overheating + decelerating GDP
    if all(r == 'OVERHEATING' for r in recent):
        if len(gdp_trend) >= 2 and float(gdp_trend.iloc[-1]) < float(gdp_trend.iloc[-2]):
            if current_regime in ('STAGFLATION', 'RECESSION'):
                return 'STAGFLATION_RISK: growth fading while inflation sticky'

    # INFLATION_BREAKTHROUGH: was high, now breaking lower
    if all(r == 'OVERHEATING' for r in recent) and current_regime == 'GOLDILOCKS':
        return 'DISINFLATION: inflation cooling into Goldilocks — equities bullish'

    return None


# ──────────────────────────────────────────────────────────────
# COMPONENT 4 — Signal adjustments
# ──────────────────────────────────────────────────────────────
def compute_adjustments(us_regime: str,
                         india_regime: str,
                         transition: str | None) -> dict:
    us_adj   = US_REGIME_ADJ.get(us_regime, US_REGIME_ADJ['UNKNOWN']).copy()
    in_adj   = INDIA_REGIME_ADJ.get(india_regime, INDIA_REGIME_ADJ['UNKNOWN']).copy()

    # Blend NIFTY: US macro has some influence (via FII, global risk)
    nifty_us_spill = us_adj.get('SP500', 0.0) * 0.30   # 30% US spill-over
    nifty_india    = in_adj.get('NIFTY', 0.0)
    nifty_total    = round(nifty_india + nifty_us_spill, 4)

    adj = {
        'SP500':  us_adj.get('SP500',  0.0),
        'Gold':   us_adj.get('Gold',   0.0),
        'Silver': us_adj.get('Silver', 0.0),
        'Crude':  us_adj.get('Crude',  0.0),
        'NIFTY':  nifty_total,
    }

    # Transition modifier: dampen adjustments during transitions
    if transition and 'POTENTIAL' in transition:
        adj = {k: round(v * 0.5, 4) for k, v in adj.items()}

    return adj


# ──────────────────────────────────────────────────────────────
# Prev regime from DB
# ──────────────────────────────────────────────────────────────
def get_prev_regimes(conn) -> tuple[str, str]:
    try:
        df = pd.read_sql(
            "SELECT us_regime, india_regime FROM MACRO_REGIME "
            "ORDER BY date DESC LIMIT 1",
            conn)
        if not df.empty:
            return str(df.iloc[0]['us_regime']), str(df.iloc[0]['india_regime'])
    except Exception:
        pass
    return 'UNKNOWN', 'UNKNOWN'


# ──────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────
def save_results(conn, today_str: str, d: dict):
    cur = conn.cursor()
    cur.execute("DELETE FROM MACRO_REGIME WHERE date = ?", (today_str,))
    cur.execute("""
        INSERT INTO MACRO_REGIME
          (date,
           us_cpi_yoy, us_gdp_qoq_ann, us_gdp_yoy,
           us_fed_rate, yield_spread_10y2y,
           india_cpi_yoy, india_gdp_growth,
           us_regime, india_regime,
           us_prev_regime, india_prev_regime,
           us_regime_changed, india_regime_changed,
           divergence, divergence_note,
           transition_flag,
           adj_sp500, adj_gold, adj_silver, adj_crude, adj_nifty,
           us_regime_desc, india_regime_desc)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        today_str,
        d['us_cpi_yoy'],    d['us_gdp_qoq'],   d['us_gdp_yoy'],
        d['us_fed_rate'],   d['yield_spread'],
        d['india_cpi'],     d['india_gdp'],
        d['us_regime'],     d['india_regime'],
        d['us_prev'],       d['india_prev'],
        int(d['us_changed']),   int(d['india_changed']),
        int(d['divergence']),   d['divergence_note'],
        d['transition'],
        d['adj']['SP500'],  d['adj']['Gold'],
        d['adj']['Silver'], d['adj']['Crude'],
        d['adj']['NIFTY'],
        d['us_desc'],       d['india_desc'],
    ))
    conn.commit()
    print(f"  ✅ Saved macro regime record")


# ──────────────────────────────────────────────────────────────
# Print report
# ──────────────────────────────────────────────────────────────
def regime_sparkline(history: list[dict]) -> str:
    """One-char per month regime indicator."""
    chars = {
        'GOLDILOCKS':  '★',
        'OVERHEATING': '▲',
        'STAGFLATION': '▼',
        'RECESSION':   '↓',
        'RECOVERY':    '↑',
        'UNKNOWN':     '?',
    }
    return '  '.join(f"{h['date']} {chars.get(h['us_regime'],'?')}"
                     for h in history[-6:])


def print_report(today_str: str, d: dict,
                 us_history: list[dict]):

    ui  = REGIME_COLORS.get(d['us_regime'], '⚪')
    ii  = REGIME_COLORS.get(INDIA_TO_GENERIC.get(d['india_regime'], 'UNKNOWN'), '⚪')

    print(f"\n{'='*70}")
    print(f"MACRO REGIME — DAILY REPORT")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*70}")

    # ── Component 1 — Data ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 1 — MACRO DATA INPUTS")
    print(f"{'─'*70}")
    print(f"\n  US:")
    cpi_s = f"{d['us_cpi_yoy']:.2f}%" if not np.isnan(d['us_cpi_yoy']) else "N/A"
    gdp_s = f"{d['us_gdp_qoq']:.2f}% QoQ ann" if not np.isnan(d['us_gdp_qoq']) else "N/A"
    yoy_s = f"{d['us_gdp_yoy']:.2f}% YoY" if not np.isnan(d['us_gdp_yoy']) else "N/A"
    fed_s = f"{d['us_fed_rate']:.2f}%" if not np.isnan(d['us_fed_rate']) else "N/A"
    ysp_s = f"{d['yield_spread']:+.3f}%" if not np.isnan(d['yield_spread']) else "N/A"
    print(f"    CPI YoY:        {cpi_s:>12}")
    print(f"    GDP (QoQ ann):  {gdp_s:>20}")
    print(f"    GDP (YoY):      {yoy_s:>16}")
    print(f"    Fed Rate:       {fed_s:>12}")
    print(f"    Yield curve:    10Y-2Y = {ysp_s}  "
          f"({'NORMAL' if not np.isnan(d['yield_spread']) and d['yield_spread'] > 0 else 'INVERTED'})")

    print(f"\n  India:")
    in_cpi_s = f"{d['india_cpi']:.2f}%" if not np.isnan(d['india_cpi']) else "N/A"
    in_gdp_s = f"{d['india_gdp']:.2f}% QoQ SA ann" if not np.isnan(d['india_gdp']) else "N/A"
    print(f"    CPI YoY:        {in_cpi_s:>12}")
    print(f"    GDP growth:     {in_gdp_s:>24}")

    # ── Component 2 — Regimes ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 2 — REGIME CLASSIFICATION")
    print(f"{'─'*70}")

    # US
    print(f"\n  {ui} US REGIME:  {d['us_regime']}")
    print(f"     {d['us_desc']}")
    if d['us_changed']:
        print(f"     ⚡ CHANGE: {d['us_prev']} → {d['us_regime']}")

    # India
    india_generic = INDIA_TO_GENERIC.get(d['india_regime'], d['india_regime'])
    print(f"\n  {ii} INDIA REGIME:  {d['india_regime']} ({india_generic})")
    print(f"     {d['india_desc']}")
    if d['india_changed']:
        print(f"     ⚡ CHANGE: {d['india_prev']} → {d['india_regime']}")

    # Divergence
    print(f"\n  {'⚠️  DIVERGENCE' if d['divergence'] else '✅ ALIGNED'}:")
    print(f"  {d['divergence_note']}")

    # Quadrant diagram
    print(f"\n  Quadrant reference:")
    quads = [
        ('GOLDILOCKS',  'High GDP + Low CPI',   '★', 'Equities 🟢 / Gold 🔴'),
        ('OVERHEATING', 'High GDP + High CPI',  '▲', 'Commodities 🟢 / Equities 🟡'),
        ('STAGFLATION', 'Low GDP + High CPI',   '▼', 'Gold 🟢 / Equities 🔴'),
        ('RECESSION',   'Low GDP + Low CPI',    '↓', 'Bonds/Gold 🟢 / Equities 🔴'),
        ('RECOVERY',    'GDP accelerating ↑',   '↑', 'Equities 🟢 / Cyclicals 🟢'),
    ]
    for name, cond, sym, assets in quads:
        marker = '  ◄ US' if name == d['us_regime'] else (
                 '  ◄ INDIA' if name == india_generic else '')
        if d['us_regime'] == name and india_generic == name:
            marker = '  ◄ US + INDIA'
        ri = REGIME_COLORS.get(name, '⚪')
        print(f"    {ri} {sym} {name:<14} {cond:<24} {assets}{marker}")

    # ── Component 3 — History ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 3 — US REGIME HISTORY (6 months)")
    print(f"{'─'*70}")
    if us_history:
        print(f"\n  {regime_sparkline(us_history)}")
        print(f"\n  {'Month':<10} {'CPI%':>7} {'GDP%':>8}  US Regime")
        print(f"  {'-'*45}")
        for h in us_history[-6:]:
            ic = REGIME_COLORS.get(h['us_regime'], '⚪')
            print(f"  {h['date']:<10} {h['cpi_yoy']:>6.2f}%  "
                  f"{h['gdp_qoq']:>6.2f}%  {ic} {h['us_regime']}")
    else:
        print(f"  No history available.")

    if d.get('transition'):
        print(f"\n  ⚡ Transition flag: {d['transition']}")

    # ── Component 4 — Adjustments ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 4 — SIGNAL ADJUSTMENTS")
    print(f"{'─'*70}")
    print(f"\n  US regime ({d['us_regime']}) drives SP500/Gold/Silver/Crude:")
    print(f"  India regime ({d['india_regime']}) drives NIFTY:")
    if d.get('transition') and 'POTENTIAL' in d['transition']:
        print(f"  ℹ️  Dampened 50% due to transition uncertainty")
    print(f"\n  {'Asset':<10} {'Adj':>8}  Reasoning")
    print(f"  {'-'*55}")
    asset_reasons = {
        'SP500':  f"US {d['us_regime']} → SP500 weight",
        'Gold':   f"US {d['us_regime']} → Gold weight",
        'Silver': f"US {d['us_regime']} → Silver weight",
        'Crude':  f"US {d['us_regime']} → Crude weight",
        'NIFTY':  (f"India {india_generic} + "
                   f"30% US spill ({d['us_regime']})"),
    }
    for asset in ['SP500', 'Gold', 'Silver', 'Crude', 'NIFTY']:
        adj  = d['adj'][asset]
        bar  = '▲' if adj > 0 else ('▼' if adj < 0 else '=')
        print(f"  {asset:<10} {bar}{adj:>+7.4f}  {asset_reasons[asset]}")

    print(f"\n{'='*70}\n")


# ──────────────────────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────────────────────
def build_telegram_message(today_str: str, d: dict) -> str:
    ui = REGIME_COLORS.get(d['us_regime'], '⚪')
    ii = REGIME_COLORS.get(INDIA_TO_GENERIC.get(d['india_regime'], ''), '⚪')
    changed_str = []
    if d['us_changed']:
        changed_str.append(f"US: {d['us_prev']} → {d['us_regime']}")
    if d['india_changed']:
        changed_str.append(f"India: {d['india_prev']} → {d['india_regime']}")
    lines = [
        f"🌍 <b>GMIS MODULE 41 — MACRO REGIME CHANGE</b>",
        f"📅 {today_str}",
        "",
    ]
    for cs in changed_str:
        lines.append(f"⚡ {cs}")
    lines += [
        "",
        f"{ui} US: <b>{d['us_regime']}</b>  {d['us_desc'][:60]}",
        f"{ii} India: <b>{d['india_regime']}</b>  {d['india_desc'][:60]}",
        "",
    ]
    if d['divergence']:
        lines.append(f"⚠️ Divergence: {d['divergence_note'][:80]}")
    if d.get('transition'):
        lines.append(f"🔄 Transition: {d['transition'][:80]}")
    lines += [
        "",
        "<b>Signal adjustments:</b>",
        f"  SP500: {d['adj']['SP500']:+.2f}  Gold: {d['adj']['Gold']:+.2f}  "
        f"Silver: {d['adj']['Silver']:+.2f}",
        f"  NIFTY: {d['adj']['NIFTY']:+.2f}  Crude: {d['adj']['Crude']:+.2f}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*65}")
    print(f"GMIS MODULE 41 — MACRO REGIME UPGRADE")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*65}")

    conn = get_conn()
    setup_table(conn)

    # ── Load data ─────────────────────────────────────────────
    print("\nLoading macro data...")
    cpi_series = load_us_cpi(conn)
    gdp_df     = load_us_gdp(conn)
    india      = load_india_macro(conn)
    fed_rate   = load_fed_rate(conn)
    yld_spread = load_yield_spread(conn)

    us_cpi_yoy  = float(cpi_series.iloc[-1])
    us_gdp_qoq  = float(gdp_df['qoq_ann'].iloc[-1])
    us_gdp_yoy  = float(gdp_df['yoy'].iloc[-1])
    india_cpi   = india.get('cpi_yoy', np.nan)
    india_gdp   = india.get('gdp_growth', np.nan)

    print(f"  US CPI YoY: {us_cpi_yoy:.2f}%  "
          f"GDP QoQ ann: {us_gdp_qoq:.2f}%  "
          f"Fed: {fed_rate:.2f}%  "
          f"10Y-2Y: {yld_spread:+.3f}%")
    print(f"  India CPI: {india_cpi:.2f}%  "
          f"India GDP: {india_gdp:.2f}%  "
          f"[{india.get('regime_m37','')}]")

    # ── Component 1 — Classify US ─────────────────────────────
    print("\nComponent 1 — US regime classification...")
    us_regime, us_desc = classify_us_regime(
        us_cpi_yoy, us_gdp_qoq, us_gdp_yoy,
        yld_spread, gdp_df['qoq_ann'])
    print(f"  US regime: {us_regime}")

    # ── Component 2 — India + divergence ─────────────────────
    print("Component 1 — India regime classification...")
    india_regime, india_generic, india_desc = classify_india_regime(
        india_cpi, india_gdp, india.get('regime_m37', ''))
    print(f"  India regime: {india_regime} ({india_generic})")

    divergence, diverge_note = detect_divergence(us_regime, india_generic)
    print(f"  Divergence: {divergence} — {diverge_note[:60]}")

    # ── Component 3 — History & transitions ───────────────────
    print("\nComponent 3 — Regime history & transitions...")
    us_history = build_historical_regimes(cpi_series, gdp_df, n_months=6)
    db_history = load_db_regime_history(conn, 6)
    # Merge DB history (more recent) with derived history
    hist_regimes = ([h['us_regime'] for h in us_history]
                    + db_history)[-6:]
    transition = detect_transition(
        hist_regimes, us_regime, gdp_df['qoq_ann'])
    print(f"  Transition flag: {transition or 'None'}")

    # Prev regimes
    us_prev, india_prev = get_prev_regimes(conn)
    us_changed    = (us_prev    != 'UNKNOWN' and us_prev    != us_regime)
    india_changed = (india_prev != 'UNKNOWN' and india_prev != india_regime)

    # ── Component 4 — Adjustments ────────────────────────────
    adj = compute_adjustments(us_regime, india_regime, transition)

    # ── Assemble ──────────────────────────────────────────────
    d = {
        'us_cpi_yoy':    us_cpi_yoy,
        'us_gdp_qoq':    us_gdp_qoq,
        'us_gdp_yoy':    us_gdp_yoy,
        'us_fed_rate':   fed_rate,
        'yield_spread':  yld_spread,
        'india_cpi':     india_cpi,
        'india_gdp':     india_gdp,
        'us_regime':     us_regime,
        'india_regime':  india_regime,
        'us_desc':       us_desc,
        'india_desc':    india_desc,
        'us_prev':       us_prev,
        'india_prev':    india_prev,
        'us_changed':    us_changed,
        'india_changed': india_changed,
        'divergence':    divergence,
        'divergence_note': diverge_note,
        'transition':    transition,
        'adj':           adj,
    }

    # ── Save ──────────────────────────────────────────────────
    print("\nSaving results...")
    save_results(conn, today_str, d)

    # ── Report ────────────────────────────────────────────────
    print_report(today_str, d, us_history)

    # ── Telegram ─────────────────────────────────────────────
    should_alert = us_changed or india_changed
    if should_alert and not NO_TELEGRAM:
        msg = build_telegram_message(today_str, d)
        asyncio.run(send_telegram(msg))
        print("  📱 Telegram regime-change alert sent")
    elif NO_TELEGRAM:
        print("  Telegram skipped (--no-telegram)")
    else:
        print(f"  No regime change — Telegram not triggered")

    conn.close()


if __name__ == '__main__':
    main()
