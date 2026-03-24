# ============================================================
# GMIS 2.0 — MODULE 31 — YIELD-EQUITY BETA MONITOR
#
# COMPONENT 1 — Rolling Yield-Equity Beta
#   Beta = Cov(SP500_ret%, Yield_chg%) / Var(Yield_chg%)
#   Both series in percentage units.
#   Interpretation: beta = -2.0 means a 10bps yield rise
#   causes a -0.20% SP500 move.
#
# COMPONENT 2 — Yield Sensitivity Regime
#   HIGH   : beta < -2.0 (market highly rate-sensitive)
#   MEDIUM : -2.0 ≤ beta < -0.5
#   LOW    : beta ≥ -0.5 (yields not driving market)
#   POSITIVE: beta > +2.0 (growth/reflationary — equities
#             and yields rising together)
#
# COMPONENT 3 — Yield Direction (10-day & 30-day momentum)
#   Combine with sensitivity for SP500 overlay
#
# COMPONENT 4 — Signal Adjustment
#   Rising yields + HIGH_SENSITIVITY  → -0.15 on SP500
#   Falling yields + HIGH_SENSITIVITY → +0.10 on SP500
#   LOW_SENSITIVITY                   → 0.00
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

# ── Beta windows ──────────────────────────────────────────────
BETA_WINDOW_FAST   = 20    # fast — catches regime shifts
BETA_WINDOW_MAIN   = 60    # primary beta (spec requirement)
BETA_WINDOW_SLOW   = 120   # slow — structural relationship

# ── Yield momentum windows ────────────────────────────────────
YIELD_MOM_SHORT = 10   # days
YIELD_MOM_LONG  = 30   # days

# ── Sensitivity thresholds ────────────────────────────────────
BETA_HIGH_SENSITIVITY  = -2.0
BETA_MEDIUM_LOW        = -0.5
BETA_POSITIVE_HIGH     =  2.0    # growth/reflationary regime

# ── Signal adjustments ────────────────────────────────────────
ADJ_RISING_HIGH    = -0.15
ADJ_FALLING_HIGH   = +0.10
ADJ_RISING_MEDIUM  = -0.05
ADJ_FALLING_MEDIUM = +0.05
ADJ_LOW_SENS       =  0.00

# ── Yield direction thresholds (in %) ─────────────────────────
YIELD_RISING_THRESHOLD  = +0.10   # > 10bps = "rising"
YIELD_FALLING_THRESHOLD = -0.10   # < -10bps = "falling"


# ═════════════════════════════════════════════════════════════
# SECTION 1 — DATABASE
# ═════════════════════════════════════════════════════════════

def create_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS YIELD_EQUITY_BETA (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            date                 TEXT NOT NULL UNIQUE,

            -- SP500 beta
            beta_sp500_20d       REAL,
            beta_sp500_60d       REAL,
            beta_sp500_120d      REAL,
            beta_sp500_pct_rank  REAL,

            -- NIFTY beta (note: NIFTY ≠ IT sector, crude proxy)
            beta_nifty_60d       REAL,
            beta_nifty_pct_rank  REAL,

            -- Yield levels and momentum
            yield_10y            REAL,
            yield_10d_chg        REAL,
            yield_30d_chg        REAL,
            yield_direction      TEXT,

            -- Regime
            sensitivity_regime   TEXT,
            yield_direction_flag TEXT,

            -- Output
            sp500_adjustment     REAL,
            combined_signal      TEXT,

            -- Previous regime for change detection
            prev_regime          TEXT
        )
    ''')
    conn.commit()


# ═════════════════════════════════════════════════════════════
# SECTION 2 — LOAD DATA
# ═════════════════════════════════════════════════════════════

def load_yield_series():
    """Load US 10Y yield from saved CSV (from module 02)."""
    path = os.path.join(DATA_PATH, 'US_10Y_YIELD.csv')
    df   = pd.read_csv(path, index_col=0, parse_dates=True)
    s    = df.iloc[:, 0].dropna().sort_index()
    return s


def load_price_csv(filename):
    """Load OHLCV CSV (same pattern as modules 20/23/25)."""
    path = os.path.join(DATA_PATH, filename)
    df   = pd.read_csv(path)
    df   = df[~df['Price'].astype(str).str.match(
        r'^[A-Za-z]', na=False)]
    df   = df.rename(columns={'Price': 'Date'})
    df['Date']  = pd.to_datetime(df['Date'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df   = (df.dropna(subset=['Date', 'Close'])
              .sort_values('Date')
              .set_index('Date'))
    return df['Close']


# ═════════════════════════════════════════════════════════════
# SECTION 3 — COMPONENT 1: ROLLING BETA
# ═════════════════════════════════════════════════════════════

def rolling_beta(ret_pct: pd.Series,
                 yield_chg: pd.Series,
                 window: int) -> pd.Series:
    """
    Compute rolling beta of ret_pct vs yield_chg.

    Both series must be in percentage units:
      ret_pct   : daily return × 100  (1% = 1.0)
      yield_chg : daily change in yield (10bps = 0.10)

    Formula: β = Cov(ret, Δyield) / Var(Δyield)
    Interpretation: β = -2.0 → 10bps yield rise = -0.20% equity.
    """
    cov = ret_pct.rolling(window).cov(yield_chg)
    var = yield_chg.rolling(window).var()
    beta = cov / var
    return beta


def compute_all_betas(sp_close: pd.Series,
                      nifty_close: pd.Series,
                      yield_10y: pd.Series):
    """
    Compute betas for SP500 and NIFTY at multiple windows.
    Returns a dict of scalar current values and the full series.
    """
    yield_chg = yield_10y.diff()           # % change (10bps = 0.10)
    sp_ret    = sp_close.pct_change() * 100
    nifty_ret = nifty_close.pct_change() * 100

    # Align all on common trading days
    aligned = pd.concat(
        [sp_ret.rename('sp'),
         nifty_ret.rename('nifty'),
         yield_chg.rename('y_chg')],
        axis=1
    ).dropna()

    # SP500 betas at all three windows
    b20  = rolling_beta(aligned['sp'], aligned['y_chg'],
                        BETA_WINDOW_FAST)
    b60  = rolling_beta(aligned['sp'], aligned['y_chg'],
                        BETA_WINDOW_MAIN)
    b120 = rolling_beta(aligned['sp'], aligned['y_chg'],
                        BETA_WINDOW_SLOW)

    # NIFTY 60-day beta
    b_nifty = rolling_beta(aligned['nifty'], aligned['y_chg'],
                           BETA_WINDOW_MAIN)

    # Percentile ranks within full history
    b60_clean = b60.dropna()
    b_nifty_clean = b_nifty.dropna()

    current_b60    = float(b60.iloc[-1])
    current_bnifty = float(b_nifty.iloc[-1])

    pct_sp500 = float((b60_clean <= current_b60).mean() * 100)
    pct_nifty = float(
        (b_nifty_clean <= current_bnifty).mean() * 100)

    return {
        'beta_sp500_20d':      round(float(b20.iloc[-1]),   4),
        'beta_sp500_60d':      round(current_b60,            4),
        'beta_sp500_120d':     round(float(b120.iloc[-1]),  4),
        'beta_sp500_pct_rank': round(pct_sp500, 1),
        'beta_nifty_60d':      round(current_bnifty,         4),
        'beta_nifty_pct_rank': round(pct_nifty, 1),
        '_b60_series':         b60,          # for context
        '_b_nifty_series':     b_nifty,
    }


# ═════════════════════════════════════════════════════════════
# SECTION 4 — COMPONENT 2: SENSITIVITY REGIME
# ═════════════════════════════════════════════════════════════

def classify_sensitivity(beta_60d: float) -> tuple[str, str]:
    """
    Returns (regime_code, description).
    """
    if beta_60d < BETA_HIGH_SENSITIVITY:
        regime = 'HIGH_SENSITIVITY'
        desc   = (f'beta {beta_60d:.2f} — every 10bps yield rise '
                  f'drops SP500 by {abs(beta_60d)*0.10:.2f}%; '
                  f'SP500 Longs at HIGH RISK during yield spikes')
    elif beta_60d < BETA_MEDIUM_LOW:
        regime = 'MEDIUM_SENSITIVITY'
        desc   = (f'beta {beta_60d:.2f} — moderate yield sensitivity; '
                  f'normal caution warranted')
    elif beta_60d > BETA_POSITIVE_HIGH:
        regime = 'POSITIVE_CORRELATION'
        desc   = (f'beta {beta_60d:.2f} — yields and equities '
                  f'rising together (growth/reflationary narrative)')
    else:
        regime = 'LOW_SENSITIVITY'
        desc   = (f'beta {beta_60d:.2f} — SP500 not currently '
                  f'driven by yield moves; other factors dominate')

    return regime, desc


# ═════════════════════════════════════════════════════════════
# SECTION 5 — COMPONENT 3: YIELD DIRECTION
# ═════════════════════════════════════════════════════════════

def analyse_yield_direction(yield_10y: pd.Series) -> dict:
    """
    Calculate 10-day and 30-day yield momentum.
    Returns a directional flag and numeric changes.
    """
    if len(yield_10y) < YIELD_MOM_LONG + 1:
        return {
            'yield_10y':       None,
            'yield_10d_chg':   None,
            'yield_30d_chg':   None,
            'yield_direction': 'UNKNOWN',
        }

    current  = float(yield_10y.iloc[-1])
    chg_10d  = float(yield_10y.iloc[-1] - yield_10y.iloc[-11])
    chg_30d  = float(yield_10y.iloc[-1] - yield_10y.iloc[-31])

    # Use the shorter momentum for direction (more responsive)
    if chg_10d >= YIELD_RISING_THRESHOLD:
        direction = 'RISING'
    elif chg_10d <= YIELD_FALLING_THRESHOLD:
        direction = 'FALLING'
    else:
        # Use 30d as tiebreaker
        direction = ('RISING'  if chg_30d > 0.05 else
                     'FALLING' if chg_30d < -0.05 else
                     'FLAT')

    return {
        'yield_10y':       round(current, 3),
        'yield_10d_chg':   round(chg_10d, 4),
        'yield_30d_chg':   round(chg_30d, 4),
        'yield_direction': direction,
    }


# ═════════════════════════════════════════════════════════════
# SECTION 6 — COMPONENT 4: SIGNAL ADJUSTMENT
# ═════════════════════════════════════════════════════════════

def compute_sp500_adjustment(regime: str,
                              yield_dir: str) -> tuple[float, str]:
    """
    Derive the SP500 score overlay and combined signal text.
    """
    if regime == 'HIGH_SENSITIVITY':
        if yield_dir == 'RISING':
            adj = ADJ_RISING_HIGH
            sig = ('⚠️  DOUBLE WARNING: High yield sensitivity + '
                   'rising yields → strong headwind for SP500 Longs')
        elif yield_dir == 'FALLING':
            adj = ADJ_FALLING_HIGH
            sig = ('✅ High sensitivity + falling yields → '
                   'tailwind for SP500 Longs')
        else:
            adj = 0.0
            sig = ('HIGH_SENSITIVITY but yields flat — '
                   'watching for direction')

    elif regime == 'MEDIUM_SENSITIVITY':
        if yield_dir == 'RISING':
            adj = ADJ_RISING_MEDIUM
            sig = ('MEDIUM sensitivity + rising yields → '
                   'mild SP500 headwind')
        elif yield_dir == 'FALLING':
            adj = ADJ_FALLING_MEDIUM
            sig = ('MEDIUM sensitivity + falling yields → '
                   'mild SP500 tailwind')
        else:
            adj = 0.0
            sig = ('MEDIUM sensitivity, yields flat — '
                   'no adjustment')

    elif regime == 'POSITIVE_CORRELATION':
        if yield_dir == 'RISING':
            adj = +0.05
            sig = ('POSITIVE beta + rising yields → '
                   'growth narrative, small SP500 boost')
        elif yield_dir == 'FALLING':
            adj = -0.05
            sig = ('POSITIVE beta + falling yields → '
                   'growth scare, small SP500 headwind')
        else:
            adj = 0.0
            sig = ('POSITIVE correlation, yields flat — '
                   'no adjustment')

    else:  # LOW_SENSITIVITY
        adj = ADJ_LOW_SENS
        sig = ('LOW sensitivity — yields not driving SP500; '
               'no adjustment applied')

    return round(adj, 2), sig


def _is_alert_condition(regime: str, yield_dir: str) -> bool:
    """
    Alert when entering HIGH_SENSITIVITY with rising yields.
    """
    return (regime == 'HIGH_SENSITIVITY' and
            yield_dir == 'RISING')


# ═════════════════════════════════════════════════════════════
# SECTION 7 — HISTORICAL CONTEXT
# ═════════════════════════════════════════════════════════════

def get_beta_context(b60_series: pd.Series, current_beta: float):
    """
    Show when similar betas occurred in history and what
    followed for SP500.
    """
    b   = b60_series.dropna()
    pct = float((b <= current_beta).mean() * 100)

    # Find notable episodes of the current regime
    regime_now = classify_sensitivity(current_beta)[0]
    if regime_now == 'HIGH_SENSITIVITY':
        band = b[b < BETA_HIGH_SENSITIVITY]
    elif regime_now == 'MEDIUM_SENSITIVITY':
        band = b[(b >= BETA_HIGH_SENSITIVITY) & (b < BETA_MEDIUM_LOW)]
    elif regime_now == 'POSITIVE_CORRELATION':
        band = b[b > BETA_POSITIVE_HIGH]
    else:
        band = b[(b >= BETA_MEDIUM_LOW) & (b <= BETA_POSITIVE_HIGH)]

    # Most recent period outside last 30 days
    hist = band.iloc[:-30] if len(band) > 30 else band
    recent_date = hist.index[-1] if not hist.empty else None

    return {
        'pct_rank':      pct,
        'total_days':    len(band),
        'recent_date':   recent_date,
    }


# ═════════════════════════════════════════════════════════════
# SECTION 8 — SAVE & LOAD
# ═════════════════════════════════════════════════════════════

def load_prev_regime(conn, today):
    try:
        rows = conn.execute(
            "SELECT sensitivity_regime FROM YIELD_EQUITY_BETA "
            "WHERE date < ? ORDER BY date DESC LIMIT 1",
            (today,)
        ).fetchall()
        return rows[0][0] if rows else None
    except Exception:
        return None


def save_results(conn, today, betas, yield_dir_data,
                 regime, adjustment, combined_sig, prev_regime):
    row = {
        'date':                today,
        'beta_sp500_20d':      betas['beta_sp500_20d'],
        'beta_sp500_60d':      betas['beta_sp500_60d'],
        'beta_sp500_120d':     betas['beta_sp500_120d'],
        'beta_sp500_pct_rank': betas['beta_sp500_pct_rank'],
        'beta_nifty_60d':      betas['beta_nifty_60d'],
        'beta_nifty_pct_rank': betas['beta_nifty_pct_rank'],
        'yield_10y':           yield_dir_data['yield_10y'],
        'yield_10d_chg':       yield_dir_data['yield_10d_chg'],
        'yield_30d_chg':       yield_dir_data['yield_30d_chg'],
        'yield_direction':     yield_dir_data['yield_direction'],
        'sensitivity_regime':  regime,
        'yield_direction_flag':yield_dir_data['yield_direction'],
        'sp500_adjustment':    adjustment,
        'combined_signal':     combined_sig,
        'prev_regime':         prev_regime,
    }
    try:
        conn.execute(
            "DELETE FROM YIELD_EQUITY_BETA WHERE date=?",
            (today,)
        )
        pd.DataFrame([row]).to_sql(
            'YIELD_EQUITY_BETA', conn,
            if_exists='append', index=False
        )
        conn.commit()
        print(f"  ✅ Yield-equity beta saved "
              f"(regime={regime}, adj={adjustment:+.2f})")
    except Exception as e:
        print(f"  ❌ Save failed: {e}")


# ═════════════════════════════════════════════════════════════
# SECTION 9 — PRINT REPORT
# ═════════════════════════════════════════════════════════════

def _beta_bar(beta, width=24):
    """Visual gauge — centre at 0, scale ±5."""
    scale = 5.0
    mid   = width // 2
    pos   = int((beta / scale) * mid)
    pos   = max(-mid, min(mid, pos))
    if pos >= 0:
        bar = '─' * mid + '█' * pos + '░' * (mid - pos)
    else:
        bar = '░' * (mid + pos) + '█' * (-pos) + '─' * mid
    return f'[{bar}]'


def _regime_emoji(regime):
    return {
        'HIGH_SENSITIVITY':     '🔴',
        'MEDIUM_SENSITIVITY':   '🟡',
        'LOW_SENSITIVITY':      '⚪',
        'POSITIVE_CORRELATION': '🟢',
    }.get(regime, '❓')


def _dir_emoji(direction):
    return {'RISING': '↑', 'FALLING': '↓', 'FLAT': '→'}.get(
        direction, '?')


def print_report(betas, yield_dir_data, regime, regime_desc,
                 yield_dir, adjustment, combined_sig,
                 context, prev_regime, regime_changed):
    b60     = betas['beta_sp500_60d']
    b20     = betas['beta_sp500_20d']
    b120    = betas['beta_sp500_120d']
    pct     = betas['beta_sp500_pct_rank']
    bnifty  = betas['beta_nifty_60d']
    y_now   = yield_dir_data['yield_10y']
    y10d    = yield_dir_data['yield_10d_chg']
    y30d    = yield_dir_data['yield_30d_chg']

    print("\n" + "="*70)
    print("YIELD-EQUITY BETA MONITOR — DAILY REPORT")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*70)

    # ── Banner ────────────────────────────────────────────────
    e = _regime_emoji(regime)
    print(f"\n  {e}  SENSITIVITY REGIME: {regime}")
    print(f"      {regime_desc}")
    if regime_changed:
        print(f"\n  ⚡ REGIME CHANGE: {prev_regime} → {regime}")

    # ── Component 1: Beta table ───────────────────────────────
    print("\n📊 COMPONENT 1 — ROLLING YIELD-EQUITY BETA (SP500)")
    print("-"*60)
    print(f"  {'Window':<12} {'Beta':>8}  {'Bar'}")
    print(f"  {'20-day':<12} {b20:>+8.3f}  {_beta_bar(b20)}")
    print(f"  {'60-day ★':<12} {b60:>+8.3f}  {_beta_bar(b60)}")
    print(f"  {'120-day':<12} {b120:>+8.3f}  {_beta_bar(b120)}")
    print(f"\n  60-day percentile rank: {pct:.1f}th "
          f"(vs {context['total_days']:,} historical days "
          f"in current regime)")
    print(f"\n  Interpretation:")
    print(f"    10bps yield rise → "
          f"{b60 * 0.10:+.3f}% SP500 move")
    print(f"    25bps yield rise → "
          f"{b60 * 0.25:+.3f}% SP500 move")
    print(f"    50bps yield rise → "
          f"{b60 * 0.50:+.3f}% SP500 move")

    # NIFTY beta note
    print(f"\n  NIFTY 60-day beta vs 10Y: {bnifty:>+.3f}  "
          f"{_beta_bar(bnifty)}")
    print(f"  ⚠️  Note: NIFTY used as NIFTY IT proxy (crude approx); "
          f"no IT sector data available")

    # ── Component 2: Regime ───────────────────────────────────
    print("\n🎯 COMPONENT 2 — SENSITIVITY REGIME CLASSIFICATION")
    print("-"*60)
    regimes = [
        ('HIGH_SENSITIVITY',     f'beta < {BETA_HIGH_SENSITIVITY}',
         'SP500 Longs at risk during yield spikes'),
        ('MEDIUM_SENSITIVITY',
         f'{BETA_HIGH_SENSITIVITY} ≤ beta < {BETA_MEDIUM_LOW}',
         'Normal caution warranted'),
        ('LOW_SENSITIVITY',
         f'beta ≥ {BETA_MEDIUM_LOW}',
         'Yields not driving market'),
        ('POSITIVE_CORRELATION',
         f'beta > {BETA_POSITIVE_HIGH}',
         'Growth/reflationary — yields & equities together'),
    ]
    for r_code, r_range, r_desc in regimes:
        marker = '► ' if r_code == regime else '  '
        e2     = _regime_emoji(r_code)
        print(f"  {marker}{e2} {r_code:<22} ({r_range})")
        if r_code == regime:
            print(f"     → {r_desc}")

    # ── Component 3: Yield direction ──────────────────────────
    print("\n📈 COMPONENT 3 — YIELD DIRECTION & MOMENTUM")
    print("-"*60)
    d_emoji = _dir_emoji(yield_dir)
    print(f"  Current 10Y Yield: {y_now:.3f}%")
    print(f"  10-day change:     {y10d:+.3f}% "
          f"({y10d*100:+.0f}bps)  "
          f"{_dir_emoji('RISING' if y10d > 0 else 'FALLING')}")
    print(f"  30-day change:     {y30d:+.3f}% "
          f"({y30d*100:+.0f}bps)  "
          f"{_dir_emoji('RISING' if y30d > 0 else 'FALLING')}")
    print(f"\n  Direction Flag: {d_emoji} {yield_dir}")

    # ── Component 4: Adjustment ───────────────────────────────
    print("\n⚙️  COMPONENT 4 — SP500 SIGNAL ADJUSTMENT")
    print("-"*60)
    if adjustment > 0:
        adj_label = f"✅ POSITIVE  ({adjustment:+.2f})"
    elif adjustment < 0:
        adj_label = f"⚠️  NEGATIVE  ({adjustment:+.2f})"
    else:
        adj_label = f"⚪ NONE       (0.00)"
    print(f"  SP500 Adjustment: {adj_label}")
    print(f"  {combined_sig}")

    # ── Historical context ────────────────────────────────────
    print("\n🕰️  HISTORICAL CONTEXT")
    print("-"*60)
    if context['recent_date']:
        print(f"  Current regime ({regime}) last seen: "
              f"{context['recent_date'].strftime('%b %Y')}")
    print(f"  Historical occurrence: "
          f"{context['total_days']:,} trading days "
          f"({context['total_days']/40:.0f}% of history)")
    print(f"  Beta percentile: {pct:.1f}th "
          f"(lower = more negative beta = more rate-sensitive)")

    # ── Beta trend ────────────────────────────────────────────
    if b20 > b60 > b120:
        trend = "Beta trending MORE POSITIVE (decreasing rate sensitivity)"
    elif b20 < b60 < b120:
        trend = "Beta trending MORE NEGATIVE (increasing rate sensitivity)"
    else:
        trend = "Beta stable — no clear trend in sensitivity"
    print(f"\n  Trend: {trend}")
    print(f"  20d={b20:+.2f}  60d={b60:+.2f}  120d={b120:+.2f}")

    print("\n" + "="*70)


# ═════════════════════════════════════════════════════════════
# SECTION 10 — TELEGRAM
# ═════════════════════════════════════════════════════════════

async def _send_telegram(msg):
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID, text=msg, parse_mode='HTML')
        print("  ✅ Telegram sent")
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")


def build_telegram_message(regime, regime_desc, yield_dir,
                            b60, y_now, y10d,
                            adjustment, combined_sig):
    date = datetime.now().strftime('%d %b %Y %H:%M')
    e    = _regime_emoji(regime)
    d    = _dir_emoji(yield_dir)
    lines = [
        f"📈 <b>GMIS YIELD-EQUITY BETA MONITOR</b>",
        f"{date}",
        f"{'─' * 30}",
        "",
        f"🚨 <b>ALERT: HIGH SENSITIVITY + RISING YIELDS</b>",
        "",
        f"{e} Regime: <b>{regime}</b>",
        f"   60d Beta: {b60:+.3f}",
        f"   {regime_desc[:80]}",
        "",
        f"📈 10Y Yield: {y_now:.3f}%  ({d} {y10d*100:+.0f}bps in 10d)",
        "",
        f"⚙️  SP500 Adjustment: <b>{adjustment:+.2f}</b>",
        f"   {combined_sig}",
        "",
        "<i>GMIS Yield-Equity Beta Monitor</i>",
    ]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_yield_equity_beta(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 31 — YIELD-EQUITY BETA MONITOR")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    conn  = sqlite3.connect(DB_PATH)
    create_table(conn)
    today = datetime.now().strftime('%Y-%m-%d')

    prev_regime = load_prev_regime(conn, today)

    # ── Load data ─────────────────────────────────────────────
    print("\nLoading price and yield data...")
    try:
        yield_10y   = load_yield_series()
        sp_close    = load_price_csv('SP500.csv')
        nifty_close = load_price_csv('NIFTY50.csv')
        print(f"  Yield: {len(yield_10y)} obs, "
              f"latest {yield_10y.index[-1].date()}, "
              f"val={yield_10y.iloc[-1]:.3f}%")
        print(f"  SP500: {len(sp_close)} obs")
        print(f"  NIFTY: {len(nifty_close)} obs")
    except Exception as e:
        print(f"  ❌ Data load failed: {e}")
        conn.close()
        return None

    # ── Component 1: Betas ────────────────────────────────────
    print("\nComponent 1 — Computing rolling betas...")
    betas = compute_all_betas(sp_close, nifty_close, yield_10y)
    b60   = betas['beta_sp500_60d']
    print(f"  SP500 20d beta:  {betas['beta_sp500_20d']:+.4f}")
    print(f"  SP500 60d beta:  {b60:+.4f}  "
          f"(pct rank: {betas['beta_sp500_pct_rank']:.1f}th)")
    print(f"  SP500 120d beta: {betas['beta_sp500_120d']:+.4f}")
    print(f"  NIFTY 60d beta:  {betas['beta_nifty_60d']:+.4f}")

    # ── Component 2: Regime ───────────────────────────────────
    print("\nComponent 2 — Classifying sensitivity regime...")
    regime, regime_desc = classify_sensitivity(b60)
    print(f"  Regime: {regime}")
    print(f"  → {regime_desc}")

    regime_changed = (prev_regime is not None and
                      prev_regime != regime)
    if regime_changed:
        print(f"  ⚡ REGIME CHANGE: {prev_regime} → {regime}")

    # ── Component 3: Yield direction ──────────────────────────
    print("\nComponent 3 — Yield direction analysis...")
    yield_dir_data = analyse_yield_direction(yield_10y)
    yield_dir      = yield_dir_data['yield_direction']
    print(f"  10Y yield:  {yield_dir_data['yield_10y']:.3f}%")
    print(f"  10d change: {yield_dir_data['yield_10d_chg']:+.4f}%  "
          f"({yield_dir_data['yield_10d_chg']*100:+.1f}bps)")
    print(f"  30d change: {yield_dir_data['yield_30d_chg']:+.4f}%  "
          f"({yield_dir_data['yield_30d_chg']*100:+.1f}bps)")
    print(f"  Direction:  {yield_dir}")

    # ── Component 4: Adjustment ───────────────────────────────
    print("\nComponent 4 — Computing SP500 signal adjustment...")
    adjustment, combined_sig = compute_sp500_adjustment(
        regime, yield_dir)
    print(f"  Adjustment: {adjustment:+.2f}")
    print(f"  Signal: {combined_sig}")

    # ── Historical context ────────────────────────────────────
    context = get_beta_context(betas['_b60_series'], b60)

    # ── Save ─────────────────────────────────────────────────
    print("\nSaving results...")
    save_results(conn, today, betas, yield_dir_data,
                 regime, adjustment, combined_sig, prev_regime)
    conn.close()

    # ── Print full report ─────────────────────────────────────
    print_report(betas, yield_dir_data, regime, regime_desc,
                 yield_dir, adjustment, combined_sig,
                 context, prev_regime, regime_changed)

    # ── Telegram: only HIGH_SENSITIVITY + RISING yields ───────
    alert_condition = _is_alert_condition(regime, yield_dir)
    if send_telegram_flag and BOT_TOKEN and alert_condition:
        print("\n⚡ Alert condition met — sending Telegram...")
        msg = build_telegram_message(
            regime, regime_desc, yield_dir, b60,
            yield_dir_data['yield_10y'],
            yield_dir_data['yield_10d_chg'],
            adjustment, combined_sig
        )
        asyncio.run(_send_telegram(msg))
    elif send_telegram_flag and not alert_condition:
        print(f"\n  Alert condition not met "
              f"(regime={regime}, yields={yield_dir}) — "
              f"no Telegram")
    elif not send_telegram_flag:
        print("\n  Telegram skipped (--no-telegram)")

    return {
        'regime':        regime,
        'beta_60d':      b60,
        'yield_dir':     yield_dir,
        'adjustment':    adjustment,
        'combined_sig':  combined_sig,
    }


if __name__ == "__main__":
    no_telegram = '--no-telegram' in sys.argv
    run_yield_equity_beta(send_telegram_flag=not no_telegram)
