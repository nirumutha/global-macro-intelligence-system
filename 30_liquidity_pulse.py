# ============================================================
# GMIS 2.0 — MODULE 30 — LIQUIDITY PULSE MONITOR
#
# COMPONENT 1 — TGA Balance (WTREGEN)
#   Treasury cash falling → money into markets = bullish
#   Treasury cash rising  → money out of markets = bearish
#   Signal threshold: ±$100B over 4 weeks
#
# COMPONENT 2 — Term Premium (THREEFYTP10 = NY Fed ACM model)
#   Proxy for ACMTP10 — same underlying ACM model
#   > 1.0% → duration risk on equities (DURATION_RISK)
#   < 0.0% → compressed premium (COMPRESSED_PREMIUM)
#
# COMPONENT 3 — Composite Liquidity Score [-1, +1]
#   Inputs: TGA change, Fed BS change, term premium, yield spread
#   Output: overlay for Decision Engine
#   > +0.3 → boost Long signals +0.05
#   < -0.3 → reduce Long signals -0.05
#
# COMPONENT 4 — Fiscal Dominance Indicator
#   MTSDS133FMS: Monthly Treasury deficit
#   Accelerating deficit = fiscal stimulus = supports equities
#   Output: STIMULATIVE / NEUTRAL / RESTRICTIVE
# ============================================================

import sqlite3
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import asyncio
import telegram
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID   = int(os.getenv('TELEGRAM_CHAT_ID', '0'))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')

START_DATE = '2020-01-01'        # enough history for percentiles

# ── Signal thresholds ─────────────────────────────────────────
TGA_SIGNAL_THRESHOLD  = 100_000  # $100B (values in $M)
TGA_WEEKS             = 4        # rolling window

TERM_PREMIUM_HIGH     =  1.00    # above → DURATION_RISK
TERM_PREMIUM_POSITIVE =  0.00    # below → COMPRESSED_PREMIUM

LIQUIDITY_BOOST_THRESHOLD  =  0.30
LIQUIDITY_DRAIN_THRESHOLD  = -0.30
SIGNAL_OVERLAY             =  0.05  # ±0.05 applied to decisions

FISCAL_ACCEL_THRESHOLD = 1.20   # 20% acceleration in deficit
FISCAL_DECEL_THRESHOLD = 0.80   # 20% deceleration

HY_SPREAD_STRESS  = 5.00        # above → credit stress
HY_SPREAD_BENIGN  = 3.50        # below → benign credit

YIELD_SPREAD_INVERSION = 0.00   # below → inverted


# ═════════════════════════════════════════════════════════════
# SECTION 1 — DATABASE
# ═════════════════════════════════════════════════════════════

def create_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS LIQUIDITY_PULSE (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            date                TEXT NOT NULL UNIQUE,

            -- TGA
            tga_current         REAL,
            tga_4wk_ago         REAL,
            tga_4wk_change      REAL,
            tga_signal          TEXT,

            -- Term Premium
            term_premium        REAL,
            term_premium_signal TEXT,

            -- Fed Balance Sheet
            fed_bs_current      REAL,
            fed_bs_4wk_change   REAL,
            fed_bs_signal       TEXT,

            -- Yield Spread
            yield_spread_10y3m  REAL,
            yield_spread_signal TEXT,

            -- HY Credit Spread
            hy_spread           REAL,
            hy_spread_signal    TEXT,

            -- Composite
            liquidity_score     REAL,
            liquidity_regime    TEXT,
            signal_overlay      REAL,

            -- Fiscal
            deficit_latest      REAL,
            deficit_3m_ago      REAL,
            deficit_acceleration REAL,
            fiscal_stance       TEXT,

            -- Previous regime (for change detection)
            prev_regime         TEXT
        )
    ''')
    conn.commit()


# ═════════════════════════════════════════════════════════════
# SECTION 2 — FETCH FRED DATA
# ═════════════════════════════════════════════════════════════

def fetch_series(series_id, start=START_DATE, retries=2):
    """Fetch a FRED series via pandas_datareader. Returns Series."""
    end = datetime.now().strftime('%Y-%m-%d')
    for attempt in range(retries + 1):
        try:
            df = pdr.DataReader(series_id, 'fred', start, end)
            s  = df.iloc[:, 0].dropna()
            return s
        except Exception as e:
            if attempt < retries:
                import time; time.sleep(5)
            else:
                print(f"    ⚠️  {series_id} fetch failed: "
                      f"{type(e).__name__}")
                return pd.Series(dtype=float)


def fetch_all():
    """Fetch all required FRED series. Returns dict of Series."""
    print("  Fetching FRED series...")
    data = {}
    series_map = {
        'tga':          'WTREGEN',
        'fed_bs':       'WALCL',
        'term_premium': 'THREEFYTP10',
        'yield_spread': 'T10Y3M',
        'hy_spread':    'BAMLH0A0HYM2',
        'deficit':      'MTSDS133FMS',
    }
    for key, sid in series_map.items():
        s = fetch_series(sid)
        if not s.empty:
            data[key] = s
            latest_date = s.index[-1].strftime('%Y-%m-%d')
            print(f"    {sid:<20} → {len(s):>4} obs, "
                  f"latest {latest_date}, "
                  f"val={s.iloc[-1]:,.2f}")
        else:
            print(f"    {sid:<20} → UNAVAILABLE")
    return data


# ═════════════════════════════════════════════════════════════
# SECTION 3 — COMPONENT 1: TGA ANALYSIS
# ═════════════════════════════════════════════════════════════

def analyse_tga(tga: pd.Series) -> dict:
    """
    Calculate 4-week TGA change and derive signal.
    TGA drops → Treasury spending down cash → liquidity injection.
    TGA rises → Treasury hoarding cash → liquidity drain.
    """
    if tga.empty or len(tga) < TGA_WEEKS + 1:
        return {'tga_signal': 'UNAVAILABLE',
                'tga_score': 0.0}

    current  = float(tga.iloc[-1])
    wk4_ago  = float(tga.iloc[-(TGA_WEEKS + 1)])
    change   = current - wk4_ago          # positive = rising = drain
    change_b = change / 1_000             # convert M → B for display

    # Score: negative change (falling TGA) = positive liquidity
    # Normalize: ±$200B → ±1.0
    score = -change / 200_000
    score = max(-1.0, min(1.0, score))

    if change < -TGA_SIGNAL_THRESHOLD:
        signal = (f'LIQUIDITY_INJECTION '
                  f'(TGA fell ${abs(change_b):,.0f}B in 4wk)')
    elif change > TGA_SIGNAL_THRESHOLD:
        signal = (f'LIQUIDITY_DRAIN '
                  f'(TGA rose ${change_b:,.0f}B in 4wk)')
    elif change < -50_000:
        signal = f'MILD_INJECTION (TGA fell ${abs(change_b):,.0f}B)'
    elif change > 50_000:
        signal = f'MILD_DRAIN (TGA rose ${change_b:,.0f}B)'
    else:
        signal = f'NEUTRAL (TGA change ${change_b:+,.0f}B)'

    return {
        'tga_current':    current,
        'tga_4wk_ago':    wk4_ago,
        'tga_4wk_change': change,
        'tga_signal':     signal,
        'tga_score':      score,
    }


# ═════════════════════════════════════════════════════════════
# SECTION 4 — COMPONENT 2: TERM PREMIUM
# ═════════════════════════════════════════════════════════════

def analyse_term_premium(tp: pd.Series) -> dict:
    """
    THREEFYTP10 = NY Fed ACM 10Y term premium.
    Positive and rising = bond investors demanding more
    compensation = pressure on equities.
    """
    if tp.empty:
        return {'term_premium_signal': 'UNAVAILABLE',
                'tp_score': 0.0}

    current = float(tp.iloc[-1])

    # Percentile rank over full history
    pct_rank = float((tp <= current).mean() * 100)

    # Score: high TP = negative for equities
    # Normalize: TP of 1.0% → score -1.0, TP of -0.5% → score +0.5
    score = -current / 1.0
    score = max(-1.0, min(1.0, score))

    if current >= TERM_PREMIUM_HIGH:
        signal = (f'DURATION_RISK '
                  f'(term premium {current:.2f}% — '
                  f'bond investors demanding high compensation; '
                  f'growth/tech equities under pressure)')
    elif current < TERM_PREMIUM_POSITIVE:
        signal = (f'COMPRESSED_PREMIUM '
                  f'(term premium {current:.2f}% — '
                  f'bonds expensive relative to cash; '
                  f'unusual environment)')
    elif current >= 0.70:
        signal = (f'ELEVATED '
                  f'(term premium {current:.2f}%, '
                  f'{pct_rank:.0f}th pct)')
    else:
        signal = (f'NORMAL '
                  f'(term premium {current:.2f}%, '
                  f'{pct_rank:.0f}th pct)')

    return {
        'term_premium':        current,
        'term_premium_pct':    pct_rank,
        'term_premium_signal': signal,
        'tp_score':            score,
    }


# ═════════════════════════════════════════════════════════════
# SECTION 5 — FED BALANCE SHEET
# ═════════════════════════════════════════════════════════════

def analyse_fed_bs(fed_bs: pd.Series) -> dict:
    """
    Fed BS rising  = QE-like injection = positive liquidity.
    Fed BS falling = QT drain = negative liquidity.
    4-week change; normalise ±$100B → ±1.0
    """
    if fed_bs.empty or len(fed_bs) < TGA_WEEKS + 1:
        return {'fed_bs_signal': 'UNAVAILABLE', 'bs_score': 0.0}

    current  = float(fed_bs.iloc[-1])
    wk4_ago  = float(fed_bs.iloc[-(TGA_WEEKS + 1)])
    change   = current - wk4_ago            # positive = expanding
    change_b = change / 1_000

    # Score: positive change (expanding BS) = positive
    score = change / 100_000
    score = max(-1.0, min(1.0, score))

    if change > 50_000:
        signal = f'QE_TAILWIND (Fed BS +${change_b:,.0f}B in 4wk)'
    elif change < -50_000:
        signal = f'QT_HEADWIND (Fed BS -${abs(change_b):,.0f}B in 4wk)'
    else:
        signal = f'STABLE (Fed BS {change_b:+,.0f}B in 4wk)'

    return {
        'fed_bs_current':   current,
        'fed_bs_4wk_change': change,
        'fed_bs_signal':    signal,
        'bs_score':         score,
    }


# ═════════════════════════════════════════════════════════════
# SECTION 6 — YIELD SPREAD & CREDIT
# ═════════════════════════════════════════════════════════════

def analyse_yield_spread(ys: pd.Series) -> dict:
    """
    T10Y3M: 10Y minus 3M Treasury yield.
    Positive = normal upward slope = growth expected.
    Negative = inversion = recession warning.
    """
    if ys.empty:
        return {'yield_spread_signal': 'UNAVAILABLE',
                'ys_score': 0.0}

    current = float(ys.iloc[-1])

    # Score: steeper curve = better for liquidity
    # Normalize: +2.0% → +1.0, -0.5% → -0.5
    score = current / 2.0
    score = max(-1.0, min(1.0, score))

    if current < YIELD_SPREAD_INVERSION:
        signal = (f'INVERTED '
                  f'(10Y-3M = {current:.2f}% — recession warning)')
    elif current < 0.50:
        signal = (f'FLAT_CURVE '
                  f'(10Y-3M = {current:.2f}% — mild caution)')
    else:
        signal = (f'NORMAL_SLOPE '
                  f'(10Y-3M = {current:.2f}%)')

    return {
        'yield_spread_10y3m':  current,
        'yield_spread_signal': signal,
        'ys_score':            score,
    }


def analyse_hy_spread(hy: pd.Series) -> dict:
    """
    BAMLH0A0HYM2: ICE HY OAS spread.
    High spread = credit stress = risk-off.
    Low spread  = credit benign = risk-on.
    """
    if hy.empty:
        return {'hy_spread_signal': 'UNAVAILABLE',
                'hy_score': 0.0}

    current  = float(hy.iloc[-1])
    pct_rank = float((hy <= current).mean() * 100)

    # Score: low spread = positive; normalize 2%→+1, 8%→-1
    score = -(current - 4.0) / 4.0
    score = max(-1.0, min(1.0, score))

    if current >= HY_SPREAD_STRESS:
        signal = (f'CREDIT_STRESS '
                  f'(HY spread {current:.2f}%, '
                  f'{pct_rank:.0f}th pct — risk-off)')
    elif current <= HY_SPREAD_BENIGN:
        signal = (f'CREDIT_BENIGN '
                  f'(HY spread {current:.2f}%, '
                  f'{pct_rank:.0f}th pct — risk-on)')
    else:
        signal = (f'CREDIT_NEUTRAL '
                  f'(HY spread {current:.2f}%, '
                  f'{pct_rank:.0f}th pct)')

    return {
        'hy_spread':        current,
        'hy_spread_signal': signal,
        'hy_score':         score,
    }


# ═════════════════════════════════════════════════════════════
# SECTION 7 — COMPONENT 3: COMPOSITE LIQUIDITY SCORE
# ═════════════════════════════════════════════════════════════

def compute_composite_score(tga_r, bs_r, tp_r, ys_r, hy_r):
    """
    Weighted composite of all liquidity sub-scores.
    Weights reflect importance in 2025-2026 macro regime:
      TGA:         30%  (fiscal dominance era)
      Fed BS:      25%  (QT still ongoing)
      Term premium:20%  (equity valuation pressure)
      Yield spread:15%  (recession leading indicator)
      HY spread:   10%  (coincident credit indicator)
    """
    weights = {
        'tga': 0.30,
        'bs':  0.25,
        'tp':  0.20,
        'ys':  0.15,
        'hy':  0.10,
    }
    scores = {
        'tga': tga_r.get('tga_score', 0.0),
        'bs':  bs_r.get('bs_score', 0.0),
        'tp':  tp_r.get('tp_score', 0.0),
        'ys':  ys_r.get('ys_score', 0.0),
        'hy':  hy_r.get('hy_score', 0.0),
    }

    # Weight only available components
    total_w = sum(w for k, w in weights.items()
                  if scores.get(k) is not None)
    if total_w == 0:
        return 0.0

    weighted = sum(scores[k] * weights[k]
                   for k in weights
                   if scores.get(k) is not None)
    composite = weighted / total_w
    return round(max(-1.0, min(1.0, composite)), 4)


def classify_liquidity_regime(score):
    """Map composite score to a named regime."""
    if score >= 0.50:
        return 'HIGHLY_ACCOMMODATIVE'
    elif score >= LIQUIDITY_BOOST_THRESHOLD:
        return 'ACCOMMODATIVE'
    elif score >= -0.10:
        return 'NEUTRAL'
    elif score >= LIQUIDITY_DRAIN_THRESHOLD:
        return 'TIGHTENING'
    else:
        return 'HIGHLY_RESTRICTIVE'


def compute_signal_overlay(score):
    """
    Return the decision-engine overlay value.
    Applied to all Long signals in the decision engine.
    """
    if score >= LIQUIDITY_BOOST_THRESHOLD:
        return +SIGNAL_OVERLAY
    elif score <= LIQUIDITY_DRAIN_THRESHOLD:
        return -SIGNAL_OVERLAY
    return 0.0


# ═════════════════════════════════════════════════════════════
# SECTION 8 — COMPONENT 4: FISCAL STANCE
# ═════════════════════════════════════════════════════════════

def analyse_fiscal_stance(deficit: pd.Series) -> dict:
    """
    MTSDS133FMS: monthly deficit (negative = deficit).
    Compare 3-month trailing average to prior 3-month period.
    Acceleration in deficit = more fiscal stimulus = STIMULATIVE.
    """
    if deficit.empty or len(deficit) < 7:
        return {
            'fiscal_stance': 'UNAVAILABLE',
            'deficit_latest': None,
            'deficit_3m_ago': None,
            'deficit_acceleration': None,
        }

    # Use 3-month rolling sums (deficit values are negative)
    recent_3m = float(deficit.iloc[-3:].mean())    # most negative = most spend
    prior_3m  = float(deficit.iloc[-6:-3].mean())

    # Acceleration ratio: more negative recent = more spending
    # ratio > 1 means spending accelerated
    if prior_3m != 0:
        accel = abs(recent_3m) / abs(prior_3m)
    else:
        accel = 1.0

    latest = float(deficit.iloc[-1])

    if accel >= FISCAL_ACCEL_THRESHOLD:
        stance = (f'STIMULATIVE '
                  f'(deficit spending accelerating '
                  f'{accel:.1f}x — fiscal tailwind for equities)')
    elif accel <= FISCAL_DECEL_THRESHOLD:
        stance = (f'RESTRICTIVE '
                  f'(deficit spending decelerating '
                  f'{accel:.1f}x — fiscal drag)')
    else:
        stance = (f'NEUTRAL '
                  f'(deficit pace stable, accel={accel:.2f}x)')

    return {
        'deficit_latest':        latest,
        'deficit_3m_ago':        prior_3m,
        'deficit_acceleration':  round(accel, 3),
        'fiscal_stance':         stance,
    }


# ═════════════════════════════════════════════════════════════
# SECTION 9 — SAVE TO DATABASE
# ═════════════════════════════════════════════════════════════

def load_prev_regime(conn, today):
    """Load yesterday's regime from DB for change detection."""
    try:
        rows = conn.execute(
            "SELECT liquidity_regime FROM LIQUIDITY_PULSE "
            "WHERE date < ? ORDER BY date DESC LIMIT 1",
            (today,)
        ).fetchall()
        return rows[0][0] if rows else None
    except Exception:
        return None


def save_results(conn, today, tga_r, tp_r, bs_r, ys_r,
                 hy_r, score, regime, overlay, fiscal_r,
                 prev_regime):
    row = {
        'date':                today,
        'tga_current':         tga_r.get('tga_current'),
        'tga_4wk_ago':         tga_r.get('tga_4wk_ago'),
        'tga_4wk_change':      tga_r.get('tga_4wk_change'),
        'tga_signal':          tga_r.get('tga_signal'),
        'term_premium':        tp_r.get('term_premium'),
        'term_premium_signal': tp_r.get('term_premium_signal'),
        'fed_bs_current':      bs_r.get('fed_bs_current'),
        'fed_bs_4wk_change':   bs_r.get('fed_bs_4wk_change'),
        'fed_bs_signal':       bs_r.get('fed_bs_signal'),
        'yield_spread_10y3m':  ys_r.get('yield_spread_10y3m'),
        'yield_spread_signal': ys_r.get('yield_spread_signal'),
        'hy_spread':           hy_r.get('hy_spread'),
        'hy_spread_signal':    hy_r.get('hy_spread_signal'),
        'liquidity_score':     score,
        'liquidity_regime':    regime,
        'signal_overlay':      overlay,
        'deficit_latest':      fiscal_r.get('deficit_latest'),
        'deficit_3m_ago':      fiscal_r.get('deficit_3m_ago'),
        'deficit_acceleration':fiscal_r.get('deficit_acceleration'),
        'fiscal_stance':       fiscal_r.get('fiscal_stance'),
        'prev_regime':         prev_regime,
    }
    try:
        conn.execute("DELETE FROM LIQUIDITY_PULSE WHERE date=?",
                     (today,))
        pd.DataFrame([row]).to_sql(
            'LIQUIDITY_PULSE', conn,
            if_exists='append', index=False
        )
        conn.commit()
        print(f"  ✅ Liquidity pulse saved (regime={regime})")
    except Exception as e:
        print(f"  ❌ Save failed: {e}")


# ═════════════════════════════════════════════════════════════
# SECTION 10 — PRINT REPORT
# ═════════════════════════════════════════════════════════════

def _score_bar(score, width=24):
    mid = width // 2
    pos = int(score * mid)
    pos = max(-mid, min(mid, pos))
    if pos >= 0:
        bar = '─' * mid + '█' * pos + '░' * (mid - pos)
    else:
        bar = '░' * (mid + pos) + '█' * (-pos) + '─' * mid
    return f'[{bar}]'


def _regime_emoji(regime):
    return {
        'HIGHLY_ACCOMMODATIVE': '💧',
        'ACCOMMODATIVE':        '🟢',
        'NEUTRAL':              '⚪',
        'TIGHTENING':           '🟡',
        'HIGHLY_RESTRICTIVE':   '🔴',
    }.get(regime, '❓')


def print_report(tga_r, tp_r, bs_r, ys_r, hy_r,
                 score, regime, overlay, fiscal_r,
                 prev_regime, regime_changed):
    print("\n" + "="*70)
    print("LIQUIDITY PULSE MONITOR — DAILY REPORT")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*70)

    # ── Banner ────────────────────────────────────────────────
    e = _regime_emoji(regime)
    print(f"\n  {e}  LIQUIDITY REGIME: {regime}")
    print(f"  Composite Score: {score:+.3f}  "
          f"{_score_bar(score)}")
    if overlay != 0:
        direction = 'BOOST' if overlay > 0 else 'REDUCE'
        print(f"  Signal Overlay:  {overlay:+.2f} "
              f"({direction} all Long signals by "
              f"{abs(overlay):.0%})")
    else:
        print(f"  Signal Overlay:  0.00 (no adjustment)")

    if regime_changed:
        print(f"\n  ⚡ REGIME CHANGE: {prev_regime} → {regime}")

    # ── Component 1: TGA ──────────────────────────────────────
    print("\n🏦 COMPONENT 1 — US TREASURY (TGA) CASH BALANCE")
    print("-"*55)
    if tga_r.get('tga_current') is not None:
        cur = tga_r['tga_current'] / 1_000
        chg = tga_r['tga_4wk_change'] / 1_000
        print(f"  Current balance: ${cur:>8,.1f}B")
        print(f"  4-week change:   ${chg:>+8,.1f}B")
        print(f"  Signal: {tga_r['tga_signal']}")
        print(f"  Score:  {tga_r['tga_score']:+.3f}  "
              f"{_score_bar(tga_r['tga_score'])}")
    else:
        print("  ⚠️  Data unavailable")

    # ── Component 2: Term Premium ─────────────────────────────
    print("\n📊 COMPONENT 2 — TERM PREMIUM (NY Fed ACM 10Y)")
    print("-"*55)
    if tp_r.get('term_premium') is not None:
        tp  = tp_r['term_premium']
        pct = tp_r.get('term_premium_pct', 0)
        print(f"  Current:  {tp:+.3f}%  ({pct:.0f}th percentile)")
        print(f"  Signal: {tp_r['term_premium_signal']}")
        print(f"  Score:  {tp_r['tp_score']:+.3f}  "
              f"{_score_bar(tp_r['tp_score'])}")
    else:
        print("  ⚠️  Data unavailable")

    # ── Fed BS & Yield Spread ─────────────────────────────────
    print("\n🏛️  COMPONENT 3 INPUTS — FED BALANCE SHEET & SPREADS")
    print("-"*55)

    if bs_r.get('fed_bs_current') is not None:
        bs   = bs_r['fed_bs_current'] / 1_000_000  # → $T
        bchg = bs_r['fed_bs_4wk_change'] / 1_000
        print(f"  Fed BS:         ${bs:.3f}T  "
              f"(4wk chg: ${bchg:+,.0f}B)")
        print(f"  BS Signal:      {bs_r['fed_bs_signal']}")

    if ys_r.get('yield_spread_10y3m') is not None:
        ys = ys_r['yield_spread_10y3m']
        print(f"  10Y-3M Spread:  {ys:+.2f}%  "
              f"→ {ys_r['yield_spread_signal']}")

    if hy_r.get('hy_spread') is not None:
        hy = hy_r['hy_spread']
        print(f"  HY OAS Spread:  {hy:.2f}%  "
              f"→ {hy_r['hy_spread_signal']}")

    # ── Composite ─────────────────────────────────────────────
    print("\n⚖️  COMPONENT 3 — COMPOSITE LIQUIDITY SCORE")
    print("-"*55)
    print(f"  Score breakdown (weighted):")
    items = [
        ('TGA (30%)',         tga_r.get('tga_score', 0)),
        ('Fed BS (25%)',      bs_r.get('bs_score', 0)),
        ('Term Prem (20%)',   tp_r.get('tp_score', 0)),
        ('Yield Sprd (15%)', ys_r.get('ys_score', 0)),
        ('HY Spread (10%)',   hy_r.get('hy_score', 0)),
    ]
    for label, s in items:
        bar_s = '█' * int(abs(s) * 8)
        sign  = '+' if s >= 0 else '-'
        print(f"    {label:<22} {s:>+6.3f}  {sign}{bar_s}")
    print(f"  ──────────────────────────")
    print(f"  COMPOSITE                  {score:>+6.3f}  "
          f"{_score_bar(score)}")

    if score >= LIQUIDITY_BOOST_THRESHOLD:
        print(f"\n  ✅ Liquidity supportive — "
              f"boosting Long signals by {overlay:+.2f}")
    elif score <= LIQUIDITY_DRAIN_THRESHOLD:
        print(f"\n  ⚠️  Liquidity tightening — "
              f"reducing Long signals by {overlay:.2f}")
    else:
        print(f"\n  Neutral — no signal overlay applied")

    # ── Fiscal ────────────────────────────────────────────────
    print("\n🏛️  COMPONENT 4 — FISCAL DOMINANCE INDICATOR")
    print("-"*55)
    if fiscal_r.get('deficit_latest') is not None:
        d   = fiscal_r['deficit_latest'] / 1_000
        acl = fiscal_r.get('deficit_acceleration', 1.0)
        print(f"  Latest monthly deficit: ${d:>+,.1f}B")
        print(f"  Deficit acceleration:   {acl:.2f}x "
              f"(3m vs prior 3m)")
        print(f"  Fiscal Stance: "
              f"{fiscal_r['fiscal_stance']}")
    else:
        print("  ⚠️  Data unavailable")

    print("\n" + "="*70)


# ═════════════════════════════════════════════════════════════
# SECTION 11 — TELEGRAM
# ═════════════════════════════════════════════════════════════

async def _send_telegram(msg):
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID, text=msg, parse_mode='HTML')
        print("  ✅ Telegram sent")
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")


def build_telegram_message(score, regime, overlay,
                            tga_r, tp_r, fiscal_r,
                            prev_regime, regime_changed):
    date = datetime.now().strftime('%d %b %Y %H:%M')
    e    = _regime_emoji(regime)
    lines = [
        f"💧 <b>GMIS LIQUIDITY PULSE MONITOR</b>",
        f"{date}",
        f"{'─' * 30}",
        "",
        f"{e} <b>REGIME: {regime}</b>",
        f"Composite Score: <b>{score:+.3f}</b>",
    ]

    if regime_changed:
        lines.append(
            f"⚡ <b>REGIME CHANGE:</b> {prev_regime} → {regime}"
        )

    overlay_str = (f"{overlay:+.2f} (boost Longs)"
                   if overlay > 0 else
                   f"{overlay:+.2f} (reduce Longs)"
                   if overlay < 0 else "0.00 (no change)")
    lines.append(f"Signal Overlay: {overlay_str}")
    lines.append("")

    # TGA
    if tga_r.get('tga_current') is not None:
        chg = tga_r['tga_4wk_change'] / 1_000
        lines.append(
            f"🏦 TGA 4wk: ${chg:+,.0f}B "
            f"→ score {tga_r['tga_score']:+.2f}"
        )

    # Term Premium
    if tp_r.get('term_premium') is not None:
        lines.append(
            f"📊 Term Premium: {tp_r['term_premium']:+.3f}% "
            f"→ score {tp_r['tp_score']:+.2f}"
        )

    # Fiscal
    if fiscal_r.get('fiscal_stance') is not None:
        lines.append(
            f"🏛️  Fiscal: {fiscal_r['fiscal_stance'][:50]}"
        )

    lines.append("")
    lines.append("<i>GMIS Liquidity Pulse Monitor</i>")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_liquidity_pulse(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 30 — LIQUIDITY PULSE MONITOR")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    conn  = sqlite3.connect(DB_PATH)
    create_table(conn)
    today = datetime.now().strftime('%Y-%m-%d')

    prev_regime = load_prev_regime(conn, today)

    # ── Fetch all FRED data ───────────────────────────────────
    print("\nFetching FRED data...")
    data = fetch_all()

    # ── Component 1: TGA ──────────────────────────────────────
    print("\nComponent 1 — TGA analysis...")
    tga_r = analyse_tga(data.get('tga', pd.Series()))
    print(f"  TGA signal: {tga_r.get('tga_signal','N/A')}")
    print(f"  TGA score:  {tga_r.get('tga_score', 0):+.3f}")

    # ── Component 2: Term Premium ─────────────────────────────
    print("\nComponent 2 — Term premium analysis...")
    tp_r = analyse_term_premium(
        data.get('term_premium', pd.Series()))
    print(f"  TP signal: {tp_r.get('term_premium_signal','N/A')}")
    print(f"  TP score:  {tp_r.get('tp_score', 0):+.3f}")

    # ── Component 3 inputs ────────────────────────────────────
    print("\nComponent 3 — Fed BS, yield spread, credit...")
    bs_r = analyse_fed_bs(data.get('fed_bs', pd.Series()))
    ys_r = analyse_yield_spread(
        data.get('yield_spread', pd.Series()))
    hy_r = analyse_hy_spread(data.get('hy_spread', pd.Series()))

    print(f"  Fed BS:  {bs_r.get('fed_bs_signal','N/A')}")
    print(f"  YldSprd: {ys_r.get('yield_spread_signal','N/A')}")
    print(f"  HY:      {hy_r.get('hy_spread_signal','N/A')}")

    # ── Composite score ───────────────────────────────────────
    score   = compute_composite_score(tga_r, bs_r, tp_r,
                                      ys_r, hy_r)
    regime  = classify_liquidity_regime(score)
    overlay = compute_signal_overlay(score)

    print(f"\n  Composite liquidity score: {score:+.4f}")
    print(f"  Regime: {regime}")
    print(f"  Signal overlay: {overlay:+.2f}")

    regime_changed = (prev_regime is not None and
                      prev_regime != regime)
    if regime_changed:
        print(f"  ⚡ REGIME CHANGE: {prev_regime} → {regime}")

    # ── Component 4: Fiscal ───────────────────────────────────
    print("\nComponent 4 — Fiscal dominance analysis...")
    fiscal_r = analyse_fiscal_stance(
        data.get('deficit', pd.Series()))
    print(f"  Fiscal stance: {fiscal_r.get('fiscal_stance','N/A')}")

    # ── Save ─────────────────────────────────────────────────
    print("\nSaving results...")
    save_results(conn, today, tga_r, tp_r, bs_r, ys_r, hy_r,
                 score, regime, overlay, fiscal_r, prev_regime)

    # ── Print report ─────────────────────────────────────────
    print_report(tga_r, tp_r, bs_r, ys_r, hy_r,
                 score, regime, overlay, fiscal_r,
                 prev_regime, regime_changed)

    conn.close()

    # ── Telegram: fire on regime change ──────────────────────
    if send_telegram_flag and BOT_TOKEN:
        should_send = (
            regime_changed or
            regime in ('HIGHLY_ACCOMMODATIVE',
                       'HIGHLY_RESTRICTIVE') or
            '--force-send' in sys.argv
        )
        if should_send:
            reason = ('regime change' if regime_changed
                      else f'extreme regime: {regime}')
            print(f"\nAlert ({reason}) — sending Telegram...")
            msg = build_telegram_message(
                score, regime, overlay, tga_r, tp_r,
                fiscal_r, prev_regime, regime_changed
            )
            asyncio.run(_send_telegram(msg))
        else:
            print(f"\n  Regime stable ({regime}) — "
                  f"no Telegram alert")
    elif not send_telegram_flag:
        print("\n  Telegram skipped (--no-telegram)")

    return {
        'score':        score,
        'regime':       regime,
        'overlay':      overlay,
        'tga':          tga_r,
        'term_premium': tp_r,
        'fed_bs':       bs_r,
        'yield_spread': ys_r,
        'hy_spread':    hy_r,
        'fiscal':       fiscal_r,
    }


if __name__ == "__main__":
    no_telegram = '--no-telegram' in sys.argv
    run_liquidity_pulse(send_telegram_flag=not no_telegram)
