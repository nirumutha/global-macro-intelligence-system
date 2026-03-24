# ============================================================
# GMIS MODULE 40 — COMPOSITE LIQUIDITY INDEX
# Aggregates signals from Modules 28, 30, 31, 22 into a
# single daily liquidity score with cross-asset adjustments.
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

# ── Component weights (must sum to 1.0) ─────────────────────
WEIGHTS = {
    'tga':         0.25,
    'fed_bs':      0.20,
    'term_premium':0.20,
    'credit':      0.15,
    'correlation': 0.10,
    'yield_dir':   0.10,
}

# ── Score thresholds → regime ────────────────────────────────
REGIME_BANDS = [
    (+0.40,  1.0,  'ABUNDANT',   +0.05),
    (+0.10, +0.40, 'AMPLE',       0.00),
    (-0.10, +0.10, 'NEUTRAL',     0.00),
    (-0.40, -0.10, 'TIGHTENING', -0.05),
    (-1.0,  -0.40, 'SCARCE',     -0.10),
]

# ── Asset-specific sensitivity multipliers (Component 4) ─────
ASSET_SENSITIVITY = {
    'SP500':  1.5,
    'NIFTY':  1.2,
    'Gold':   1.0,
    'Silver': 1.0,
    'Crude':  0.8,
}

# ── Regime change alert: send Telegram also for SCARCE ───────
ALERT_REGIMES = {'SCARCE'}

SPARKLINE_WIDTH = 40   # characters wide


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
        CREATE TABLE IF NOT EXISTS COMPOSITE_LIQUIDITY (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            date                TEXT UNIQUE,
            -- sub-component scores
            tga_score           REAL,
            fed_bs_score        REAL,
            term_premium_score  REAL,
            credit_score        REAL,
            corr_regime_score   REAL,
            yield_dir_score     REAL,
            model_health_avg    REAL,
            -- composite
            composite_score     REAL,
            rolling_30d_avg     REAL,
            regime              TEXT,
            prev_regime         TEXT,
            regime_changed      INTEGER,
            signal_adjustment   REAL,
            -- asset-specific adjustments
            adj_sp500           REAL,
            adj_nifty           REAL,
            adj_gold            REAL,
            adj_silver          REAL,
            adj_crude           REAL,
            -- context
            corr_regime_label   TEXT,
            yield_direction     TEXT,
            liquidity_pulse_raw REAL
        )
    """)
    conn.commit()


# ──────────────────────────────────────────────────────────────
# Signal-text → numeric score parser
# ──────────────────────────────────────────────────────────────
def signal_to_score(text: str) -> float:
    """Convert Module 30 signal text to [-1, +1] numeric score."""
    if not text:
        return 0.0
    t = text.upper()
    # TGA
    if 'STRONG_INJECTION' in t:   return +1.0
    if 'MILD_INJECTION'   in t:   return +0.5
    if 'STRONG_DRAIN'     in t:   return -1.0
    if 'MILD_DRAIN'       in t:   return -0.5
    # Fed BS
    if 'EXPANDING' in t and 'BS' not in t: pass  # avoid false positive
    if 'QE'        in t:                   return +1.0
    if 'QT'        in t or 'CONTRACTING' in t: return -1.0
    # Term premium
    if 'NEGATIVE_PREMIUM'  in t:  return +0.5   # low/negative TP = accommodative
    if 'ELEVATED_PREMIUM'  in t:  return -0.5
    if 'EXTREME_PREMIUM'   in t:  return -1.0
    # Credit spreads
    if 'CREDIT_BENIGN'     in t:  return +0.5
    if 'CREDIT_ELEVATED'   in t:  return -0.5
    if 'CREDIT_STRESSED'   in t:  return -1.0
    # Yield spread
    if 'INVERTED'          in t:  return -0.5
    if 'STEEP'             in t:  return +0.5
    # Generic
    if 'STABLE'  in t:            return  0.0
    if 'NORMAL'  in t:            return  0.0
    if 'NEUTRAL' in t:            return  0.0
    return 0.0


# ──────────────────────────────────────────────────────────────
# COMPONENT 1 — Pull source data
# ──────────────────────────────────────────────────────────────
def load_liquidity_pulse(conn) -> dict:
    try:
        df = pd.read_sql(
            "SELECT * FROM LIQUIDITY_PULSE ORDER BY date DESC LIMIT 1",
            conn)
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            'tga_signal':          str(row['tga_signal']),
            'fed_bs_signal':       str(row['fed_bs_signal']),
            'term_premium_signal': str(row['term_premium_signal']),
            'hy_spread_signal':    str(row['hy_spread_signal']),
            'liquidity_score_raw': float(row['liquidity_score']),
            'liquidity_regime_raw':str(row['liquidity_regime']),
            'date':                str(row['date']),
        }
    except Exception as e:
        print(f"  ⚠️  LIQUIDITY_PULSE load: {e}")
        return {}


def load_yield_beta(conn) -> dict:
    try:
        df = pd.read_sql(
            "SELECT yield_direction, sensitivity_regime "
            "FROM YIELD_EQUITY_BETA ORDER BY date DESC LIMIT 1",
            conn)
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            'yield_direction':   str(row['yield_direction']),
            'sensitivity_regime':str(row['sensitivity_regime']),
        }
    except Exception as e:
        print(f"  ⚠️  YIELD_EQUITY_BETA load: {e}")
        return {}


def load_correlation_regime(conn) -> dict:
    try:
        df = pd.read_sql(
            "SELECT regime, notes FROM REGIME_SUMMARY "
            "ORDER BY date DESC LIMIT 1",
            conn)
        if df.empty:
            # fallback to CORRELATION_REGIMES majority
            df2 = pd.read_sql(
                "SELECT regime, COUNT(*) as n FROM CORRELATION_REGIMES "
                "WHERE date = (SELECT MAX(date) FROM CORRELATION_REGIMES) "
                "GROUP BY regime ORDER BY n DESC LIMIT 1",
                conn)
            regime = str(df2.iloc[0]['regime']) if not df2.empty else 'UNKNOWN'
            return {'regime': regime, 'notes': ''}
        row = df.iloc[0]
        return {'regime': str(row['regime']), 'notes': str(row['notes'])}
    except Exception as e:
        print(f"  ⚠️  REGIME_SUMMARY load: {e}")
        return {'regime': 'UNKNOWN', 'notes': ''}


def load_model_health(conn) -> dict:
    try:
        df = pd.read_sql(
            "SELECT asset, rolling_sharpe FROM MODEL_HEALTH "
            "WHERE date = (SELECT MAX(date) FROM MODEL_HEALTH)",
            conn)
        if df.empty:
            return {'avg_sharpe': np.nan}
        avg = float(df['rolling_sharpe'].mean())
        return {'avg_sharpe': avg, 'per_asset': df.set_index('asset')['rolling_sharpe'].to_dict()}
    except Exception as e:
        print(f"  ⚠️  MODEL_HEALTH load: {e}")
        return {'avg_sharpe': np.nan}


# ──────────────────────────────────────────────────────────────
# COMPONENT 2 — Composite score
# ──────────────────────────────────────────────────────────────
def corr_regime_to_score(regime: str) -> float:
    mapping = {
        'RISK_ON':    +0.5,
        'NORMAL':      0.0,
        'RISK_OFF':   -1.0,
        'DECOUPLING':  0.0,   # ambiguous — neutral
        'UNKNOWN':     0.0,
    }
    return mapping.get(regime.upper(), 0.0)


def yield_direction_to_score(direction: str) -> float:
    mapping = {
        'FALLING': +0.5,
        'STABLE':   0.0,
        'RISING':  -0.5,
    }
    return mapping.get(direction.upper(), 0.0)


def classify_regime(score: float) -> tuple[str, float]:
    """Returns (regime_label, signal_adjustment)."""
    for lo, hi, label, adj in REGIME_BANDS:
        if lo <= score <= hi:
            return label, adj
    if score > 0.40:
        return 'ABUNDANT', +0.05
    return 'SCARCE', -0.10


def build_composite(lp: dict, yb: dict,
                     corr: dict) -> dict:
    """Compute sub-scores and weighted composite."""
    tga_score   = signal_to_score(lp.get('tga_signal', ''))
    fed_score   = signal_to_score(lp.get('fed_bs_signal', ''))
    tp_score    = signal_to_score(lp.get('term_premium_signal', ''))
    cr_score    = signal_to_score(lp.get('hy_spread_signal', ''))
    corr_score  = corr_regime_to_score(corr.get('regime', 'UNKNOWN'))
    yield_score = yield_direction_to_score(yb.get('yield_direction', 'STABLE'))

    composite = (tga_score   * WEIGHTS['tga']
               + fed_score   * WEIGHTS['fed_bs']
               + tp_score    * WEIGHTS['term_premium']
               + cr_score    * WEIGHTS['credit']
               + corr_score  * WEIGHTS['correlation']
               + yield_score * WEIGHTS['yield_dir'])
    composite = float(np.clip(composite, -1.0, 1.0))

    regime, adj = classify_regime(composite)

    return {
        'tga_score':         tga_score,
        'fed_bs_score':      fed_score,
        'term_premium_score':tp_score,
        'credit_score':      cr_score,
        'corr_regime_score': corr_score,
        'yield_dir_score':   yield_score,
        'composite_score':   composite,
        'regime':            regime,
        'signal_adjustment': adj,
        'corr_regime_label': corr.get('regime', 'UNKNOWN'),
        'yield_direction':   yb.get('yield_direction', 'UNKNOWN'),
    }


# ──────────────────────────────────────────────────────────────
# COMPONENT 3 — Rolling history & sparkline
# ──────────────────────────────────────────────────────────────
def load_history(conn, days: int = 30) -> pd.DataFrame:
    try:
        df = pd.read_sql(
            f"SELECT date, composite_score, regime FROM COMPOSITE_LIQUIDITY "
            f"ORDER BY date DESC LIMIT {days}",
            conn)
        return df.sort_values('date').reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def rolling_30d_avg(conn) -> float:
    hist = load_history(conn, 30)
    if hist.empty:
        return np.nan
    return float(hist['composite_score'].mean())


def ascii_sparkline(scores: list[float],
                    width: int = SPARKLINE_WIDTH) -> str:
    """Build a fixed-width ASCII line chart."""
    if not scores:
        return '(no history)'
    mn, mx = min(scores), max(scores)
    rng    = mx - mn if mx != mn else 1.0
    bars   = '▁▂▃▄▅▆▇█'
    result = []
    for v in scores[-width:]:
        idx = int((v - mn) / rng * (len(bars) - 1))
        result.append(bars[max(0, min(len(bars)-1, idx))])
    return ''.join(result)


def regime_change_detected(conn, new_regime: str) -> tuple[bool, str]:
    hist = load_history(conn, 5)
    if hist.empty:
        return False, 'UNKNOWN'
    prev = str(hist.iloc[-1]['regime']) if not hist.empty else 'UNKNOWN'
    return (prev != new_regime and prev != 'UNKNOWN'), prev


# ──────────────────────────────────────────────────────────────
# COMPONENT 4 — Asset-specific adjustments
# ──────────────────────────────────────────────────────────────
def compute_asset_adjustments(base_adj: float) -> dict:
    return {
        asset: round(base_adj * mult, 4)
        for asset, mult in ASSET_SENSITIVITY.items()
    }


# ──────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────
def save_results(conn, today_str: str, data: dict):
    cur = conn.cursor()
    cur.execute("DELETE FROM COMPOSITE_LIQUIDITY WHERE date = ?",
                (today_str,))
    cur.execute("""
        INSERT INTO COMPOSITE_LIQUIDITY
          (date,
           tga_score, fed_bs_score, term_premium_score,
           credit_score, corr_regime_score, yield_dir_score,
           model_health_avg,
           composite_score, rolling_30d_avg,
           regime, prev_regime, regime_changed, signal_adjustment,
           adj_sp500, adj_nifty, adj_gold, adj_silver, adj_crude,
           corr_regime_label, yield_direction, liquidity_pulse_raw)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        today_str,
        data['tga_score'],          data['fed_bs_score'],
        data['term_premium_score'], data['credit_score'],
        data['corr_regime_score'],  data['yield_dir_score'],
        data.get('model_health_avg', None),
        data['composite_score'],    data.get('rolling_30d_avg', None),
        data['regime'],             data.get('prev_regime', None),
        int(data.get('regime_changed', False)),
        data['signal_adjustment'],
        data['adj_sp500'],  data['adj_nifty'],
        data['adj_gold'],   data['adj_silver'],
        data['adj_crude'],
        data['corr_regime_label'],  data['yield_direction'],
        data.get('liquidity_pulse_raw', None),
    ))
    conn.commit()
    print(f"  ✅ Saved composite liquidity record")


# ──────────────────────────────────────────────────────────────
# Print report
# ──────────────────────────────────────────────────────────────
def regime_icon(regime: str) -> str:
    return {
        'ABUNDANT':   '🟢',
        'AMPLE':      '🟢',
        'NEUTRAL':    '🟡',
        'TIGHTENING': '🟠',
        'SCARCE':     '🔴',
    }.get(regime, '⚪')


def score_bar(score: float, width: int = 40) -> str:
    """Centered bar: negative left of center, positive right."""
    center = width // 2
    pos    = int(round((score + 1.0) / 2.0 * width))
    pos    = max(0, min(width, pos))
    bar    = ['░'] * width
    if pos >= center:
        for i in range(center, pos):
            bar[i] = '█'
    else:
        for i in range(pos, center):
            bar[i] = '█'
    bar[center] = '│'
    return ''.join(bar)


def print_report(today_str: str, data: dict,
                 lp: dict, mh: dict,
                 hist: pd.DataFrame):

    score  = data['composite_score']
    regime = data['regime']
    icon   = regime_icon(regime)
    adj    = data['signal_adjustment']

    print(f"\n{'='*70}")
    print(f"COMPOSITE LIQUIDITY INDEX — DAILY REPORT")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*70}")

    # ── Component 1 — sub-components ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 1 — LIQUIDITY SIGNAL INPUTS")
    print(f"{'─'*70}")
    sub = [
        ('TGA signal',        'tga_score',          WEIGHTS['tga'],
         lp.get('tga_signal','')),
        ('Fed balance sheet', 'fed_bs_score',        WEIGHTS['fed_bs'],
         lp.get('fed_bs_signal','')),
        ('Term premium',      'term_premium_score',  WEIGHTS['term_premium'],
         lp.get('term_premium_signal','')),
        ('Credit spreads',    'credit_score',        WEIGHTS['credit'],
         lp.get('hy_spread_signal','')),
        ('Correlation regime','corr_regime_score',   WEIGHTS['correlation'],
         data.get('corr_regime_label','')),
        ('Yield direction',   'yield_dir_score',     WEIGHTS['yield_dir'],
         data.get('yield_direction','')),
    ]
    print(f"\n  {'Component':<22} {'Wt':>5}  {'Score':>6}  "
          f"{'Contribution':>13}  Source")
    print(f"  {'-'*68}")
    for label, key, wt, source in sub:
        sc   = data.get(key, 0.0)
        cont = sc * wt
        bar  = '▲' if sc > 0 else ('▼' if sc < 0 else ' ')
        src  = str(source)[:38] + '…' if len(str(source)) > 38 else str(source)
        print(f"  {label:<22} {wt:>4.0%}  {bar}{sc:>5.2f}  "
              f"{cont:>+12.4f}  {src}")

    # Model health (context only — not in score)
    avg_sh = mh.get('avg_sharpe', np.nan)
    sh_s   = f"{avg_sh:+.3f}" if not np.isnan(avg_sh) else "N/A"
    print(f"\n  Model health (context):  avg Sharpe {sh_s}  "
          f"({'degraded' if not np.isnan(avg_sh) and avg_sh < 0 else 'healthy'})")
    if 'per_asset' in mh:
        for asset, sh in mh['per_asset'].items():
            print(f"    {asset:<8}: {sh:+.3f}")

    # ── Component 2 — Composite score ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 2 — COMPOSITE LIQUIDITY SCORE")
    print(f"{'─'*70}")
    print(f"\n  Score:  {score:+.4f}")
    print(f"  {score_bar(score)}")
    print(f"  -1.0 (SCARCE)         0.0         +1.0 (ABUNDANT)")
    print(f"\n  {icon} Regime: {regime}  "
          f"→ signal adjustment: {adj:+.2f}")
    if data.get('regime_changed'):
        print(f"\n  ⚡ REGIME CHANGE: {data['prev_regime']} → {regime}")

    roll = data.get('rolling_30d_avg')
    if roll is not None and not np.isnan(float(roll)):
        roll = float(roll)
        dev  = score - roll
        print(f"\n  30d avg: {roll:+.4f}  |  "
              f"Deviation from avg: {dev:+.4f}"
              + ("  ⚠️  Significant shift" if abs(dev) > 0.20 else ""))
    else:
        print(f"\n  30d avg: N/A (insufficient history)")

    # Module 30 raw score for cross-reference
    raw = lp.get('liquidity_score_raw', np.nan)
    if not np.isnan(raw):
        print(f"  Module 30 raw score (cross-ref): {raw:+.4f}  "
              f"[{lp.get('liquidity_regime_raw','')}]")

    # Regime reference table
    print(f"\n  Regime bands:")
    band_rows = [
        ('> +0.40', 'ABUNDANT',   '+0.05', '🟢'),
        ('+0.10 to +0.40', 'AMPLE',  ' 0.00', '🟢'),
        ('±0.10',   'NEUTRAL',    ' 0.00', '🟡'),
        ('-0.10 to -0.40', 'TIGHTENING','-0.05','🟠'),
        ('< -0.40', 'SCARCE',     '-0.10', '🔴'),
    ]
    for rng, rlabel, radj, ricn in band_rows:
        marker = '  ◄' if rlabel == regime else ''
        print(f"    {ricn} {rng:<22} {rlabel:<12} adj {radj}{marker}")

    # ── Component 3 — History & sparkline ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 3 — LIQUIDITY REGIME HISTORY")
    print(f"{'─'*70}")
    if not hist.empty:
        spark = ascii_sparkline(hist['composite_score'].tolist())
        n     = len(hist)
        oldest = hist.iloc[0]['date']
        print(f"\n  {n}-day history  ({oldest} → {today_str})")
        print(f"  {spark}")
        print(f"  Low                               High")
        print(f"\n  Recent readings:")
        for _, row in hist.tail(5).iterrows():
            ic = regime_icon(row['regime'])
            print(f"    {row['date']}  {row['composite_score']:+.4f}  "
                  f"{ic} {row['regime']}")
    else:
        print(f"\n  First run — no history yet.  "
              f"Sparkline available after 2+ daily runs.")

    # ── Component 4 — Asset adjustments ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 4 — CROSS-ASSET LIQUIDITY ADJUSTMENTS")
    print(f"{'─'*70}")
    print(f"\n  Base signal adj: {adj:+.2f}")
    print(f"\n  {'Asset':<10} {'Sensitivity':>12} {'Liquidity Adj':>14}")
    print(f"  {'-'*40}")
    for asset, mult in ASSET_SENSITIVITY.items():
        key  = f"adj_{asset.lower()}"
        aadj = data.get(key, adj * mult)
        bar  = '▲' if aadj > 0 else ('▼' if aadj < 0 else '=')
        print(f"  {asset:<10} {mult:>10.1f}×  {bar}{aadj:>+12.4f}")

    print(f"\n{'='*70}\n")


# ──────────────────────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────────────────────
def build_telegram_message(today_str: str, data: dict,
                            reason: str) -> str:
    score  = data['composite_score']
    regime = data['regime']
    icon   = regime_icon(regime)
    adj    = data['signal_adjustment']
    changed = data.get('regime_changed', False)
    lines = [
        f"💧 <b>GMIS MODULE 40 — LIQUIDITY ALERT</b>",
        f"📅 {today_str}",
        "",
    ]
    if changed:
        lines += [
            f"⚡ Regime change: "
            f"<b>{data.get('prev_regime','?')} → {regime}</b>",
        ]
    else:
        lines.append(f"{icon} Regime: <b>{regime}</b>  (score: {score:+.3f})")
    lines += [
        "",
        f"Signal adjustment: <b>{adj:+.2f}</b> to all long signals",
        f"Reason: {reason}",
        "",
        "<b>Asset-specific:</b>",
        f"  SP500:  {data['adj_sp500']:+.3f}",
        f"  NIFTY:  {data['adj_nifty']:+.3f}",
        f"  Gold:   {data['adj_gold']:+.3f}",
        f"  Silver: {data['adj_silver']:+.3f}",
        f"  Crude:  {data['adj_crude']:+.3f}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*65}")
    print(f"GMIS MODULE 40 — COMPOSITE LIQUIDITY INDEX")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*65}")

    conn = get_conn()
    setup_table(conn)

    # ── Component 1 — Pull source data ───────────────────────
    print("\nComponent 1 — Loading source data...")
    lp   = load_liquidity_pulse(conn)
    yb   = load_yield_beta(conn)
    corr = load_correlation_regime(conn)
    mh   = load_model_health(conn)

    print(f"  LP date: {lp.get('date','?')}  "
          f"raw_score={lp.get('liquidity_score_raw',np.nan):.4f}  "
          f"regime={lp.get('liquidity_regime_raw','?')}")
    print(f"  Yield dir: {yb.get('yield_direction','?')}  "
          f"Sensitivity: {yb.get('sensitivity_regime','?')}")
    print(f"  Corr regime: {corr.get('regime','?')}")
    avg_sh = mh.get('avg_sharpe', np.nan)
    print(f"  Model health avg Sharpe: "
          f"{avg_sh:+.3f}" if not np.isnan(avg_sh) else "  Model health: N/A")

    # ── Component 2 — Composite score ────────────────────────
    print("\nComponent 2 — Computing composite score...")
    scores = build_composite(lp, yb, corr)
    print(f"  Sub-scores: TGA={scores['tga_score']:+.2f}  "
          f"FedBS={scores['fed_bs_score']:+.2f}  "
          f"TP={scores['term_premium_score']:+.2f}  "
          f"Credit={scores['credit_score']:+.2f}  "
          f"Corr={scores['corr_regime_score']:+.2f}  "
          f"Yield={scores['yield_dir_score']:+.2f}")
    print(f"  Composite: {scores['composite_score']:+.4f}  "
          f"Regime: {scores['regime']}  "
          f"Adj: {scores['signal_adjustment']:+.2f}")

    # ── Component 3 — History ─────────────────────────────────
    r30    = rolling_30d_avg(conn)
    changed, prev_regime = regime_change_detected(conn, scores['regime'])
    hist   = load_history(conn, 30)
    print(f"\nComponent 3 — History: {len(hist)} rows  "
          f"30d_avg={r30:+.4f}" if not np.isnan(r30) else
          "\nComponent 3 — History: first run")
    if changed:
        print(f"  ⚡ Regime change: {prev_regime} → {scores['regime']}")

    # ── Component 4 — Asset adjustments ─────────────────────
    asset_adj = compute_asset_adjustments(scores['signal_adjustment'])

    # ── Assemble full data dict ───────────────────────────────
    data = {
        **scores,
        'rolling_30d_avg':    r30 if not np.isnan(r30) else None,
        'regime_changed':     changed,
        'prev_regime':        prev_regime,
        'model_health_avg':   avg_sh if not np.isnan(avg_sh) else None,
        'liquidity_pulse_raw':lp.get('liquidity_score_raw', None),
        'adj_sp500':  asset_adj['SP500'],
        'adj_nifty':  asset_adj['NIFTY'],
        'adj_gold':   asset_adj['Gold'],
        'adj_silver': asset_adj['Silver'],
        'adj_crude':  asset_adj['Crude'],
    }

    # ── Save ──────────────────────────────────────────────────
    print("\nSaving results...")
    save_results(conn, today_str, data)

    # ── Report ────────────────────────────────────────────────
    print_report(today_str, data, lp, mh, hist)

    # ── Telegram ─────────────────────────────────────────────
    should_alert = (changed or scores['regime'] in ALERT_REGIMES)
    reason = (f"Regime change {prev_regime}→{scores['regime']}"
              if changed
              else f"{scores['regime']} liquidity detected")

    if should_alert and not NO_TELEGRAM:
        msg = build_telegram_message(today_str, data, reason)
        asyncio.run(send_telegram(msg))
        print("  📱 Telegram alert sent")
    elif NO_TELEGRAM:
        print("  Telegram skipped (--no-telegram)")
    else:
        print(f"  No alert triggered (regime={scores['regime']}, changed={changed})")

    conn.close()


if __name__ == '__main__':
    main()
