# ============================================================
# GMIS 2.0 — MODULE 28 — CORRELATION REGIME MONITOR
#
# COMPONENT 1 — Regime Shift Detection
#   Rolling 20d vs 60d correlation for all key pairs
#   Flag REGIME_SHIFT when |20d - 60d| > 0.30
#
# COMPONENT 2 — Regime Classification
#   RISK_ON    : SP500+NIFTY correlated (>0.7), Gold negative
#   RISK_OFF   : Gold-SP500 goes positive — liquidity crunch
#   DECOUPLING : NIFTY-SP500 < 0.4 — India-specific event
#   NORMAL     : No unusual patterns
#
# COMPONENT 3 — Signal Reliability Adjustment
#   Adds confidence warnings based on correlation regime
#
# COMPONENT 4 — Historical Context
#   Finds previous episodes with similar correlation readings
# ============================================================

import sqlite3
import pandas as pd
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
DATA_PATH = os.path.join(BASE_PATH, 'data')

ASSETS = ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']

ASSET_FILES = {
    'NIFTY':  'NIFTY50.csv',
    'SP500':  'SP500.csv',
    'Gold':   'GOLD.csv',
    'Silver': 'SILVER.csv',
    'Crude':  'CRUDE_WTI.csv',
}

# ── Correlation windows ───────────────────────────────────────
SHORT_WINDOW = 20   # recent — captures regime shifts fast
LONG_WINDOW  = 60   # baseline

# ── Shift detection threshold ─────────────────────────────────
SHIFT_THRESHOLD = 0.30   # 20d vs 60d change that flags a shift

# ── Regime thresholds ─────────────────────────────────────────
RISK_OFF_THRESHOLD  =  0.20  # Gold-SP500 above this → RISK_OFF
RISK_ON_THRESHOLD   =  0.70  # NIFTY-SP500 above this → RISK_ON
DECOUPLE_THRESHOLD  =  0.40  # NIFTY-SP500 below this → DECOUPLING
GOLD_SILVER_FLOOR   =  0.60  # below this → industrial divergence

# ── Key pairs with economic rationale ────────────────────────
KEY_PAIRS = [
    ('Gold',  'SP500', 'Safe haven vs equities'),
    ('NIFTY', 'SP500', 'EM-US alignment'),
    ('Gold',  'Silver','Precious metals cohesion'),
    ('Crude', 'NIFTY', 'Oil cost pressure on India'),
]

# ── Historical context window ─────────────────────────────────
# How close does past corr need to be to "match" current?
HIST_MATCH_BAND = 0.15
MIN_EPISODE_DAYS = 10


# ═════════════════════════════════════════════════════════════
# SECTION 1 — DATABASE
# ═════════════════════════════════════════════════════════════

def create_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS CORRELATION_REGIMES (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT NOT NULL,
            pair         TEXT NOT NULL,
            asset1       TEXT,
            asset2       TEXT,
            corr_20d     REAL,
            corr_60d     REAL,
            shift        REAL,
            shift_flag   TEXT,
            regime       TEXT,
            reliability  TEXT,
            note         TEXT,
            UNIQUE(date, pair)
        )
    ''')
    # Regime-level summary (one row per day)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS REGIME_SUMMARY (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT NOT NULL UNIQUE,
            regime       TEXT,
            shift_count  INTEGER,
            key_shifts   TEXT,
            notes        TEXT
        )
    ''')
    conn.commit()


# ═════════════════════════════════════════════════════════════
# SECTION 2 — LOAD PRICE DATA
# ═════════════════════════════════════════════════════════════

def load_all_closes():
    """Load full history of daily closes for all assets."""
    closes = {}
    for asset, fname in ASSET_FILES.items():
        path = os.path.join(DATA_PATH, fname)
        try:
            df = pd.read_csv(path)
            df = df[~df['Price'].astype(str).str.match(
                r'^[A-Za-z]', na=False)]
            df = df.rename(columns={'Price': 'Date'})
            df['Date']  = pd.to_datetime(
                df['Date'], errors='coerce')
            df['Close'] = pd.to_numeric(
                df['Close'], errors='coerce')
            df = (df.dropna(subset=['Date', 'Close'])
                    .sort_values('Date')
                    .set_index('Date'))
            closes[asset] = df['Close']
        except Exception as e:
            print(f"  ⚠️  Could not load {asset}: {e}")

    if not closes:
        return pd.DataFrame(), pd.DataFrame()

    prices  = pd.DataFrame(closes).dropna()
    returns = prices.pct_change().dropna()
    return prices, returns


# ═════════════════════════════════════════════════════════════
# SECTION 3 — COMPONENT 1: ROLLING CORRELATIONS & SHIFT
# ═════════════════════════════════════════════════════════════

def compute_pair_correlations(returns):
    """
    For every key pair compute:
      corr_20d, corr_60d, shift (20d - 60d), shift_flag
    Returns a dict keyed by pair label.
    """
    results = {}

    for a1, a2, rationale in KEY_PAIRS:
        if a1 not in returns.columns or \
           a2 not in returns.columns:
            continue

        r1 = returns[a1]
        r2 = returns[a2]

        corr_20d = float(
            r1.rolling(SHORT_WINDOW).corr(r2).iloc[-1]
        )
        corr_60d = float(
            r1.rolling(LONG_WINDOW).corr(r2).iloc[-1]
        )
        shift    = corr_20d - corr_60d

        shift_flag = 'SHIFT' if abs(shift) >= SHIFT_THRESHOLD \
                     else 'STABLE'

        results[f'{a1}_{a2}'] = {
            'pair':       f'{a1}_{a2}',
            'asset1':     a1,
            'asset2':     a2,
            'rationale':  rationale,
            'corr_20d':   round(corr_20d, 4),
            'corr_60d':   round(corr_60d, 4),
            'shift':      round(shift, 4),
            'shift_flag': shift_flag,
        }

    return results


def compute_full_corr_matrix(returns, window=SHORT_WINDOW):
    """Rolling correlation matrix (window days)."""
    recent = returns.tail(window)
    return recent.corr()


# ═════════════════════════════════════════════════════════════
# SECTION 4 — COMPONENT 2: REGIME CLASSIFICATION
# ═════════════════════════════════════════════════════════════

def classify_regime(pair_corrs):
    """
    Derive the overall correlation regime from key pairs.

    Priority order:
      1. RISK_OFF  — Gold-SP500 goes positive (liquidity crunch)
      2. DECOUPLING — NIFTY-SP500 falls below 0.40
      3. RISK_ON   — NIFTY-SP500 > 0.70 AND Gold negative
      4. NORMAL
    """
    gs  = pair_corrs.get('Gold_SP500', {}).get('corr_20d', 0)
    ns  = pair_corrs.get('NIFTY_SP500', {}).get('corr_20d', 0.5)
    gsilv = pair_corrs.get('Gold_Silver', {}).get('corr_20d', 0.8)

    notes = []

    if gs > RISK_OFF_THRESHOLD:
        regime = 'RISK_OFF'
        notes.append(
            f'Gold-SP500 correlation {gs:+.2f} '
            f'(liquidity crunch — both selling off together)'
        )
    elif ns < DECOUPLE_THRESHOLD:
        regime = 'DECOUPLING'
        notes.append(
            f'NIFTY-SP500 correlation {ns:.2f} '
            f'(India decoupled from US — India-specific drivers)'
        )
    elif ns > RISK_ON_THRESHOLD and gs < 0:
        regime = 'RISK_ON'
        notes.append(
            f'NIFTY-SP500 {ns:.2f} (synchronized), '
            f'Gold negative {gs:+.2f} (safe haven not needed)'
        )
    else:
        regime = 'NORMAL'
        notes.append('No extreme correlation patterns detected')

    # Add supplementary notes
    if gsilv < GOLD_SILVER_FLOOR:
        notes.append(
            f'Gold-Silver diverging ({gsilv:.2f}) — '
            f'industrial demand separating from safe haven'
        )

    cn = pair_corrs.get('Crude_NIFTY', {}).get('corr_20d', 0)
    if cn < -0.40:
        notes.append(
            f'Crude-NIFTY strongly negative ({cn:.2f}) — '
            f'oil pressuring Indian equities'
        )

    return regime, notes


# ═════════════════════════════════════════════════════════════
# SECTION 5 — COMPONENT 3: SIGNAL RELIABILITY
# ═════════════════════════════════════════════════════════════

def derive_reliability(pair_label, corr_20d, regime):
    """
    Returns (reliability_flag, note) for one pair.
    """
    a1, a2 = pair_label.split('_')

    # Gold as safe haven — only reliable when negatively corr
    if pair_label == 'Gold_SP500':
        if corr_20d > RISK_OFF_THRESHOLD:
            return ('DEGRADED',
                    'Gold no longer acting as safe haven — '
                    'moves with equities in liquidity crunch')
        elif corr_20d > 0:
            return ('REDUCED',
                    'Gold mildly positive to equities — '
                    'safe haven hedge partially intact')
        else:
            return ('FULL', 'Gold acting as safe haven (negative corr)')

    # NIFTY signals
    if pair_label == 'NIFTY_SP500':
        if corr_20d < DECOUPLE_THRESHOLD:
            return ('INDEPENDENT',
                    'India decoupling from US — use India-specific '
                    'signals; US signal spill-over reduced')
        elif corr_20d > RISK_ON_THRESHOLD:
            return ('US_LINKED',
                    'NIFTY highly correlated to SP500 — '
                    'global risk-on/off dominates')
        else:
            return ('PARTIAL', 'Moderate NIFTY-SP500 alignment')

    # Gold-Silver cohesion
    if pair_label == 'Gold_Silver':
        if corr_20d > 0.80:
            return ('FULL', 'PM cohesion strong — signals confirm each other')
        elif corr_20d > GOLD_SILVER_FLOOR:
            return ('PARTIAL', 'PM cohesion moderate')
        else:
            return ('DIVERGED',
                    'Gold and Silver diverging — '
                    'industrial vs safe-haven demand split')

    # Crude-NIFTY
    if pair_label == 'Crude_NIFTY':
        if corr_20d < -0.40:
            return ('BEARISH_PRESSURE',
                    'Strong oil headwind on NIFTY — '
                    'rising crude hurts India economy')
        elif corr_20d > 0.30:
            return ('GROWTH_ALIGNED',
                    'NIFTY and Crude moving together — '
                    'global growth narrative dominant')
        else:
            return ('NEUTRAL', 'Crude-NIFTY relationship mixed')

    return ('FULL', '')


# ═════════════════════════════════════════════════════════════
# SECTION 6 — COMPONENT 4: HISTORICAL CONTEXT
# ═════════════════════════════════════════════════════════════

def find_historical_analogs(returns, pair_label, current_corr):
    """
    Find previous episodes where the 20-day rolling correlation
    for this pair was within HIST_MATCH_BAND of current_corr.
    Returns list of (start_date, end_date, label) tuples,
    most recent 5 notable episodes only.
    """
    a1, a2 = pair_label.split('_')
    if a1 not in returns.columns or a2 not in returns.columns:
        return []

    series = (returns[a1]
              .rolling(SHORT_WINDOW)
              .corr(returns[a2])
              .dropna())

    # Exclude last 30 days (current)
    hist = series.iloc[:-30]

    in_ep  = False
    start  = None
    ep_corr = []
    episodes = []

    for date, val in hist.items():
        match = abs(val - current_corr) <= HIST_MATCH_BAND
        if match and not in_ep:
            in_ep   = True
            start   = date
            ep_corr = [val]
        elif match and in_ep:
            ep_corr.append(val)
        elif not match and in_ep:
            in_ep = False
            dur   = (date - start).days
            if dur >= MIN_EPISODE_DAYS:
                episodes.append({
                    'start':    start,
                    'end':      date,
                    'duration': dur,
                    'avg_corr': float(np.mean(ep_corr)),
                })

    # Sort by recency and length; keep 5 most notable
    episodes.sort(key=lambda x: (x['start'].year,
                                  x['duration']), reverse=True)
    return episodes[:5]


def format_episode_label(ep, pair_label):
    """Convert an episode dict to a readable string."""
    s   = ep['start'].strftime('%b %Y')
    e   = ep['end'].strftime('%b %Y')
    dur = ep['duration']
    c   = ep['avg_corr']
    return f"{s}→{e} ({dur}d, avg {c:+.2f})"


def _regime_label_for_date(gs_series, ns_series, date):
    """Quick regime label for a historical date."""
    try:
        gs = gs_series.loc[:date].iloc[-1]
        ns = ns_series.loc[:date].iloc[-1]
        if gs > RISK_OFF_THRESHOLD:
            return 'RISK_OFF'
        elif ns < DECOUPLE_THRESHOLD:
            return 'DECOUPLING'
        elif ns > RISK_ON_THRESHOLD:
            return 'RISK_ON'
        return 'NORMAL'
    except Exception:
        return '?'


# ═════════════════════════════════════════════════════════════
# SECTION 7 — SAVE TO DATABASE
# ═════════════════════════════════════════════════════════════

def save_results(conn, today, pair_corrs, regime, notes):
    # Save per-pair rows
    for pair_label, p in pair_corrs.items():
        rel, rel_note = derive_reliability(
            pair_label, p['corr_20d'], regime
        )
        try:
            conn.execute('''
                INSERT OR REPLACE INTO CORRELATION_REGIMES
                (date, pair, asset1, asset2,
                 corr_20d, corr_60d, shift, shift_flag,
                 regime, reliability, note)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                today, pair_label, p['asset1'], p['asset2'],
                p['corr_20d'], p['corr_60d'],
                p['shift'],    p['shift_flag'],
                regime, rel, rel_note
            ))
        except Exception as e:
            print(f"  ❌ Save pair {pair_label}: {e}")

    # Save regime summary
    shift_pairs = [k for k, v in pair_corrs.items()
                   if v['shift_flag'] == 'SHIFT']
    try:
        conn.execute('''
            INSERT OR REPLACE INTO REGIME_SUMMARY
            (date, regime, shift_count, key_shifts, notes)
            VALUES (?,?,?,?,?)
        ''', (
            today, regime, len(shift_pairs),
            ', '.join(shift_pairs),
            ' | '.join(notes)
        ))
    except Exception as e:
        print(f"  ❌ Save regime summary: {e}")

    conn.commit()
    print(f"  ✅ Correlation regimes saved "
          f"({len(pair_corrs)} pairs, regime={regime})")


# ═════════════════════════════════════════════════════════════
# SECTION 8 — PRINT REPORT
# ═════════════════════════════════════════════════════════════

def _regime_emoji(regime):
    return {
        'RISK_OFF':   '🔴',
        'DECOUPLING': '🟡',
        'RISK_ON':    '🟢',
        'NORMAL':     '⚪',
    }.get(regime, '❓')


def _corr_bar(corr, width=20):
    """Visual bar: center = 0, left = negative, right = positive."""
    mid  = width // 2
    pos  = int(corr * mid)
    pos  = max(-mid, min(mid, pos))
    if pos >= 0:
        bar = '─' * mid + '█' * pos + '░' * (mid - pos)
    else:
        bar = '░' * (mid + pos) + '█' * (-pos) + '─' * mid
    return f'[{bar}]'


def _shift_arrow(shift):
    if shift >= SHIFT_THRESHOLD:
        return '↑↑ SHIFT'
    elif shift <= -SHIFT_THRESHOLD:
        return '↓↓ SHIFT'
    elif shift > 0.15:
        return '↑'
    elif shift < -0.15:
        return '↓'
    return '→'


def print_report(pair_corrs, corr_matrix, regime, notes,
                 historical_context, shift_pairs):
    print("\n" + "="*75)
    print("CORRELATION REGIME MONITOR — DAILY REPORT")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*75)

    # ── Regime banner ─────────────────────────────────────────
    e = _regime_emoji(regime)
    print(f"\n  {e}  CURRENT REGIME: {regime}")
    for note in notes:
        print(f"      • {note}")

    # ── Key pair table ────────────────────────────────────────
    print("\n📊 COMPONENT 1 — KEY PAIR CORRELATIONS")
    print("-"*75)
    hdr = (f"  {'Pair':<18} {'20d':>6}  {'60d':>6}  "
           f"{'Shift':>7}  {'Direction':<12} Reliability")
    print(hdr)
    print("-"*75)

    for pair_label, p in pair_corrs.items():
        a1, a2  = p['asset1'], p['asset2']
        c20     = p['corr_20d']
        c60     = p['corr_60d']
        shift   = p['shift']
        sflag   = p['shift_flag']
        rat     = p['rationale']

        rel, rel_note = derive_reliability(pair_label, c20, regime)

        arrow = _shift_arrow(shift)
        star  = ' ⚡' if sflag == 'SHIFT' else ''

        print(f"  {a1+'/'+a2:<18} {c20:>+6.3f}  {c60:>+6.3f}  "
              f"{shift:>+7.3f}  {arrow:<12} {rel}{star}")
        print(f"  {'':18} {_corr_bar(c20)}")
        if sflag == 'SHIFT':
            print(f"  {'':18} ⚡ REGIME SHIFT: {rat}")
        if rel_note:
            print(f"  {'':18}   → {rel_note}")
        print()

    # ── Full correlation matrix ───────────────────────────────
    if not corr_matrix.empty:
        print("🔗 COMPONENT 2 — 20-DAY ROLLING CORRELATION MATRIX")
        print("-"*55)
        cols = [a for a in ASSETS if a in corr_matrix.columns]
        print(f"  {'':8}" +
              "".join(f"{c:>8}" for c in cols))
        for r in cols:
            row = f"  {r:<8}"
            for c in cols:
                v    = float(corr_matrix.loc[r, c])
                mark = ('!' if r != c and
                         abs(v) > 0.70 else ' ')
                row += f"{v:>7.2f}{mark}"
            print(row)
        print("    ! = |corr| > 0.70")

    # ── Signal reliability ────────────────────────────────────
    print("\n⚠️  COMPONENT 3 — SIGNAL RELIABILITY ADJUSTMENTS")
    print("-"*75)

    warnings_found = False
    rel_map = {
        'Gold_SP500': ('Gold safe-haven hedge',
                       'Gold signals'),
        'NIFTY_SP500': ('NIFTY-US linkage',
                        'NIFTY signals'),
        'Gold_Silver': ('Precious metals cohesion',
                        'Silver signals'),
        'Crude_NIFTY': ('Oil-India relationship',
                        'NIFTY oil sensitivity'),
    }
    for pair_label, p in pair_corrs.items():
        rel, rel_note = derive_reliability(
            pair_label, p['corr_20d'], regime)
        label, signal = rel_map.get(pair_label, (pair_label, ''))
        if rel not in ('FULL', 'NEUTRAL', 'US_LINKED', 'NORMAL'):
            icon = ('🔴' if rel in
                    ('DEGRADED', 'DIVERGED', 'INDEPENDENT')
                    else '🟡')
            print(f"  {icon} [{label}] {rel}: {rel_note}")
            warnings_found = True

    if regime == 'RISK_OFF':
        print(f"  🚨 RISK_OFF REGIME: Gold unreliable as hedge — "
              f"reduce Gold Long positions")
    elif regime == 'DECOUPLING':
        print(f"  🟡 DECOUPLING: NIFTY signals independent of US — "
              f"treat NIFTY separately")

    if not warnings_found and regime == 'NORMAL':
        print("  ✅ All signals operating with full reliability")

    # ── Historical context ────────────────────────────────────
    print("\n🕰️  COMPONENT 4 — HISTORICAL ANALOGS")
    print("-"*75)

    for pair_label, episodes in historical_context.items():
        if not episodes:
            continue
        a1, a2  = pair_label.split('_')
        c_now   = pair_corrs[pair_label]['corr_20d']
        print(f"\n  {a1}/{a2} (current 20d: {c_now:+.2f})")
        print(f"  Similar readings in history:")
        for ep in episodes:
            print(f"    • {format_episode_label(ep, pair_label)}")

    print("\n" + "="*75)


# ═════════════════════════════════════════════════════════════
# SECTION 9 — TELEGRAM
# ═════════════════════════════════════════════════════════════

async def _send_telegram(msg):
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID,
            text=msg,
            parse_mode='HTML'
        )
        print("  ✅ Telegram sent")
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")


def build_telegram_message(pair_corrs, regime, notes,
                            shift_pairs):
    date = datetime.now().strftime('%d %b %Y %H:%M')
    e    = _regime_emoji(regime)
    lines = [
        f"🔗 <b>GMIS CORRELATION REGIME MONITOR</b>",
        f"{date}",
        f"{'─' * 30}",
        "",
        f"{e} <b>REGIME: {regime}</b>",
    ]
    for note in notes:
        lines.append(f"  • {note}")
    lines.append("")

    if shift_pairs:
        lines.append(f"⚡ <b>REGIME SHIFTS DETECTED:</b>")
        for pair_label in shift_pairs:
            p    = pair_corrs[pair_label]
            a1, a2 = p['asset1'], p['asset2']
            lines.append(
                f"  {a1}/{a2}: 20d={p['corr_20d']:+.2f}  "
                f"60d={p['corr_60d']:+.2f}  "
                f"Δ={p['shift']:+.2f}"
            )
        lines.append("")

    # Reliability warnings
    rel_issues = []
    for pair_label, p in pair_corrs.items():
        rel, rel_note = derive_reliability(
            pair_label, p['corr_20d'], regime)
        if rel not in ('FULL', 'NEUTRAL', 'US_LINKED', 'NORMAL'):
            rel_issues.append(f"  ⚠️ {rel}: {rel_note}")

    if rel_issues:
        lines.append(f"<b>Signal Reliability Warnings:</b>")
        lines.extend(rel_issues)
        lines.append("")

    lines.append("<i>GMIS Correlation Regime Monitor</i>")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_correlation_regime(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 28 — CORRELATION REGIME MONITOR")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    conn = sqlite3.connect(DB_PATH)
    create_table(conn)
    today = datetime.now().strftime('%Y-%m-%d')

    # ── Load price data ───────────────────────────────────────
    print("\nLoading full price history...")
    prices, returns = load_all_closes()

    if returns.empty:
        print("  ❌ No price data — aborting")
        conn.close()
        return None

    print(f"  {len(returns)} trading days loaded "
          f"({returns.index[0].date()} → "
          f"{returns.index[-1].date()})")

    # ── Component 1: Pair correlations ───────────────────────
    print("\nComponent 1 — Computing rolling correlations...")
    pair_corrs = compute_pair_correlations(returns)

    for pair_label, p in pair_corrs.items():
        flag = '⚡ SHIFT' if p['shift_flag'] == 'SHIFT' else ''
        print(f"  {p['asset1']}/{p['asset2']}: "
              f"20d={p['corr_20d']:+.3f}  "
              f"60d={p['corr_60d']:+.3f}  "
              f"Δ={p['shift']:+.3f}  {flag}")

    # ── Full matrix ───────────────────────────────────────────
    corr_matrix = compute_full_corr_matrix(returns)

    # ── Component 2: Regime classification ───────────────────
    print("\nComponent 2 — Classifying regime...")
    regime, notes = classify_regime(pair_corrs)
    print(f"  Regime: {regime}")
    for note in notes:
        print(f"  → {note}")

    # ── Component 3: Reliability is printed in the report ────
    shift_pairs = [k for k, v in pair_corrs.items()
                   if v['shift_flag'] == 'SHIFT']
    print(f"\nComponent 3 — Shift detection: "
          f"{len(shift_pairs)} shift(s) found")
    for p in shift_pairs:
        print(f"  ⚡ {p}")

    # ── Component 4: Historical analogs ──────────────────────
    print("\nComponent 4 — Building historical context...")
    historical_context = {}
    for pair_label, p in pair_corrs.items():
        eps = find_historical_analogs(
            returns, pair_label, p['corr_20d']
        )
        historical_context[pair_label] = eps
        print(f"  {pair_label}: {len(eps)} analog episode(s)")

    # ── Save ─────────────────────────────────────────────────
    print("\nSaving to database...")
    save_results(conn, today, pair_corrs, regime, notes)
    conn.close()

    # ── Print report ─────────────────────────────────────────
    print_report(pair_corrs, corr_matrix, regime, notes,
                 historical_context, shift_pairs)

    # ── Telegram: only on regime shift ───────────────────────
    if send_telegram_flag and BOT_TOKEN:
        if shift_pairs or regime in ('RISK_OFF', 'DECOUPLING'):
            reason = ('regime shifts: ' + ', '.join(shift_pairs)
                      if shift_pairs else f'regime = {regime}')
            print(f"\nAlert condition met ({reason}) "
                  f"— sending Telegram...")
            msg = build_telegram_message(
                pair_corrs, regime, notes, shift_pairs
            )
            asyncio.run(_send_telegram(msg))
        else:
            print(f"\n  NORMAL regime, no shifts — "
                  f"no Telegram alert")
    elif not send_telegram_flag:
        print("\n  Telegram skipped (--no-telegram)")

    return {
        'regime':      regime,
        'notes':       notes,
        'pair_corrs':  pair_corrs,
        'corr_matrix': corr_matrix,
        'shift_pairs': shift_pairs,
    }


if __name__ == "__main__":
    no_telegram = '--no-telegram' in sys.argv
    run_correlation_regime(send_telegram_flag=not no_telegram)
