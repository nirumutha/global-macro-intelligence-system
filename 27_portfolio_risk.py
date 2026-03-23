# ============================================================
# GMIS 2.0 — MODULE 27 — PORTFOLIO RISK ENGINE
#
# COMPONENT 1 — Kelly Criterion Position Sizing
#   p  = win rate from SIGNALS_V3 hit_rate
#   b  = analog 30d median return / (3×ATR as % of price)
#   f  = (p×b - q) / b   →  half-Kelly  →  cap [2%, 20%]
#
# COMPONENT 2 — Correlation-Adjusted Risk
#   60-day rolling correlation matrix from CSV prices
#   Corr > 0.70 between two active longs → reduce both 30%
#   Corr < 0.30 → full Kelly allowed (genuinely diversified)
#
# COMPONENT 3 — Portfolio Drawdown Estimator
#   Per-asset expected max-DD  = position% × ATR%×√252 × 1.5
#   Portfolio DD adjusted for correlation
#   > 15%  → WARNING   > 25%  → CRITICAL (cut all by 50%)
#
# COMPONENT 4 — Daily Risk Summary
#   Saved to PORTFOLIO_RISK table
#   Telegram fired when active positions exist
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

# ── Risk parameters ───────────────────────────────────────────
HALF_KELLY          = True    # apply half-Kelly safety factor
KELLY_MAX_PCT       = 20.0   # cap position at 20 % of portfolio
KELLY_MIN_PCT       =  2.0   # floor for any active signal
CORR_HIGH_THRESHOLD =  0.70  # above this → high-corr warning
CORR_LOW_THRESHOLD  =  0.30  # below this → full Kelly
CORR_REDUCTION      =  0.30  # reduce both positions by 30 %
DD_WARNING_PCT      = 15.0   # portfolio DD warning level
DD_CRITICAL_PCT     = 25.0   # portfolio DD critical level
ATR_PERIOD          = 14
CORR_WINDOW         = 60     # rolling correlation window (days)
ANNUALISE           = 252    # trading days per year


# ═════════════════════════════════════════════════════════════
# SECTION 1 — DATABASE
# ═════════════════════════════════════════════════════════════

def create_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS PORTFOLIO_RISK (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            asset           TEXT NOT NULL,
            bias            TEXT,
            kelly_raw       REAL,
            kelly_half      REAL,
            position_pct    REAL,
            corr_flag       TEXT,
            corr_pair       TEXT,
            corr_value      REAL,
            adj_position    REAL,
            daily_vol_pct   REAL,
            atr_pct         REAL,
            max_dd_est      REAL,
            portfolio_dd    REAL,
            dd_level        TEXT,
            action          TEXT,
            UNIQUE(date, asset)
        )
    ''')
    conn.commit()


# ═════════════════════════════════════════════════════════════
# SECTION 2 — LOAD SOURCE DATA
# ═════════════════════════════════════════════════════════════

def load_decisions(conn):
    """Latest bias per asset from DECISIONS table."""
    try:
        df = pd.read_sql(
            '''SELECT asset, bias, confidence, combined
               FROM DECISIONS
               WHERE date = (SELECT MAX(date) FROM DECISIONS)
               ORDER BY asset''',
            conn
        )
        return {row['asset']: dict(row)
                for _, row in df.iterrows()}
    except Exception as e:
        print(f"  ⚠️  Could not load decisions: {e}")
        return {}


def load_hit_rates(conn):
    """Latest hit rates from SIGNALS_V3."""
    try:
        df = pd.read_sql(
            'SELECT * FROM SIGNALS_V3 ORDER BY Date DESC LIMIT 1',
            conn
        )
        if df.empty:
            return {}
        row = df.iloc[0]
        return {a: float(row.get(f'{a}_hit_rate', 0.5) or 0.5)
                for a in ASSETS}
    except Exception as e:
        print(f"  ⚠️  Could not load hit rates: {e}")
        return {a: 0.5 for a in ASSETS}


def load_analog_outcomes(conn):
    """30-day analog median returns per asset."""
    try:
        df = pd.read_sql(
            '''SELECT asset, median_return, prob_positive
               FROM ANALOG_OUTCOMES
               WHERE forward_days = 30
                 AND run_date = (SELECT MAX(run_date)
                                 FROM ANALOG_OUTCOMES)''',
            conn
        )
        return {row['asset']: {
            'median_return': float(row['median_return']),
            'prob_positive': float(row['prob_positive']),
        } for _, row in df.iterrows()}
    except Exception as e:
        print(f"  ⚠️  Could not load analog outcomes: {e}")
        return {}


def load_price_series(asset, days=120):
    """Return a DataFrame of daily closes for an asset."""
    path = os.path.join(DATA_PATH, ASSET_FILES[asset])
    try:
        df = pd.read_csv(path)
        # Drop non-date header rows
        df = df[~df['Price'].astype(str).str.match(
            r'^[A-Za-z]', na=False
        )]
        df = df.rename(columns={'Price': 'Date'})
        df['Date']  = pd.to_datetime(df['Date'], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if 'High' in df.columns:
            df['High'] = pd.to_numeric(df['High'], errors='coerce')
        if 'Low' in df.columns:
            df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        df = (df.dropna(subset=['Date', 'Close'])
                .sort_values('Date')
                .set_index('Date'))
        return df.tail(max(days, CORR_WINDOW + 30))
    except Exception as e:
        print(f"    ⚠️  Could not load {asset} prices: {e}")
        return pd.DataFrame()


# ═════════════════════════════════════════════════════════════
# SECTION 3 — COMPONENT 1: KELLY CRITERION
# ═════════════════════════════════════════════════════════════

def calc_atr(df):
    """14-period Average True Range."""
    if 'High' not in df.columns or 'Low' not in df.columns:
        # Approximate ATR from daily returns
        closes = df['Close'].tail(ATR_PERIOD + 1)
        return float(closes.pct_change().abs().mean() *
                     closes.iloc[-1])

    high  = df['High']
    low   = df['Low']
    close = df['Close']
    tr    = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(ATR_PERIOD).mean().iloc[-1])
    return atr if not np.isnan(atr) else float(
        df['Close'].pct_change().abs().tail(ATR_PERIOD).mean()
        * df['Close'].iloc[-1]
    )


def calc_daily_vol(df):
    """Annualised daily volatility (last 60 days)."""
    returns = df['Close'].pct_change().dropna().tail(60)
    return float(returns.std() * np.sqrt(ANNUALISE))


def kelly_size(p, b):
    """
    Kelly fraction:  f = (p·b − q) / b
    Clipped to [0, 1] before half-Kelly and cap/floor.
    """
    if b <= 0 or p <= 0:
        return 0.0
    q = 1.0 - p
    f = (p * b - q) / b
    return max(0.0, min(1.0, f))


def compute_kelly_for_asset(asset, bias, hit_rate,
                             analog, df):
    """
    Return dict with Kelly sizing details for one asset.
    bias must be 'LONG' or 'SHORT'.
    """
    result = {
        'asset':         asset,
        'bias':          bias,
        'kelly_raw':     0.0,
        'kelly_half':    0.0,
        'position_pct':  0.0,
        'atr_pct':       0.0,
        'daily_vol_pct': 0.0,
        'max_dd_est':    0.0,
    }

    if df.empty:
        return result

    current = float(df['Close'].iloc[-1])
    atr     = calc_atr(df)
    ann_vol = calc_daily_vol(df)

    atr_pct  = (atr / current * 100) if current > 0 else 2.0
    result['atr_pct']       = round(atr_pct, 3)
    result['daily_vol_pct'] = round(ann_vol * 100, 2)

    # Win estimate: analog 30d median return (absolute %)
    if analog:
        win_pct = abs(analog.get('median_return', 2.0))
    else:
        win_pct = 2.0
    win_pct = max(win_pct, 0.5)       # floor at 0.5 %

    # Loss estimate: 3×ATR as % of current price
    loss_pct = atr_pct * 3.0
    loss_pct = max(loss_pct, 0.5)

    b = win_pct / loss_pct            # reward-to-risk ratio
    p = hit_rate

    raw  = kelly_size(p, b)
    half = raw / 2.0 if HALF_KELLY else raw

    # Convert fraction → % of portfolio; apply cap and floor
    pct = half * 100.0
    if pct > 0:
        pct = min(max(pct, KELLY_MIN_PCT), KELLY_MAX_PCT)

    result.update({
        'kelly_raw':    round(raw * 100, 2),
        'kelly_half':   round(half * 100, 2),
        'position_pct': round(pct, 2),
    })

    # Per-asset expected max drawdown = position% × ann_vol × 1.5
    max_dd = pct * ann_vol * 1.5
    result['max_dd_est'] = round(max_dd, 2)

    return result


# ═════════════════════════════════════════════════════════════
# SECTION 4 — COMPONENT 2: CORRELATION-ADJUSTED RISK
# ═════════════════════════════════════════════════════════════

def build_corr_matrix():
    """
    Build 60-day rolling correlation matrix from CSV closes.
    Returns a DataFrame (assets × assets).
    """
    closes = {}
    for asset in ASSETS:
        df = load_price_series(asset, days=CORR_WINDOW + 10)
        if not df.empty:
            closes[asset] = df['Close'].tail(CORR_WINDOW)

    if len(closes) < 2:
        return pd.DataFrame()

    price_df = pd.DataFrame(closes).dropna()
    returns  = price_df.pct_change().dropna()
    return returns.corr()


def apply_correlation_adjustments(sizing, corr_matrix):
    """
    For each pair of active Long or Short signals,
    check correlation and reduce positions if > threshold.
    Adds 'corr_flag', 'corr_pair', 'corr_value',
    'adj_position' to each sizing dict.
    """
    # Initialise adj_position = position_pct
    for a in sizing:
        sizing[a]['corr_flag']   = 'OK'
        sizing[a]['corr_pair']   = ''
        sizing[a]['corr_value']  = 0.0
        sizing[a]['adj_position'] = sizing[a]['position_pct']

    if corr_matrix.empty:
        return sizing

    active = [a for a in sizing
              if sizing[a]['bias'] in ('LONG', 'SHORT')
              and sizing[a]['position_pct'] > 0]

    already_reduced = set()

    for i, a1 in enumerate(active):
        for a2 in active[i+1:]:
            pair_key = tuple(sorted([a1, a2]))
            if pair_key in already_reduced:
                continue

            if a1 not in corr_matrix.index or \
               a2 not in corr_matrix.index:
                continue

            corr = float(corr_matrix.loc[a1, a2])

            # Only warn if both are in the SAME direction
            same_dir = (sizing[a1]['bias'] ==
                        sizing[a2]['bias'])

            if corr > CORR_HIGH_THRESHOLD and same_dir:
                factor = 1.0 - CORR_REDUCTION
                pair_label = f"{a1}/{a2}"
                for a in [a1, a2]:
                    orig = sizing[a]['adj_position']
                    sizing[a]['adj_position'] = round(
                        orig * factor, 2
                    )
                    sizing[a]['corr_flag']  = 'HIGH_CORR_WARNING'
                    sizing[a]['corr_pair']  = pair_label
                    sizing[a]['corr_value'] = round(corr, 3)
                already_reduced.add(pair_key)

            elif corr < CORR_LOW_THRESHOLD and same_dir:
                # Truly diversified — note but no reduction
                for a in [a1, a2]:
                    if sizing[a]['corr_flag'] == 'OK':
                        sizing[a]['corr_flag'] = 'DIVERSIFIED'
                        sizing[a]['corr_pair'] = (
                            f"{a1}/{a2}"
                        )
                        sizing[a]['corr_value'] = round(
                            corr, 3
                        )

    return sizing


# ═════════════════════════════════════════════════════════════
# SECTION 5 — COMPONENT 3: PORTFOLIO DRAWDOWN ESTIMATOR
# ═════════════════════════════════════════════════════════════

def estimate_portfolio_drawdown(sizing, corr_matrix):
    """
    Portfolio max-DD estimate uses asset DDs and average
    pairwise correlation to apply a diversification benefit.

    Formula:
      portfolio_var = Σ (w_i × σ_i)² +
                      Σ_{i≠j} w_i × σ_i × w_j × σ_j × ρ_ij
      portfolio_vol = √portfolio_var
      max_DD_est    = portfolio_vol × √(2 × ln(n_periods)) / √252
                      (approximation for geometric BM)

    Simplified practical version:
      portfolio_vol = √( Σ Σ w_i w_j σ_i σ_j ρ_ij )
      max_DD        = portfolio_vol × 1.65   (95th pct)
    """
    active = {a: s for a, s in sizing.items()
              if s['bias'] in ('LONG', 'SHORT')
              and s['adj_position'] > 0}

    if not active:
        return 0.0, 'NONE'

    # weights as fractions of total portfolio
    weights = {a: s['adj_position'] / 100.0
               for a, s in active.items()}

    # annualised vol per asset as fraction
    vols = {a: s['daily_vol_pct'] / 100.0
            for a, s in active.items()}

    # Build covariance matrix subset
    assets_list = list(active.keys())
    n = len(assets_list)
    port_var = 0.0

    for i, a1 in enumerate(assets_list):
        for j, a2 in enumerate(assets_list):
            w1  = weights.get(a1, 0.0)
            w2  = weights.get(a2, 0.0)
            s1  = vols.get(a1, 0.15)
            s2  = vols.get(a2, 0.15)

            if (not corr_matrix.empty
                    and a1 in corr_matrix.index
                    and a2 in corr_matrix.index):
                rho = float(corr_matrix.loc[a1, a2])
            else:
                rho = 1.0 if i == j else 0.3  # assume mild corr

            port_var += w1 * w2 * s1 * s2 * rho

    port_vol = np.sqrt(max(port_var, 0.0))  # annualised

    # Expected max DD ≈ portfolio vol × 1.65 (95th pct Gaussian)
    max_dd_pct = port_vol * 1.65 * 100.0

    if max_dd_pct >= DD_CRITICAL_PCT:
        level = 'CRITICAL'
    elif max_dd_pct >= DD_WARNING_PCT:
        level = 'WARNING'
    else:
        level = 'OK'

    return round(max_dd_pct, 2), level


# ═════════════════════════════════════════════════════════════
# SECTION 6 — DERIVE RECOMMENDED ACTION
# ═════════════════════════════════════════════════════════════

def recommended_action(s, dd_level):
    """Plain-English recommended action for one asset."""
    bias = s['bias']
    pos  = s['adj_position']
    flag = s['corr_flag']

    if bias == 'NO TRADE' or pos == 0:
        return 'WAIT — no active signal'

    if dd_level == 'CRITICAL':
        return (f'{bias} {pos:.1f}% → ⚠️ CUT TO '
                f'{pos*0.5:.1f}% (portfolio DD critical)')

    parts = [f'{bias} {pos:.1f}% of portfolio']

    if flag == 'HIGH_CORR_WARNING':
        parts.append(
            f'(reduced from {s["position_pct"]:.1f}% '
            f'— high corr with {s["corr_pair"]})'
        )
    elif flag == 'DIVERSIFIED':
        parts.append('(full Kelly — diversified)')

    return ' '.join(parts)


# ═════════════════════════════════════════════════════════════
# SECTION 7 — SAVE TO DATABASE
# ═════════════════════════════════════════════════════════════

def save_portfolio_risk(conn, sizing, portfolio_dd, dd_level):
    today = datetime.now().strftime('%Y-%m-%d')

    try:
        conn.execute(
            "DELETE FROM PORTFOLIO_RISK WHERE date = ?",
            (today,)
        )
    except Exception:
        pass

    rows = []
    for asset, s in sizing.items():
        action = recommended_action(s, dd_level)
        rows.append({
            'date':          today,
            'asset':         asset,
            'bias':          s.get('bias', 'NO TRADE'),
            'kelly_raw':     s.get('kelly_raw', 0),
            'kelly_half':    s.get('kelly_half', 0),
            'position_pct':  s.get('position_pct', 0),
            'corr_flag':     s.get('corr_flag', 'OK'),
            'corr_pair':     s.get('corr_pair', ''),
            'corr_value':    s.get('corr_value', 0),
            'adj_position':  s.get('adj_position', 0),
            'daily_vol_pct': s.get('daily_vol_pct', 0),
            'atr_pct':       s.get('atr_pct', 0),
            'max_dd_est':    s.get('max_dd_est', 0),
            'portfolio_dd':  portfolio_dd,
            'dd_level':      dd_level,
            'action':        action,
        })

    if rows:
        df = pd.DataFrame(rows)
        try:
            df.to_sql('PORTFOLIO_RISK', conn,
                      if_exists='append', index=False)
        except Exception as e:
            print(f"  ❌ Save failed: {e}")
            df.to_sql('PORTFOLIO_RISK', conn,
                      if_exists='replace', index=False)
        conn.commit()
        print(f"  ✅ Portfolio risk saved ({len(rows)} assets)")


# ═════════════════════════════════════════════════════════════
# SECTION 8 — PRINT REPORT
# ═════════════════════════════════════════════════════════════

def _dd_bar(pct, width=20):
    """Visual bar for drawdown level."""
    filled = int(min(pct / 30 * width, width))
    bar    = '█' * filled + '░' * (width - filled)
    return f'[{bar}] {pct:.1f}%'


def print_report(sizing, corr_matrix, portfolio_dd, dd_level):
    print("\n" + "="*75)
    print("PORTFOLIO RISK ENGINE — DAILY SUMMARY")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*75)

    # ── Kelly sizing table ───────────────────────────────────
    print("\n📐 COMPONENT 1+2 — KELLY SIZING & CORRELATION RISK")
    print("-"*75)
    header = (f"{'Asset':<8} {'Bias':<10} {'Kelly½%':>7} "
              f"{'AdjSize%':>8} {'AdjSize':>9} "
              f"{'ATR%':>6} {'Vol%':>6}  Corr Flag")
    print(header)
    print("-"*75)

    active_count = 0
    for asset in ASSETS:
        s    = sizing.get(asset, {})
        bias = s.get('bias', 'NO TRADE')
        k2   = s.get('kelly_half', 0)
        adj  = s.get('adj_position', 0)
        atr  = s.get('atr_pct', 0)
        vol  = s.get('daily_vol_pct', 0)
        flag = s.get('corr_flag', 'OK')
        pair = s.get('corr_pair', '')
        corr = s.get('corr_value', 0)

        if bias in ('LONG', 'SHORT') and adj > 0:
            active_count += 1
            b_emoji = '🟢' if bias == 'LONG' else '🔴'
        else:
            b_emoji = '⬜'

        corr_note = ''
        if flag == 'HIGH_CORR_WARNING':
            corr_note = f'⚠️  HIGH CORR {pair} ({corr:.2f})'
        elif flag == 'DIVERSIFIED':
            corr_note = f'✅ DIVERSIFIED {pair} ({corr:.2f})'

        print(f"{b_emoji} {asset:<7} {bias:<10} {k2:>7.1f}%"
              f" {adj:>8.1f}%  "
              f"{'█' * int(adj/2):<9}  "
              f"{atr:>5.2f}% {vol:>5.1f}%  {corr_note}")

    print("-"*75)
    print(f"  Active positions: {active_count} / {len(ASSETS)}")

    # ── Correlation matrix ───────────────────────────────────
    if not corr_matrix.empty:
        print("\n🔗 COMPONENT 2 — 60-DAY CORRELATION MATRIX")
        print("-"*50)
        cols = [a for a in ASSETS if a in corr_matrix.columns]
        header_row = f"{'':8}" + "".join(f"{c:>8}" for c in cols)
        print(header_row)
        for r in cols:
            row_str = f"{r:<8}"
            for c in cols:
                v    = corr_matrix.loc[r, c]
                mark = ''
                if r != c and abs(v) > CORR_HIGH_THRESHOLD:
                    mark = '!'
                row_str += f"{v:>7.2f}{mark}"
            print(row_str)
        print("  ! = correlation > 0.70 (high)")

    # ── Drawdown ─────────────────────────────────────────────
    print("\n📉 COMPONENT 3 — PORTFOLIO DRAWDOWN ESTIMATE")
    print("-"*50)
    dd_emoji = {'OK': '✅', 'WARNING': '⚠️',
                'CRITICAL': '🚨'}.get(dd_level, '❓')
    print(f"  Expected Max Drawdown: {dd_emoji} "
          f"{_dd_bar(portfolio_dd)}")
    print(f"  Status: {dd_level}")

    if dd_level == 'CRITICAL':
        print(f"\n  🚨 CRITICAL: Portfolio DD above "
              f"{DD_CRITICAL_PCT}%")
        print(f"     RECOMMENDATION: Cut all positions by 50%")
    elif dd_level == 'WARNING':
        print(f"\n  ⚠️  WARNING: Portfolio DD above "
              f"{DD_WARNING_PCT}%")
        print(f"     Review position sizes carefully")

    # ── Action summary ───────────────────────────────────────
    print("\n📋 COMPONENT 4 — RECOMMENDED ACTIONS")
    print("-"*75)
    for asset in ASSETS:
        s      = sizing.get(asset, {})
        bias   = s.get('bias', 'NO TRADE')
        action = recommended_action(s, dd_level)
        icon   = '🟢' if bias == 'LONG' else (
                 '🔴' if bias == 'SHORT' else '⬜')
        print(f"  {icon} {asset:<8}: {action}")

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


def build_telegram_message(sizing, portfolio_dd, dd_level):
    date  = datetime.now().strftime('%d %b %Y %H:%M')
    dd_emoji = {'OK': '✅', 'WARNING': '⚠️',
                'CRITICAL': '🚨'}.get(dd_level, '❓')
    lines = [
        f"🎯 <b>GMIS PORTFOLIO RISK ENGINE</b>",
        f"{date}",
        f"{'─' * 30}",
        "",
        f"<b>Position Sizing (Half-Kelly)</b>",
    ]

    for asset in ASSETS:
        s    = sizing.get(asset, {})
        bias = s.get('bias', 'NO TRADE')
        adj  = s.get('adj_position', 0)
        flag = s.get('corr_flag', 'OK')

        if bias == 'NO TRADE' or adj == 0:
            lines.append(f"  ⬜ {asset}: NO TRADE")
            continue

        icon = '🟢' if bias == 'LONG' else '🔴'
        corr_note = ''
        if flag == 'HIGH_CORR_WARNING':
            corr_note = f' ⚠️ high corr {s["corr_pair"]}'

        lines.append(
            f"  {icon} <b>{asset}</b>: {bias} "
            f"<b>{adj:.1f}%</b>{corr_note}"
        )

    lines.append("")
    lines.append(
        f"{dd_emoji} <b>Portfolio Max DD: "
        f"{portfolio_dd:.1f}% — {dd_level}</b>"
    )

    if dd_level == 'CRITICAL':
        lines.append(
            f"🚨 <b>CRITICAL: Cut all positions by 50%</b>"
        )
    elif dd_level == 'WARNING':
        lines.append(
            f"⚠️ <b>WARNING: Review position sizes</b>"
        )

    lines.append("")
    lines.append("<i>GMIS Portfolio Risk Engine</i>")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_portfolio_risk(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 27 — PORTFOLIO RISK ENGINE")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    # ── Load source data ─────────────────────────────────────
    print("\nLoading source data...")
    decisions = load_decisions(conn)
    hit_rates = load_hit_rates(conn)
    analogs   = load_analog_outcomes(conn)

    if not decisions:
        print("  ⚠️  No decisions found — run 19_decision_engine.py")
        conn.close()
        return

    print(f"  Decisions: {len(decisions)} assets")
    print(f"  Hit rates: "
          + ", ".join(f"{a}={v:.2f}"
                      for a, v in hit_rates.items()))

    # ── Load price series for all assets ─────────────────────
    print("\nLoading price data...")
    price_data = {}
    for asset in ASSETS:
        df = load_price_series(asset)
        if not df.empty:
            price_data[asset] = df
            print(f"  {asset}: {len(df)} rows, "
                  f"latest {df.index[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"  ⚠️  {asset}: no data")

    # ── COMPONENT 1: Kelly sizing ─────────────────────────────
    print("\nComponent 1 — Kelly Criterion sizing...")
    sizing = {}
    for asset in ASSETS:
        d    = decisions.get(asset, {})
        bias = d.get('bias', 'NO TRADE')

        base = {
            'asset':         asset,
            'bias':          bias,
            'kelly_raw':     0.0,
            'kelly_half':    0.0,
            'position_pct':  0.0,
            'atr_pct':       0.0,
            'daily_vol_pct': 0.0,
            'max_dd_est':    0.0,
        }

        if bias not in ('LONG', 'SHORT'):
            sizing[asset] = base
            print(f"  {asset}: NO TRADE — position = 0%")
            continue

        df     = price_data.get(asset, pd.DataFrame())
        hr     = hit_rates.get(asset, 0.5)
        analog = analogs.get(asset)

        s = compute_kelly_for_asset(
            asset, bias, hr, analog, df
        )

        print(f"  {asset}: {bias} | p={hr:.2f} | "
              f"Kelly½={s['kelly_half']:.1f}% → "
              f"pos={s['position_pct']:.1f}% | "
              f"ATR={s['atr_pct']:.2f}%")

        sizing[asset] = s

    # ── COMPONENT 2: Correlation adjustments ─────────────────
    print("\nComponent 2 — Correlation matrix (60d)...")
    corr_matrix = build_corr_matrix()

    if not corr_matrix.empty:
        active_assets = [a for a in ASSETS
                         if sizing[a]['bias'] in ('LONG','SHORT')]
        for i, a1 in enumerate(active_assets):
            for a2 in active_assets[i+1:]:
                if (a1 in corr_matrix.index and
                        a2 in corr_matrix.index):
                    corr = corr_matrix.loc[a1, a2]
                    print(f"  Corr({a1},{a2}) = {corr:.2f}")

    sizing = apply_correlation_adjustments(sizing, corr_matrix)

    for asset, s in sizing.items():
        if s['corr_flag'] == 'HIGH_CORR_WARNING':
            print(f"  ⚠️  {asset}: HIGH CORR with "
                  f"{s['corr_pair']} ({s['corr_value']:.2f}) "
                  f"→ size cut to {s['adj_position']:.1f}%")

    # ── COMPONENT 3: Portfolio drawdown ───────────────────────
    print("\nComponent 3 — Portfolio drawdown estimate...")
    portfolio_dd, dd_level = estimate_portfolio_drawdown(
        sizing, corr_matrix
    )
    print(f"  Expected Max DD: {portfolio_dd:.1f}% — {dd_level}")

    if dd_level == 'CRITICAL':
        print(f"  🚨 CRITICAL: recommend cutting all positions 50%")
    elif dd_level == 'WARNING':
        print(f"  ⚠️  WARNING: portfolio risk elevated")

    # Back-apply CRITICAL 50% cut to dd estimates on sizing
    for s in sizing.values():
        s['portfolio_dd'] = portfolio_dd

    # ── Save ─────────────────────────────────────────────────
    print("\nSaving portfolio risk...")
    save_portfolio_risk(conn, sizing, portfolio_dd, dd_level)
    conn.close()

    # ── Print report ─────────────────────────────────────────
    print_report(sizing, corr_matrix, portfolio_dd, dd_level)

    # ── Telegram ─────────────────────────────────────────────
    has_active = any(
        s['bias'] in ('LONG', 'SHORT') and s['adj_position'] > 0
        for s in sizing.values()
    )

    if send_telegram_flag and BOT_TOKEN and has_active:
        print("\nActive positions detected — sending Telegram...")
        msg = build_telegram_message(
            sizing, portfolio_dd, dd_level
        )
        asyncio.run(_send_telegram(msg))
    elif send_telegram_flag and BOT_TOKEN and dd_level in (
            'WARNING', 'CRITICAL'):
        print("\nDD alert — sending Telegram...")
        msg = build_telegram_message(
            sizing, portfolio_dd, dd_level
        )
        asyncio.run(_send_telegram(msg))
    elif not send_telegram_flag:
        print("\n  Telegram skipped (--no-telegram)")
    else:
        print("\n  No active positions — no Telegram")

    return {
        'sizing':       sizing,
        'corr_matrix':  corr_matrix,
        'portfolio_dd': portfolio_dd,
        'dd_level':     dd_level,
    }


if __name__ == "__main__":
    no_telegram = '--no-telegram' in sys.argv
    run_portfolio_risk(send_telegram_flag=not no_telegram)
