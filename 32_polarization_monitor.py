# ============================================================
# GMIS 2.0 — MODULE 32 — POLARIZATION MONITOR
#
# COMPONENT 1 — Mag-7 vs Broad Market (Russell 2000)
#   Equal-weight Mag-7 index vs ^RUT, 20-day rolling correlation
#   Data via yfinance (90-day lookback)
#
# COMPONENT 2 — Breadth Analysis
#   Rolling R² of Mag-7 explaining SP500 daily returns
#   > 60% explained = NARROW (fragile)
#   < 40% explained = BROAD (genuine participation)
#
# COMPONENT 3 — Polarization Regime
#   HEALTHY_BREADTH:    corr > 0.60 AND explained < 50%
#   MILD_POLARIZATION:  corr 0.30–0.60, or corr > 0.60 but
#                       explained ≥ 50% (high concentration)
#   EXTREME_POLARIZATION: corr < 0.30 (Mag-7 diverging from RUT)
#
# COMPONENT 4 — SP500 Signal Adjustment
#   HEALTHY:  +0.05  MILD: 0.00  EXTREME: -0.15
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
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

# ── Tickers ───────────────────────────────────────────────────
MAG7_TICKERS  = ['AAPL', 'MSFT', 'NVDA', 'GOOGL',
                  'AMZN', 'META', 'TSLA']
SP500_TICKER  = '^GSPC'
RUT_TICKER    = '^RUT'
LOOKBACK_DAYS = '90d'       # enough for 60+ trading days
ROLL_WINDOW   = 20          # rolling window for all metrics

# ── Regime thresholds ─────────────────────────────────────────
CORR_HEALTHY   = 0.60
CORR_EXTREME   = 0.30
EXPL_NARROW    = 0.60    # > 60% Mag-7 explaining SP500 = NARROW
EXPL_BROAD     = 0.40    # < 40% = BROAD

# ── Signal adjustments ────────────────────────────────────────
ADJ_HEALTHY   = +0.05
ADJ_MILD      =  0.00
ADJ_EXTREME   = -0.15


# ═════════════════════════════════════════════════════════════
# SECTION 1 — DATABASE
# ═════════════════════════════════════════════════════════════

def create_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS POLARIZATION_DATA (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            date               TEXT NOT NULL UNIQUE,

            -- Mag-7 index level and return
            mag7_index         REAL,
            mag7_return        REAL,

            -- Individual Mag-7 returns (for diagnostics)
            ret_AAPL           REAL, ret_MSFT  REAL,
            ret_NVDA           REAL, ret_GOOGL REAL,
            ret_AMZN           REAL, ret_META  REAL,
            ret_TSLA           REAL,

            -- SP500 and Russell 2000
            sp500_return       REAL,
            rut_return         REAL,

            -- Rolling metrics
            corr_mag7_rut_20d  REAL,
            mag7_expl_sp500    REAL,    -- R² (20-day rolling)
            mag7_sp500_spread  REAL,    -- cumulative 20d performance gap

            -- Regime
            breadth_regime     TEXT,
            concentration_flag TEXT,    -- NARROW / MODERATE / BROAD
            sp500_adjustment   REAL,
            combined_signal    TEXT,

            -- Change detection
            prev_regime        TEXT
        )
    ''')
    conn.commit()


# ═════════════════════════════════════════════════════════════
# SECTION 2 — DOWNLOAD DATA
# ═════════════════════════════════════════════════════════════

def download_data():
    """
    Download Mag-7, SP500, Russell 2000 via yfinance.
    Returns dict of Close price Series.
    """
    all_tickers = MAG7_TICKERS + [SP500_TICKER, RUT_TICKER]
    try:
        df = yf.download(
            all_tickers,
            period=LOOKBACK_DAYS,
            auto_adjust=True,
            progress=False,
        )['Close']

        # yfinance may return MultiIndex or single-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(how='all')
        return df
    except Exception as e:
        print(f"  ❌ yfinance download failed: {e}")
        return pd.DataFrame()


# ═════════════════════════════════════════════════════════════
# SECTION 3 — COMPONENT 1: MAG-7 INDEX & CORRELATION
# ═════════════════════════════════════════════════════════════

def build_mag7_index(prices: pd.DataFrame) -> pd.Series:
    """
    Equal-weight Mag-7 index, normalised to 100 on first day.
    """
    mag7 = prices[MAG7_TICKERS].copy()
    mag7_norm  = mag7.div(mag7.iloc[0]) * 100
    mag7_index = mag7_norm.mean(axis=1)
    return mag7_index


def rolling_corr_mag7_rut(mag7_ret: pd.Series,
                           rut_ret: pd.Series,
                           window: int = ROLL_WINDOW) -> pd.Series:
    return mag7_ret.rolling(window).corr(rut_ret)


# ═════════════════════════════════════════════════════════════
# SECTION 4 — COMPONENT 2: BREADTH (EXPLAINED %)
# ═════════════════════════════════════════════════════════════

def rolling_r2_mag7_sp500(sp500_ret: pd.Series,
                           mag7_ret: pd.Series,
                           window: int = ROLL_WINDOW) -> pd.Series:
    """
    Rolling window OLS R² of mag7_ret explaining sp500_ret.
    R² = 1 - SS_res / SS_tot
    """
    results = []
    index   = []

    for i in range(window, len(sp500_ret) + 1):
        sp = sp500_ret.iloc[i - window:i].values
        m7 = mag7_ret.iloc[i - window:i].values

        ss_tot = np.sum((sp - sp.mean()) ** 2)
        if ss_tot < 1e-12:
            results.append(np.nan)
        else:
            # OLS fitted values: m7 * beta + intercept
            m7_col = np.column_stack([m7, np.ones(window)])
            coeffs, _, _, _ = np.linalg.lstsq(m7_col, sp,
                                               rcond=None)
            fitted = m7_col @ coeffs
            ss_res = np.sum((sp - fitted) ** 2)
            r2     = 1.0 - ss_res / ss_tot
            results.append(max(0.0, r2))

        index.append(sp500_ret.index[i - 1])

    return pd.Series(results, index=index)


def rolling_performance_spread(mag7_ret: pd.Series,
                                sp500_ret: pd.Series,
                                window: int = ROLL_WINDOW
                                ) -> pd.Series:
    """
    Cumulative 20-day return of Mag-7 minus SP500.
    Positive = Mag-7 outperforming (concentration building).
    """
    mag7_cum  = (1 + mag7_ret).rolling(window).apply(
        np.prod) - 1
    sp500_cum = (1 + sp500_ret).rolling(window).apply(
        np.prod) - 1
    return (mag7_cum - sp500_cum) * 100   # in %


# ═════════════════════════════════════════════════════════════
# SECTION 5 — COMPONENT 3: POLARIZATION REGIME
# ═════════════════════════════════════════════════════════════

def classify_regime(corr: float,
                    r2: float) -> tuple[str, str, str]:
    """
    Returns (regime, concentration_flag, description).

    Regime logic (spec-aligned):
      HEALTHY_BREADTH:      corr > 0.60 AND r2 < 0.50
      EXTREME_POLARIZATION: corr < 0.30
      MILD_POLARIZATION:    everything else
        (includes: corr > 0.60 but r2 ≥ 0.50 — high concentration)
    """
    if corr < CORR_EXTREME:
        regime = 'EXTREME_POLARIZATION'
        desc   = (f'Mag-7 and Russell 2000 diverging '
                  f'(corr={corr:.2f} < {CORR_EXTREME}) — '
                  f'AI bubble risk, rally driven by handful '
                  f'of mega-caps, not the economy')
    elif corr >= CORR_HEALTHY and r2 < 0.50:
        regime = 'HEALTHY_BREADTH'
        desc   = (f'Broad participation '
                  f'(corr={corr:.2f} ≥ {CORR_HEALTHY}, '
                  f'Mag-7 explains only {r2*100:.0f}% of SP500) — '
                  f'genuine economic participation, '
                  f'SP500 Longs reliable')
    else:
        regime = 'MILD_POLARIZATION'
        if corr >= CORR_HEALTHY and r2 >= 0.50:
            desc = (f'Correlated but concentrated '
                    f'(corr={corr:.2f} is healthy but Mag-7 '
                    f'explains {r2*100:.0f}% of SP500) — '
                    f'caution: index returns dominated by few names')
        else:
            desc = (f'Moderate polarization '
                    f'(corr={corr:.2f}) — '
                    f'caution on SP500 Longs')

    # Concentration flag (independent of correlation)
    if r2 >= EXPL_NARROW:
        conc = f'NARROW (Mag-7 → {r2*100:.0f}% of SP500 variance)'
    elif r2 <= EXPL_BROAD:
        conc = f'BROAD (Mag-7 → {r2*100:.0f}% of SP500 variance)'
    else:
        conc = f'MODERATE (Mag-7 → {r2*100:.0f}% of SP500 variance)'

    return regime, conc, desc


# ═════════════════════════════════════════════════════════════
# SECTION 6 — COMPONENT 4: SIGNAL ADJUSTMENT
# ═════════════════════════════════════════════════════════════

def compute_adjustment(regime: str,
                       r2: float,
                       corr: float) -> tuple[float, str]:
    """
    Returns (adjustment, signal_text).
    """
    if regime == 'HEALTHY_BREADTH':
        adj = ADJ_HEALTHY
        sig = ('HEALTHY_BREADTH: broad participation confirmed — '
               f'+{ADJ_HEALTHY:.2f} boost to SP500 Long signals')
    elif regime == 'EXTREME_POLARIZATION':
        adj = ADJ_EXTREME
        sig = ('⚠️  EXTREME_POLARIZATION: AI/mega-cap bubble risk — '
               f'{ADJ_EXTREME:.2f} reduction on SP500 Long signals; '
               f'rally not confirmed by small-cap breadth')
    else:
        adj = ADJ_MILD
        # Add extra context for high-concentration mild case
        if r2 >= EXPL_NARROW:
            sig = ('MILD_POLARIZATION (high concentration): '
                   f'Mag-7 drives {r2*100:.0f}% of SP500 — '
                   'no adjustment but monitor for extreme pivot')
        else:
            sig = ('MILD_POLARIZATION: moderate breadth — '
                   'no adjustment applied')
    return adj, sig


# ═════════════════════════════════════════════════════════════
# SECTION 7 — SAVE & LOAD
# ═════════════════════════════════════════════════════════════

def load_prev_regime(conn, today):
    try:
        rows = conn.execute(
            "SELECT breadth_regime FROM POLARIZATION_DATA "
            "WHERE date < ? ORDER BY date DESC LIMIT 1",
            (today,)
        ).fetchall()
        return rows[0][0] if rows else None
    except Exception:
        return None


def save_results(conn, today, prices, mag7_index,
                 sp500_ret, rut_ret, mag7_ret,
                 corr_val, r2_val, spread_val,
                 regime, conc_flag, adj, sig, prev_regime):
    # Individual Mag-7 returns (today)
    ind_rets = {}
    for t in MAG7_TICKERS:
        try:
            ind_rets[f'ret_{t}'] = float(
                prices[t].pct_change().iloc[-1]
            )
        except Exception:
            ind_rets[f'ret_{t}'] = None

    row = {
        'date':               today,
        'mag7_index':         round(float(mag7_index.iloc[-1]), 4),
        'mag7_return':        round(float(mag7_ret.iloc[-1]), 6),
        **ind_rets,
        'sp500_return':       round(float(sp500_ret.iloc[-1]), 6),
        'rut_return':         round(float(rut_ret.iloc[-1]), 6),
        'corr_mag7_rut_20d':  round(corr_val, 4),
        'mag7_expl_sp500':    round(r2_val, 4),
        'mag7_sp500_spread':  round(spread_val, 4)
            if spread_val is not None else None,
        'breadth_regime':     regime,
        'concentration_flag': conc_flag,
        'sp500_adjustment':   adj,
        'combined_signal':    sig,
        'prev_regime':        prev_regime,
    }

    try:
        conn.execute(
            "DELETE FROM POLARIZATION_DATA WHERE date=?",
            (today,)
        )
        pd.DataFrame([row]).to_sql(
            'POLARIZATION_DATA', conn,
            if_exists='append', index=False
        )
        conn.commit()
        print(f"  ✅ Polarization data saved "
              f"(regime={regime})")
    except Exception as e:
        print(f"  ❌ Save failed: {e}")


# ═════════════════════════════════════════════════════════════
# SECTION 8 — PRINT REPORT
# ═════════════════════════════════════════════════════════════

def _regime_emoji(regime):
    return {
        'HEALTHY_BREADTH':      '🟢',
        'MILD_POLARIZATION':    '🟡',
        'EXTREME_POLARIZATION': '🔴',
    }.get(regime, '❓')


def _bar(val, lo, hi, width=20):
    """Horizontal bar scaled between lo and hi."""
    frac = (val - lo) / (hi - lo) if hi != lo else 0.5
    frac = max(0.0, min(1.0, frac))
    n    = int(frac * width)
    return '[' + '█' * n + '░' * (width - n) + ']'


def print_report(prices, mag7_index, mag7_ret, sp500_ret,
                 rut_ret, corr_series, r2_series, spread_series,
                 regime, regime_desc, conc_flag, adj, sig,
                 prev_regime, regime_changed):

    corr_now   = float(corr_series.iloc[-1])
    r2_now     = float(r2_series.iloc[-1])
    spread_now = float(spread_series.iloc[-1]) \
                 if not spread_series.isna().iloc[-1] else 0.0

    print("\n" + "="*70)
    print("POLARIZATION MONITOR — DAILY REPORT")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*70)

    # ── Banner ────────────────────────────────────────────────
    e = _regime_emoji(regime)
    print(f"\n  {e}  POLARIZATION REGIME: {regime}")
    print(f"      {regime_desc}")
    if regime_changed:
        print(f"\n  ⚡ REGIME CHANGE: {prev_regime} → {regime}")

    # ── Component 1: Mag-7 index & correlation ────────────────
    print("\n📊 COMPONENT 1 — MAG-7 INDEX vs RUSSELL 2000")
    print("-"*60)

    # Individual Mag-7 returns today
    today_rets = {}
    for t in MAG7_TICKERS:
        today_rets[t] = float(prices[t].pct_change().iloc[-1]) * 100

    mag7_today = float(mag7_ret.iloc[-1]) * 100
    sp_today   = float(sp500_ret.iloc[-1]) * 100
    rut_today  = float(rut_ret.iloc[-1]) * 100

    print(f"  Today's returns:")
    for t, r in today_rets.items():
        bar = ('▲' if r > 0 else '▼')
        print(f"    {t:<6} {r:>+7.2f}%  {bar}")

    print(f"\n  Equal-weight Mag-7: {mag7_today:>+7.2f}%")
    print(f"  SP500 (^GSPC):      {sp_today:>+7.2f}%")
    print(f"  Russell 2000 (^RUT):{rut_today:>+7.2f}%")

    print(f"\n  Rolling 20-day Mag-7 ↔ Russell 2000 correlation:")
    print(f"  {corr_now:.3f}  {_bar(corr_now, -1, 1, 24)}")
    print(f"  Thresholds: "
          f"EXTREME < {CORR_EXTREME} | "
          f"HEALTHY > {CORR_HEALTHY}")

    # ── Component 2: Breadth ──────────────────────────────────
    print("\n📈 COMPONENT 2 — BREADTH ANALYSIS")
    print("-"*60)
    print(f"  Mag-7 explains {r2_now*100:.1f}% of SP500 20-day variance (R²)")
    print(f"  {_bar(r2_now, 0, 1, 24)}")
    print(f"  Thresholds: BROAD < {EXPL_BROAD*100:.0f}% | "
          f"NARROW > {EXPL_NARROW*100:.0f}%")
    print(f"  Concentration: {conc_flag}")

    print(f"\n  Mag-7 vs SP500 cumulative 20-day spread:")
    spread_label = (
        f"Mag-7 outperforming by {spread_now:+.1f}%"
        if spread_now > 0 else
        f"Mag-7 underperforming by {spread_now:.1f}%"
    )
    print(f"  {spread_label}")

    # 5-day R² trend
    if len(r2_series.dropna()) >= 5:
        r2_5d = r2_series.dropna().tail(5)
        trend = ("increasing" if r2_5d.iloc[-1] > r2_5d.iloc[0]
                 else "decreasing")
        print(f"\n  R² 5-day trend ({trend}):")
        for dt, v in r2_5d.items():
            bar = '█' * int(v * 10)
            print(f"    {dt.strftime('%d %b')}: "
                  f"{v*100:>5.1f}%  {bar}")

    # ── Component 3: Regime ───────────────────────────────────
    print("\n🎯 COMPONENT 3 — POLARIZATION REGIME")
    print("-"*60)
    regimes = [
        ('HEALTHY_BREADTH',
         f'corr > {CORR_HEALTHY} AND explained < 50%',
         'SP500 Longs reliable'),
        ('MILD_POLARIZATION',
         f'corr {CORR_EXTREME}–{CORR_HEALTHY} or high concentration',
         'Caution on SP500 Longs'),
        ('EXTREME_POLARIZATION',
         f'corr < {CORR_EXTREME}',
         'AI bubble risk — reduce SP500 confidence'),
    ]
    for r_code, r_range, r_desc in regimes:
        marker = '► ' if r_code == regime else '  '
        e2     = _regime_emoji(r_code)
        print(f"  {marker}{e2} {r_code:<24} ({r_range})")
        if r_code == regime:
            print(f"     → {r_desc}")

    # ── Component 4: Adjustment ───────────────────────────────
    print("\n⚙️  COMPONENT 4 — SP500 SIGNAL ADJUSTMENT")
    print("-"*60)
    if adj > 0:
        label = f"✅ BOOST      ({adj:+.2f})"
    elif adj < 0:
        label = f"⚠️  REDUCTION  ({adj:+.2f})"
    else:
        label = f"⚪ NONE        (0.00)"
    print(f"  Adjustment: {label}")
    print(f"  {sig}")

    print("\n" + "="*70)


# ═════════════════════════════════════════════════════════════
# SECTION 9 — TELEGRAM
# ═════════════════════════════════════════════════════════════

async def _send_telegram(msg):
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID, text=msg, parse_mode='HTML')
        print("  ✅ Telegram sent")
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")


def build_telegram_message(regime, regime_desc, corr_val,
                            r2_val, adj, sig, prev_regime,
                            regime_changed):
    date = datetime.now().strftime('%d %b %Y %H:%M')
    e    = _regime_emoji(regime)
    lines = [
        f"🧩 <b>GMIS POLARIZATION MONITOR</b>",
        f"{date}",
        f"{'─' * 30}",
        "",
    ]

    if regime == 'EXTREME_POLARIZATION':
        lines.append("🚨 <b>AI BUBBLE RISK — NARROW MARKET</b>")
        lines.append("")

    lines += [
        f"{e} Regime: <b>{regime}</b>",
        f"   {regime_desc[:90]}",
        "",
        f"📊 Mag-7 ↔ Russell 2000 corr: <b>{corr_val:.3f}</b>",
        f"📈 Mag-7 explains SP500: <b>{r2_val*100:.1f}%</b>",
        "",
        f"⚙️  SP500 Adjustment: <b>{adj:+.2f}</b>",
        f"   {sig[:100]}",
    ]

    if regime_changed:
        lines += [
            "",
            f"⚡ <b>REGIME CHANGE: {prev_regime} → {regime}</b>",
        ]

    lines += ["", "<i>GMIS Polarization Monitor</i>"]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_polarization_monitor(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 32 — POLARIZATION MONITOR")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    conn  = sqlite3.connect(DB_PATH)
    create_table(conn)
    today = datetime.now().strftime('%Y-%m-%d')

    prev_regime = load_prev_regime(conn, today)

    # ── Download data ─────────────────────────────────────────
    print("\nDownloading market data via yfinance...")
    prices = download_data()

    if prices.empty:
        print("  ❌ No data downloaded — aborting")
        conn.close()
        return None

    # Verify all required columns
    required = MAG7_TICKERS + [SP500_TICKER, RUT_TICKER]
    missing  = [t for t in required if t not in prices.columns]
    if missing:
        print(f"  ❌ Missing tickers: {missing}")
        conn.close()
        return None

    print(f"  {len(prices)} trading days downloaded "
          f"({prices.index[0].date()} → "
          f"{prices.index[-1].date()})")

    # ── Component 1: Index & correlation ─────────────────────
    print("\nComponent 1 — Building Mag-7 index...")
    mag7_index = build_mag7_index(prices)
    mag7_ret   = mag7_index.pct_change().dropna()
    sp500_ret  = prices[SP500_TICKER].pct_change().dropna()
    rut_ret    = prices[RUT_TICKER].pct_change().dropna()

    # Align all on common index
    common = mag7_ret.index.intersection(
        sp500_ret.index).intersection(rut_ret.index)
    mag7_ret  = mag7_ret.loc[common]
    sp500_ret = sp500_ret.loc[common]
    rut_ret   = rut_ret.loc[common]

    corr_series = rolling_corr_mag7_rut(mag7_ret, rut_ret)
    corr_val    = float(corr_series.dropna().iloc[-1])
    print(f"  Mag-7 index: "
          f"{float(mag7_index.iloc[-1]):.2f} "
          f"(+100 base on {prices.index[0].date()})")
    print(f"  Mag-7 vs RUT 20d corr: {corr_val:.3f}")

    # ── Component 2: Breadth ──────────────────────────────────
    print("\nComponent 2 — Breadth analysis (rolling R²)...")
    r2_series   = rolling_r2_mag7_sp500(sp500_ret, mag7_ret)
    r2_val      = float(r2_series.dropna().iloc[-1])
    spread_series = rolling_performance_spread(mag7_ret, sp500_ret)
    spread_val  = float(spread_series.dropna().iloc[-1]) \
                  if not spread_series.dropna().empty else 0.0

    print(f"  Mag-7 explains {r2_val*100:.1f}% of SP500 variance")
    print(f"  Mag-7 vs SP500 20d spread: {spread_val:+.2f}%")

    # ── Component 3: Regime ───────────────────────────────────
    print("\nComponent 3 — Classifying polarization regime...")
    regime, conc_flag, regime_desc = classify_regime(
        corr_val, r2_val)
    print(f"  Regime: {regime}")
    print(f"  Concentration: {conc_flag}")

    regime_changed = (prev_regime is not None and
                      prev_regime != regime)
    if regime_changed:
        print(f"  ⚡ REGIME CHANGE: {prev_regime} → {regime}")

    # ── Component 4: Adjustment ───────────────────────────────
    print("\nComponent 4 — Computing SP500 adjustment...")
    adj, sig = compute_adjustment(regime, r2_val, corr_val)
    print(f"  SP500 adjustment: {adj:+.2f}")
    print(f"  {sig[:80]}")

    # ── Save ─────────────────────────────────────────────────
    print("\nSaving results...")
    save_results(conn, today, prices, mag7_index,
                 sp500_ret, rut_ret, mag7_ret,
                 corr_val, r2_val, spread_val,
                 regime, conc_flag, adj, sig, prev_regime)

    # ── Print full report ─────────────────────────────────────
    print_report(prices, mag7_index, mag7_ret, sp500_ret,
                 rut_ret, corr_series, r2_series, spread_series,
                 regime, regime_desc, conc_flag, adj, sig,
                 prev_regime, regime_changed)

    conn.close()

    # ── Telegram: fire on EXTREME or regime change ────────────
    if send_telegram_flag and BOT_TOKEN:
        should_send = (
            regime == 'EXTREME_POLARIZATION' or
            (regime_changed and
             regime in ('EXTREME_POLARIZATION',
                        'HEALTHY_BREADTH')) or
            '--force-send' in sys.argv
        )
        if should_send:
            print("\nAlert condition met — sending Telegram...")
            msg = build_telegram_message(
                regime, regime_desc, corr_val, r2_val,
                adj, sig, prev_regime, regime_changed
            )
            asyncio.run(_send_telegram(msg))
        else:
            print(f"\n  No alert condition "
                  f"(regime={regime}) — no Telegram")
    elif not send_telegram_flag:
        print("\n  Telegram skipped (--no-telegram)")

    return {
        'regime':     regime,
        'corr':       corr_val,
        'r2':         r2_val,
        'adjustment': adj,
        'signal':     sig,
    }


if __name__ == "__main__":
    no_telegram = '--no-telegram' in sys.argv
    run_polarization_monitor(send_telegram_flag=not no_telegram)
