# ============================================================
# GMIS 2.0 — MODULE 33 — GEOPOLITICAL RISK PREMIUM MONITOR
#
# COMPONENT 1 — Oil Geopolitical Risk Premium
#   Fair value = 2-year rolling mean + EIA inventory signal
#   EIA WCESTUS1 via EIA API v2 (free, no key needed)
#   Inventory surplus  → fair value lower than spot
#   Inventory deficit  → fair value closer to or above spot
#   Premium = Spot − Fair Value
#
# COMPONENT 2 — Premium Classification (Crude & Gold)
#   LOW_PREMIUM:      < 5%   fundamentally justified
#   MODERATE_PREMIUM: 5–15%  some risk priced in
#   HIGH_PREMIUM:     15–30% significant war/supply premium
#   EXTREME_PREMIUM:  > 30%  sentiment/fear rally
#
# COMPONENT 3 — Gold Geopolitical (Fear) Premium
#   Fair value = MA126 × (1 − real_yield × 0.05)
#   Baseline: 126-day MA (6-month structural trend)
#   Sensitivity: 0.05 (calibrated to post-2020 gold behaviour)
#   Real yield: DFII10 (10Y TIPS, direct from FRED)
#
# COMPONENT 4 — Premium Trend & Signal
#   Expanding → hold; Contracting → exit
#   Telegram on: LOW→HIGH crossing (new risk event)
#                HIGH→MODERATE crossing (de-escalation signal)
# ============================================================

import sqlite3
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import requests
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

# ── Model parameters ─────────────────────────────────────────
CRUDE_FV_WINDOW     = 504      # 2-year rolling mean baseline
CRUDE_INV_SCALE     = 50_000   # kbbl per $5 fair-value adj
CRUDE_INV_ADJ_PER   = 5.0      # $5 per CRUDE_INV_SCALE kbbl

GOLD_MA_WINDOW      = 126      # 6-month structural baseline
GOLD_REAL_SENS      = 0.05     # 5% discount per 1% real yield

# ── Premium thresholds (% of fair value) ─────────────────────
THRESH_EXTREME  = 30.0
THRESH_HIGH     = 15.0
THRESH_MODERATE =  5.0

# ── EIA API ───────────────────────────────────────────────────
EIA_URL         = ('https://api.eia.gov/v2/petroleum/'
                   'sum/sndw/data/')
EIA_SERIES      = 'WCESTUS1'
EIA_LOOKBACK    = 400          # weekly observations (~7.7 years)

FRED_START      = '2015-01-01'

# ── Alert crossing thresholds ─────────────────────────────────
ALERT_ESCALATE   = ('LOW_PREMIUM',      'HIGH_PREMIUM')
ALERT_DEESCALATE = ('HIGH_PREMIUM',     'MODERATE_PREMIUM')
ALERT_EXTREME    = 'EXTREME_PREMIUM'


# ═════════════════════════════════════════════════════════════
# SECTION 1 — DATABASE
# ═════════════════════════════════════════════════════════════

def create_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS GEOPOLITICAL_PREMIUM (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            date                 TEXT NOT NULL UNIQUE,

            -- Crude model
            crude_spot           REAL,
            crude_fair_value     REAL,
            crude_premium_usd    REAL,
            crude_premium_pct    REAL,
            crude_pct_rank       REAL,
            crude_classification TEXT,
            crude_trend          TEXT,

            -- Inventory context
            inv_latest_mbbl      REAL,
            inv_5yr_avg_mbbl     REAL,
            inv_dev_mbbl         REAL,
            inv_signal_usd       REAL,

            -- Gold model
            gold_spot            REAL,
            gold_fair_value      REAL,
            gold_premium_usd     REAL,
            gold_premium_pct     REAL,
            gold_pct_rank        REAL,
            gold_classification  TEXT,
            gold_trend           TEXT,

            -- Gold inputs
            real_yield           REAL,
            gold_ma126           REAL,

            -- Alert state
            prev_crude_class     TEXT,
            prev_gold_class      TEXT,
            alert_type           TEXT
        )
    ''')
    conn.commit()


# ═════════════════════════════════════════════════════════════
# SECTION 2 — LOAD DATA
# ═════════════════════════════════════════════════════════════

def load_price_csv(filename):
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


def fetch_eia_inventories() -> pd.Series:
    """
    Fetch US crude oil stocks (excl. SPR) from EIA API v2.
    Returns weekly Series in thousand barrels.
    """
    params = {
        'frequency':            'weekly',
        'data[0]':              'value',
        'facets[series][]':     EIA_SERIES,
        'sort[0][column]':      'period',
        'sort[0][direction]':   'desc',
        'length':               EIA_LOOKBACK,
        'api_key':              'DEMO_KEY',
    }
    try:
        r   = requests.get(EIA_URL, params=params, timeout=20)
        r.raise_for_status()
        raw = r.json()['response']['data']
        df  = pd.DataFrame(raw)
        df['period'] = pd.to_datetime(df['period'])
        df['value']  = pd.to_numeric(df['value'], errors='coerce')
        s   = (df.set_index('period')['value']
                  .dropna()
                  .sort_index())
        return s
    except Exception as e:
        print(f"    ⚠️  EIA inventory fetch failed: {e}")
        return pd.Series(dtype=float)


def fetch_fred_series(series_id: str) -> pd.Series:
    end = datetime.now().strftime('%Y-%m-%d')
    try:
        df = pdr.DataReader(series_id, 'fred',
                            FRED_START, end)
        return df.iloc[:, 0].dropna()
    except Exception as e:
        print(f"    ⚠️  FRED {series_id} failed: "
              f"{type(e).__name__}")
        return pd.Series(dtype=float)


# ═════════════════════════════════════════════════════════════
# SECTION 3 — COMPONENT 1: CRUDE FAIR VALUE & PREMIUM
# ═════════════════════════════════════════════════════════════

def compute_crude_model(crude: pd.Series,
                         inv: pd.Series) -> dict:
    """
    Fair value = 2-year rolling mean + inventory adjustment.
    Inventory adjustment:
      deviation (kbbl above 5yr avg) → $/bbl price signal
      surplus  50M kbbl = −$5 (oversupply → lower FV)
      deficit  50M kbbl = +$5 (tightness  → higher FV)
    """
    if crude.empty:
        return {}

    # 2-year rolling mean as structural baseline
    crude_2yr_ma = crude.rolling(CRUDE_FV_WINDOW).mean()

    # Inventory 5-year average and deviation
    inv_clean = inv if not inv.empty else pd.Series(dtype=float)
    if not inv_clean.empty and len(inv_clean) >= 260:
        inv_5yr  = inv_clean.rolling(260).mean()   # 260 weeks ≈ 5yr
        inv_dev  = inv_clean - inv_5yr
        # Get latest matching date
        latest_inv_date = inv_clean.index[-1]
        latest_inv      = float(inv_clean.iloc[-1])
        latest_5yr_avg  = float(inv_5yr.iloc[-1])
        latest_dev      = float(inv_dev.iloc[-1])
        inv_signal      = -(latest_dev / CRUDE_INV_SCALE) \
                          * CRUDE_INV_ADJ_PER
    else:
        latest_inv      = None
        latest_5yr_avg  = None
        latest_dev      = None
        inv_signal      = 0.0

    crude_baseline = float(crude_2yr_ma.iloc[-1]) \
                     if not crude_2yr_ma.isna().iloc[-1] \
                     else float(crude.rolling(90).mean().iloc[-1])

    fair_value  = crude_baseline + inv_signal
    spot        = float(crude.iloc[-1])
    premium_usd = spot - fair_value
    premium_pct = (premium_usd / fair_value * 100
                   if fair_value > 0 else 0.0)

    # Historical premium series for percentile rank
    # Compute rolling FV series — inventory resampled to daily
    if not inv_clean.empty:
        inv_daily = inv_clean.resample('D').ffill()
        inv_5yr_d = inv_daily.rolling(260*7, min_periods=100) \
                              .mean()
        inv_dev_d = inv_daily - inv_5yr_d
        inv_sig_d = -(inv_dev_d / CRUDE_INV_SCALE) \
                    * CRUDE_INV_ADJ_PER
        fv_series = crude_2yr_ma + \
                    inv_sig_d.reindex(crude.index, method='ffill')
    else:
        fv_series = crude_2yr_ma

    hist_prem = ((crude - fv_series) / fv_series * 100).dropna()
    pct_rank  = float((hist_prem <= premium_pct).mean() * 100) \
                if not hist_prem.empty else 50.0

    return {
        'crude_spot':        round(spot, 2),
        'crude_fair_value':  round(fair_value, 2),
        'crude_premium_usd': round(premium_usd, 2),
        'crude_premium_pct': round(premium_pct, 2),
        'crude_pct_rank':    round(pct_rank, 1),
        'inv_latest_mbbl':   round(latest_inv / 1000, 1)
                             if latest_inv else None,
        'inv_5yr_avg_mbbl':  round(latest_5yr_avg / 1000, 1)
                             if latest_5yr_avg else None,
        'inv_dev_mbbl':      round(latest_dev / 1000, 1)
                             if latest_dev else None,
        'inv_signal_usd':    round(inv_signal, 2),
        '_hist_prem':        hist_prem,
    }


# ═════════════════════════════════════════════════════════════
# SECTION 4 — COMPONENT 3: GOLD FAIR VALUE & PREMIUM
# ═════════════════════════════════════════════════════════════

def compute_gold_model(gold: pd.Series,
                        real_yield: pd.Series) -> dict:
    """
    Fair value = MA126 × (1 − real_yield × 0.05)
    Premium = Spot − Fair Value
    """
    if gold.empty:
        return {}

    gold_ma = gold.rolling(GOLD_MA_WINDOW).mean()
    rv      = real_yield.reindex(gold.index, method='ffill')

    fv_series   = gold_ma * (1 - rv * GOLD_REAL_SENS)
    prem_series = ((gold - fv_series) / fv_series * 100).dropna()

    spot        = float(gold.iloc[-1])
    fv          = float(fv_series.iloc[-1])
    premium_usd = spot - fv
    premium_pct = float(prem_series.iloc[-1])

    pct_rank = float(
        (prem_series <= premium_pct).mean() * 100
    ) if not prem_series.empty else 50.0

    return {
        'gold_spot':        round(spot, 2),
        'gold_fair_value':  round(fv, 2),
        'gold_premium_usd': round(premium_usd, 2),
        'gold_premium_pct': round(premium_pct, 2),
        'gold_pct_rank':    round(pct_rank, 1),
        'real_yield':       round(float(rv.iloc[-1]), 3),
        'gold_ma126':       round(float(gold_ma.iloc[-1]), 2),
        '_hist_prem':       prem_series,
    }


# ═════════════════════════════════════════════════════════════
# SECTION 5 — COMPONENT 2: CLASSIFICATION & TREND
# ═════════════════════════════════════════════════════════════

def classify_premium(prem_pct: float,
                     asset: str) -> tuple[str, str]:
    """
    Returns (classification, description).
    Uses % of fair value thresholds from spec.
    """
    if prem_pct >= THRESH_EXTREME:
        cls  = 'EXTREME_PREMIUM'
        desc = (f'{prem_pct:.1f}% above fair value — pure fear/'
                f'sentiment rally; historically reverts within '
                f'30–60 days; strongly avoid new {asset} Longs')
    elif prem_pct >= THRESH_HIGH:
        cls  = 'HIGH_PREMIUM'
        desc = (f'{prem_pct:.1f}% above fair value — significant '
                f'war/supply risk priced in; '
                f'{asset} Long at risk of reversal on de-escalation')
    elif prem_pct >= THRESH_MODERATE:
        cls  = 'MODERATE_PREMIUM'
        desc = (f'{prem_pct:.1f}% above fair value — some '
                f'geopolitical risk priced in; monitor for expansion')
    elif prem_pct >= 0:
        cls  = 'LOW_PREMIUM'
        desc = (f'{prem_pct:.1f}% above fair value — '
                f'price fundamentally justified; '
                f'no unusual risk premium')
    else:
        cls  = 'DISCOUNT'
        desc = (f'{prem_pct:.1f}% below fair value — '
                f'price below modelled fundamentals; '
                f'potential value entry')

    return cls, desc


def compute_trend(hist_prem: pd.Series,
                  current: float,
                  lookback: int = 10) -> tuple[str, float]:
    """
    Compare today's premium to N-day ago.
    Returns (trend_label, change_pct).
    """
    if hist_prem.empty or len(hist_prem) < lookback + 1:
        return 'UNKNOWN', 0.0

    prev = float(hist_prem.iloc[-lookback - 1])
    chg  = current - prev

    if chg >= 3.0:
        label = 'EXPANDING'
    elif chg <= -3.0:
        label = 'CONTRACTING'
    else:
        label = 'STABLE'

    return label, round(chg, 2)


# ═════════════════════════════════════════════════════════════
# SECTION 6 — ALERT LOGIC
# ═════════════════════════════════════════════════════════════

def check_alert(cls: str, prev_cls: str,
                asset: str) -> tuple[bool, str]:
    """
    Returns (should_alert, alert_type).
    Fires on:
      LOW → HIGH (new risk event)
      HIGH → MODERATE (de-escalation)
      Any → EXTREME
    """
    if cls == ALERT_EXTREME and prev_cls != ALERT_EXTREME:
        return (True,
                f'{asset} EXTREME_PREMIUM — '
                f'pure fear rally, avoid new Longs')
    if (prev_cls in ('LOW_PREMIUM', 'DISCOUNT', 'MODERATE_PREMIUM')
            and cls == 'HIGH_PREMIUM'):
        return (True,
                f'{asset} geopolitical premium escalating → '
                f'HIGH_PREMIUM')
    if (prev_cls in ('HIGH_PREMIUM', 'EXTREME_PREMIUM')
            and cls == 'MODERATE_PREMIUM'):
        return (True,
                f'{asset} premium contracting → MODERATE — '
                f'de-escalation signal, consider exiting Longs')
    return False, ''


# ═════════════════════════════════════════════════════════════
# SECTION 7 — LOAD PREVIOUS STATE & SAVE
# ═════════════════════════════════════════════════════════════

def load_prev_state(conn, today):
    try:
        row = conn.execute(
            "SELECT crude_classification, gold_classification "
            "FROM GEOPOLITICAL_PREMIUM "
            "WHERE date < ? ORDER BY date DESC LIMIT 1",
            (today,)
        ).fetchone()
        return (row[0], row[1]) if row else (None, None)
    except Exception:
        return None, None


def save_results(conn, today, c, g,
                 crude_cls, crude_trend,
                 gold_cls, gold_trend,
                 prev_crude_cls, prev_gold_cls,
                 alert_type):
    row = {
        'date':               today,
        'crude_spot':         c.get('crude_spot'),
        'crude_fair_value':   c.get('crude_fair_value'),
        'crude_premium_usd':  c.get('crude_premium_usd'),
        'crude_premium_pct':  c.get('crude_premium_pct'),
        'crude_pct_rank':     c.get('crude_pct_rank'),
        'crude_classification': crude_cls,
        'crude_trend':        crude_trend,
        'inv_latest_mbbl':    c.get('inv_latest_mbbl'),
        'inv_5yr_avg_mbbl':   c.get('inv_5yr_avg_mbbl'),
        'inv_dev_mbbl':       c.get('inv_dev_mbbl'),
        'inv_signal_usd':     c.get('inv_signal_usd'),
        'gold_spot':          g.get('gold_spot'),
        'gold_fair_value':    g.get('gold_fair_value'),
        'gold_premium_usd':   g.get('gold_premium_usd'),
        'gold_premium_pct':   g.get('gold_premium_pct'),
        'gold_pct_rank':      g.get('gold_pct_rank'),
        'gold_classification': gold_cls,
        'gold_trend':         gold_trend,
        'real_yield':         g.get('real_yield'),
        'gold_ma126':         g.get('gold_ma126'),
        'prev_crude_class':   prev_crude_cls,
        'prev_gold_class':    prev_gold_cls,
        'alert_type':         alert_type,
    }
    try:
        conn.execute(
            "DELETE FROM GEOPOLITICAL_PREMIUM WHERE date=?",
            (today,)
        )
        pd.DataFrame([row]).to_sql(
            'GEOPOLITICAL_PREMIUM', conn,
            if_exists='append', index=False
        )
        conn.commit()
        print(f"  ✅ Geopolitical premium saved")
    except Exception as e:
        print(f"  ❌ Save failed: {e}")


# ═════════════════════════════════════════════════════════════
# SECTION 8 — PRINT REPORT
# ═════════════════════════════════════════════════════════════

def _cls_emoji(cls):
    return {
        'DISCOUNT':          '🔵',
        'LOW_PREMIUM':       '🟢',
        'MODERATE_PREMIUM':  '🟡',
        'HIGH_PREMIUM':      '🟠',
        'EXTREME_PREMIUM':   '🔴',
    }.get(cls, '❓')


def _trend_arrow(trend):
    return {'EXPANDING': '↑↑', 'CONTRACTING': '↓↓',
            'STABLE': '→', 'UNKNOWN': '?'}.get(trend, '?')


def _prem_bar(pct, width=24):
    """Bar from 0 to 50%."""
    frac = min(1.0, max(0.0, pct / 50.0))
    n    = int(frac * width)
    # Colour zones
    thresholds = [
        int(THRESH_MODERATE / 50.0 * width),
        int(THRESH_HIGH     / 50.0 * width),
        int(THRESH_EXTREME  / 50.0 * width),
    ]
    return '[' + '█' * n + '░' * (width - n) + ']'


def print_report(c, g, crude_cls, crude_desc, crude_trend,
                 crude_chg, gold_cls, gold_desc, gold_trend,
                 gold_chg, prev_crude_cls, prev_gold_cls,
                 alerts):
    print("\n" + "="*70)
    print("GEOPOLITICAL RISK PREMIUM MONITOR — DAILY REPORT")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*70)

    # ── Crude ─────────────────────────────────────────────────
    e = _cls_emoji(crude_cls)
    t = _trend_arrow(crude_trend)
    print(f"\n🛢️  COMPONENT 1+2 — CRUDE OIL RISK PREMIUM")
    print("-"*60)
    if c:
        spot = c['crude_spot']
        fv   = c['crude_fair_value']
        pusd = c['crude_premium_usd']
        ppct = c['crude_premium_pct']
        prk  = c['crude_pct_rank']
        print(f"  Spot Price:    ${spot:>8.2f} / bbl")
        print(f"  Fair Value:    ${fv:>8.2f} / bbl")
        print(f"  Risk Premium:  ${pusd:>+8.2f} / bbl  "
              f"({ppct:+.1f}% of FV)")
        print(f"  {_prem_bar(ppct)}")
        print(f"\n  Classification: {e} {crude_cls}  "
              f"{t} {crude_trend} ({crude_chg:+.1f}pp 10d)")
        print(f"  → {crude_desc}")
        print(f"  Percentile rank: {prk:.0f}th (vs full history)")

        if c.get('inv_latest_mbbl') is not None:
            print(f"\n  Inventory context (EIA weekly):")
            print(f"    US crude stocks:  "
                  f"{c['inv_latest_mbbl']:.0f}M bbl")
            print(f"    5-yr average:     "
                  f"{c['inv_5yr_avg_mbbl']:.0f}M bbl")
            print(f"    Deviation:        "
                  f"{c['inv_dev_mbbl']:+.0f}M bbl  "
                  f"→ FV signal: ${c['inv_signal_usd']:+.2f}/bbl")

        if prev_crude_cls and prev_crude_cls != crude_cls:
            print(f"\n  ⚡ CLASS CHANGE: {prev_crude_cls}"
                  f" → {crude_cls}")
    else:
        print("  ⚠️  Crude data unavailable")

    # ── Gold ──────────────────────────────────────────────────
    e = _cls_emoji(gold_cls)
    t = _trend_arrow(gold_trend)
    print(f"\n🥇 COMPONENT 3 — GOLD FEAR PREMIUM")
    print("-"*60)
    if g:
        spot = g['gold_spot']
        fv   = g['gold_fair_value']
        pusd = g['gold_premium_usd']
        ppct = g['gold_premium_pct']
        prk  = g['gold_pct_rank']
        ry   = g['real_yield']
        ma   = g['gold_ma126']
        print(f"  Spot Price:     ${spot:>8,.0f} / oz")
        print(f"  Fair Value:     ${fv:>8,.0f} / oz")
        print(f"    (MA126 ${ma:,.0f} × "
              f"(1 − {ry:.2f}% × 0.05))")
        print(f"  Fear Premium:   ${pusd:>+8,.0f} / oz  "
              f"({ppct:+.1f}% of FV)")
        print(f"  {_prem_bar(ppct)}")
        print(f"\n  Classification: {e} {gold_cls}  "
              f"{t} {gold_trend} ({gold_chg:+.1f}pp 10d)")
        print(f"  → {gold_desc}")
        print(f"  Percentile rank: {prk:.0f}th (vs full history)")
        print(f"  Real yield input: {ry:+.3f}% (DFII10)")

        if prev_gold_cls and prev_gold_cls != gold_cls:
            print(f"\n  ⚡ CLASS CHANGE: {prev_gold_cls}"
                  f" → {gold_cls}")
    else:
        print("  ⚠️  Gold data unavailable")

    # ── Premium Trend Signal ──────────────────────────────────
    print(f"\n📊 COMPONENT 4 — PREMIUM TREND & SIGNAL")
    print("-"*60)
    c_arrow = _trend_arrow(crude_trend)
    g_arrow = _trend_arrow(gold_trend)
    print(f"  Crude:  {c_arrow} {crude_trend:12} "
          f"({crude_chg:+.1f}pp over 10d)")
    print(f"  Gold:   {g_arrow} {gold_trend:12} "
          f"({gold_chg:+.1f}pp over 10d)")

    # Trading signals
    print(f"\n  Trading implications:")
    for asset, cls, trend in [
        ('Crude', crude_cls, crude_trend),
        ('Gold',  gold_cls,  gold_trend),
    ]:
        e2 = _cls_emoji(cls)
        if cls == 'EXTREME_PREMIUM':
            print(f"  {e2} {asset}: Strongly AVOID new Longs — "
                  f"pure fear premium, expect mean reversion")
        elif cls == 'HIGH_PREMIUM' and trend == 'CONTRACTING':
            print(f"  {e2} {asset}: EXIT Long — "
                  f"high premium contracting (de-escalation)")
        elif cls == 'HIGH_PREMIUM' and trend == 'EXPANDING':
            print(f"  {e2} {asset}: HOLD Long but set tight stop — "
                  f"high premium still expanding")
        elif cls == 'HIGH_PREMIUM':
            print(f"  {e2} {asset}: CAUTION on Longs — "
                  f"high premium, watch for reversal")
        elif cls == 'MODERATE_PREMIUM' and trend == 'EXPANDING':
            print(f"  {e2} {asset}: Monitor — "
                  f"premium building toward HIGH zone")
        elif cls in ('LOW_PREMIUM', 'DISCOUNT'):
            print(f"  {e2} {asset}: Fundamentally priced — "
                  f"no geopolitical distortion in signal")
        else:
            print(f"  {e2} {asset}: {cls} — normal monitoring")

    if alerts:
        print(f"\n  ⚡ ALERTS:")
        for a in alerts:
            print(f"    • {a}")

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


def build_telegram_message(c, g, crude_cls, crude_trend,
                            crude_chg, gold_cls, gold_trend,
                            gold_chg, alerts):
    date = datetime.now().strftime('%d %b %Y %H:%M')
    lines = [
        f"🌍 <b>GMIS GEOPOLITICAL PREMIUM ALERT</b>",
        f"{date}",
        f"{'─' * 30}",
        "",
    ]
    for a in alerts:
        lines.append(f"⚡ <b>{a}</b>")
    lines.append("")

    if c:
        ce = _cls_emoji(crude_cls)
        lines += [
            f"🛢️  <b>Crude</b>: "
            f"{ce} {crude_cls} ({_trend_arrow(crude_trend)})",
            f"   Spot ${c['crude_spot']:.1f}  FV ${c['crude_fair_value']:.1f}  "
            f"Prem {c['crude_premium_pct']:+.1f}%",
        ]
    if g:
        ge = _cls_emoji(gold_cls)
        lines += [
            f"🥇 <b>Gold</b>: "
            f"{ge} {gold_cls} ({_trend_arrow(gold_trend)})",
            f"   Spot ${g['gold_spot']:,.0f}  FV ${g['gold_fair_value']:,.0f}  "
            f"Prem {g['gold_premium_pct']:+.1f}%",
        ]

    lines += ["", "<i>GMIS Geopolitical Premium Monitor</i>"]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_geopolitical_premium(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 33 — GEOPOLITICAL RISK PREMIUM MONITOR")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    conn  = sqlite3.connect(DB_PATH)
    create_table(conn)
    today = datetime.now().strftime('%Y-%m-%d')

    prev_crude_cls, prev_gold_cls = load_prev_state(conn, today)

    # ── Load prices ───────────────────────────────────────────
    print("\nLoading price data...")
    crude = load_price_csv('CRUDE_WTI.csv')
    gold  = load_price_csv('GOLD.csv')
    print(f"  Crude: {len(crude)} obs, "
          f"latest {crude.index[-1].date()}, "
          f"${crude.iloc[-1]:.2f}")
    print(f"  Gold:  {len(gold)} obs, "
          f"latest {gold.index[-1].date()}, "
          f"${gold.iloc[-1]:,.0f}")

    # ── Fetch EIA inventories ─────────────────────────────────
    print("\nFetching EIA crude inventories...")
    inv = fetch_eia_inventories()
    if not inv.empty:
        print(f"  {len(inv)} weekly obs, "
              f"latest {inv.index[-1].date()}, "
              f"{inv.iloc[-1]/1000:.0f}M bbl")
    else:
        print("  ⚠️  Inventory data unavailable — "
              "using baseline only")

    # ── Fetch FRED real yield ─────────────────────────────────
    print("\nFetching real yield (DFII10)...")
    real_yield = fetch_fred_series('DFII10')
    if not real_yield.empty:
        print(f"  Real yield: {real_yield.iloc[-1]:.3f}%")
    else:
        print("  ⚠️  Real yield unavailable — using 0%")
        real_yield = pd.Series([0.0],
                                index=[pd.Timestamp.now()])

    # ── Component 1+2: Crude model ────────────────────────────
    print("\nComponent 1+2 — Crude fair value & premium...")
    c = compute_crude_model(crude, inv)
    if c:
        crude_cls, crude_desc = classify_premium(
            c['crude_premium_pct'], 'Crude')
        crude_trend, crude_chg = compute_trend(
            c.get('_hist_prem', pd.Series()), c['crude_premium_pct'])
        print(f"  Crude spot: ${c['crude_spot']:.2f}  "
              f"FV: ${c['crude_fair_value']:.2f}  "
              f"Premium: {c['crude_premium_pct']:+.1f}%")
        print(f"  Classification: {crude_cls}  "
              f"Trend: {crude_trend}")
    else:
        crude_cls = 'UNKNOWN'
        crude_desc = 'Data unavailable'
        crude_trend, crude_chg = 'UNKNOWN', 0.0

    # ── Component 3: Gold model ───────────────────────────────
    print("\nComponent 3 — Gold fair value & fear premium...")
    g = compute_gold_model(gold, real_yield)
    if g:
        gold_cls, gold_desc = classify_premium(
            g['gold_premium_pct'], 'Gold')
        gold_trend, gold_chg = compute_trend(
            g.get('_hist_prem', pd.Series()), g['gold_premium_pct'])
        print(f"  Gold spot: ${g['gold_spot']:,.0f}  "
              f"FV: ${g['gold_fair_value']:,.0f}  "
              f"Premium: {g['gold_premium_pct']:+.1f}%")
        print(f"  Real yield: {g['real_yield']:+.3f}%  "
              f"MA126: ${g['gold_ma126']:,.0f}")
        print(f"  Classification: {gold_cls}  "
              f"Trend: {gold_trend}")
    else:
        gold_cls = 'UNKNOWN'
        gold_desc = 'Data unavailable'
        gold_trend, gold_chg = 'UNKNOWN', 0.0

    # ── Component 4: Alert check ──────────────────────────────
    alerts = []
    for asset, cls, prev in [
        ('Crude', crude_cls, prev_crude_cls),
        ('Gold',  gold_cls,  prev_gold_cls),
    ]:
        if prev:
            should_alert, alert_txt = check_alert(
                cls, prev, asset)
            if should_alert:
                alerts.append(alert_txt)

    alert_type = '; '.join(alerts) if alerts else None
    print(f"\nAlerts: {alerts if alerts else 'none'}")

    # ── Save ─────────────────────────────────────────────────
    print("\nSaving results...")
    save_results(conn, today, c or {}, g or {},
                 crude_cls, crude_trend,
                 gold_cls,  gold_trend,
                 prev_crude_cls, prev_gold_cls, alert_type)

    # ── Full report ───────────────────────────────────────────
    print_report(c, g, crude_cls, crude_desc, crude_trend,
                 crude_chg, gold_cls, gold_desc, gold_trend,
                 gold_chg, prev_crude_cls, prev_gold_cls, alerts)

    conn.close()

    # ── Telegram ──────────────────────────────────────────────
    if send_telegram_flag and BOT_TOKEN:
        should_send = bool(alerts) or '--force-send' in sys.argv
        if should_send:
            print("\nAlert detected — sending Telegram...")
            msg = build_telegram_message(
                c, g, crude_cls, crude_trend, crude_chg,
                gold_cls, gold_trend, gold_chg, alerts
            )
            asyncio.run(_send_telegram(msg))
        else:
            print(f"\n  No alert crossings — no Telegram")
    elif not send_telegram_flag:
        print("\n  Telegram skipped (--no-telegram)")

    return {
        'crude_cls':    crude_cls,
        'crude_pct':    c.get('crude_premium_pct') if c else None,
        'gold_cls':     gold_cls,
        'gold_pct':     g.get('gold_premium_pct') if g else None,
        'alerts':       alerts,
    }


if __name__ == "__main__":
    no_telegram = '--no-telegram' in sys.argv
    run_geopolitical_premium(send_telegram_flag=not no_telegram)
