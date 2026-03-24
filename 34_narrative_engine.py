# ============================================================
# GMIS 2.0 — MODULE 34 — NARRATIVE ENGINE
#
# COMPONENT 1 — Narrative Detection (12 core narratives)
#   Scans SENTIMENT_DAILY headlines + EXTERNAL_INTELLIGENCE
#   keywords for 12 named macro narratives over 7/14/30-day
#   rolling windows.
#
# COMPONENT 2 — Narrative Strength Scoring (0–100)
#   Raw hit count → 7-day rolling average → normalised score
#   DOMINANT  : score > 60
#   EMERGING  : score rising > 20% in 7 days
#   FADING    : score falling > 20% in 7 days
#
# COMPONENT 3 — Narrative-to-Asset Impact
#   Maps dominant/emerging narratives to bull/bear implications
#   for NIFTY, SP500, Gold, Silver, Crude
#
# COMPONENT 4 — Narrative Shift Alert
#   Telegram when narrative flips FADING→EMERGING or
#   DOMINANT→FADING (earliest trend-change warning)
#
# Note: Uses all available dates in SENTIMENT_DAILY.
#   On first run (1 day) momentum is marked 'INSUFFICIENT_HISTORY'.
#   Rolling analysis activates once ≥ 7 days of data exist.
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import json
import asyncio
import telegram
import os
import sys
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID   = int(os.getenv('TELEGRAM_CHAT_ID', '0'))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')

# ── Thresholds ────────────────────────────────────────────────
DOMINANT_SCORE      = 60    # normalised score → DOMINANT
EMERGING_MOMENTUM   = 20    # % gain in 7d → EMERGING
FADING_MOMENTUM     = -20   # % drop in 7d → FADING
MIN_DAYS_FOR_TREND  = 7     # need at least this many days

# ── 12 Core Narratives ───────────────────────────────────────
NARRATIVES = {
    'INFLATION_FEAR': [
        'inflation', 'cpi', 'price rise', 'price surge',
        'fed hawkish', 'rate hike', 'hot inflation',
        'inflationary', 'pce', 'sticky inflation',
        'rising prices', 'stagflation',
    ],
    'RATE_CUT_HOPE': [
        'rate cut', 'fed pivot', 'dovish', 'easing',
        'lower rates', 'fed pause', 'rate reduction',
        'rate relief', 'fed cut', 'fomc cut',
        'cutting rates', 'rate cycle',
    ],
    'RECESSION_FEAR': [
        'recession', 'slowdown', 'contraction',
        'gdp miss', 'unemployment rising', 'hard landing',
        'economic downturn', 'job losses', 'gdp decline',
        'negative growth', 'economic weakness',
    ],
    'SOFT_LANDING': [
        'soft landing', 'goldilocks', 'resilient',
        'strong jobs', 'gdp beat', 'robust economy',
        'economic resilience', 'no recession',
        'healthy economy', 'economy on track',
    ],
    'GEOPOLITICAL_RISK': [
        'war', 'conflict', 'sanctions', 'iran',
        'russia', 'china taiwan', 'geopolitical',
        'military', 'strike', 'missile', 'attack',
        'escalation', 'ceasefire', 'troops',
        'invasion', 'tension', 'tariff', 'trade war',
    ],
    'DOLLAR_STRENGTH': [
        'dollar rally', 'dxy', 'strong dollar',
        'dollar index', 'usd strength', 'dollar surge',
        'dollar gains', 'king dollar', 'dollar bull',
        'greenback', 'dollar rises',
    ],
    'GOLD_DEMAND': [
        'gold rally', 'safe haven', 'gold demand',
        'central bank buying', 'gold record', 'gold surge',
        'gold hits', 'bullion', 'gold all-time',
        'gold price', 'precious metals rise',
    ],
    'OIL_SUPPLY': [
        'oil supply', 'opec', 'crude inventory',
        'oil production', 'energy crisis', 'wti',
        'brent', 'oil cut', 'oil output',
        'petroleum', 'energy supply', 'oil stockpiles',
        'crude stocks',
    ],
    'AI_EUPHORIA': [
        'ai', 'artificial intelligence', 'nvidia',
        'chatgpt', 'ai revenue', 'ai capex',
        'machine learning', 'generative ai',
        'ai spending', 'ai boom', 'data center',
        'ai infrastructure', 'llm',
    ],
    'CHINA_RISK': [
        'china slowdown', 'china property', 'china stimulus',
        'yuan', 'pboc', 'china growth', 'chinese economy',
        'beijing', 'xi jinping', 'china gdp',
        'evergrande', 'china debt', 'china deflation',
    ],
    'INDIA_GROWTH': [
        'india gdp', 'india growth', 'modi',
        'rbi', 'india economy', 'nifty rally',
        'indian market', 'india stocks', 'sensex',
        'india inflation', 'india rate', 'bse',
        'india manufacturing',
    ],
    'FISCAL_STIMULUS': [
        'stimulus', 'spending bill', 'deficit',
        'fiscal', 'government spending', 'tax cut',
        'budget', 'infrastructure bill', 'debt ceiling',
        'treasury', 'fiscal policy', 'spending package',
        'federal spending',
    ],
}

# ── Narrative → asset impact map ─────────────────────────────
NARRATIVE_IMPACT = {
    'INFLATION_FEAR':    {
        'Gold': 'BULLISH', 'SP500': 'BEARISH',
        'NIFTY': 'BEARISH', 'Crude': 'NEUTRAL',
        'Silver': 'BULLISH',
    },
    'RATE_CUT_HOPE':     {
        'SP500': 'BULLISH', 'Gold': 'BULLISH',
        'NIFTY': 'BULLISH', 'Crude': 'NEUTRAL',
        'Silver': 'BULLISH',
    },
    'RECESSION_FEAR':    {
        'Gold': 'BULLISH', 'Crude': 'BEARISH',
        'SP500': 'BEARISH', 'NIFTY': 'BEARISH',
        'Silver': 'BEARISH',
    },
    'SOFT_LANDING':      {
        'SP500': 'BULLISH', 'NIFTY': 'BULLISH',
        'Crude': 'BULLISH', 'Gold': 'NEUTRAL',
        'Silver': 'BULLISH',
    },
    'GEOPOLITICAL_RISK': {
        'Gold': 'BULLISH', 'Crude': 'BULLISH',
        'NIFTY': 'BEARISH', 'SP500': 'BEARISH',
        'Silver': 'BULLISH',
    },
    'DOLLAR_STRENGTH':   {
        'Gold': 'BEARISH', 'Crude': 'BEARISH',
        'SP500': 'MIXED',  'NIFTY': 'BEARISH',
        'Silver': 'BEARISH',
    },
    'GOLD_DEMAND':       {
        'Gold': 'BULLISH', 'Silver': 'BULLISH',
        'SP500': 'NEUTRAL', 'NIFTY': 'NEUTRAL',
        'Crude': 'NEUTRAL',
    },
    'OIL_SUPPLY':        {
        'Crude': 'BULLISH', 'NIFTY': 'BEARISH',
        'SP500': 'MIXED',   'Gold': 'NEUTRAL',
        'Silver': 'NEUTRAL',
    },
    'AI_EUPHORIA':       {
        'SP500': 'BULLISH (narrow)', 'NIFTY': 'NEUTRAL',
        'Gold': 'BEARISH', 'Crude': 'NEUTRAL',
        'Silver': 'NEUTRAL',
    },
    'CHINA_RISK':        {
        'Gold': 'BULLISH', 'Crude': 'BEARISH',
        'SP500': 'BEARISH', 'NIFTY': 'BEARISH',
        'Silver': 'BEARISH',
    },
    'INDIA_GROWTH':      {
        'NIFTY': 'BULLISH', 'Gold': 'NEUTRAL',
        'SP500': 'NEUTRAL', 'Crude': 'BULLISH',
        'Silver': 'NEUTRAL',
    },
    'FISCAL_STIMULUS':   {
        'SP500': 'BULLISH', 'NIFTY': 'BULLISH',
        'Crude': 'BULLISH', 'Gold': 'MIXED',
        'Silver': 'NEUTRAL',
    },
}


# ═════════════════════════════════════════════════════════════
# SECTION 1 — DATABASE
# ═════════════════════════════════════════════════════════════

def create_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS NARRATIVE_SCORES (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT NOT NULL,
            narrative   TEXT NOT NULL,
            raw_hits    INTEGER,
            score_7d    REAL,
            score_14d   REAL,
            score_30d   REAL,
            normalised  REAL,
            momentum_7d REAL,
            status      TEXT,
            UNIQUE(date, narrative)
        )
    ''')
    conn.commit()


# ═════════════════════════════════════════════════════════════
# SECTION 2 — LOAD HEADLINES
# ═════════════════════════════════════════════════════════════

def load_headlines(conn) -> pd.DataFrame:
    """
    Load all available headlines from SENTIMENT_DAILY,
    plus top_themes text from EXTERNAL_INTELLIGENCE.
    Returns DataFrame with columns: date, text
    """
    frames = []

    # Primary: SENTIMENT_DAILY headlines
    try:
        df = pd.read_sql(
            "SELECT date, headline AS text "
            "FROM SENTIMENT_DAILY "
            "ORDER BY date",
            conn
        )
        if not df.empty:
            frames.append(df)
    except Exception as e:
        print(f"    ⚠️  SENTIMENT_DAILY load: {e}")

    # Supplement: EXTERNAL_INTELLIGENCE top_themes
    # (these are already theme labels, not raw text,
    #  but we can include them for extra signal weight)
    try:
        df2 = pd.read_sql(
            "SELECT date, top_themes AS text "
            "FROM EXTERNAL_INTELLIGENCE "
            "WHERE top_themes IS NOT NULL "
            "ORDER BY date",
            conn
        )
        if not df2.empty:
            frames.append(df2)
    except Exception:
        pass

    if not frames:
        return pd.DataFrame(columns=['date', 'text'])

    out = pd.concat(frames, ignore_index=True)
    out['date'] = pd.to_datetime(out['date']).dt.strftime('%Y-%m-%d')
    return out


# ═════════════════════════════════════════════════════════════
# SECTION 3 — COMPONENT 1: NARRATIVE SCORING
# ═════════════════════════════════════════════════════════════

def _count_hits(text: str, keywords: list[str]) -> int:
    """Case-insensitive keyword match count in text."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def score_headlines_daily(headlines: pd.DataFrame) -> pd.DataFrame:
    """
    For each day and each narrative, count total keyword hits.
    Returns DataFrame: date × narrative_name = hit_count.
    """
    if headlines.empty:
        return pd.DataFrame()

    records = []
    for date, group in headlines.groupby('date'):
        all_text = ' '.join(
            group['text'].dropna().astype(str).tolist()
        ).lower()
        row = {'date': date}
        for name, keywords in NARRATIVES.items():
            row[name] = sum(
                len(re.findall(re.escape(kw), all_text))
                for kw in keywords
            )
        records.append(row)

    df = pd.DataFrame(records).set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def compute_rolling_scores(daily: pd.DataFrame
                            ) -> dict[str, pd.Series]:
    """
    Rolling averages for 7, 14, 30-day windows.
    Returns dict: narrative → Series of 7d rolling avg.
    """
    result = {}
    for name in NARRATIVES:
        if name not in daily.columns:
            continue
        s = daily[name].astype(float)
        result[name] = {
            '7d':  s.rolling(7,  min_periods=1).mean(),
            '14d': s.rolling(14, min_periods=1).mean(),
            '30d': s.rolling(30, min_periods=1).mean(),
            'raw': s,
        }
    return result


def normalise_scores(rolling: dict) -> pd.DataFrame:
    """
    Normalise 7-day rolling averages to 0–100.
    Uses max across all narratives on each day as the scale.
    If max is 0 (no data), all scores stay 0.
    """
    if not rolling:
        return pd.DataFrame()

    # Build matrix of 7d scores
    scores_7d = pd.DataFrame({
        name: vals['7d']
        for name, vals in rolling.items()
    })

    # Row-wise max for normalisation (prevent /0)
    row_max = scores_7d.max(axis=1).replace(0, np.nan)
    normalised = scores_7d.div(row_max, axis=0) * 100
    normalised = normalised.fillna(0).clip(0, 100)
    return normalised


def compute_momentum(rolling: dict,
                     norm_scores: pd.DataFrame
                     ) -> pd.DataFrame:
    """
    7-day momentum = % change in normalised score
    between today and 7 days ago.
    Returns DataFrame: date × narrative.
    """
    if len(norm_scores) < MIN_DAYS_FOR_TREND:
        return pd.DataFrame(
            np.nan,
            index=norm_scores.index,
            columns=norm_scores.columns
        )
    # pct_change over 7 rows (not calendar days)
    momentum = norm_scores.pct_change(
        min(7, len(norm_scores) - 1)
    ) * 100
    return momentum


def classify_narrative(score: float,
                        momentum: float,
                        n_days_history: int) -> str:
    """
    Returns status string for a single narrative on a given day.
    """
    if n_days_history < MIN_DAYS_FOR_TREND:
        if score >= DOMINANT_SCORE:
            return 'DOMINANT'
        elif score > 20:
            return 'ACTIVE'
        return 'NEUTRAL'

    if np.isnan(momentum):
        return 'NEUTRAL'

    if score >= DOMINANT_SCORE:
        return 'DOMINANT'
    elif momentum >= EMERGING_MOMENTUM and score > 10:
        return 'EMERGING'
    elif momentum <= FADING_MOMENTUM and score > 5:
        return 'FADING'
    elif score > 20:
        return 'ACTIVE'
    return 'NEUTRAL'


# ═════════════════════════════════════════════════════════════
# SECTION 4 — SAVE TO DATABASE
# ═════════════════════════════════════════════════════════════

def save_results(conn, today: str,
                 daily: pd.DataFrame,
                 rolling: dict,
                 norm_scores: pd.DataFrame,
                 momentum: pd.DataFrame,
                 n_days: int):
    # Save today's row for each narrative
    conn.execute(
        "DELETE FROM NARRATIVE_SCORES WHERE date=?", (today,)
    )
    for name in NARRATIVES:
        if name not in norm_scores.columns:
            continue
        try:
            today_ts  = pd.Timestamp(today)
            norm_val  = float(
                norm_scores.loc[today_ts, name]
            ) if today_ts in norm_scores.index else 0.0
            mom_val   = (
                float(momentum.loc[today_ts, name])
                if (not momentum.empty and
                    today_ts in momentum.index and
                    not np.isnan(momentum.loc[today_ts, name]))
                else None
            )
            raw_hits  = int(
                daily.loc[today_ts, name]
            ) if today_ts in daily.index else 0

            r7d  = float(rolling[name]['7d'].iloc[-1]) \
                   if name in rolling else None
            r14d = float(rolling[name]['14d'].iloc[-1]) \
                   if name in rolling else None
            r30d = float(rolling[name]['30d'].iloc[-1]) \
                   if name in rolling else None

            status = classify_narrative(
                norm_val, mom_val or 0.0, n_days)

            conn.execute('''
                INSERT OR REPLACE INTO NARRATIVE_SCORES
                (date, narrative, raw_hits, score_7d,
                 score_14d, score_30d, normalised,
                 momentum_7d, status)
                VALUES (?,?,?,?,?,?,?,?,?)
            ''', (today, name, raw_hits, r7d, r14d, r30d,
                  round(norm_val, 2),
                  round(mom_val, 2) if mom_val else None,
                  status))
        except Exception as e:
            pass

    conn.commit()
    print(f"  ✅ Narrative scores saved ({len(NARRATIVES)} narratives)")


# ═════════════════════════════════════════════════════════════
# SECTION 5 — COMPONENT 3: ASSET IMPACT SUMMARY
# ═════════════════════════════════════════════════════════════

def build_asset_impact(today_scores: dict,
                        today_status: dict) -> dict:
    """
    For each asset aggregate bullish/bearish signals
    from all active (DOMINANT/EMERGING) narratives.
    Returns: asset → {bull_count, bear_count, net, signal, drivers}
    """
    assets  = ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']
    impacts = {a: {'bull': 0, 'bear': 0, 'drivers': []}
               for a in assets}

    for narrative, status in today_status.items():
        if status not in ('DOMINANT', 'EMERGING', 'ACTIVE'):
            continue
        score = today_scores.get(narrative, 0)
        if score < 10:
            continue
        asset_map = NARRATIVE_IMPACT.get(narrative, {})
        for asset in assets:
            direction = asset_map.get(asset, 'NEUTRAL')
            weight    = score / 100
            if 'BULLISH' in direction:
                impacts[asset]['bull'] += weight
                impacts[asset]['drivers'].append(
                    f"+{narrative}({score:.0f})")
            elif 'BEARISH' in direction:
                impacts[asset]['bear'] += weight
                impacts[asset]['drivers'].append(
                    f"-{narrative}({score:.0f})")

    result = {}
    for asset in assets:
        bull = impacts[asset]['bull']
        bear = impacts[asset]['bear']
        net  = bull - bear
        if net > 0.3:
            sig = 'NARRATIVE_BULLISH'
        elif net < -0.3:
            sig = 'NARRATIVE_BEARISH'
        else:
            sig = 'NARRATIVE_NEUTRAL'
        result[asset] = {
            'bull':    round(bull, 2),
            'bear':    round(bear, 2),
            'net':     round(net, 2),
            'signal':  sig,
            'drivers': impacts[asset]['drivers'][:3],
        }
    return result


# ═════════════════════════════════════════════════════════════
# SECTION 6 — COMPONENT 4: SHIFT DETECTION
# ═════════════════════════════════════════════════════════════

def detect_shifts(conn, today: str,
                   today_status: dict) -> list[str]:
    """
    Compare today's status to yesterday's stored status.
    Alert on FADING→EMERGING or DOMINANT→FADING flips.
    """
    alerts = []
    try:
        rows = conn.execute(
            "SELECT narrative, status FROM NARRATIVE_SCORES "
            "WHERE date < ? ORDER BY date DESC",
            (today,)
        ).fetchall()
    except Exception:
        return []

    # Build prev status dict (latest row per narrative)
    prev_status = {}
    seen = set()
    for narrative, status in rows:
        if narrative not in seen:
            prev_status[narrative] = status
            seen.add(narrative)

    ALERT_TRANSITIONS = {
        ('FADING',    'EMERGING'):  'turned EMERGING',
        ('FADING',    'DOMINANT'):  'turned DOMINANT',
        ('NEUTRAL',   'DOMINANT'):  'turned DOMINANT',
        ('ACTIVE',    'DOMINANT'):  'turned DOMINANT',
        ('DOMINANT',  'FADING'):    'now FADING',
        ('DOMINANT',  'NEUTRAL'):   'collapsed to NEUTRAL',
        ('EMERGING',  'FADING'):    'reversed to FADING',
    }

    for narrative, new_status in today_status.items():
        old_status = prev_status.get(narrative, 'NEUTRAL')
        key = (old_status, new_status)
        if key in ALERT_TRANSITIONS:
            label     = ALERT_TRANSITIONS[key]
            asset_map = NARRATIVE_IMPACT.get(narrative, {})
            impacted  = ', '.join(
                f"{a} {d}"
                for a, d in asset_map.items()
                if d not in ('NEUTRAL', 'MIXED')
            )
            alerts.append(
                f"{narrative} {label} "
                f"({old_status}→{new_status}) | {impacted}"
            )
    return alerts


# ═════════════════════════════════════════════════════════════
# SECTION 7 — PRINT REPORT
# ═════════════════════════════════════════════════════════════

def _status_emoji(status):
    return {
        'DOMINANT':  '🔥',
        'EMERGING':  '📈',
        'FADING':    '📉',
        'ACTIVE':    '✅',
        'NEUTRAL':   '⚪',
    }.get(status, '❓')


def _impact_emoji(signal):
    return {
        'NARRATIVE_BULLISH':  '🟢',
        'NARRATIVE_BEARISH':  '🔴',
        'NARRATIVE_NEUTRAL':  '⚪',
    }.get(signal, '❓')


def _score_bar(score, width=20):
    n = int(min(100, max(0, score)) / 100 * width)
    return '[' + '█' * n + '░' * (width - n) + ']'


def print_report(today: str,
                  norm_scores: pd.DataFrame,
                  momentum: pd.DataFrame,
                  today_status: dict,
                  today_scores: dict,
                  today_momentum: dict,
                  asset_impacts: dict,
                  alerts: list[str],
                  n_days: int,
                  top_n: int = 5):

    print("\n" + "="*70)
    print("NARRATIVE ENGINE — DAILY REPORT")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print(f"  Data history: {n_days} day(s) in DB")
    print("="*70)

    # ── Component 1+2: Ranked narratives ─────────────────────
    print(f"\n📣 COMPONENT 1+2 — TOP NARRATIVES (by normalised score)")
    print("-"*70)

    # Sort by score descending
    scored = sorted(
        today_scores.items(), key=lambda x: x[1], reverse=True
    )
    active_narratives = [(n, s) for n, s in scored if s > 0]

    if not active_narratives:
        print("  No active narratives detected today")
    else:
        print(f"  {'Narrative':<25} {'Score':>6}  "
              f"{'Momentum':>8}  {'Status':<22} Bar")
        print("  " + "-"*66)
        for i, (name, score) in enumerate(active_narratives[:top_n]):
            status = today_status.get(name, 'NEUTRAL')
            mom    = today_momentum.get(name)
            mom_s  = (f"{mom:>+6.1f}%" if mom is not None
                      and not np.isnan(mom) else "   N/A")
            e      = _status_emoji(status)
            print(f"  {name:<25} {score:>5.1f}  "
                  f"{mom_s:>8}  {e} {status:<20} "
                  f"{_score_bar(score, 14)}")

        # Show remaining active with just scores
        remaining = active_narratives[top_n:]
        if remaining:
            print(f"\n  Others active: " +
                  ", ".join(
                      f"{n}({s:.0f})"
                      for n, s in remaining
                      if s > 5
                  ))

    print(f"\n  Status legend: "
          f"🔥 DOMINANT(>60) 📈 EMERGING 📉 FADING ✅ ACTIVE")
    if n_days < MIN_DAYS_FOR_TREND:
        print(f"\n  ℹ️  Only {n_days} day(s) of history — "
              f"momentum available after {MIN_DAYS_FOR_TREND} days")

    # ── Component 3: Asset impacts ────────────────────────────
    print(f"\n🎯 COMPONENT 3 — NARRATIVE-TO-ASSET IMPACT")
    print("-"*70)
    print(f"  {'Asset':<8} {'Signal':<22} {'Bull':>5} "
          f"{'Bear':>5} {'Net':>6}  Drivers")
    print("  " + "-"*65)
    for asset in ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']:
        imp  = asset_impacts[asset]
        e    = _impact_emoji(imp['signal'])
        drv  = ' | '.join(imp['drivers']) if imp['drivers'] else '—'
        print(f"  {asset:<8} {e} {imp['signal']:<20} "
              f"{imp['bull']:>5.2f} {imp['bear']:>5.2f} "
              f"{imp['net']:>+6.2f}  {drv}")

    # ── Component 4: Shift alerts ─────────────────────────────
    print(f"\n⚡ COMPONENT 4 — NARRATIVE SHIFT ALERTS")
    print("-"*70)
    if alerts:
        for a in alerts:
            print(f"  ⚡ {a}")
    elif n_days < MIN_DAYS_FOR_TREND:
        print(f"  ℹ️  Shift detection active after "
              f"{MIN_DAYS_FOR_TREND} days of data")
    else:
        print("  No narrative shifts detected today")

    # ── Dominant narrative analysis ───────────────────────────
    dominant = [(n, s) for n, s in active_narratives
                if today_status.get(n) == 'DOMINANT']
    if dominant:
        print(f"\n🔥 DOMINANT NARRATIVE DEEP DIVE")
        print("-"*70)
        for name, score in dominant[:3]:
            impact = NARRATIVE_IMPACT.get(name, {})
            print(f"\n  {name} (score: {score:.1f})")
            print(f"  Asset implications:")
            for asset in ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']:
                d = impact.get(asset, 'NEUTRAL')
                if d != 'NEUTRAL':
                    emoji = '🟢' if 'BULLISH' in d else ('🔴' if 'BEARISH' in d else '⚪')
                    print(f"    {emoji} {asset}: {d}")

    print("\n" + "="*70)


# ═════════════════════════════════════════════════════════════
# SECTION 8 — TELEGRAM
# ═════════════════════════════════════════════════════════════

async def _send_telegram(msg):
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID, text=msg, parse_mode='HTML')
        print("  ✅ Telegram sent")
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")


def build_telegram_message(today_scores: dict,
                            today_status: dict,
                            asset_impacts: dict,
                            alerts: list[str]) -> str:
    date = datetime.now().strftime('%d %b %Y %H:%M')
    lines = [
        f"📣 <b>GMIS NARRATIVE ENGINE</b>",
        f"{date}",
        f"{'─' * 30}",
        "",
    ]

    if alerts:
        lines.append("⚡ <b>NARRATIVE SHIFTS:</b>")
        for a in alerts:
            lines.append(f"  • {a}")
        lines.append("")

    # Top narratives
    top = sorted(today_scores.items(),
                 key=lambda x: x[1], reverse=True)[:5]
    lines.append("<b>Top Narratives:</b>")
    for name, score in top:
        if score < 5:
            continue
        e      = _status_emoji(today_status.get(name, 'NEUTRAL'))
        status = today_status.get(name, 'NEUTRAL')
        lines.append(f"  {e} {name}: {score:.0f}/100 [{status}]")
    lines.append("")

    # Asset signals
    lines.append("<b>Narrative → Asset Impact:</b>")
    for asset in ['NIFTY', 'SP500', 'Gold', 'Crude']:
        imp = asset_impacts[asset]
        e   = _impact_emoji(imp['signal'])
        if imp['signal'] != 'NARRATIVE_NEUTRAL':
            lines.append(f"  {e} {asset}: {imp['signal']} "
                         f"(net {imp['net']:+.2f})")
    lines += ["", "<i>GMIS Narrative Engine</i>"]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_narrative_engine(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 34 — NARRATIVE ENGINE")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    conn  = sqlite3.connect(DB_PATH)
    create_table(conn)
    today = datetime.now().strftime('%Y-%m-%d')

    # ── Load headlines ────────────────────────────────────────
    print("\nLoading headlines from DB...")
    headlines = load_headlines(conn)
    dates_available = sorted(headlines['date'].unique())
    n_days = len(dates_available)
    print(f"  {len(headlines)} headlines across "
          f"{n_days} day(s) "
          f"({dates_available[0] if dates_available else 'N/A'} "
          f"→ {dates_available[-1] if dates_available else 'N/A'})")

    if headlines.empty:
        print("  ⚠️  No headlines — "
              "run 15_finbert_sentiment.py first")
        conn.close()
        return None

    # ── Component 1: Score headlines ─────────────────────────
    print("\nComponent 1 — Scoring headlines against narratives...")
    daily_hits = score_headlines_daily(headlines)
    total_hits = daily_hits.sum().sum() if not daily_hits.empty \
                 else 0
    print(f"  Total keyword hits: {total_hits:,}")
    if not daily_hits.empty:
        top_today = daily_hits.iloc[-1].nlargest(5)
        print("  Top narratives today (raw hits):")
        for name, val in top_today.items():
            print(f"    {name}: {int(val)} hits")

    # ── Component 2: Rolling scores & normalisation ───────────
    print("\nComponent 2 — Computing rolling averages & scores...")
    rolling     = compute_rolling_scores(daily_hits)
    norm_scores = normalise_scores(rolling)
    momentum    = compute_momentum(rolling, norm_scores)

    # Extract today's values
    today_ts    = pd.Timestamp(today)
    # Find the closest available date (in case today has no
    # sentiment data yet — use the latest available)
    if today_ts not in norm_scores.index:
        latest_ts = norm_scores.index[-1] \
                    if not norm_scores.empty else None
        if latest_ts is None:
            print("  ⚠️  No scores computed — aborting")
            conn.close()
            return None
        print(f"  Using latest available: "
              f"{latest_ts.strftime('%Y-%m-%d')}")
        use_ts = latest_ts
    else:
        use_ts = today_ts

    today_scores = {
        name: float(norm_scores.loc[use_ts, name])
        for name in NARRATIVES
        if name in norm_scores.columns
    }
    today_momentum = {}
    if not momentum.empty and use_ts in momentum.index:
        for name in NARRATIVES:
            if name in momentum.columns:
                v = momentum.loc[use_ts, name]
                today_momentum[name] = \
                    float(v) if not np.isnan(v) else None

    today_status = {
        name: classify_narrative(
            today_scores.get(name, 0),
            today_momentum.get(name) or 0.0,
            n_days
        )
        for name in NARRATIVES
    }

    active_count = sum(
        1 for s in today_status.values()
        if s != 'NEUTRAL'
    )
    print(f"  Active narratives: {active_count}/{len(NARRATIVES)}")

    # ── Component 3: Asset impact ─────────────────────────────
    print("\nComponent 3 — Building asset impact map...")
    asset_impacts = build_asset_impact(
        today_scores, today_status)
    for asset, imp in asset_impacts.items():
        if imp['signal'] != 'NARRATIVE_NEUTRAL':
            print(f"  {asset}: {imp['signal']} "
                  f"(net {imp['net']:+.2f})")

    # ── Component 4: Shift detection ─────────────────────────
    print("\nComponent 4 — Shift detection...")
    alerts = detect_shifts(conn, today, today_status)
    if alerts:
        print(f"  ⚡ {len(alerts)} shift(s) detected:")
        for a in alerts:
            print(f"    {a}")
    else:
        print("  No shifts detected")

    # ── Save ─────────────────────────────────────────────────
    print("\nSaving narrative scores...")
    save_results(conn, today, daily_hits, rolling,
                 norm_scores, momentum, n_days)

    # ── Print report ─────────────────────────────────────────
    print_report(today, norm_scores, momentum, today_status,
                 today_scores, today_momentum, asset_impacts,
                 alerts, n_days)

    conn.close()

    # ── Telegram: on shifts or dominant narratives ────────────
    if send_telegram_flag and BOT_TOKEN:
        has_dominant = any(
            s == 'DOMINANT' for s in today_status.values()
        )
        should_send = (bool(alerts) or
                       has_dominant or
                       '--force-send' in sys.argv)
        if should_send:
            print("\nSending Telegram (shifts / dominant)...")
            msg = build_telegram_message(
                today_scores, today_status,
                asset_impacts, alerts
            )
            asyncio.run(_send_telegram(msg))
        else:
            print("\n  No dominant narratives or shifts — "
                  "no Telegram")
    elif not send_telegram_flag:
        print("\n  Telegram skipped (--no-telegram)")

    return {
        'scores':   today_scores,
        'status':   today_status,
        'impacts':  asset_impacts,
        'alerts':   alerts,
    }


if __name__ == "__main__":
    no_telegram = '--no-telegram' in sys.argv
    run_narrative_engine(send_telegram_flag=not no_telegram)
