# ============================================================
# GMIS MODULE 42 — EVENT CLASSIFICATION ENGINE
# Upgrades sentiment from tone detection to causal market
# impact prediction via event type + directional mapping.
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import asyncio
import os
import sys
import re
from datetime import datetime, timedelta
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────
BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_PATH, 'data', 'macro_system.db')
NO_TELEGRAM = '--no-telegram' in sys.argv

ASSETS = ['SP500', 'NIFTY', 'Gold', 'Silver', 'Crude']

# High-impact threshold for Telegram
TELEGRAM_CONFIDENCE_THRESHOLD = 0.65
TELEGRAM_IMPACT_THRESHOLD     = 0.10   # abs asset impact

# ── Confidence model ─────────────────────────────────────────
def headline_count_to_confidence(n: int) -> float:
    if n == 0:   return 0.0
    if n == 1:   return 0.40
    if n <= 3:   return 0.65
    return 0.85


# ──────────────────────────────────────────────────────────────
# EVENT DEFINITIONS
# Each event type has:
#   keywords:    terms triggering this type
#   pos_words:   terms indicating bullish direction for the type
#   neg_words:   terms indicating bearish direction for the type
#   impacts_pos: asset adjustments if direction = positive
#   impacts_neg: asset adjustments if direction = negative
# ──────────────────────────────────────────────────────────────
EVENTS = {

    'MONETARY_POLICY': {
        'keywords': [
            'rate decision', 'fomc', 'federal reserve', 'fed meeting', 'fed chair',
            'rbi', 'ecb', 'boj', 'bank of japan', 'rate hike', 'rate cut',
            'hawkish', 'dovish', 'quantitative', 'balance sheet', 'basis points',
            'jerome powell', 'powell', 'interest rate', 'policy rate',
        ],
        'pos_words': [   # dovish / rate cut → positive for risk assets
            'rate cut', 'dovish', 'easing', 'cut rates', 'lower rates',
            'pause', 'hold', 'accommodation', 'stimulus',
        ],
        'neg_words': [   # hawkish / rate hike → negative for risk assets
            'rate hike', 'hawkish', 'tightening', 'raise rates', 'hike',
            'hot ppi', 'hot inflation', 'inflation risk',
        ],
        'impacts_pos': {'Gold':+0.20, 'SP500':+0.15, 'NIFTY':+0.10,
                        'Crude':+0.10, 'Silver':+0.15},
        'impacts_neg': {'Gold':-0.20, 'SP500':-0.15, 'NIFTY':-0.10,
                        'Crude':-0.05, 'Silver':-0.15},
        'label': 'Monetary Policy',
    },

    'GEOPOLITICAL': {
        'keywords': [
            'war', 'attack', 'strike', 'sanctions', 'conflict', 'invasion',
            'military', 'bomb', 'missile', 'ceasefire', 'peace talks',
            'iran', 'russia', 'ukraine', 'mideast', 'middle east',
            'west asia', 'tensions', 'geopolit',
        ],
        'pos_words': [   # de-escalation → positive for risk assets
            'ceasefire', 'peace', 'de-escalat', 'agreement', 'deal',
            'withdrawal', 'truce',
        ],
        'neg_words': [   # escalation → positive for safe havens
            'war', 'attack', 'invasion', 'strike', 'bomb', 'missile',
            'escalat', 'conflict', 'sanctions', 'tensions',
        ],
        'impacts_pos': {'Gold':-0.20, 'Crude':-0.15,
                        'SP500':+0.10, 'NIFTY':+0.05, 'Silver':-0.05},
        'impacts_neg': {'Gold':+0.25, 'Crude':+0.20,
                        'NIFTY':-0.15, 'SP500':-0.10, 'Silver':+0.10},
        'label': 'Geopolitical',
    },

    'SUPPLY_SHOCK': {
        'keywords': [
            'opec', 'production cut', 'supply disruption', 'pipeline',
            'refinery', 'shortage', 'inventory', 'output cut', 'saudi',
            'cartel', 'supply cut', 'lock out', 'lockout', 'strike',
            'supply increase', 'output increase',
        ],
        'pos_words': [   # supply increase → positive for consumers
            'supply increase', 'output increase', 'more supply',
            'inventory build', 'excess supply', 'glut',
        ],
        'neg_words': [   # supply cut → oil price up → inflationary
            'production cut', 'supply disruption', 'shortage', 'output cut',
            'lock out', 'lockout', 'refinery', 'pipeline leak',
            'supply cut', 'opec cut',
        ],
        'impacts_pos': {'Crude':-0.20, 'SP500':+0.05,
                        'Gold': 0.00, 'NIFTY':+0.03, 'Silver': 0.00},
        'impacts_neg': {'Crude':+0.25, 'SP500':-0.05,
                        'Gold': 0.00, 'NIFTY':-0.10, 'Silver':+0.05},
        'label': 'Supply Shock',
    },

    'INFLATION_DATA': {
        'keywords': [
            'cpi', 'inflation', 'price index', 'pce', 'core inflation',
            'headline inflation', 'price pressure', 'producer price', 'ppi',
            'consumer price', 'disinflation', 'deflation',
        ],
        'pos_words': [   # cool inflation → positive for risk
            'cool', 'fall', 'lower', 'easing', 'below expect', 'soft',
            'disinflation', 'deflation', 'slowing inflation', 'tame',
        ],
        'neg_words': [   # hot inflation → negative for risk
            'hot', 'rise', 'surge', 'jump', 'higher', 'above expect',
            'sticky', 'persistent', 'accelerat', 'heat',
        ],
        'impacts_pos': {'Gold':-0.10, 'SP500':+0.15,
                        'NIFTY':+0.10, 'Crude': 0.00, 'Silver':-0.05},
        'impacts_neg': {'Gold':+0.15, 'SP500':-0.15,
                        'NIFTY':-0.10, 'Crude':+0.10, 'Silver':+0.05},
        'label': 'Inflation Data',
    },

    'GROWTH_DATA': {
        'keywords': [
            'gdp', 'growth', 'pmi', 'manufacturing', 'services',
            'retail sales', 'industrial production', 'economic output',
            'expansion', 'contraction',
        ],
        'pos_words': [
            'strong', 'beat', 'above', 'expansion', 'surge', 'better',
            'exceeds', 'accelerat', 'boost',
        ],
        'neg_words': [
            'weak', 'miss', 'below', 'contraction', 'slowdown', 'decline',
            'slump', 'recession', 'shrink',
        ],
        'impacts_pos': {'SP500':+0.10, 'NIFTY':+0.08,
                        'Crude':+0.05, 'Gold':-0.05, 'Silver':+0.03},
        'impacts_neg': {'SP500':-0.10, 'NIFTY':-0.08,
                        'Crude':-0.05, 'Gold':+0.10, 'Silver':-0.03},
        'label': 'Growth Data',
    },

    'EMPLOYMENT_DATA': {
        'keywords': [
            'jobs', 'payrolls', 'unemployment', 'nfp', 'jobless claims',
            'labor market', 'labour market', 'hiring', 'job', 'employment',
            'non-farm', 'nonfarm',
        ],
        'pos_words': [
            'strong', 'surge', 'beat', 'above', 'low unemployment',
            'job gains', 'better than', 'robust',
        ],
        'neg_words': [
            'weak', 'miss', 'rise in unemployment', 'layoffs', 'job cuts',
            'below', 'decline', 'slowdown',
        ],
        'impacts_pos': {'SP500':+0.08, 'NIFTY':+0.03,
                        'Gold':-0.05, 'Crude':+0.05, 'Silver': 0.00},
        'impacts_neg': {'SP500':-0.08, 'NIFTY':-0.03,
                        'Gold':+0.08, 'Crude':-0.05, 'Silver': 0.00},
        'label': 'Employment Data',
    },

    'EARNINGS': {
        'keywords': [
            'earnings', 'revenue', 'profit', 'eps', 'quarterly results',
            'beat estimates', 'miss estimates', 'guidance', 'outlook',
            'q1', 'q2', 'q3', 'q4', 'results',
        ],
        'pos_words': [
            'beat', 'strong', 'surge', 'record', 'above', 'top-line',
            'better than', 'upside', 'bullish',
        ],
        'neg_words': [
            'miss', 'weak', 'decline', 'below', 'disappoint', 'loss',
            'cut guidance', 'downside',
        ],
        'impacts_pos': {'SP500':+0.08, 'NIFTY':+0.05,
                        'Gold': 0.00, 'Crude': 0.00, 'Silver': 0.00},
        'impacts_neg': {'SP500':-0.08, 'NIFTY':-0.05,
                        'Gold': 0.00, 'Crude': 0.00, 'Silver': 0.00},
        'label': 'Earnings',
    },

    'REGULATORY': {
        'keywords': [
            'tariff', 'trade war', 'regulation', 'ban', 'restriction',
            'policy change', 'tax', 'sanction', 'import duty',
            'trade deal', 'free trade',
        ],
        'pos_words': [
            'trade deal', 'free trade', 'lift ban', 'remove tariff',
            'deregulat', 'easing', 'agreement',
        ],
        'neg_words': [
            'tariff', 'ban', 'restriction', 'trade war', 'sanction',
            'new tax', 'import duty', 'trade barrier',
        ],
        'impacts_pos': {'SP500':+0.08, 'NIFTY':+0.05,
                        'Gold':-0.05, 'Crude':+0.03, 'Silver': 0.00},
        'impacts_neg': {'SP500':-0.10, 'NIFTY':-0.05,
                        'Gold':+0.05, 'Crude':-0.05, 'Silver': 0.00},
        'label': 'Regulatory',
    },
}

# ── Impact tier labels ───────────────────────────────────────
def impact_tier(adj: float) -> str:
    a = abs(adj)
    if a >= 0.20: return 'VERY HIGH'
    if a >= 0.12: return 'HIGH'
    if a >= 0.07: return 'MODERATE'
    if a >= 0.03: return 'LOW'
    return 'MINIMAL'


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
        CREATE TABLE IF NOT EXISTS EVENT_CLASSIFICATIONS (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT,
            headlines_date  TEXT,
            total_headlines INTEGER,
            event_type      TEXT,
            event_direction TEXT,
            headline_count  INTEGER,
            confidence      REAL,
            raw_impact_sp500   REAL,
            raw_impact_gold    REAL,
            raw_impact_silver  REAL,
            raw_impact_crude   REAL,
            raw_impact_nifty   REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS EVENT_NET_IMPACT (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT UNIQUE,
            headlines_date  TEXT,
            total_headlines INTEGER,
            primary_event   TEXT,
            primary_direction TEXT,
            primary_confidence REAL,
            events_detected TEXT,
            net_sp500       REAL,
            net_gold        REAL,
            net_silver      REAL,
            net_crude       REAL,
            net_nifty       REAL,
            high_impact_events TEXT
        )
    """)
    conn.commit()


# ──────────────────────────────────────────────────────────────
# COMPONENT 1 — Headline loading and event classification
# ──────────────────────────────────────────────────────────────
def load_headlines(conn, today_str: str) -> tuple[pd.DataFrame, str]:
    """
    Load today's headlines. Falls back to most recent day if none today.
    Returns (df, actual_date_used).
    """
    df = pd.read_sql(
        "SELECT headline, score, sentiment FROM SENTIMENT_DAILY "
        "WHERE date = ?",
        conn, params=(today_str,))
    if not df.empty:
        return df, today_str

    # fallback: most recent day
    df2 = pd.read_sql(
        "SELECT headline, score, sentiment, date FROM SENTIMENT_DAILY "
        "ORDER BY date DESC LIMIT 200",
        conn)
    if df2.empty:
        return pd.DataFrame(), today_str

    latest = df2['date'].iloc[0]
    df_latest = df2[df2['date'] == latest].drop(columns=['date'])
    print(f"  ℹ️  No headlines for {today_str}, using {latest} ({len(df_latest)} headlines)")
    return df_latest, latest


def keyword_match(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def classify_direction(text: str,
                        pos_words: list[str],
                        neg_words: list[str]) -> str:
    """Returns 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'."""
    t      = text.lower()
    p_hits = sum(1 for w in pos_words if w.lower() in t)
    n_hits = sum(1 for w in neg_words if w.lower() in t)
    if p_hits > n_hits:   return 'POSITIVE'
    if n_hits > p_hits:   return 'NEGATIVE'
    if n_hits > 0:        return 'NEGATIVE'   # tie-break negative
    if p_hits > 0:        return 'POSITIVE'
    return 'NEUTRAL'


def classify_headlines(headlines: pd.DataFrame) -> dict:
    """
    Returns dict: { event_type: {count, direction, headlines, confidence} }
    Direction is majority vote across matching headlines.
    """
    results   = {}
    dir_votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    matched:   dict[str, list[str]]       = defaultdict(list)

    for _, row in headlines.iterrows():
        text = str(row['headline'])
        for ev_type, ev_def in EVENTS.items():
            if keyword_match(text, ev_def['keywords']):
                d = classify_direction(text,
                                       ev_def['pos_words'],
                                       ev_def['neg_words'])
                dir_votes[ev_type][d] += 1
                matched[ev_type].append(text)

    for ev_type, votes in dir_votes.items():
        count     = len(matched[ev_type])
        direction = max(votes, key=votes.get)
        confidence = headline_count_to_confidence(count)
        results[ev_type] = {
            'count':      count,
            'direction':  direction,
            'confidence': confidence,
            'headlines':  matched[ev_type],
            'votes':      dict(votes),
        }

    return results


# ──────────────────────────────────────────────────────────────
# COMPONENT 2 — Asset impact mapping
# ──────────────────────────────────────────────────────────────
def compute_impacts(classified: dict) -> dict:
    """
    Returns:
      per_event:  { event_type: { asset: weighted_impact } }
      net:        { asset: total_net_impact }
    """
    per_event = {}
    net = defaultdict(float)

    for ev_type, ev_data in classified.items():
        ev_def    = EVENTS[ev_type]
        direction = ev_data['direction']
        conf      = ev_data['confidence']

        # Choose impact table
        if direction == 'POSITIVE':
            raw = ev_def['impacts_pos']
        elif direction == 'NEGATIVE':
            raw = ev_def['impacts_neg']
        else:   # NEUTRAL: average of pos and neg, reduced
            raw = {a: (ev_def['impacts_pos'].get(a, 0)
                       + ev_def['impacts_neg'].get(a, 0)) / 2 * 0.5
                   for a in ASSETS}

        weighted = {a: round(v * conf, 4) for a, v in raw.items()}
        per_event[ev_type] = weighted

        for asset, impact in weighted.items():
            net[asset] += impact

    return per_event, {a: round(net[a], 4) for a in ASSETS}


# ──────────────────────────────────────────────────────────────
# COMPONENT 3 — Confidence scoring (already in classify_headlines)
# ──────────────────────────────────────────────────────────────
def primary_event(classified: dict) -> tuple[str, str, float]:
    """Return (event_type, direction, confidence) of the top event."""
    if not classified:
        return 'NONE', 'NEUTRAL', 0.0
    top = max(classified.items(), key=lambda x: x[1]['count'])
    return top[0], top[1]['direction'], top[1]['confidence']


# ──────────────────────────────────────────────────────────────
# COMPONENT 4 — Telegram + save
# ──────────────────────────────────────────────────────────────
def high_impact_events(classified: dict,
                        per_event_impacts: dict) -> list[str]:
    """Events where confidence > threshold AND any asset impact > threshold."""
    hi = []
    for ev, ev_data in classified.items():
        if ev_data['confidence'] < TELEGRAM_CONFIDENCE_THRESHOLD:
            continue
        impacts = per_event_impacts.get(ev, {})
        if any(abs(v) >= TELEGRAM_IMPACT_THRESHOLD for v in impacts.values()):
            hi.append(f"{ev}({ev_data['direction']}, conf={ev_data['confidence']:.2f})")
    return hi


def save_results(conn, today_str: str, headlines_date: str,
                 total_headlines: int,
                 classified: dict, per_event_impacts: dict,
                 net: dict, prim_ev: str, prim_dir: str,
                 prim_conf: float, hi_events: list):
    cur = conn.cursor()
    cur.execute("DELETE FROM EVENT_CLASSIFICATIONS WHERE date = ?",
                (today_str,))
    for ev_type, ev_data in classified.items():
        impacts = per_event_impacts.get(ev_type, {})
        cur.execute("""
            INSERT INTO EVENT_CLASSIFICATIONS
              (date, headlines_date, total_headlines,
               event_type, event_direction, headline_count, confidence,
               raw_impact_sp500, raw_impact_gold, raw_impact_silver,
               raw_impact_crude, raw_impact_nifty)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            today_str, headlines_date, total_headlines,
            ev_type, ev_data['direction'],
            ev_data['count'], ev_data['confidence'],
            impacts.get('SP500', 0), impacts.get('Gold',   0),
            impacts.get('Silver', 0), impacts.get('Crude', 0),
            impacts.get('NIFTY', 0),
        ))

    cur.execute("DELETE FROM EVENT_NET_IMPACT WHERE date = ?", (today_str,))
    cur.execute("""
        INSERT INTO EVENT_NET_IMPACT
          (date, headlines_date, total_headlines,
           primary_event, primary_direction, primary_confidence,
           events_detected,
           net_sp500, net_gold, net_silver, net_crude, net_nifty,
           high_impact_events)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        today_str, headlines_date, total_headlines,
        prim_ev, prim_dir, prim_conf,
        ', '.join(classified.keys()),
        net['SP500'], net['Gold'], net['Silver'],
        net['Crude'], net['NIFTY'],
        '; '.join(hi_events),
    ))
    conn.commit()
    print(f"  ✅ Saved {len(classified)} event type(s), net impact per asset")


def build_telegram_message(today_str: str, headlines_date: str,
                            classified: dict,
                            per_event_impacts: dict,
                            net: dict, hi_events: list) -> str:
    lines = [
        f"📰 <b>GMIS MODULE 42 — EVENT ALERT</b>",
        f"📅 {today_str}  (headlines: {headlines_date})",
        "",
        f"⚡ <b>High-impact events detected:</b>",
    ]
    for hi in hi_events:
        lines.append(f"  • {hi}")
    lines += ["", "<b>Net asset impact:</b>"]
    for asset in ASSETS:
        v = net[asset]
        bar = '▲' if v > 0 else ('▼' if v < 0 else '=')
        tier = impact_tier(v)
        lines.append(f"  {bar} {asset}: {v:+.3f}  [{tier}]")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Print report
# ──────────────────────────────────────────────────────────────
def print_report(today_str: str, headlines_date: str,
                 total_headlines: int,
                 classified: dict,
                 per_event_impacts: dict,
                 net: dict,
                 prim_ev: str, prim_dir: str, prim_conf: float,
                 hi_events: list):

    dir_icons = {'POSITIVE':'🟢', 'NEGATIVE':'🔴', 'NEUTRAL':'🟡'}
    ev_icons  = {
        'MONETARY_POLICY':'🏦', 'GEOPOLITICAL':'🌍',
        'SUPPLY_SHOCK':'⛽',    'INFLATION_DATA':'📊',
        'GROWTH_DATA':'📈',     'EMPLOYMENT_DATA':'👷',
        'EARNINGS':'💰',        'REGULATORY':'⚖️',
    }

    print(f"\n{'='*70}")
    print(f"EVENT CLASSIFICATION ENGINE — DAILY REPORT")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*70}")
    print(f"  Headlines: {total_headlines} from {headlines_date}")

    # ── Component 1 — Event types detected ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 1 — EVENT TYPES DETECTED")
    print(f"{'─'*70}")
    if not classified:
        print(f"\n  No events classified from available headlines.")
    else:
        print(f"\n  {'Event Type':<22} {'N':>3}  {'Direction':<10}  "
              f"{'Confidence':>11}  {'Vote breakdown'}")
        print(f"  {'-'*65}")
        for ev, data in sorted(classified.items(),
                               key=lambda x: -x[1]['count']):
            ic   = ev_icons.get(ev, '•')
            dic  = dir_icons.get(data['direction'], '⚪')
            conf = data['confidence']
            votes_s = ' '.join(f"{d}:{n}"
                               for d, n in sorted(data['votes'].items()))
            marker = '  ← PRIMARY' if ev == prim_ev else ''
            print(f"  {ic} {ev:<20} {data['count']:>3}  "
                  f"{dic} {data['direction']:<9}  "
                  f"{conf:>10.2f}  {votes_s}{marker}")

        # Sample headlines for primary event
        prim_data = classified.get(prim_ev, {})
        if prim_data and prim_data.get('headlines'):
            print(f"\n  Sample headlines for {prim_ev}:")
            for h in prim_data['headlines'][:3]:
                print(f"    • {h[:80]}")

    # ── Component 2 — Asset impact mapping ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 2 — ASSET IMPACT BY EVENT TYPE")
    print(f"{'─'*70}")
    if classified:
        print(f"\n  {'Event':<22}", end="")
        for a in ASSETS:
            print(f"  {a:>7}", end="")
        print(f"  {'Conf':>6}")
        print(f"  {'-'*65}")
        for ev, data in sorted(classified.items(),
                               key=lambda x: -x[1]['count']):
            impacts = per_event_impacts.get(ev, {})
            ic      = ev_icons.get(ev, '•')
            print(f"  {ic} {ev:<20}", end="")
            for a in ASSETS:
                v    = impacts.get(a, 0)
                sign = '+' if v > 0 else ('-' if v < 0 else ' ')
                bar  = '█' * min(3, int(abs(v) * 10))
                print(f"  {sign}{abs(v):.3f}", end="")
            print(f"  {data['confidence']:.2f}")

    # ── Component 3 — Confidence ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 3 — CONFIDENCE SCORES")
    print(f"{'─'*70}")
    print(f"\n  Confidence model: 1 headline=0.40  2-3=0.65  4+=0.85")
    print(f"\n  {'Event':<22} {'Headlines':>10}  {'Confidence':>11}  Tier")
    print(f"  {'-'*55}")
    for ev, data in sorted(classified.items(),
                            key=lambda x: -x[1]['confidence']):
        conf  = data['confidence']
        tier  = 'HIGH' if conf >= 0.85 else ('MED' if conf >= 0.65 else 'LOW')
        print(f"  {ev:<22} {data['count']:>10}  {conf:>10.2f}  {tier}")

    # ── Component 4 — Net impact & summary ──
    print(f"\n{'─'*70}")
    print(f"COMPONENT 4 — NET ASSET IMPACT SUMMARY")
    print(f"{'─'*70}")
    if prim_ev != 'NONE':
        pi = dir_icons.get(prim_dir, '⚪')
        pc = ev_icons.get(prim_ev, '•')
        print(f"\n  {pc} PRIMARY EVENT: {prim_ev}  "
              f"{pi} {prim_dir}  (confidence: {prim_conf:.2f})")
    print(f"\n  Net impact (all events combined, confidence-weighted):")
    print(f"\n  {'Asset':<10} {'Net Adj':>9}  {'Tier':<12}  "
          f"{'Bar':>20}")
    print(f"  {'-'*55}")
    for asset in ASSETS:
        v    = net[asset]
        tier = impact_tier(v)
        bar  = ('█' * min(20, int(abs(v) * 40))).ljust(20)
        sign = '▲' if v > 0 else ('▼' if v < 0 else '=')
        print(f"  {asset:<10} {sign}{abs(v):>7.4f}  {tier:<12}  {bar}")

    if hi_events:
        print(f"\n  ⚡ HIGH-IMPACT EVENTS (conf > {TELEGRAM_CONFIDENCE_THRESHOLD}):")
        for hi in hi_events:
            print(f"    • {hi}")
    else:
        print(f"\n  ℹ️  No single event exceeds high-impact threshold "
              f"(conf>{TELEGRAM_CONFIDENCE_THRESHOLD} and impact>{TELEGRAM_IMPACT_THRESHOLD})")

    print(f"\n{'='*70}\n")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*65}")
    print(f"GMIS MODULE 42 — EVENT CLASSIFICATION ENGINE")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*65}")

    conn = get_conn()
    setup_table(conn)

    # ── Component 1 — Load + classify ────────────────────────
    print(f"\nComponent 1 — Loading headlines for {today_str}...")
    headlines, headlines_date = load_headlines(conn, today_str)
    total = len(headlines)
    print(f"  Loaded {total} headline(s) from {headlines_date}")

    if total == 0:
        print("  ⚠️  No headlines available. Exiting.")
        conn.close()
        return

    classified = classify_headlines(headlines)
    print(f"  Detected {len(classified)} event type(s): "
          f"{', '.join(classified.keys())}")

    # ── Component 2 — Impacts ────────────────────────────────
    print("\nComponent 2 — Computing asset impacts...")
    per_event_impacts, net = compute_impacts(classified)

    # ── Component 3 — Primary event ──────────────────────────
    prim_ev, prim_dir, prim_conf = primary_event(classified)
    print(f"  Primary event: {prim_ev} ({prim_dir}, conf={prim_conf:.2f})")

    # ── Component 4 — High-impact + save ─────────────────────
    hi_events = high_impact_events(classified, per_event_impacts)
    print(f"  High-impact events: {len(hi_events)}")

    # ── Save ──────────────────────────────────────────────────
    print("\nSaving results...")
    save_results(conn, today_str, headlines_date, total,
                 classified, per_event_impacts, net,
                 prim_ev, prim_dir, prim_conf, hi_events)

    # ── Report ────────────────────────────────────────────────
    print_report(today_str, headlines_date, total,
                 classified, per_event_impacts, net,
                 prim_ev, prim_dir, prim_conf, hi_events)

    # ── Telegram ─────────────────────────────────────────────
    if hi_events and not NO_TELEGRAM:
        msg = build_telegram_message(
            today_str, headlines_date, classified,
            per_event_impacts, net, hi_events)
        asyncio.run(send_telegram(msg))
        print(f"  📱 Telegram sent ({len(hi_events)} high-impact events)")
    elif NO_TELEGRAM:
        print("  Telegram skipped (--no-telegram)")
    else:
        print("  No high-impact events — Telegram not sent")

    conn.close()


if __name__ == '__main__':
    main()
