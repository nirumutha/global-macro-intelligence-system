# ============================================================
# GMIS 2.0 — MODULE 29 — EXTERNAL INTELLIGENCE LAYER
#
# COMPONENT 1 — Tiered RSS Collection
#   Tier 1 (weight 0.90): Central bank press releases (Fed, ECB)
#   Tier 2 (weight 0.70): Quality financial media
#   Tier 3 (weight 0.50): Read from existing SENTIMENT_DAILY
#
# COMPONENT 2 — Source Trust Weighting
#   Staleness penalty: -3d → 50%, -7d → 90% reduction
#   Crowded consensus: |mean| > 0.80 → CROWDED flag
#
# COMPONENT 3 — Asset-Level External Score
#   Weighted score -1 to +1 per asset, saved to DB
#
# COMPONENT 4 — Key Theme Extraction
#   Counts 8 macro themes in headlines, outputs top 3
#   Feeds into Module 34 Narrative Engine
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import feedparser
import requests
import asyncio
import telegram
import os
import sys
import json
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID   = int(os.getenv('TELEGRAM_CHAT_ID', '0'))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')

ASSETS = ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']

# ── Trust tier configuration ──────────────────────────────────
TIER_WEIGHTS = {1: 0.90, 2: 0.70, 3: 0.50}

TIER1_FEEDS = [
    {
        'name': 'Fed_FOMC',
        'url':  'https://www.federalreserve.gov/feeds/press_monetary.xml',
        'tier': 1,
    },
    {
        'name': 'Fed_All_Press',
        'url':  'https://www.federalreserve.gov/feeds/press_all.xml',
        'tier': 1,
    },
    {
        'name': 'ECB_Speeches',
        'url':  'https://www.ecb.europa.eu/rss/press.html',
        'tier': 1,
    },
]

TIER2_FEEDS = [
    {
        'name': 'CNBC_Top',
        'url':  'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'tier': 2,
    },
    {
        'name': 'CNBC_Economy',
        'url':  'https://www.cnbc.com/id/20910258/device/rss/rss.html',
        'tier': 2,
    },
    {
        'name': 'MarketWatch',
        'url':  'https://feeds.marketwatch.com/marketwatch/topstories/',
        'tier': 2,
    },
    {
        'name': 'WSJ_Markets',
        'url':  'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
        'tier': 2,
    },
    {
        'name': 'Yahoo_Finance',
        'url':  'https://finance.yahoo.com/news/rssindex',
        'tier': 2,
    },
]

# ── Asset keyword mapping ─────────────────────────────────────
ASSET_KEYWORDS = {
    'NIFTY':  ['nifty', 'sensex', 'india', 'bse', 'nse',
               'indian market', 'rupee', 'rbi', 'sebi',
               'india growth', 'mumbai'],
    'SP500':  ['s&p', 'sp500', 'nasdaq', 'dow jones', 'fed',
               'federal reserve', 'wall street', 'us market',
               'us stocks', 'nyse', 'trump', 'tariff',
               'fomc', 'powell', 'us economy'],
    'Gold':   ['gold', 'xau', 'precious metal', 'bullion',
               'safe haven', 'central bank gold'],
    'Silver': ['silver', 'xag', 'precious metals'],
    'Crude':  ['crude', 'oil', 'wti', 'brent', 'opec',
               'petroleum', 'energy', 'iran', 'saudi',
               'oil supply', 'barrel'],
}

# ── Macro theme keywords ──────────────────────────────────────
THEME_KEYWORDS = {
    'inflation':       ['inflation', 'cpi', 'price pressure',
                        'hawkish', 'price rise', 'pce', 'prices'],
    'rate_cuts':       ['rate cut', 'dovish', 'pivot', 'easing',
                        'lower rates', 'cut rates', 'rate reduction'],
    'recession':       ['recession', 'contraction', 'slowdown',
                        'gdp decline', 'layoffs', 'unemployment rise'],
    'geopolitical':    ['war', 'sanctions', 'trade war', 'tariff',
                        'conflict', 'geopolitical', 'military',
                        'tension', 'crisis'],
    'dollar_strength': ['dollar', 'dxy', 'usd', 'dollar rally',
                        'strong dollar', 'dollar index'],
    'oil_supply':      ['opec', 'oil production', 'supply cut',
                        'iran oil', 'oil output', 'oil supply'],
    'gold_demand':     ['gold demand', 'gold buying', 'safe haven',
                        'gold rally', 'bullion demand',
                        'central bank gold'],
    'india_growth':    ['india gdp', 'rbi policy', 'india growth',
                        'india economy', 'nifty rally', 'india inflation'],
}

# ── Staleness penalty schedule ────────────────────────────────
STALENESS_SCHEDULE = [
    (1,   1.00),   # ≤ 1 day old  → full weight
    (3,   0.75),   # ≤ 3 days old → 75%
    (7,   0.50),   # ≤ 7 days old → 50%
    (999, 0.10),   # > 7 days old → 10%
]

# ── Crowded consensus threshold ───────────────────────────────
CROWDED_THRESHOLD = 0.80

# ── Tier 1 extreme reading threshold for Telegram ────────────
TIER1_ALERT_THRESHOLD = 0.60


# ═════════════════════════════════════════════════════════════
# SECTION 1 — DATABASE
# ═════════════════════════════════════════════════════════════

def create_tables(conn):
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS EXTERNAL_INTELLIGENCE (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            asset           TEXT NOT NULL,
            tier1_score     REAL,
            tier1_count     INTEGER,
            tier2_score     REAL,
            tier2_count     INTEGER,
            tier3_score     REAL,
            tier3_count     INTEGER,
            weighted_score  REAL,
            total_count     INTEGER,
            crowded_flag    TEXT,
            top_themes      TEXT,
            UNIQUE(date, asset)
        );

        CREATE TABLE IF NOT EXISTS THEME_COUNTS (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT NOT NULL UNIQUE,
            themes_json TEXT,
            top3        TEXT,
            total_headlines INTEGER
        );
    ''')
    conn.commit()


# ═════════════════════════════════════════════════════════════
# SECTION 2 — FinBERT LOADER
# ═════════════════════════════════════════════════════════════

def load_finbert():
    """Load FinBERT (cached from Module 15 run)."""
    try:
        from transformers import pipeline
        model = pipeline(
            'sentiment-analysis',
            model='ProsusAI/finbert',
            truncation=True,
            max_length=512,
        )
        print("  ✅ FinBERT loaded")
        return model, 'finbert'
    except Exception as e:
        print(f"  ⚠️  FinBERT failed ({e}) — falling back to VADER")
        try:
            from vaderSentiment.vaderSentiment import (
                SentimentIntensityAnalyzer)
            return SentimentIntensityAnalyzer(), 'vader'
        except Exception:
            return None, None


def score_headline(headline, model, model_type):
    """Score a headline, return float in [-1, +1]."""
    if model is None or not headline:
        return 0.0
    try:
        if model_type == 'finbert':
            res   = model(headline[:512])[0]
            label = res['label']
            conf  = res['score']
            if label == 'positive':
                return conf
            elif label == 'negative':
                return -conf
            return 0.0
        else:  # vader
            scores = model.polarity_scores(headline)
            return float(scores['compound'])
    except Exception:
        return 0.0


# ═════════════════════════════════════════════════════════════
# SECTION 3 — RSS FETCH
# ═════════════════════════════════════════════════════════════

def _staleness_factor(pub_date_str):
    """
    Parse publication date string, return staleness factor.
    Handles RFC 2822 (RSS), ISO 8601, and plain dates.
    """
    now = datetime.now(tz=timezone.utc)
    dt  = None

    if not pub_date_str:
        return 0.75  # assume 1-2 days old if unknown

    for parser in [
        lambda s: parsedate_to_datetime(s),
        lambda s: datetime.fromisoformat(
            s.replace('Z', '+00:00')),
    ]:
        try:
            dt = parser(pub_date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            break
        except Exception:
            continue

    if dt is None:
        return 0.75

    age_days = (now - dt).total_seconds() / 86400.0

    for threshold, factor in STALENESS_SCHEDULE:
        if age_days <= threshold:
            return factor
    return 0.10


def _map_to_assets(text):
    """Return list of assets mentioned in text."""
    text_lower = text.lower()
    matched = []
    for asset, keywords in ASSET_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(asset)
    return matched if matched else ['General']


def _count_themes(text):
    """Return dict of {theme: count} for text."""
    text_lower = text.lower()
    counts = {}
    for theme, keywords in THEME_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits:
            counts[theme] = hits
    return counts


def fetch_rss_articles(feeds):
    """
    Fetch all articles from a list of feed configs.
    Returns list of dicts: {source, tier, headline,
    pub_date, staleness, score_raw, assets_mentioned}
    """
    articles = []
    headers  = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

    for feed in feeds:
        name = feed['name']
        url  = feed['url']
        tier = feed['tier']
        try:
            resp = requests.get(url, headers=headers,
                                timeout=15)
            resp.raise_for_status()
            parsed = feedparser.parse(resp.text)
            n      = len(parsed.entries)

            for entry in parsed.entries:
                title   = getattr(entry, 'title', '')
                summary = getattr(entry, 'summary', '')
                text    = f"{title} {summary}".strip()
                pub     = getattr(entry, 'published', '') or \
                          getattr(entry, 'updated', '')

                articles.append({
                    'source':   name,
                    'tier':     tier,
                    'headline': title,
                    'text':     text,
                    'pub_date': pub,
                    'assets':   _map_to_assets(text),
                })

            print(f"    {name} (Tier {tier}): {n} articles")

        except Exception as e:
            print(f"    ⚠️  {name}: {type(e).__name__} — skipped")

    return articles


# ═════════════════════════════════════════════════════════════
# SECTION 4 — COMPONENT 1+2: SCORE & WEIGHT ARTICLES
# ═════════════════════════════════════════════════════════════

def score_articles(articles, model, model_type):
    """Score and weight all fetched articles."""
    scored = []
    for art in articles:
        raw        = score_headline(
            art['headline'], model, model_type)
        staleness  = _staleness_factor(art['pub_date'])
        tier_w     = TIER_WEIGHTS[art['tier']]
        effective_w = tier_w * staleness

        scored.append({
            **art,
            'raw_score':   raw,
            'staleness':   staleness,
            'tier_weight': tier_w,
            'eff_weight':  effective_w,
        })
    return scored


def load_tier3_from_db(conn, today):
    """
    Read today's SENTIMENT_DAILY rows (already scored by
    Module 15). Map 'markets' column → asset.
    Returns dict: {asset: {'scores': [...], 'count': n}}
    """
    try:
        df = pd.read_sql(
            "SELECT headline, score, markets, date "
            "FROM SENTIMENT_DAILY WHERE date = ?",
            conn, params=(today,)
        )
    except Exception:
        # Try yesterday if today's run hasn't happened yet
        yesterday = (datetime.now() - timedelta(days=1)
                     ).strftime('%Y-%m-%d')
        try:
            df = pd.read_sql(
                "SELECT headline, score, markets, date "
                "FROM SENTIMENT_DAILY WHERE date = ?",
                conn, params=(yesterday,)
            )
        except Exception:
            return {}

    if df.empty:
        return {}

    by_asset = {}
    for _, row in df.iterrows():
        mkt   = str(row['markets'])
        score = float(row['score'])
        for asset in ASSETS:
            if asset.lower() in mkt.lower() or mkt == 'General':
                by_asset.setdefault(asset, []).append(score)

    return {a: {'scores': v, 'count': len(v)}
            for a, v in by_asset.items()}


# ═════════════════════════════════════════════════════════════
# SECTION 5 — COMPONENT 3: ASSET-LEVEL SCORES
# ═════════════════════════════════════════════════════════════

def aggregate_scores(scored_articles, tier3_by_asset):
    """
    For each asset, compute tier 1, 2, 3 scores and a
    single weighted_score. Also detect crowded consensus.

    Returns dict: {asset: {...}}
    """
    results = {}

    for asset in ASSETS:
        tier_scores = {1: [], 2: [], 3: []}

        # Tier 1 and 2 from RSS
        for art in scored_articles:
            if asset in art['assets'] or (
                'General' in art['assets'] and
                asset in ('SP500', 'NIFTY')   # general → broad equity
            ):
                tier_scores[art['tier']].append(
                    (art['raw_score'], art['eff_weight'])
                )

        # Tier 3 from SENTIMENT_DAILY
        t3 = tier3_by_asset.get(asset, {})
        t3_scores  = t3.get('scores', [])
        t3_weight  = TIER_WEIGHTS[3]
        for s in t3_scores:
            tier_scores[3].append((s, t3_weight * 1.0))

        def _weighted_mean(pairs):
            if not pairs:
                return None, 0
            ws = sum(w for _, w in pairs)
            if ws == 0:
                return 0.0, len(pairs)
            wm = sum(s * w for s, w in pairs) / ws
            return round(wm, 4), len(pairs)

        t1_score, t1_n = _weighted_mean(tier_scores[1])
        t2_score, t2_n = _weighted_mean(tier_scores[2])
        t3_score, t3_n = _weighted_mean(tier_scores[3])

        # Combined weighted score (tiers weighted by count×tier_weight)
        all_pairs = (
            tier_scores[1] + tier_scores[2] + tier_scores[3]
        )
        weighted_score, total_n = _weighted_mean(all_pairs)
        weighted_score = weighted_score if weighted_score is not None else 0.0

        # Crowded consensus
        crowded = 'NO'
        if total_n >= 3 and abs(weighted_score) >= CROWDED_THRESHOLD:
            direction = 'BULLISH' if weighted_score > 0 else 'BEARISH'
            crowded = (f'CROWDED_{direction} — consensus '
                       f'extremely one-sided ({weighted_score:+.2f}); '
                       f'contrarian risk elevated')

        results[asset] = {
            'tier1_score': t1_score,
            'tier1_count': t1_n,
            'tier2_score': t2_score,
            'tier2_count': t2_n,
            'tier3_score': t3_score,
            'tier3_count': t3_n,
            'weighted_score': weighted_score,
            'total_count': total_n,
            'crowded_flag': crowded,
        }

    return results


# ═════════════════════════════════════════════════════════════
# SECTION 6 — COMPONENT 4: THEME EXTRACTION
# ═════════════════════════════════════════════════════════════

def extract_themes(articles, tier3_headlines=None):
    """
    Count theme occurrences across all article headlines.
    Returns (theme_counts, top3_themes).
    """
    total_counts = {theme: 0 for theme in THEME_KEYWORDS}
    total_headlines = 0

    all_texts = [art['text'] for art in articles]
    if tier3_headlines:
        all_texts.extend(tier3_headlines)

    for text in all_texts:
        if not text:
            continue
        total_headlines += 1
        for theme, count in _count_themes(text).items():
            total_counts[theme] += count

    # Remove zero-count themes
    active = {k: v for k, v in total_counts.items() if v > 0}
    top3   = sorted(active, key=active.get, reverse=True)[:3]

    return total_counts, top3, total_headlines


# ═════════════════════════════════════════════════════════════
# SECTION 7 — SAVE TO DATABASE
# ═════════════════════════════════════════════════════════════

def save_results(conn, today, asset_scores,
                 theme_counts, top3_themes, total_headlines):
    # Asset scores
    for asset, s in asset_scores.items():
        top = json.dumps(top3_themes)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO EXTERNAL_INTELLIGENCE
                (date, asset,
                 tier1_score, tier1_count,
                 tier2_score, tier2_count,
                 tier3_score, tier3_count,
                 weighted_score, total_count,
                 crowded_flag, top_themes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                today, asset,
                s['tier1_score'], s['tier1_count'],
                s['tier2_score'], s['tier2_count'],
                s['tier3_score'], s['tier3_count'],
                s['weighted_score'], s['total_count'],
                s['crowded_flag'], top
            ))
        except Exception as e:
            print(f"  ❌ Save {asset}: {e}")

    # Theme counts
    try:
        conn.execute('''
            INSERT OR REPLACE INTO THEME_COUNTS
            (date, themes_json, top3, total_headlines)
            VALUES (?,?,?,?)
        ''', (
            today,
            json.dumps(theme_counts),
            ', '.join(top3_themes),
            total_headlines
        ))
    except Exception as e:
        print(f"  ❌ Save themes: {e}")

    conn.commit()
    print(f"  ✅ External intelligence saved "
          f"({len(asset_scores)} assets, "
          f"{total_headlines} headlines)")


# ═════════════════════════════════════════════════════════════
# SECTION 8 — PRINT REPORT
# ═════════════════════════════════════════════════════════════

def _score_bar(score, width=16):
    """Visual bar: centre=0, negative=left, positive=right."""
    mid  = width // 2
    pos  = int(score * mid)
    pos  = max(-mid, min(mid, pos))
    if pos >= 0:
        bar = '─' * mid + '█' * pos + '░' * (mid - pos)
    else:
        bar = '░' * (mid + pos) + '█' * (-pos) + '─' * mid
    return f'[{bar}]'


def _score_emoji(score):
    if score is None:
        return '⬜'
    if score >= 0.30:
        return '🟢'
    if score <= -0.30:
        return '🔴'
    return '🟡'


def print_report(asset_scores, theme_counts, top3_themes,
                 total_headlines, tier1_articles):
    print("\n" + "="*75)
    print("EXTERNAL INTELLIGENCE — DAILY REPORT")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*75)

    # ── Tiered source summary ─────────────────────────────────
    tier1_names = {a['source'] for a in tier1_articles
                   if a['tier'] == 1}
    tier2_names = {a['source'] for a in tier1_articles
                   if a['tier'] == 2}
    print(f"\n  Tier 1 (weight 0.90): {', '.join(tier1_names) or 'none'}")
    print(f"  Tier 2 (weight 0.70): {', '.join(tier2_names) or 'none'}")
    print(f"  Tier 3 (weight 0.50): SENTIMENT_DAILY (Module 15)")
    print(f"  Total headlines scored: {total_headlines}")

    # ── Asset scores table ────────────────────────────────────
    print("\n📊 COMPONENT 3 — EXTERNAL INTELLIGENCE SCORES")
    print("-"*75)
    hdr = (f"  {'Asset':<8} {'T1':>6} {'T2':>6} {'T3':>6} "
           f"{'Combined':>9}  Bar              Count")
    print(hdr)
    print("-"*75)

    tier1_alerts = []
    for asset in ASSETS:
        s   = asset_scores.get(asset, {})
        t1  = s.get('tier1_score')
        t2  = s.get('tier2_score')
        t3  = s.get('tier3_score')
        ws  = s.get('weighted_score', 0.0) or 0.0
        n   = s.get('total_count', 0)
        c   = s.get('crowded_flag', 'NO')

        t1_str = f"{t1:+.2f}" if t1 is not None else '  n/a'
        t2_str = f"{t2:+.2f}" if t2 is not None else '  n/a'
        t3_str = f"{t3:+.2f}" if t3 is not None else '  n/a'

        icon = _score_emoji(ws)
        print(f"  {icon} {asset:<7} {t1_str:>6} {t2_str:>6} "
              f"{t3_str:>6} {ws:>+9.3f}  "
              f"{_score_bar(ws)}  n={n}")

        if c != 'NO':
            print(f"     ⚠️  {c}")

        # Collect Tier 1 alerts
        if (t1 is not None and
                abs(t1) >= TIER1_ALERT_THRESHOLD):
            tier1_alerts.append((asset, t1))

    print("-"*75)

    # ── Theme extraction ──────────────────────────────────────
    print("\n🔍 COMPONENT 4 — DOMINANT MACRO THEMES")
    print("-"*50)
    active_themes = [(t, c) for t, c in theme_counts.items() if c > 0]
    active_themes.sort(key=lambda x: x[1], reverse=True)

    if active_themes:
        print(f"  Top themes today:")
        for i, (theme, count) in enumerate(active_themes[:8]):
            star = ' ◄' if theme in top3_themes else ''
            bar  = '█' * min(count, 20)
            print(f"  {'★' if theme in top3_themes[:3] else ' '} "
                  f"{theme:<18} {bar:<20} ({count}){star}")
    else:
        print("  No themes detected")

    print(f"\n  Top 3 themes → Narrative Engine (Module 34):")
    for i, t in enumerate(top3_themes, 1):
        print(f"  {i}. {t}")

    # ── Tier 1 alerts ─────────────────────────────────────────
    if tier1_alerts:
        print("\n⚡ TIER 1 EXTREME READINGS:")
        for asset, score in tier1_alerts:
            direction = 'BULLISH' if score > 0 else 'BEARISH'
            print(f"  [{asset}] Central bank signal: "
                  f"{direction} ({score:+.2f})")

    print("\n" + "="*75)
    return tier1_alerts


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


def build_telegram_message(asset_scores, top3_themes,
                            tier1_alerts):
    date  = datetime.now().strftime('%d %b %Y %H:%M')
    lines = [
        f"🧠 <b>GMIS EXTERNAL INTELLIGENCE</b>",
        f"{date}",
        f"{'─' * 30}",
        "",
        f"<b>Asset Sentiment Scores</b>",
        f"<i>(T1=CentralBank T2=Media T3=Yahoo/ET)</i>",
        "",
    ]

    for asset in ASSETS:
        s  = asset_scores.get(asset, {})
        ws = s.get('weighted_score', 0.0) or 0.0
        t1 = s.get('tier1_score')
        n  = s.get('total_count', 0)
        em = _score_emoji(ws)
        t1_str = f" | T1:{t1:+.2f}" if t1 is not None else ''
        lines.append(
            f"  {em} <b>{asset}</b>: {ws:+.3f}{t1_str} "
            f"({n} articles)"
        )
        c = s.get('crowded_flag', 'NO')
        if c != 'NO':
            lines.append(f"    ⚠️ CROWDED consensus")

    if top3_themes:
        lines.append("")
        lines.append(
            f"<b>Top Themes:</b> {', '.join(top3_themes)}"
        )

    if tier1_alerts:
        lines.append("")
        lines.append(f"⚡ <b>TIER 1 ALERT (Central Bank):</b>")
        for asset, score in tier1_alerts:
            d = 'BULLISH' if score > 0 else 'BEARISH'
            lines.append(
                f"  [{asset}] {d} signal: {score:+.2f}"
            )

    lines.append("")
    lines.append("<i>GMIS External Intelligence Engine</i>")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_external_intelligence(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 29 — EXTERNAL INTELLIGENCE LAYER")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    conn  = sqlite3.connect(DB_PATH)
    create_tables(conn)
    today = datetime.now().strftime('%Y-%m-%d')

    # ── Load FinBERT ──────────────────────────────────────────
    print("\nLoading FinBERT...")
    model, model_type = load_finbert()
    if model is None:
        print("  ❌ No scoring model available — aborting")
        conn.close()
        return None

    # ── Component 1: Fetch RSS ────────────────────────────────
    print("\nComponent 1 — Fetching Tier 1 feeds (central banks)...")
    tier1_raw = fetch_rss_articles(TIER1_FEEDS)

    print("\nFetching Tier 2 feeds (financial media)...")
    tier2_raw = fetch_rss_articles(TIER2_FEEDS)

    all_raw = tier1_raw + tier2_raw
    print(f"  Total raw articles: {len(all_raw)}")

    # ── Component 2: Score + stale weighting ─────────────────
    print("\nComponent 2 — Scoring & staleness-weighting...")
    scored = score_articles(all_raw, model, model_type)
    print(f"  Scored {len(scored)} articles")

    # ── Tier 3: Load from SENTIMENT_DAILY ────────────────────
    print("\nLoading Tier 3 from SENTIMENT_DAILY (Module 15)...")
    tier3_by_asset = load_tier3_from_db(conn, today)
    total_t3 = sum(v['count'] for v in tier3_by_asset.values())
    print(f"  Tier 3: {total_t3} entries "
          f"across {len(tier3_by_asset)} asset buckets")

    # ── Component 3: Asset-level scores ──────────────────────
    print("\nComponent 3 — Aggregating asset-level scores...")
    asset_scores = aggregate_scores(scored, tier3_by_asset)

    for asset in ASSETS:
        s = asset_scores[asset]
        print(f"  {asset}: combined={s['weighted_score']:+.3f} "
              f"(T1:{s['tier1_score'] if s['tier1_score'] is not None else 'n/a'} "
              f"T2:{s['tier2_score'] if s['tier2_score'] is not None else 'n/a'} "
              f"T3:{s['tier3_score'] if s['tier3_score'] is not None else 'n/a'} "
              f"n={s['total_count']})")

    # ── Component 4: Theme extraction ────────────────────────
    print("\nComponent 4 — Extracting macro themes...")
    t3_headlines = []
    for art in tier3_by_asset.values():
        pass  # headlines not stored, themes extracted from RSS

    theme_counts, top3_themes, total_headlines = extract_themes(
        scored, t3_headlines
    )
    print(f"  Top themes: {', '.join(top3_themes) or 'none'}")

    # ── Save ─────────────────────────────────────────────────
    print("\nSaving to database...")
    save_results(conn, today, asset_scores,
                 theme_counts, top3_themes, total_headlines)

    # ── Print report ─────────────────────────────────────────
    tier1_alerts = print_report(
        asset_scores, theme_counts, top3_themes,
        total_headlines, scored
    )

    conn.close()

    # ── Telegram: only on Tier 1 extreme readings ─────────────
    if send_telegram_flag and BOT_TOKEN and tier1_alerts:
        print(f"\nTier 1 alert ({len(tier1_alerts)} asset(s)) "
              f"— sending Telegram...")
        msg = build_telegram_message(
            asset_scores, top3_themes, tier1_alerts
        )
        asyncio.run(_send_telegram(msg))
    elif not send_telegram_flag:
        print("\n  Telegram skipped (--no-telegram)")
    else:
        print("\n  No Tier 1 extremes — no Telegram alert")

    return {
        'asset_scores':  asset_scores,
        'theme_counts':  theme_counts,
        'top3_themes':   top3_themes,
        'tier1_alerts':  tier1_alerts,
    }


if __name__ == "__main__":
    no_telegram = '--no-telegram' in sys.argv
    run_external_intelligence(send_telegram_flag=not no_telegram)
