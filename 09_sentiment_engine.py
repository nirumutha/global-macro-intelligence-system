# ============================================================
# MODULE 7 — NEWS SENTIMENT ENGINE (Updated v2)
# Context-aware sentiment scoring for financial headlines
# Fixes: Gold and Crude macro context awareness
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import os
import feedparser
from datetime import datetime

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')
OUT_PATH  = os.path.join(BASE_PATH, 'outputs')
DB_PATH   = os.path.join(DATA_PATH, 'macro_system.db')
os.makedirs(OUT_PATH, exist_ok=True)

# ── Install VADER if needed ───────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'vaderSentiment'])
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# ── RSS Feed Sources ──────────────────────────────────────────
RSS_FEEDS = {
    'Reuters_Business':  'https://feeds.reuters.com/reuters/businessNews',
    'Reuters_Markets':   'https://feeds.reuters.com/reuters/companyNews',
    'Yahoo_Finance':     'https://finance.yahoo.com/rss/',
    'Economic_Times':    'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
    'Moneycontrol':      'https://www.moneycontrol.com/rss/marketreports.xml',
    'LiveMint':          'https://www.livemint.com/rss/markets',
    'Investing_Gold':    'https://www.investing.com/rss/news_25.rss',
    'Investing_Oil':     'https://www.investing.com/rss/news_2.rss',
}

# ── Market Keywords ───────────────────────────────────────────
MARKET_KEYWORDS = {
    'NIFTY':  ['nifty', 'sensex', 'bse', 'nse', 'india market', 'indian stock',
               'dalal street', 'sebi', 'rbi', 'rupee', 'india stocks'],
    'SP500':  ['s&p 500', 'sp500', 'nasdaq', 'dow jones', 'wall street', 'fed',
               'federal reserve', 'us market', 'us stocks', 'powell', 'us equities'],
    'Gold':   ['gold', 'bullion', 'precious metal', 'xau', 'comex gold', 'gold price',
               'gold futures', 'gold rally', 'gold surge', 'gold falls'],
    'Silver': ['silver', 'xag', 'silver price', 'silver futures'],
    'Crude':  ['oil', 'crude', 'brent', 'wti', 'opec', 'petroleum', 'energy',
               'oil price', 'oil futures', 'oil rally', 'oil falls'],
}

# ── Context Modifiers ─────────────────────────────────────────
# VADER reads words literally but misses macro context.
# These rules fix the most common misclassifications.

GOLD_BULLISH_CONTEXTS = [
    'tariff', 'trade war', 'uncertainty', 'inflation', 'crisis',
    'geopolitical', 'sanctions', 'recession', 'fear', 'safe haven',
    'dollar falls', 'dollar weakens', 'rate cut', 'fed pivot',
    'war', 'conflict', 'surge', 'rally', 'record high', 'all time high',
    'haven demand', 'risk off', 'flight to safety', 'debt ceiling',
    'banking crisis', 'central bank buying', 'gold demand',
]

GOLD_BEARISH_CONTEXTS = [
    'rate hike', 'dollar rises', 'dollar strengthens', 'risk on',
    'strong dollar', 'hawkish', 'yields rise', 'sell off',
    'profit taking', 'dollar index rises', 'dxy rises',
]

CRUDE_BULLISH_CONTEXTS = [
    'opec cut', 'supply cut', 'inventory draw', 'demand surge',
    'production cut', 'supply disruption', 'geopolitical tension',
    'iran', 'russia sanctions', 'gulf conflict', 'hurricane',
    'pipeline disruption', 'supply risk', 'draw down',
]

CRUDE_BEARISH_CONTEXTS = [
    'tariff', 'trade war', 'recession', 'demand concern',
    'inventory build', 'oversupply', 'slowdown', 'opec increase',
    'demand falls', 'economic slowdown', 'china slowdown',
    'demand weakness', 'build up', 'surplus', 'increase output',
    'production increase', 'demand drop',
]

SP500_BULLISH_CONTEXTS = [
    'rate cut', 'fed pivot', 'earnings beat', 'jobs growth',
    'gdp growth', 'soft landing', 'trade deal', 'stimulus',
    'record high', 'rally', 'bull market', 'strong earnings',
]

SP500_BEARISH_CONTEXTS = [
    'tariff', 'trade war', 'recession', 'layoffs', 'earnings miss',
    'rate hike', 'inflation surge', 'sanctions', 'war', 'crisis',
    'sell off', 'correction', 'bear market', 'job cuts',
    'default', 'debt crisis', 'banking crisis',
]

NIFTY_BULLISH_CONTEXTS = [
    'fii inflow', 'fdi', 'india gdp', 'rbi rate cut', 'budget boost',
    'reform', 'india growth', 'rupee strengthens', 'earnings beat',
    'bull run', 'record high', 'sensex rally',
]

NIFTY_BEARISH_CONTEXTS = [
    'fii outflow', 'rbi rate hike', 'rupee weakens', 'inflation india',
    'monsoon deficit', 'fiscal deficit', 'earnings miss india',
    'foreign sell', 'india slowdown',
]

# ── Context-Aware Scoring Function ────────────────────────────
def score_headline(text, market='General'):
    """
    Score a headline using VADER base + macro context adjustment.
    Returns a score between -1.0 (very bearish) and +1.0 (very bullish).
    """
    scores     = analyzer.polarity_scores(str(text))
    base       = scores['compound']
    text_lower = text.lower()
    adjustment = 0.0

    if market == 'Gold':
        bullish_hits = sum(1 for kw in GOLD_BULLISH_CONTEXTS if kw in text_lower)
        bearish_hits = sum(1 for kw in GOLD_BEARISH_CONTEXTS if kw in text_lower)
        adjustment   = (bullish_hits * 0.12) - (bearish_hits * 0.12)

    elif market == 'Crude':
        bullish_hits = sum(1 for kw in CRUDE_BULLISH_CONTEXTS if kw in text_lower)
        bearish_hits = sum(1 for kw in CRUDE_BEARISH_CONTEXTS if kw in text_lower)
        adjustment   = (bullish_hits * 0.12) - (bearish_hits * 0.12)

    elif market == 'SP500':
        bullish_hits = sum(1 for kw in SP500_BULLISH_CONTEXTS if kw in text_lower)
        bearish_hits = sum(1 for kw in SP500_BEARISH_CONTEXTS if kw in text_lower)
        adjustment   = (bullish_hits * 0.10) - (bearish_hits * 0.10)

    elif market == 'NIFTY':
        bullish_hits = sum(1 for kw in NIFTY_BULLISH_CONTEXTS if kw in text_lower)
        bearish_hits = sum(1 for kw in NIFTY_BEARISH_CONTEXTS if kw in text_lower)
        adjustment   = (bullish_hits * 0.10) - (bearish_hits * 0.10)

    final = float(np.clip(base + adjustment, -1.0, 1.0))
    return round(final, 4)

def classify_sentiment(score):
    """Convert numeric score to label."""
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def tag_markets(text):
    """Tag which markets a headline relates to."""
    text_lower = text.lower()
    tags = []
    for market, keywords in MARKET_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(market)
    return tags if tags else ['General']

# ════════════════════════════════════════════════════════════
# STEP 1 — Fetch Live Headlines from RSS Feeds
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("STEP 1: Fetching live headlines from RSS feeds...")
print("=" * 55)

all_headlines = []

for source, url in RSS_FEEDS.items():
    try:
        feed  = feedparser.parse(url)
        count = 0
        for entry in feed.entries[:20]:
            title = entry.get('title', '').strip()
            if not title:
                continue

            # Tag markets first so we can pass to scorer
            markets_tagged  = tag_markets(title)
            primary_market  = markets_tagged[0] if markets_tagged else 'General'

            # Context-aware score
            score           = score_headline(title, market=primary_market)
            label           = classify_sentiment(score)

            all_headlines.append({
                'source':    source,
                'headline':  title,
                'score':     score,
                'sentiment': label,
                'markets':   ', '.join(markets_tagged),
                'date':      str(datetime.now().date()),
            })
            count += 1

        print(f"  {source:<25} {count} headlines fetched")

    except Exception as e:
        print(f"  {source:<25} Error: {e}")

headlines_df = pd.DataFrame(all_headlines)
print(f"\nTotal headlines collected: {len(headlines_df)}")

if headlines_df.empty:
    print("WARNING: No headlines collected. Check internet connection.")
else:
    # ── Save headlines to CSV ─────────────────────────────────
    headlines_path = os.path.join(DATA_PATH, 'sentiment_today.csv')
    headlines_df.to_csv(headlines_path, index=False)
    print(f"Saved to data/sentiment_today.csv")

    # ── Save to database ──────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    headlines_df.to_sql('SENTIMENT_DAILY', conn, if_exists='replace', index=False)

    # ════════════════════════════════════════════════════════
    # STEP 2 — Load price data for charts
    # ════════════════════════════════════════════════════════
    def load_close(table):
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        close_col = [c for c in df.columns if 'Close' in c or 'close' in c]
        return df[close_col[0]] if close_col else df.iloc[:, 0]

    vix_hist   = load_close('VIX_US')
    nifty_hist = load_close('NIFTY50')
    conn.close()

    # ════════════════════════════════════════════════════════
    # CHART 1 — Today's Sentiment Dashboard
    # ════════════════════════════════════════════════════════
    print("\nCreating Chart 1: Today's Sentiment Dashboard...")

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Live Market Sentiment Dashboard — {datetime.now().strftime('%d %B %Y')}\n"
        f"Global Macro Intelligence System (Context-Aware v2)",
        fontsize=13, fontweight='bold'
    )

    # Panel 1: Overall sentiment pie chart
    ax1 = fig.add_subplot(2, 3, 1)
    sentiment_counts = headlines_df['sentiment'].value_counts()
    colors_pie = {'Positive': '#1E6B3C', 'Neutral': '#2E75B6', 'Negative': '#C00000'}
    pie_colors  = [colors_pie.get(s, 'gray') for s in sentiment_counts.index]
    ax1.pie(sentiment_counts, labels=sentiment_counts.index,
            colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Sentiment\nDistribution', fontsize=11, fontweight='bold')

    # Panel 2: Sentiment by market
    ax2 = fig.add_subplot(2, 3, 2)
    market_sentiment = {}
    for market in ['NIFTY', 'SP500', 'Gold', 'Crude', 'General']:
        mask = headlines_df['markets'].str.contains(market, na=False)
        if mask.sum() > 0:
            market_sentiment[market] = headlines_df[mask]['score'].mean()

    if market_sentiment:
        markets_list = list(market_sentiment.keys())
        scores_list  = list(market_sentiment.values())
        bar_colors   = ['#1E6B3C' if s >= 0.05 else '#C00000' if s <= -0.05
                         else '#C55A11' for s in scores_list]
        bars = ax2.bar(markets_list, scores_list, color=bar_colors, alpha=0.85)
        ax2.axhline(0,     color='black',  linewidth=0.8)
        ax2.axhline(0.05,  color='green',  linewidth=0.6, linestyle='--', alpha=0.6)
        ax2.axhline(-0.05, color='red',    linewidth=0.6, linestyle='--', alpha=0.6)
        ax2.set_ylabel('Avg Sentiment Score', fontsize=9)
        ax2.set_title('Sentiment Score\nby Market (Context-Aware)', fontsize=11, fontweight='bold')
        ax2.tick_params(axis='x', rotation=30)
        for bar, val in zip(bars, scores_list):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel 3: Score distribution
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(headlines_df['score'], bins=20, color='#2E75B6', alpha=0.8, edgecolor='white')
    ax3.axvline(0, color='black', linestyle='--', linewidth=0.8)
    mean_score = headlines_df['score'].mean()
    ax3.axvline(mean_score, color='red', linestyle='-', linewidth=1.5,
                label=f"Mean: {mean_score:.3f}")
    ax3.set_xlabel('Sentiment Score', fontsize=9)
    ax3.set_ylabel('Count', fontsize=9)
    ax3.set_title('Score Distribution\nAll Headlines', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)

    # Panel 4: Headlines table
    ax4 = fig.add_subplot(2, 1, 2)
    ax4.axis('off')

    top_pos = headlines_df.nlargest(5, 'score')
    top_neg = headlines_df.nsmallest(5, 'score')

    y_pos = 0.95
    ax4.text(0, y_pos, "TOP POSITIVE HEADLINES (Context-Aware):",
             fontsize=10, fontweight='bold', color='#1E6B3C', transform=ax4.transAxes)
    for _, row in top_pos.iterrows():
        y_pos -= 0.12
        h = row['headline'][:80] + '...' if len(row['headline']) > 80 else row['headline']
        ax4.text(0, y_pos, f"▲ [{row['score']:+.3f}] [{row['markets']}] {h}",
                 fontsize=8, color='#1E6B3C', transform=ax4.transAxes)

    y_pos -= 0.06
    ax4.text(0.5, y_pos + 0.12, "TOP NEGATIVE HEADLINES (Context-Aware):",
             fontsize=10, fontweight='bold', color='#C00000', transform=ax4.transAxes)
    for _, row in top_neg.iterrows():
        y_pos -= 0.12
        h = row['headline'][:80] + '...' if len(row['headline']) > 80 else row['headline']
        ax4.text(0.5, y_pos + 0.12,
                 f"▼ [{row['score']:+.3f}] [{row['markets']}] {h}",
                 fontsize=8, color='#C00000', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, '21_sentiment_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved → outputs/21_sentiment_dashboard.png ✓")

    # ════════════════════════════════════════════════════════
    # CHART 2 — Sentiment by Source
    # ════════════════════════════════════════════════════════
    print("Creating Chart 2: Sentiment by Source...")

    source_stats = headlines_df.groupby('source')['score'].agg(
        ['mean', 'count', 'std']
    ).reset_index().sort_values('mean', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#1E6B3C' if v >= 0.05 else '#C00000' if v <= -0.05
               else '#C55A11' for v in source_stats['mean']]
    bars = ax.barh(source_stats['source'], source_stats['mean'],
                    color=colors, alpha=0.85, edgecolor='white')
    ax.axvline(0,     color='black', linewidth=0.8)
    ax.axvline(0.05,  color='green', linewidth=0.6, linestyle='--', alpha=0.6)
    ax.axvline(-0.05, color='red',   linewidth=0.6, linestyle='--', alpha=0.6)

    for bar, val, cnt in zip(bars, source_stats['mean'], source_stats['count']):
        ax.text(
            val + 0.003 if val >= 0 else val - 0.003,
            bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f} (n={cnt})',
            ha='left' if val >= 0 else 'right',
            va='center', fontsize=9
        )

    ax.set_xlabel('Average Sentiment Score', fontsize=10)
    ax.set_title(
        f'Sentiment Score by News Source — {datetime.now().strftime("%d %B %Y")}\n'
        f'Context-Aware Scoring v2 | Global Macro Intelligence System',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, '22_sentiment_by_source.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved → outputs/22_sentiment_by_source.png ✓")

    # ════════════════════════════════════════════════════════
    # CHART 3 — Historical Sentiment Proxy (VIX-based)
    # ════════════════════════════════════════════════════════
    print("Creating Chart 3: Historical Sentiment Proxy...")

    vix_series = vix_hist.dropna()
    sentiment_proxy = -(
        (vix_series - vix_series.rolling(252).mean()) /
        vix_series.rolling(252).std()
    ).clip(-3, 3)

    roll_min = sentiment_proxy.rolling(252).min()
    roll_max = sentiment_proxy.rolling(252).max()
    sentiment_norm = ((sentiment_proxy - roll_min) /
                      (roll_max - roll_min).replace(0, np.nan))
    sentiment_smooth = sentiment_norm.rolling(20).mean()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        'Market Sentiment Proxy — VIX-Based Historical Sentiment (2010–2026)\n'
        'Global Macro Intelligence System',
        fontsize=13, fontweight='bold'
    )

    ax1.plot(nifty_hist.index, nifty_hist, color='#1F3864', linewidth=1.2)
    ax1.set_ylabel('NIFTY 50', fontsize=10)
    ax1.set_title('NIFTY 50 Price', fontsize=11)

    ax2.plot(sentiment_smooth.index, sentiment_smooth,
             color='#2E75B6', linewidth=1.2, label='Sentiment Proxy (20d smoothed)')
    ax2.axhline(0.7, color='green', linestyle='--', linewidth=0.8,
                label='Bullish zone (0.7)')
    ax2.axhline(0.3, color='red',   linestyle='--', linewidth=0.8,
                label='Bearish zone (0.3)')
    ax2.fill_between(sentiment_smooth.index, sentiment_smooth, 0.5,
                     where=sentiment_smooth >= 0.5, alpha=0.2, color='green')
    ax2.fill_between(sentiment_smooth.index, sentiment_smooth, 0.5,
                     where=sentiment_smooth < 0.5,  alpha=0.2, color='red')
    ax2.set_ylabel('Sentiment (0–1)', fontsize=10)
    ax2.set_title('Sentiment Proxy — Higher = More Positive Market Mood', fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)

    ax3.plot(vix_series.index, vix_series, color='#C00000', linewidth=1)
    ax3.axhline(20, color='orange', linestyle='--', linewidth=0.8, label='Caution (20)')
    ax3.axhline(30, color='red',    linestyle='--', linewidth=0.8, label='Fear (30)')
    ax3.set_ylabel('VIX', fontsize=10)
    ax3.set_title('VIX — input to sentiment proxy (inverted)', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, '23_sentiment_proxy_historical.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved → outputs/23_sentiment_proxy_historical.png ✓")

    # ════════════════════════════════════════════════════════
    # KEY INSIGHTS
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("KEY INSIGHTS — SENTIMENT ENGINE (Context-Aware v2)")
    print("=" * 55)

    overall_score = headlines_df['score'].mean()
    positive_pct  = (headlines_df['sentiment'] == 'Positive').mean() * 100
    negative_pct  = (headlines_df['sentiment'] == 'Negative').mean() * 100
    neutral_pct   = (headlines_df['sentiment'] == 'Neutral').mean()  * 100

    print(f"\nToday's Market Sentiment ({datetime.now().strftime('%d %B %Y')}):")
    print(f"  Overall score:    {overall_score:+.4f}  ", end="")
    if overall_score > 0.05:
        print("→ POSITIVE market mood")
    elif overall_score < -0.05:
        print("→ NEGATIVE market mood")
    else:
        print("→ NEUTRAL market mood")

    print(f"\n  Positive headlines:  {positive_pct:.1f}%")
    print(f"  Neutral headlines:   {neutral_pct:.1f}%")
    print(f"  Negative headlines:  {negative_pct:.1f}%")
    print(f"  Total headlines:     {len(headlines_df)}")

    print(f"\nSentiment by Market (Context-Aware):")
    for market, score in sorted(market_sentiment.items(),
                                 key=lambda x: x[1], reverse=True):
        label = "POSITIVE" if score > 0.05 else \
                "NEGATIVE" if score < -0.05 else "NEUTRAL"
        print(f"  {market:<10} {score:+.4f}  → {label}")

    if len(source_stats) > 0:
        most_pos = source_stats.iloc[-1]
        most_neg = source_stats.iloc[0]
        print(f"\nMost Positive Source: {most_pos['source']} ({most_pos['mean']:+.4f})")
        print(f"Most Negative Source: {most_neg['source']} ({most_neg['mean']:+.4f})")

    print("\nTop 5 Most Positive Headlines:")
    for _, row in headlines_df.nlargest(5, 'score').iterrows():
        print(f"  [{row['score']:+.3f}] [{row['markets']}] {row['headline'][:70]}")

    print("\nTop 5 Most Negative Headlines:")
    for _, row in headlines_df.nsmallest(5, 'score').iterrows():
        print(f"  [{row['score']:+.3f}] [{row['markets']}] {row['headline'][:70]}")

    print("=" * 55)
    print(f"\nAll 3 charts saved to your outputs/ folder.")
    print(f"Total charts built so far: 23")
    print(f"\nSentiment data saved to:")
    print(f"  data/sentiment_today.csv")
    print(f"  database table: SENTIMENT_DAILY")
