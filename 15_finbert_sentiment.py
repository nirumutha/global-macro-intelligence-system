# ============================================================
# GMIS 2.0 — MODULE 15 — FINBERT SENTIMENT ENGINE
# Replaces VADER with financial-grade NLP
# Maintains backward compatibility with existing database
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import feedparser
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')

# ── RSS Feed Sources ──────────────────────────────────────────
RSS_FEEDS = {
    'Yahoo_Finance':    'https://finance.yahoo.com/news/rssindex',
    'Economic_Times':   'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
    'LiveMint':         'https://www.livemint.com/rss/markets',
    'Investing_Gold':   'https://www.investing.com/rss/news_25.rss',
    'Investing_Oil':    'https://www.investing.com/rss/news_8.rss',
    'Reuters_Business': 'https://feeds.reuters.com/reuters/businessNews',
    'Reuters_Markets':  'https://feeds.reuters.com/reuters/marketsNews',
    'Moneycontrol':     'https://www.moneycontrol.com/rss/marketreports.xml',
}

# ── Asset keyword mapping ─────────────────────────────────────
ASSET_KEYWORDS = {
    'NIFTY':   ['nifty', 'sensex', 'india', 'bse', 'nse',
                'indian market', 'rupee', 'rbi', 'sebi'],
    'SP500':   ['s&p', 'sp500', 'nasdaq', 'dow', 'fed',
                'federal reserve', 'wall street', 'us market',
                'us stocks', 'nyse', 'trump', 'tariff'],
    'Gold':    ['gold', 'xau', 'precious metal', 'bullion'],
    'Silver':  ['silver', 'xag'],
    'Crude':   ['crude', 'oil', 'wti', 'brent', 'opec',
                'petroleum', 'energy', 'iran', 'saudi'],
    'General': ['market', 'stock', 'economy', 'inflation',
                'gdp', 'recession', 'central bank', 'rate'],
}

# ═════════════════════════════════════════════════════════════
# SECTION 1 — LOAD FINBERT
# ═════════════════════════════════════════════════════════════

def load_finbert():
    try:
        from transformers import pipeline
        print("  Loading FinBERT model...")
        model = pipeline(
            'sentiment-analysis',
            model='ProsusAI/finbert',
            truncation=True,
            max_length=512
        )
        print("  ✅ FinBERT loaded successfully")
        return model, 'finbert'
    except Exception as e:
        print(f"  ⚠️ FinBERT unavailable ({e})")
        print("  Falling back to VADER...")
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        model = SentimentIntensityAnalyzer()
        return model, 'vader'

# ═════════════════════════════════════════════════════════════
# SECTION 2 — SCORE A HEADLINE
# ═════════════════════════════════════════════════════════════

def score_headline_finbert(headline, model):
    try:
        result = model(headline)[0]
        label  = result['label']
        conf   = result['score']
        if label == 'positive':
            return conf
        elif label == 'negative':
            return -conf
        else:
            return 0.0
    except:
        return 0.0

def score_headline_vader(headline, model):
    try:
        scores = model.polarity_scores(headline)
        return scores['compound']
    except:
        return 0.0

def score_headline(headline, model, model_type):
    if model_type == 'finbert':
        return score_headline_finbert(headline, model)
    else:
        return score_headline_vader(headline, model)

# ═════════════════════════════════════════════════════════════
# SECTION 3 — CLASSIFY SENTIMENT
# ═════════════════════════════════════════════════════════════

def classify_sentiment(score):
    if score >= 0.15:
        return 'Positive'
    elif score <= -0.15:
        return 'Negative'
    else:
        return 'Neutral'

def detect_markets(headline):
    headline_lower = headline.lower()
    detected = []
    for market, keywords in ASSET_KEYWORDS.items():
        if any(kw in headline_lower for kw in keywords):
            detected.append(market)
    return detected if detected else ['General']

# ═════════════════════════════════════════════════════════════
# SECTION 4 — FETCH HEADLINES
# ═════════════════════════════════════════════════════════════

def fetch_headlines():
    all_headlines = []
    print("\n  Fetching headlines from RSS feeds...")
    for source, url in RSS_FEEDS.items():
        try:
            feed  = feedparser.parse(url)
            count = 0
            for entry in feed.entries[:20]:
                headline = entry.get('title', '').strip()
                if headline and len(headline) > 10:
                    all_headlines.append({
                        'headline': headline,
                        'source':   source,
                    })
                    count += 1
            print(f"    {source}: {count} headlines")
        except Exception as e:
            print(f"    {source}: failed ({e})")
    print(f"  Total: {len(all_headlines)} headlines collected")
    return all_headlines

# ═════════════════════════════════════════════════════════════
# SECTION 5 — SCORE ALL HEADLINES
# ═════════════════════════════════════════════════════════════

def score_all_headlines(headlines, model, model_type):
    print(f"\n  Scoring {len(headlines)} headlines "
          f"using {model_type.upper()}...")
    results = []
    for i, item in enumerate(headlines):
        headline = item['headline']
        score    = score_headline(headline, model, model_type)
        markets  = detect_markets(headline)
        label    = classify_sentiment(score)
        results.append({
            'date':      datetime.now().strftime('%Y-%m-%d'),
            'headline':  headline,
            'source':    item['source'],
            'score':     round(score, 4),
            'sentiment': label,
            'markets':   ', '.join(markets),
        })
        if (i + 1) % 20 == 0:
            print(f"    Scored {i+1}/{len(headlines)}...")
    return pd.DataFrame(results)

# ═════════════════════════════════════════════════════════════
# SECTION 6 — SAVE TO DATABASE
# ═════════════════════════════════════════════════════════════

def save_to_database(df):
    conn = sqlite3.connect(DB_PATH)
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        conn.execute(
            "DELETE FROM SENTIMENT_DAILY WHERE date = ?", (today,)
        )
        # Only save columns that exist in original schema
        df_save = df[['date', 'headline', 'source',
                      'score', 'sentiment', 'markets']]
        df_save.to_sql('SENTIMENT_DAILY', conn,
                       if_exists='append', index=False)
        conn.commit()
        print(f"\n  ✅ {len(df)} headlines saved to database")
    except Exception as e:
        print(f"\n  ❌ Database save failed: {e}")
    finally:
        conn.close()

# ═════════════════════════════════════════════════════════════
# SECTION 7 — PRINT INSIGHTS
# ═════════════════════════════════════════════════════════════

def print_insights(df, model_type):
    print("\n" + "="*55)
    print(f"FINBERT SENTIMENT ENGINE — {datetime.now().strftime('%d %B %Y')}")
    print("="*55)

    overall = df['score'].mean()
    pos_pct = (df['sentiment'] == 'Positive').mean() * 100
    neu_pct = (df['sentiment'] == 'Neutral').mean()  * 100
    neg_pct = (df['sentiment'] == 'Negative').mean() * 100
    label   = 'POSITIVE' if overall > 0.05 else \
              'NEGATIVE' if overall < -0.05 else 'NEUTRAL'

    print(f"\nModel used: {model_type.upper()}")
    print(f"Overall score: {overall:+.4f} → {label}")
    print(f"Positive: {pos_pct:.1f}% | "
          f"Neutral: {neu_pct:.1f}% | "
          f"Negative: {neg_pct:.1f}%")

    print("\nSentiment by Market:")
    for market in ['NIFTY', 'SP500', 'Gold', 'Crude', 'General']:
        mask = df['markets'].str.contains(market, na=False)
        if mask.sum() > 0:
            avg = df[mask]['score'].mean()
            lbl = 'POSITIVE' if avg > 0.05 else \
                  'NEGATIVE' if avg < -0.05 else 'NEUTRAL'
            print(f"  {market:<10} {avg:+.4f}  → {lbl}")

    print("\nTop 3 Most Positive Headlines:")
    for _, row in df.nlargest(3, 'score').iterrows():
        print(f"  [{row['score']:+.3f}] {row['headline'][:65]}")

    print("\nTop 3 Most Negative Headlines:")
    for _, row in df.nsmallest(3, 'score').iterrows():
        print(f"  [{row['score']:+.3f}] {row['headline'][:65]}")

    print("="*55)

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_finbert_sentiment():
    print("\n" + "="*55)
    print("GMIS MODULE 15 — FINBERT SENTIMENT ENGINE")
    print("="*55)

    model, model_type = load_finbert()
    headlines = fetch_headlines()

    if not headlines:
        print("  ❌ No headlines fetched")
        return

    df = score_all_headlines(headlines, model, model_type)
    save_to_database(df)

    csv_path = os.path.join(BASE_PATH, 'data', 'sentiment_today.csv')
    df.to_csv(csv_path, index=False)
    print(f"  ✅ CSV saved to {csv_path}")

    print_insights(df, model_type)
    return df

if __name__ == "__main__":
    run_finbert_sentiment()
