# ============================================================
# GMIS 2.0 — MODULE 17 — HISTORICAL ANALOG ENGINE
# Finds past periods most similar to current conditions
# Outputs directional tendencies and probability ranges
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')

ASSETS          = ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']
TOP_N_ANALOGS   = 5
MIN_SIMILARITY  = 0.70
FORWARD_DAYS    = [10, 30, 60]

# ═════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ═════════════════════════════════════════════════════════════

def load_all_data():
    """Load all required data from database."""
    conn = sqlite3.connect(DB_PATH)

    # Prices
    prices = {}
    for asset, table in [
        ('NIFTY',  'NIFTY50'),
        ('SP500',  'SP500'),
        ('Gold',   'GOLD'),
        ('Silver', 'SILVER'),
        ('Crude',  'CRUDE_WTI'),
        ('VIX',    'VIX_US'),
    ]:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        close_col = [c for c in df.columns
                     if 'Close' in c or 'close' in c]
        col = close_col[0] if close_col else df.columns[0]
        prices[asset] = df[col].dropna()

    # Yields
    y10 = pd.read_sql("SELECT * FROM US_10Y_YIELD", conn)
    y10['Date'] = pd.to_datetime(y10['Date'])
    y10 = y10.set_index('Date').sort_index().iloc[:, 0].dropna()

    y2 = pd.read_sql("SELECT * FROM US_2Y_YIELD", conn)
    y2['Date'] = pd.to_datetime(y2['Date'])
    y2 = y2.set_index('Date').sort_index().iloc[:, 0].dropna()

    # Signals
    signals = pd.read_sql("SELECT * FROM SIGNALS", conn)
    signals['Date'] = pd.to_datetime(signals['Date'])
    signals = signals.set_index('Date').sort_index()

    conn.close()

    return prices, y10, y2, signals

# ═════════════════════════════════════════════════════════════
# SECTION 2 — BUILD FEATURE MATRIX
# ═════════════════════════════════════════════════════════════

def build_feature_matrix(prices, y10, y2, signals):
    """
    For every historical date, build a feature vector
    that describes the macro state on that day.

    Features (all normalised):
    1.  VIX level (percentile)
    2.  VIX 20-day change
    3.  Yield spread (10Y - 2Y)
    4.  Yield spread 20-day change
    5.  Gold 20-day momentum
    6.  Gold 60-day momentum
    7.  SP500 20-day momentum
    8.  SP500 60-day momentum
    9.  NIFTY 20-day momentum
    10. NIFTY 60-day momentum
    11. Crude 20-day momentum
    12. SP500 vs 200-day MA ratio
    """
    print("  Building feature matrix...")

    vix    = prices['VIX']
    gold   = prices['Gold']
    sp500  = prices['SP500']
    nifty  = prices['NIFTY']
    crude  = prices['Crude']

    # Align all series to common dates
    df = pd.DataFrame({
        'VIX':       vix,
        'Gold':      gold,
        'SP500':     sp500,
        'NIFTY':     nifty,
        'Crude':     crude,
    }).dropna()

    # Add yields on same dates
    spread = (y10 - y2).reindex(df.index).ffill()
    df['Yield_Spread'] = spread

    # Feature engineering
    features = pd.DataFrame(index=df.index)

    # VIX features
    vix_pct = df['VIX'].rank(pct=True)
    features['vix_pct']      = vix_pct
    features['vix_chg_20d']  = df['VIX'].pct_change(20)

    # Yield features
    features['yield_spread']      = df['Yield_Spread']
    features['yield_spread_chg']  = df['Yield_Spread'].diff(20)

    # Momentum features (% change over N days)
    for asset in ['Gold', 'SP500', 'NIFTY', 'Crude']:
        features[f'{asset}_mom_20d'] = df[asset].pct_change(20)
        features[f'{asset}_mom_60d'] = df[asset].pct_change(60)

    # SP500 vs 200-day MA
    ma200 = df['SP500'].rolling(200).mean()
    features['sp500_vs_ma200'] = (df['SP500'] / ma200) - 1

    # Drop NaN rows (first ~200 days needed for MA200)
    features = features.dropna()

    print(f"  Feature matrix: {len(features)} rows × "
          f"{len(features.columns)} features")
    print(f"  Date range: {features.index[0].date()} → "
          f"{features.index[-1].date()}")

    return features, df

# ═════════════════════════════════════════════════════════════
# SECTION 3 — FIND ANALOGS
# ═════════════════════════════════════════════════════════════

def find_historical_analogs(features, current_date=None,
                             top_n=TOP_N_ANALOGS,
                             min_similarity=MIN_SIMILARITY):
    """
    Find the N most similar historical dates to today.

    Uses cosine similarity on normalised feature vectors.
    Excludes recent 90 days to avoid near-duplicate matches.
    """
    if current_date is None:
        current_date = features.index[-1]

    # Current state vector
    current_vector = features.loc[current_date].values.reshape(1, -1)

    # Historical states — exclude recent 90 days
    cutoff = current_date - pd.Timedelta(days=90)
    historical = features[features.index < cutoff]

    if len(historical) < 100:
        print("  ⚠️ Not enough historical data for analogs")
        return pd.DataFrame()

    # Normalise using StandardScaler
    scaler = StandardScaler()
    hist_scaled    = scaler.fit_transform(historical.values)
    current_scaled = scaler.transform(current_vector)

    # Calculate cosine similarity
    similarities = cosine_similarity(
        current_scaled, hist_scaled
    )[0]

    # Get top N
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_dates   = historical.index[top_indices]
    top_scores  = similarities[top_indices]

    # Build results dataframe
    analogs = pd.DataFrame({
        'analog_date': top_dates,
        'similarity':  top_scores,
    })

    # Filter by minimum similarity
    analogs = analogs[analogs['similarity'] >= min_similarity]

    return analogs

# ═════════════════════════════════════════════════════════════
# SECTION 4 — EXTRACT OUTCOMES
# ═════════════════════════════════════════════════════════════

def extract_analog_outcomes(analogs, prices):
    """
    For each analog date, look at what happened
    to each asset in the following 10, 30, 60 days.
    """
    if analogs.empty:
        return {}

    outcomes = {asset: {days: [] for days in FORWARD_DAYS}
                for asset in ASSETS}

    for _, row in analogs.iterrows():
        analog_date  = row['analog_date']
        similarity   = row['similarity']

        for asset in ASSETS:
            price_s = prices[asset]

            # Get price at analog date
            try:
                base_price = price_s.loc[analog_date]
            except:
                # Find nearest date
                idx = price_s.index.get_indexer(
                    [analog_date], method='nearest'
                )[0]
                base_price = price_s.iloc[idx]
                analog_date = price_s.index[idx]

            # Forward returns
            future_prices = price_s[price_s.index > analog_date]

            for days in FORWARD_DAYS:
                if len(future_prices) >= days:
                    future_price  = future_prices.iloc[days - 1]
                    forward_return = (future_price / base_price - 1) * 100
                    outcomes[asset][days].append({
                        'analog_date':    analog_date,
                        'similarity':     similarity,
                        'forward_return': forward_return,
                    })

    return outcomes

# ═════════════════════════════════════════════════════════════
# SECTION 5 — SUMMARISE OUTCOMES
# ═════════════════════════════════════════════════════════════

def summarise_outcomes(outcomes):
    """
    Convert raw outcomes into probability statistics.
    """
    summary = {}

    for asset in ASSETS:
        summary[asset] = {}
        for days in FORWARD_DAYS:
            data = outcomes[asset][days]
            if not data:
                continue

            returns = [d['forward_return'] for d in data]
            arr     = np.array(returns)

            positive_count = (arr > 0).sum()
            total_count    = len(arr)

            # Cap probability between 30% and 80%
            raw_prob = positive_count / total_count
            prob     = max(0.30, min(0.80, raw_prob))

            summary[asset][days] = {
                'median_return':  round(float(np.median(arr)), 2),
                'mean_return':    round(float(np.mean(arr)),   2),
                'p25_return':     round(float(np.percentile(arr, 25)), 2),
                'p75_return':     round(float(np.percentile(arr, 75)), 2),
                'p10_return':     round(float(np.percentile(arr, 10)), 2),
                'p90_return':     round(float(np.percentile(arr, 90)), 2),
                'prob_positive':  round(prob * 100, 1),
                'n_analogs':      total_count,
            }

    return summary

# ═════════════════════════════════════════════════════════════
# SECTION 6 — SAVE TO DATABASE
# ═════════════════════════════════════════════════════════════

def save_analog_results(analogs, summary):
    """Save analog results to database."""
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime('%Y-%m-%d')

    try:
        # Save analog dates
        if not analogs.empty:
            analogs_save = analogs.copy()
            analogs_save['run_date'] = today
            analogs_save['analog_date'] = analogs_save[
                'analog_date'].astype(str)
            analogs_save.to_sql('ANALOG_DATES', conn,
                                if_exists='replace', index=False)

        # Save summary statistics
        rows = []
        for asset in ASSETS:
            for days in FORWARD_DAYS:
                if days in summary.get(asset, {}):
                    s = summary[asset][days]
                    rows.append({
                        'run_date':     today,
                        'asset':        asset,
                        'forward_days': days,
                        **s
                    })

        if rows:
            df_save = pd.DataFrame(rows)
            df_save.to_sql('ANALOG_OUTCOMES', conn,
                           if_exists='replace', index=False)

        conn.commit()
        print(f"\n  ✅ Analog results saved to database")

    except Exception as e:
        print(f"\n  ❌ Save failed: {e}")
    finally:
        conn.close()

# ═════════════════════════════════════════════════════════════
# SECTION 7 — PRINT RESULTS
# ═════════════════════════════════════════════════════════════

def print_analog_results(analogs, summary, features):
    """Print the analog findings clearly."""

    print("\n" + "="*65)
    print("HISTORICAL ANALOG ENGINE — RESULTS")
    print("="*65)

    current_date = features.index[-1]
    print(f"\nCurrent date: {current_date.strftime('%d %B %Y')}")
    print(f"Analogs found: {len(analogs)}")

    if analogs.empty:
        print("  No analogs found above similarity threshold")
        return

    # Print analog dates
    print("\n📅 Most Similar Historical Periods:")
    print(f"  {'Date':<15} {'Similarity':>10} {'How Similar'}")
    print(f"  {'-'*45}")
    for _, row in analogs.iterrows():
        date = pd.Timestamp(row['analog_date'])
        sim  = row['similarity']
        bar  = '█' * int(sim * 20)
        print(f"  {date.strftime('%d %b %Y'):<15} "
              f"{sim:>9.1%}  {bar}")

    # Print forward return expectations
    print("\n📊 Forward Return Expectations (based on analogs):")
    print(f"\n  {'Asset':<8} {'Horizon':<10} "
          f"{'Prob+':<8} {'Median':>8} "
          f"{'Range (P25-P75)':>20}")
    print(f"  {'-'*58}")

    for asset in ASSETS:
        for days in FORWARD_DAYS:
            if days not in summary.get(asset, {}):
                continue
            s = summary[asset][days]

            prob   = s['prob_positive']
            median = s['median_return']
            p25    = s['p25_return']
            p75    = s['p75_return']

            # Direction indicator
            if prob >= 60:
                dir_icon = '↑'
            elif prob <= 40:
                dir_icon = '↓'
            else:
                dir_icon = '→'

            print(f"  {asset:<8} {days:>3}d ahead  "
                  f"{dir_icon} {prob:>5.1f}%  "
                  f"{median:>+7.1f}%  "
                  f"({p25:+.1f}% to {p75:+.1f}%)")

        print()

    print("="*65)
    print("\n⚠️  Note: These are historical tendencies, not predictions.")
    print("   Based on similar past conditions, not guaranteed outcomes.")
    print("="*65)

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_historical_analog():
    print("\n" + "="*65)
    print("GMIS MODULE 17 — HISTORICAL ANALOG ENGINE")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    # Load data
    print("\nLoading data...")
    prices, y10, y2, signals = load_all_data()

    # Build feature matrix
    features, price_df = build_feature_matrix(
        prices, y10, y2, signals
    )

    # Find analogs
    print("\nSearching for historical analogs...")
    analogs = find_historical_analogs(features)

    if analogs.empty:
        print("  No analogs found above threshold")
        return

    print(f"  Found {len(analogs)} analog periods")

    # Extract outcomes
    print("\nExtracting historical outcomes...")
    outcomes = extract_analog_outcomes(analogs, prices)

    # Summarise
    summary = summarise_outcomes(outcomes)

    # Save
    print("\nSaving results...")
    save_analog_results(analogs, summary)

    # Print
    print_analog_results(analogs, summary, features)

    return analogs, summary

if __name__ == "__main__":
    run_historical_analog()
