# ============================================================
# MODULE 43: OPTIONS MARKET INTELLIGENCE
# NSE FO Bhavcopy → PCR, Max Pain, IV Term Structure, Skew
# Fallback: OPTIONS_DATA table (Module 26) if Bhavcopy unavailable
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import sys
import argparse
import asyncio
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--no-telegram', action='store_true')
args = parser.parse_args()
SEND_TELEGRAM = not args.no_telegram

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')

# ── Telegram ─────────────────────────────────────────────────
async def send_telegram(msg: str):
    try:
        from utils.telegram_utils import send_message
        await send_message(msg)
    except Exception as e:
        print(f"  [Telegram] {e}")

def notify(msg: str):
    if SEND_TELEGRAM:
        asyncio.run(send_telegram(msg))

# ── Constants ────────────────────────────────────────────────
BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com",
}

# PCR thresholds
PCR_EXTREME_BEARISH = 0.60
PCR_BEARISH         = 0.80
PCR_BULLISH         = 1.20
PCR_EXTREME_BULLISH = 1.40

# Max pain deviation thresholds (% from spot)
MAX_PAIN_NEAR_PCT  = 1.5   # < 1.5% → no bias
MAX_PAIN_MED_PCT   = 3.0   # 1.5–3% → moderate bias
MAX_PAIN_FAR_PCT   = 5.0   # > 3% → strong bias

# Skew thresholds
SKEW_STEEP_BEAR = 1.20   # PE/CE ratio > 1.20 → elevated put demand
SKEW_FLAT       = 0.80   # PE/CE ratio < 0.80 → elevated call demand

# IV term structure thresholds
IV_CONTANGO_THRESHOLD   = 1.05   # far/near > 1.05 → normal contango
IV_BACKWARDATION_THRESH = 0.95   # far/near < 0.95 → backwardation


# ── 1. Fetch Bhavcopy ────────────────────────────────────────
def fetch_bhavcopy(date_str: str) -> pd.DataFrame | None:
    """Download and parse NSE FO Bhavcopy for a given date (YYYYMMDD)."""
    url = BHAVCOPY_URL.format(date=date_str)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = z.namelist()[0]
        df = pd.read_csv(z.open(csv_name), low_memory=False)
        return df
    except Exception as e:
        print(f"  [Bhavcopy] {e}")
        return None


def get_nifty_options(lookback_days: int = 5) -> tuple[pd.DataFrame | None, str | None]:
    """Try recent trading days until Bhavcopy is available."""
    today = datetime.now()
    for offset in range(lookback_days):
        d = today - timedelta(days=offset)
        if d.weekday() >= 5:          # skip weekends
            continue
        date_str = d.strftime("%Y%m%d")
        raw = fetch_bhavcopy(date_str)
        if raw is not None and len(raw) > 0:
            nifty = raw[raw['TckrSymb'] == 'NIFTY'].copy()
            if len(nifty) > 0:
                print(f"  Bhavcopy loaded: {date_str}, NIFTY rows={len(nifty)}")
                return nifty, d.strftime("%Y-%m-%d")
    return None, None


# ── 2. Fallback: OPTIONS_DATA from Module 26 ─────────────────
def get_fallback_pcr(conn: sqlite3.Connection) -> dict:
    """Read most recent PCR values from OPTIONS_DATA table."""
    try:
        row = pd.read_sql(
            "SELECT * FROM OPTIONS_DATA ORDER BY date DESC LIMIT 1", conn
        )
        if len(row) == 0:
            return {}
        r = row.iloc[0]
        return {
            'pcr_oi':   float(r.get('pcr_oi',   0)),
            'pcr_vol':  float(r.get('pcr_vol',  0)),
            'date':     str(r.get('date', '')),
        }
    except Exception as e:
        print(f"  [Fallback PCR] {e}")
        return {}


# ── 3. Core Calculations ─────────────────────────────────────
def compute_pcr(df: pd.DataFrame) -> dict:
    """Put-Call Ratio by OI and Volume (near-expiry only)."""
    # Sort expiry dates — use only the nearest two
    df['XpryDt'] = pd.to_datetime(df['XpryDt'], errors='coerce')
    expiries = sorted(df['XpryDt'].dropna().unique())
    near_expiry = expiries[:1] if len(expiries) >= 1 else expiries

    near = df[df['XpryDt'].isin(near_expiry)]
    ce = near[near['OptnTp'] == 'CE']
    pe = near[near['OptnTp'] == 'PE']

    total_ce_oi  = ce['OpnIntrst'].sum()
    total_pe_oi  = pe['OpnIntrst'].sum()
    total_ce_vol = ce['TtlTradgVol'].sum()
    total_pe_vol = pe['TtlTradgVol'].sum()

    pcr_oi  = (total_pe_oi  / total_ce_oi)  if total_ce_oi  > 0 else 0
    pcr_vol = (total_pe_vol / total_ce_vol) if total_ce_vol > 0 else 0

    return {
        'pcr_oi':       round(pcr_oi, 4),
        'pcr_vol':      round(pcr_vol, 4),
        'total_ce_oi':  int(total_ce_oi),
        'total_pe_oi':  int(total_pe_oi),
        'total_ce_vol': int(total_ce_vol),
        'total_pe_vol': int(total_pe_vol),
    }


def interpret_pcr(pcr_oi: float) -> str:
    if pcr_oi >= PCR_EXTREME_BULLISH:
        return 'EXTREME_BULLISH'
    elif pcr_oi >= PCR_BULLISH:
        return 'BULLISH'
    elif pcr_oi <= PCR_EXTREME_BEARISH:
        return 'EXTREME_BEARISH'
    elif pcr_oi <= PCR_BEARISH:
        return 'BEARISH'
    else:
        return 'NEUTRAL'


def compute_max_pain(df: pd.DataFrame, spot: float) -> dict:
    """Max pain = strike where total in-the-money loss for writers is minimised."""
    df['XpryDt'] = pd.to_datetime(df['XpryDt'], errors='coerce')
    expiries = sorted(df['XpryDt'].dropna().unique())
    near_exp = expiries[0] if expiries else None
    if near_exp is None:
        return {}

    near = df[df['XpryDt'] == near_exp].copy()
    near['StrkPric'] = pd.to_numeric(near['StrkPric'], errors='coerce')
    near = near.dropna(subset=['StrkPric'])

    strikes = sorted(near['StrkPric'].unique())
    if len(strikes) == 0:
        return {}

    pain = {}
    for s in strikes:
        ce_loss = near[(near['OptnTp'] == 'CE') & (near['StrkPric'] < s)].apply(
            lambda row: max(0, s - row['StrkPric']) * row['OpnIntrst'], axis=1
        ).sum()
        pe_loss = near[(near['OptnTp'] == 'PE') & (near['StrkPric'] > s)].apply(
            lambda row: max(0, row['StrkPric'] - s) * row['OpnIntrst'], axis=1
        ).sum()
        pain[s] = ce_loss + pe_loss

    max_pain_strike = min(pain, key=pain.get)
    deviation_pct   = ((max_pain_strike - spot) / spot) * 100

    if abs(deviation_pct) < MAX_PAIN_NEAR_PCT:
        bias = 'NEUTRAL'
    elif deviation_pct > MAX_PAIN_MED_PCT:
        bias = 'BULLISH_GRAVITATE'
    elif deviation_pct < -MAX_PAIN_MED_PCT:
        bias = 'BEARISH_GRAVITATE'
    elif deviation_pct > 0:
        bias = 'MILDLY_BULLISH'
    else:
        bias = 'MILDLY_BEARISH'

    return {
        'max_pain_strike':  float(max_pain_strike),
        'deviation_pct':    round(deviation_pct, 2),
        'max_pain_bias':    bias,
        'expiry_used':      str(near_exp.date()) if hasattr(near_exp, 'date') else str(near_exp),
    }


def compute_skew(df: pd.DataFrame, spot: float, width_pct: float = 2.0) -> dict:
    """
    Approximate skew: average PE price / CE price at equidistant strikes from spot.
    Uses near-expiry strikes within ±width_pct of spot.
    """
    df['XpryDt'] = pd.to_datetime(df['XpryDt'], errors='coerce')
    expiries = sorted(df['XpryDt'].dropna().unique())
    near_exp = expiries[0] if expiries else None
    if near_exp is None:
        return {'skew_ratio': None, 'skew_signal': 'UNAVAILABLE'}

    near = df[df['XpryDt'] == near_exp].copy()
    near['StrkPric'] = pd.to_numeric(near['StrkPric'], errors='coerce')
    near['ClsPric']  = pd.to_numeric(near['ClsPric'],  errors='coerce')
    near['LastPric'] = pd.to_numeric(near['LastPric'], errors='coerce')
    near['price']    = near['ClsPric'].fillna(near['LastPric'])
    near = near.dropna(subset=['StrkPric', 'price'])

    low_bound  = spot * (1 - width_pct / 100)
    high_bound = spot * (1 + width_pct / 100)

    puts  = near[(near['OptnTp'] == 'PE') & (near['StrkPric'] >= low_bound)  & (near['StrkPric'] < spot)]
    calls = near[(near['OptnTp'] == 'CE') & (near['StrkPric'] <= high_bound) & (near['StrkPric'] > spot)]

    if len(puts) == 0 or len(calls) == 0:
        return {'skew_ratio': None, 'skew_signal': 'INSUFFICIENT_STRIKES'}

    avg_put  = puts['price'].mean()
    avg_call = calls['price'].mean()

    skew_ratio = (avg_put / avg_call) if avg_call > 0 else None

    if skew_ratio is None:
        skew_signal = 'UNAVAILABLE'
    elif skew_ratio >= SKEW_STEEP_BEAR:
        skew_signal = 'ELEVATED_PUT_DEMAND'
    elif skew_ratio <= SKEW_FLAT:
        skew_signal = 'ELEVATED_CALL_DEMAND'
    else:
        skew_signal = 'BALANCED'

    return {
        'skew_ratio':   round(skew_ratio, 4) if skew_ratio else None,
        'skew_signal':  skew_signal,
        'avg_put_px':   round(avg_put, 2),
        'avg_call_px':  round(avg_call, 2),
    }


def compute_iv_term_structure(conn: sqlite3.Connection, df: pd.DataFrame | None) -> dict:
    """
    Near-term IV: India VIX (VIX_INDIA table).
    Far-term IV: approximated as mean of last 20-day VIX readings.
    Term structure: far/near ratio.
    Also compute 20-day VIX z-score.
    """
    try:
        vix_df = pd.read_sql(
            "SELECT Date, \"Unnamed: 4\" AS vix_close FROM VIX_INDIA ORDER BY Date DESC LIMIT 30", conn
        )
        if len(vix_df) == 0:
            raise ValueError("No VIX_INDIA data")

        vix_df = vix_df.rename(columns={'vix_close': 'close'})
        vix_df = vix_df.sort_values('Date')
        near_iv  = float(vix_df['close'].iloc[-1])
        far_iv   = float(vix_df['close'].mean())   # 30-day avg as "far" proxy
        ts_ratio = round(far_iv / near_iv, 4) if near_iv > 0 else None

        vix_mean = vix_df['close'].mean()
        vix_std  = vix_df['close'].std()
        iv_zscore = round((near_iv - vix_mean) / vix_std, 2) if vix_std > 0 else 0.0

        if ts_ratio is None:
            ts_signal = 'UNAVAILABLE'
        elif ts_ratio >= IV_CONTANGO_THRESHOLD:
            ts_signal = 'CONTANGO'     # normal; far more expensive
        elif ts_ratio <= IV_BACKWARDATION_THRESH:
            ts_signal = 'BACKWARDATION'  # stress; near more expensive
        else:
            ts_signal = 'FLAT'

        iv_regime = (
            'CRISIS'    if near_iv > 30 else
            'ELEVATED'  if near_iv > 20 else
            'NORMAL'    if near_iv > 12 else
            'COMPRESSED'
        )

        return {
            'near_iv':    round(near_iv, 2),
            'far_iv':     round(far_iv, 2),
            'ts_ratio':   ts_ratio,
            'ts_signal':  ts_signal,
            'iv_zscore':  iv_zscore,
            'iv_regime':  iv_regime,
        }
    except Exception as e:
        print(f"  [IV Term Structure] {e}")
        return {
            'near_iv':   None, 'far_iv': None,
            'ts_ratio':  None, 'ts_signal': 'UNAVAILABLE',
            'iv_zscore': 0.0,  'iv_regime': 'UNKNOWN',
        }


# ── 4. Composite Options Signal ──────────────────────────────
def build_composite_signal(pcr_signal: str, max_pain_bias: str,
                            skew_signal: str, iv_regime: str) -> dict:
    """
    Synthesise a composite directional bias for NIFTY.
    Returns score ∈ [-1, +1] and label.
    """
    score = 0.0
    components = []

    # PCR contribution (±0.35)
    pcr_map = {
        'EXTREME_BULLISH': -0.35,   # contrarian: too many puts → mean revert up... but
                                     # Actually high PCR = more puts = bearish sentiment
                                     # but contrarians see this as bullish floor
                                     # We use the contrarian interpretation
        'BULLISH':         -0.20,
        'NEUTRAL':          0.00,
        'BEARISH':         +0.20,
        'EXTREME_BEARISH': +0.35,
    }
    # Note: extreme bearish PCR (very few puts) means market is complacent → bearish for market
    # conventional: low PCR = bullish (more calls), high PCR = bearish (more puts)?
    # Standard convention: PCR_OI > 1 → more puts → bearish sentiment, but contrarian = potential support
    # We use CONTRARIAN interpretation (common for India options market)
    # High PCR OI (>1.2) = extreme put buying = bearish consensus = contrarian BULLISH signal
    pcr_contrib = {
        'EXTREME_BULLISH': +0.35,   # high PCR = tons of puts = contrarian bullish
        'BULLISH':         +0.20,
        'NEUTRAL':          0.00,
        'BEARISH':         -0.20,
        'EXTREME_BEARISH': -0.35,   # very low PCR = complacency = contrarian bearish
    }
    v = pcr_contrib.get(pcr_signal, 0.0)
    score += v
    components.append(f"PCR({pcr_signal}:{v:+.2f})")

    # Max pain contribution (±0.30)
    mp_map = {
        'BULLISH_GRAVITATE': +0.30,
        'MILDLY_BULLISH':    +0.15,
        'NEUTRAL':            0.00,
        'MILDLY_BEARISH':    -0.15,
        'BEARISH_GRAVITATE': -0.30,
    }
    v = mp_map.get(max_pain_bias, 0.0)
    score += v
    components.append(f"MaxPain({max_pain_bias}:{v:+.2f})")

    # Skew contribution (±0.20)
    skew_map = {
        'ELEVATED_PUT_DEMAND':  -0.20,   # put skew = fear = bearish
        'BALANCED':              0.00,
        'ELEVATED_CALL_DEMAND': +0.20,   # call skew = greed = bullish
        'INSUFFICIENT_STRIKES':  0.00,
        'UNAVAILABLE':           0.00,
    }
    v = skew_map.get(skew_signal, 0.0)
    score += v
    components.append(f"Skew({skew_signal}:{v:+.2f})")

    # IV regime contribution (±0.15)
    iv_map = {
        'COMPRESSED': -0.10,   # low vol = complacency = slight bearish risk
        'NORMAL':      0.00,
        'ELEVATED':   -0.05,   # some fear, slightly negative
        'CRISIS':     +0.10,   # extreme fear = contrarian positive
        'UNKNOWN':     0.00,
    }
    v = iv_map.get(iv_regime, 0.0)
    score += v
    components.append(f"IV({iv_regime}:{v:+.2f})")

    score = max(-1.0, min(1.0, round(score, 4)))

    if score >= 0.40:
        label = 'STRONG_BULLISH'
    elif score >= 0.15:
        label = 'BULLISH'
    elif score <= -0.40:
        label = 'STRONG_BEARISH'
    elif score <= -0.15:
        label = 'BEARISH'
    else:
        label = 'NEUTRAL'

    return {
        'composite_score':  score,
        'composite_signal': label,
        'components':       ' | '.join(components),
    }


# ── 5. Historical Context ────────────────────────────────────
def get_historical_context(conn: sqlite3.Connection) -> dict:
    """Pull 30-day PCR history from OPTIONS_INTELLIGENCE for context."""
    try:
        hist = pd.read_sql(
            "SELECT date, pcr_oi, composite_score FROM OPTIONS_INTELLIGENCE "
            "ORDER BY date DESC LIMIT 30", conn
        )
        if len(hist) < 3:
            return {'pcr_percentile': None, 'signal_trend': 'INSUFFICIENT_HISTORY'}

        hist = hist.sort_values('date')
        latest_pcr = hist['pcr_oi'].iloc[-1]
        pcr_pct = float(np.percentile(hist['pcr_oi'], [p for p in range(0, 101)][
            int((hist['pcr_oi'] <= latest_pcr).mean() * 100)
        ])) if len(hist) > 1 else None

        recent_scores = hist['composite_score'].dropna().tolist()
        if len(recent_scores) >= 5:
            trend = np.polyfit(range(len(recent_scores[-5:])), recent_scores[-5:], 1)[0]
            signal_trend = 'IMPROVING' if trend > 0.01 else ('DETERIORATING' if trend < -0.01 else 'STABLE')
        else:
            signal_trend = 'STABLE'

        return {
            'pcr_30d_mean':   round(hist['pcr_oi'].mean(), 4),
            'pcr_30d_std':    round(hist['pcr_oi'].std(), 4),
            'signal_trend':   signal_trend,
        }
    except Exception:
        return {'pcr_30d_mean': None, 'pcr_30d_std': None, 'signal_trend': 'NEW'}


# ── 6. DB Save ───────────────────────────────────────────────
def save_to_db(conn: sqlite3.Connection, row: dict):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS OPTIONS_INTELLIGENCE (
            date                TEXT,
            data_source         TEXT,
            spot_price          REAL,
            pcr_oi              REAL,
            pcr_vol             REAL,
            pcr_signal          TEXT,
            total_ce_oi         INTEGER,
            total_pe_oi         INTEGER,
            max_pain_strike     REAL,
            max_pain_deviation  REAL,
            max_pain_bias       TEXT,
            skew_ratio          REAL,
            skew_signal         TEXT,
            near_iv             REAL,
            far_iv              REAL,
            iv_ts_ratio         REAL,
            iv_ts_signal        TEXT,
            iv_zscore           REAL,
            iv_regime           TEXT,
            composite_score     REAL,
            composite_signal    TEXT,
            signal_components   TEXT,
            pcr_30d_mean        REAL,
            signal_trend        TEXT,
            PRIMARY KEY (date)
        )
    """)
    conn.execute("""
        INSERT OR REPLACE INTO OPTIONS_INTELLIGENCE VALUES (
            :date, :data_source, :spot_price,
            :pcr_oi, :pcr_vol, :pcr_signal,
            :total_ce_oi, :total_pe_oi,
            :max_pain_strike, :max_pain_deviation, :max_pain_bias,
            :skew_ratio, :skew_signal,
            :near_iv, :far_iv, :iv_ts_ratio, :iv_ts_signal, :iv_zscore, :iv_regime,
            :composite_score, :composite_signal, :signal_components,
            :pcr_30d_mean, :signal_trend
        )
    """, row)
    conn.commit()


# ── MAIN ─────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("MODULE 43: OPTIONS MARKET INTELLIGENCE")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    today_str = datetime.now().strftime("%Y-%m-%d")

    # ── Step 1: Try Bhavcopy ──────────────────────────────────
    nifty_df, bhavcopy_date = get_nifty_options()
    data_source = 'BHAVCOPY'

    # ── Step 2: Fallback if needed ────────────────────────────
    fallback_pcr = get_fallback_pcr(conn)
    if nifty_df is None or len(nifty_df) == 0:
        print("  Bhavcopy unavailable — using OPTIONS_DATA fallback")
        data_source = 'OPTIONS_DATA_FALLBACK'

    # ── Step 3: Spot price ────────────────────────────────────
    spot = None
    if nifty_df is not None and len(nifty_df) > 0:
        spot_vals = pd.to_numeric(nifty_df['UndrlygPric'], errors='coerce').dropna()
        if len(spot_vals) > 0:
            spot = float(spot_vals.iloc[0])

    if spot is None:
        # Fallback to NIFTY50 DB
        try:
            row = pd.read_sql(
                "SELECT close FROM NIFTY50 ORDER BY date DESC LIMIT 1", conn
            )
            if len(row) > 0:
                spot = float(row.iloc[0]['close'])
        except Exception:
            pass

    if spot is None:
        spot = 22500.0
        print("  [WARN] Spot price defaulted to 22500")

    print(f"  Spot price: {spot:,.2f}")

    # ── Step 4: PCR ───────────────────────────────────────────
    if nifty_df is not None and len(nifty_df) > 0:
        pcr_data = compute_pcr(nifty_df)
    else:
        pcr_data = {
            'pcr_oi':       fallback_pcr.get('pcr_oi', 1.0),
            'pcr_vol':      fallback_pcr.get('pcr_vol', 1.0),
            'total_ce_oi':  0,
            'total_pe_oi':  0,
            'total_ce_vol': 0,
            'total_pe_vol': 0,
        }

    pcr_signal = interpret_pcr(pcr_data['pcr_oi'])
    print(f"  PCR OI={pcr_data['pcr_oi']:.3f}  PCR Vol={pcr_data['pcr_vol']:.3f}  → {pcr_signal}")

    # ── Step 5: Max Pain ──────────────────────────────────────
    if nifty_df is not None and len(nifty_df) > 0:
        mp_data = compute_max_pain(nifty_df, spot)
    else:
        mp_data = {'max_pain_strike': spot, 'deviation_pct': 0.0, 'max_pain_bias': 'NEUTRAL', 'expiry_used': 'N/A'}

    print(f"  Max Pain: {mp_data.get('max_pain_strike', 'N/A')} "
          f"({mp_data.get('deviation_pct', 0):+.2f}% from spot) → {mp_data.get('max_pain_bias', 'N/A')}")

    # ── Step 6: Skew ──────────────────────────────────────────
    if nifty_df is not None and len(nifty_df) > 0:
        skew_data = compute_skew(nifty_df, spot)
    else:
        skew_data = {'skew_ratio': None, 'skew_signal': 'UNAVAILABLE', 'avg_put_px': 0, 'avg_call_px': 0}

    print(f"  Skew: ratio={skew_data.get('skew_ratio', 'N/A')} → {skew_data['skew_signal']}")

    # ── Step 7: IV Term Structure ─────────────────────────────
    iv_data = compute_iv_term_structure(conn, nifty_df)
    print(f"  IV: near={iv_data['near_iv']} far={iv_data['far_iv']} "
          f"TS={iv_data['ts_signal']} regime={iv_data['iv_regime']} z={iv_data['iv_zscore']}")

    # ── Step 8: Composite Signal ──────────────────────────────
    composite = build_composite_signal(
        pcr_signal,
        mp_data.get('max_pain_bias', 'NEUTRAL'),
        skew_data['skew_signal'],
        iv_data['iv_regime'],
    )
    print(f"\n  COMPOSITE: {composite['composite_signal']} (score={composite['composite_score']:+.4f})")
    print(f"  {composite['components']}")

    # ── Step 9: Historical Context ────────────────────────────
    hist_ctx = get_historical_context(conn)

    # ── Step 10: Save ─────────────────────────────────────────
    record = {
        'date':                today_str,
        'data_source':         data_source,
        'spot_price':          spot,
        'pcr_oi':              pcr_data['pcr_oi'],
        'pcr_vol':             pcr_data['pcr_vol'],
        'pcr_signal':          pcr_signal,
        'total_ce_oi':         pcr_data.get('total_ce_oi', 0),
        'total_pe_oi':         pcr_data.get('total_pe_oi', 0),
        'max_pain_strike':     mp_data.get('max_pain_strike'),
        'max_pain_deviation':  mp_data.get('deviation_pct', 0),
        'max_pain_bias':       mp_data.get('max_pain_bias', 'NEUTRAL'),
        'skew_ratio':          skew_data.get('skew_ratio'),
        'skew_signal':         skew_data['skew_signal'],
        'near_iv':             iv_data['near_iv'],
        'far_iv':              iv_data['far_iv'],
        'iv_ts_ratio':         iv_data['ts_ratio'],
        'iv_ts_signal':        iv_data['ts_signal'],
        'iv_zscore':           iv_data['iv_zscore'],
        'iv_regime':           iv_data['iv_regime'],
        'composite_score':     composite['composite_score'],
        'composite_signal':    composite['composite_signal'],
        'signal_components':   composite['components'],
        'pcr_30d_mean':        hist_ctx.get('pcr_30d_mean'),
        'signal_trend':        hist_ctx.get('signal_trend', 'NEW'),
    }

    save_to_db(conn, record)
    print(f"\n  Saved to OPTIONS_INTELLIGENCE for {today_str}")

    # ── Step 11: Telegram Alert ───────────────────────────────
    if SEND_TELEGRAM and abs(composite['composite_score']) >= 0.35:
        msg = (
            f"*Options Intelligence — {today_str}*\n"
            f"Signal: *{composite['composite_signal']}* (score={composite['composite_score']:+.2f})\n"
            f"PCR OI: {pcr_data['pcr_oi']:.3f} → {pcr_signal}\n"
            f"Max Pain: {mp_data.get('max_pain_strike', 'N/A')} "
            f"({mp_data.get('deviation_pct', 0):+.2f}%) → {mp_data.get('max_pain_bias', 'N/A')}\n"
            f"IV: {iv_data['near_iv']} ({iv_data['iv_regime']}) | TS: {iv_data['ts_signal']}\n"
            f"Skew: {skew_data['skew_signal']}"
        )
        notify(msg)

    conn.close()

    print("\n" + "=" * 60)
    print("MODULE 43 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
