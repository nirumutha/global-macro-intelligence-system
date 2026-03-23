# ============================================================
# GMIS 2.0 — MODULE 26 — LIVE INSTITUTIONAL FLOWS
#
# COMPONENT 1 — FII/DII Daily Flows (NIFTY)
#   Source: NSE India live API (today's data only)
#   History: accumulated in FII_DII_FLOWS table in DB
#   Signal: 5+ consecutive days same direction
#
# COMPONENT 2 — CFTC Commitment of Traders (Gold & Crude)
#   Source: CFTC year-file zip downloads (free)
#   Track: Managed Money (hedge fund) net position
#   Signal: percentile rank vs 3-year history (contrarian)
#
# COMPONENT 3 — NIFTY Put/Call Ratio
#   Source: NSE FO Bhavcopy (daily CSV zip, free)
#   Signal: PCR > 1.2 = fear = contrarian bullish
#           PCR < 0.7 = complacency = contrarian bearish
# ============================================================

import requests
import sqlite3
import pandas as pd
import numpy as np
import zipfile
import io
import asyncio
import telegram
import os
import sys
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID   = int(os.getenv('TELEGRAM_CHAT_ID', '0'))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')

# ── NSE headers (session required to avoid 403) ───────────────
NSE_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept':          'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer':         'https://www.nseindia.com/',
}

# ── CFTC contract names ───────────────────────────────────────
CFTC_CONTRACTS = {
    'Gold':  'GOLD - COMMODITY EXCHANGE INC.',
    'Crude': 'CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE',
}

# Percentile thresholds for extreme positioning
COT_EXTREME_HIGH = 80   # above 80th pct = crowded long  → bearish contrarian
COT_EXTREME_LOW  = 20   # below 20th pct = crowded short → bullish contrarian

# PCR thresholds
PCR_FEAR_LEVEL   = 1.20  # high put buying = fear = contrarian bullish
PCR_GREED_LEVEL  = 0.70  # low put buying  = greed = contrarian bearish

# FII consecutive-day threshold
FII_STREAK_DAYS  = 5


# ═════════════════════════════════════════════════════════════
# SECTION 1 — DATABASE SETUP
# ═════════════════════════════════════════════════════════════

def create_tables(conn):
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS FII_DII_FLOWS (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            date       TEXT NOT NULL UNIQUE,
            fii_buy    REAL, fii_sell   REAL, fii_net    REAL,
            dii_buy    REAL, dii_sell   REAL, dii_net    REAL,
            fii_streak INTEGER DEFAULT 0,
            signal     TEXT
        );

        CREATE TABLE IF NOT EXISTS COT_DATA (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT NOT NULL,
            asset        TEXT NOT NULL,
            mm_long      REAL, mm_short  REAL, mm_net    REAL,
            oi_total     REAL,
            mm_net_pct   REAL,
            signal       TEXT,
            UNIQUE(date, asset)
        );

        CREATE TABLE IF NOT EXISTS OPTIONS_DATA (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            date     TEXT NOT NULL,
            symbol   TEXT NOT NULL DEFAULT 'NIFTY',
            ce_oi    REAL, pe_oi    REAL, pcr_oi  REAL,
            ce_vol   REAL, pe_vol   REAL, pcr_vol REAL,
            signal   TEXT,
            UNIQUE(date, symbol)
        );
    ''')
    conn.commit()


# ═════════════════════════════════════════════════════════════
# SECTION 2 — COMPONENT 1: FII / DII FLOWS
# ═════════════════════════════════════════════════════════════

def _nse_session():
    """Create a warmed-up NSE session."""
    session = requests.Session()
    try:
        session.get(
            'https://www.nseindia.com',
            headers={**NSE_HEADERS, 'Accept': 'text/html'},
            timeout=15
        )
        time.sleep(0.8)
    except Exception:
        pass
    return session


def fetch_fii_dii_today():
    """
    Fetch today's FII/DII equity flow from NSE.
    Returns dict with keys: fii_buy, fii_sell, fii_net,
                             dii_buy, dii_sell, dii_net, date
    Note: NSE API only exposes today's data.
    History is accumulated in FII_DII_FLOWS table day by day.
    """
    session = _nse_session()
    url     = 'https://www.nseindia.com/api/fiidiiTradeReact'

    try:
        resp = session.get(url, headers=NSE_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    ⚠️  NSE FII/DII fetch failed: {e}")
        return None

    if not data or not isinstance(data, list):
        print("    ⚠️  NSE returned empty FII/DII data")
        return None

    result = {}
    for row in data:
        cat = row.get('category', '').upper()
        try:
            buy  = float(row.get('buyValue',  0) or 0)
            sell = float(row.get('sellValue', 0) or 0)
            net  = float(row.get('netValue',  0) or 0)
        except (ValueError, TypeError):
            continue

        if 'FII' in cat or 'FPI' in cat:
            result.update({'fii_buy': buy, 'fii_sell': sell,
                           'fii_net': net})
        elif 'DII' in cat:
            result.update({'dii_buy': buy, 'dii_sell': sell,
                           'dii_net': net})

        # Parse date from NSE (format: "23-Mar-2026")
        if 'date' not in result and row.get('date'):
            try:
                result['date'] = datetime.strptime(
                    row['date'], '%d-%b-%Y'
                ).strftime('%Y-%m-%d')
            except Exception:
                result['date'] = datetime.now().strftime('%Y-%m-%d')

    if 'fii_net' not in result:
        print("    ⚠️  FII data missing in NSE response")
        return None

    result.setdefault('date', datetime.now().strftime('%Y-%m-%d'))
    return result


def compute_fii_streak(conn, today_net):
    """
    Count consecutive days of FII net buying or selling
    from DB history (excludes today, adds today's direction).
    Returns (streak_count, direction) where direction is
    'BUY' | 'SELL' | 'MIXED'.
    """
    try:
        df = pd.read_sql(
            '''SELECT date, fii_net FROM FII_DII_FLOWS
               ORDER BY date DESC LIMIT 20''',
            conn
        )
    except Exception:
        return 1, ('BUY' if today_net >= 0 else 'SELL')

    today_dir = 'BUY' if today_net >= 0 else 'SELL'
    streak    = 1

    for _, row in df.iterrows():
        d = 'BUY' if float(row['fii_net']) >= 0 else 'SELL'
        if d == today_dir:
            streak += 1
        else:
            break

    return streak, today_dir


def derive_fii_signal(streak, direction, fii_net):
    """Map streak and direction to a trading signal."""
    if streak >= FII_STREAK_DAYS:
        if direction == 'SELL':
            return f'BEARISH ({streak}d consecutive FII selling)'
        else:
            return f'BULLISH ({streak}d consecutive FII buying)'
    elif streak >= 3:
        if direction == 'SELL':
            return f'MILD_BEARISH ({streak}d FII selling)'
        else:
            return f'MILD_BULLISH ({streak}d FII buying)'
    else:
        if fii_net < -5000:
            return 'WEAK_BEARISH (heavy single-day FII sell)'
        elif fii_net > 5000:
            return 'WEAK_BULLISH (heavy single-day FII buy)'
        return 'NEUTRAL'


def save_fii_dii(conn, data, streak, signal):
    try:
        conn.execute('''
            INSERT OR REPLACE INTO FII_DII_FLOWS
            (date, fii_buy, fii_sell, fii_net,
             dii_buy, dii_sell, dii_net,
             fii_streak, signal)
            VALUES (?,?,?,?,?,?,?,?,?)
        ''', (
            data['date'],
            data.get('fii_buy'),  data.get('fii_sell'),
            data.get('fii_net'),
            data.get('dii_buy'),  data.get('dii_sell'),
            data.get('dii_net'),
            streak, signal
        ))
        conn.commit()
        print(f"    ✅ FII/DII saved ({data['date']})")
    except Exception as e:
        print(f"    ❌ FII/DII save failed: {e}")


def run_fii_dii(conn):
    print("\n  COMPONENT 1 — FII/DII FLOWS")
    print("  " + "-"*45)

    data = fetch_fii_dii_today()
    if data is None:
        print("    ⚠️  Skipping FII/DII (data unavailable)")
        return None

    fii_net = data.get('fii_net', 0)
    dii_net = data.get('dii_net', 0)
    streak, direction = compute_fii_streak(conn, fii_net)
    signal = derive_fii_signal(streak, direction, fii_net)

    save_fii_dii(conn, data, streak, signal)

    print(f"    FII: Buy {data.get('fii_buy',0):>10,.2f}  "
          f"Sell {data.get('fii_sell',0):>10,.2f}  "
          f"Net {fii_net:>+10,.2f} Cr")
    print(f"    DII: Buy {data.get('dii_buy',0):>10,.2f}  "
          f"Sell {data.get('dii_sell',0):>10,.2f}  "
          f"Net {dii_net:>+10,.2f} Cr")
    print(f"    FII Streak: {direction} × {streak} days")
    print(f"    Signal: {signal}")

    return {
        'date':      data['date'],
        'fii_net':   fii_net,
        'dii_net':   dii_net,
        'streak':    streak,
        'direction': direction,
        'signal':    signal,
    }


# ═════════════════════════════════════════════════════════════
# SECTION 3 — COMPONENT 2: CFTC COT DATA
# ═════════════════════════════════════════════════════════════

def fetch_cftc_year(year):
    """Download CFTC disaggregated futures CSV for one year."""
    url = (f'https://www.cftc.gov/files/dea/history/'
           f'fut_disagg_txt_{year}.zip')
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        z   = zipfile.ZipFile(io.BytesIO(r.content))
        df  = pd.read_csv(
            z.open(z.namelist()[0]),
            low_memory=False
        )
        return df
    except Exception as e:
        print(f"    ⚠️  CFTC {year} download failed: {e}")
        return None


def fetch_cot_history(years_back=3):
    """
    Download CFTC data for the last N years + current year.
    Returns combined DataFrame filtered to tracked contracts.
    """
    current_year = datetime.now().year
    year_range   = range(current_year - years_back,
                         current_year + 1)

    frames = []
    for y in year_range:
        print(f"    Fetching CFTC {y}...")
        df = fetch_cftc_year(y)
        if df is not None:
            frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined['date'] = pd.to_datetime(
        combined['Report_Date_as_YYYY-MM-DD'], errors='coerce'
    )

    # Keep only the contracts we track
    mask = combined['Market_and_Exchange_Names'].isin(
        CFTC_CONTRACTS.values()
    )
    return combined[mask].copy()


def compute_cot_signals(df):
    """
    For each asset compute current MM net position,
    its percentile rank vs 3-year history,
    and derive a contrarian signal.
    """
    results = {}

    for asset, contract in CFTC_CONTRACTS.items():
        sub = df[
            df['Market_and_Exchange_Names'] == contract
        ].copy().sort_values('date')

        if sub.empty:
            print(f"    ⚠️  No COT rows for {asset}")
            results[asset] = None
            continue

        sub['mm_net'] = (
            sub['M_Money_Positions_Long_All'].astype(float) -
            sub['M_Money_Positions_Short_All'].astype(float)
        )
        sub['oi'] = sub['Open_Interest_All'].astype(float)

        latest     = sub.iloc[-1]
        mm_net     = float(latest['mm_net'])
        oi_total   = float(latest['oi'])
        report_dt  = latest['date'].strftime('%Y-%m-%d')

        # Percentile rank within the history window
        pct_rank   = float(
            (sub['mm_net'] <= mm_net).mean() * 100
        )

        # Contrarian signal
        if pct_rank >= COT_EXTREME_HIGH:
            signal = (f'CONTRARIAN_BEARISH '
                      f'(specs {pct_rank:.0f}th pct long '
                      f'— crowded trade)')
        elif pct_rank <= COT_EXTREME_LOW:
            signal = (f'CONTRARIAN_BULLISH '
                      f'(specs {pct_rank:.0f}th pct short '
                      f'— crowded short)')
        elif pct_rank >= 60:
            signal = (f'MILDLY_BEARISH '
                      f'(specs {pct_rank:.0f}th pct long)')
        elif pct_rank <= 40:
            signal = (f'MILDLY_BULLISH '
                      f'(specs {pct_rank:.0f}th pct long)')
        else:
            signal = (f'NEUTRAL '
                      f'(specs {pct_rank:.0f}th pct)')

        results[asset] = {
            'report_date': report_dt,
            'mm_long':     float(
                latest['M_Money_Positions_Long_All']
            ),
            'mm_short':    float(
                latest['M_Money_Positions_Short_All']
            ),
            'mm_net':      mm_net,
            'oi_total':    oi_total,
            'mm_net_pct':  pct_rank,
            'signal':      signal,
        }

        print(f"    {asset}: MM Net {mm_net:+,.0f} | "
              f"Pct {pct_rank:.0f}th | {signal}")

    return results


def save_cot_data(conn, cot_results):
    for asset, r in cot_results.items():
        if r is None:
            continue
        try:
            conn.execute('''
                INSERT OR REPLACE INTO COT_DATA
                (date, asset, mm_long, mm_short, mm_net,
                 oi_total, mm_net_pct, signal)
                VALUES (?,?,?,?,?,?,?,?)
            ''', (
                r['report_date'], asset,
                r['mm_long'], r['mm_short'], r['mm_net'],
                r['oi_total'], r['mm_net_pct'], r['signal']
            ))
        except Exception as e:
            print(f"    ❌ COT save {asset}: {e}")
    conn.commit()
    print(f"    ✅ COT data saved ({len(cot_results)} assets)")


def run_cot(conn):
    print("\n  COMPONENT 2 — CFTC COT DATA (Gold & Crude)")
    print("  " + "-"*45)

    df = fetch_cot_history(years_back=3)
    if df is None or df.empty:
        print("    ⚠️  Skipping COT (download failed)")
        return None

    results = compute_cot_signals(df)
    save_cot_data(conn, results)
    return results


# ═════════════════════════════════════════════════════════════
# SECTION 4 — COMPONENT 3: NIFTY PUT/CALL RATIO
# ═════════════════════════════════════════════════════════════

def _bhavcopy_url(date):
    return (
        'https://nsearchives.nseindia.com/content/fo/'
        f'BhavCopy_NSE_FO_0_0_0_{date.strftime("%Y%m%d")}_F_0000.csv.zip'
    )


def fetch_pcr_today():
    """
    Download NSE FO bhavcopy for today (or last trading day),
    compute NIFTY index options put/call ratio from OI.
    """
    session  = _nse_session()
    today    = datetime.now().date()

    # Try today and up to 5 calendar days back
    for delta in range(6):
        target = today - timedelta(days=delta)
        url    = _bhavcopy_url(target)

        try:
            resp = session.get(
                url,
                headers={**NSE_HEADERS, 'Accept': '*/*'},
                timeout=30
            )
            if resp.status_code != 200:
                continue

            z   = zipfile.ZipFile(io.BytesIO(resp.content))
            df  = pd.read_csv(z.open(z.namelist()[0]))

            # Filter NIFTY index options only
            nifty_opts = df[
                (df['FinInstrmTp'] == 'IDO') &
                (df['TckrSymb']    == 'NIFTY')
            ].copy()

            if nifty_opts.empty:
                continue

            ce = nifty_opts[
                nifty_opts['OptnTp'] == 'CE'
            ]['OpnIntrst'].astype(float).sum()

            pe = nifty_opts[
                nifty_opts['OptnTp'] == 'PE'
            ]['OpnIntrst'].astype(float).sum()

            ce_vol = nifty_opts[
                nifty_opts['OptnTp'] == 'CE'
            ]['TtlTradgVol'].astype(float).sum()

            pe_vol = nifty_opts[
                nifty_opts['OptnTp'] == 'PE'
            ]['TtlTradgVol'].astype(float).sum()

            pcr_oi  = pe / ce  if ce  > 0 else None
            pcr_vol = pe_vol / ce_vol if ce_vol > 0 else None

            return {
                'date':    target.strftime('%Y-%m-%d'),
                'ce_oi':   ce,  'pe_oi':  pe,
                'pcr_oi':  pcr_oi,
                'ce_vol':  ce_vol, 'pe_vol': pe_vol,
                'pcr_vol': pcr_vol,
            }

        except Exception as e:
            continue   # Try next day back

    print("    ⚠️  Could not fetch NSE bhavcopy (5-day lookback)")
    return None


def derive_pcr_signal(pcr_oi, pcr_vol):
    """Map PCR values to a contrarian signal."""
    # Prefer OI-based PCR; fall back to volume
    pcr = pcr_oi if pcr_oi is not None else pcr_vol
    if pcr is None:
        return 'UNKNOWN'

    if pcr > PCR_FEAR_LEVEL:
        return (f'CONTRARIAN_BULLISH — Extreme fear '
                f'(PCR OI {pcr:.2f} > {PCR_FEAR_LEVEL})')
    elif pcr > 1.0:
        return f'MILDLY_BULLISH — Mild fear (PCR OI {pcr:.2f})'
    elif pcr < PCR_GREED_LEVEL:
        return (f'CONTRARIAN_BEARISH — Excessive complacency '
                f'(PCR OI {pcr:.2f} < {PCR_GREED_LEVEL})')
    elif pcr < 0.9:
        return (f'MILDLY_BEARISH — Low put interest '
                f'(PCR OI {pcr:.2f})')
    else:
        return f'NEUTRAL (PCR OI {pcr:.2f})'


def save_pcr(conn, data, signal):
    try:
        conn.execute('''
            INSERT OR REPLACE INTO OPTIONS_DATA
            (date, symbol, ce_oi, pe_oi, pcr_oi,
             ce_vol, pe_vol, pcr_vol, signal)
            VALUES (?,?,?,?,?,?,?,?,?)
        ''', (
            data['date'], 'NIFTY',
            data['ce_oi'],  data['pe_oi'],  data['pcr_oi'],
            data['ce_vol'], data['pe_vol'], data['pcr_vol'],
            signal
        ))
        conn.commit()
        print(f"    ✅ PCR saved ({data['date']})")
    except Exception as e:
        print(f"    ❌ PCR save failed: {e}")


def run_pcr(conn):
    print("\n  COMPONENT 3 — NIFTY PUT/CALL RATIO")
    print("  " + "-"*45)

    data = fetch_pcr_today()
    if data is None:
        print("    ⚠️  Skipping PCR (bhavcopy unavailable)")
        return None

    signal = derive_pcr_signal(data['pcr_oi'], data['pcr_vol'])
    save_pcr(conn, data, signal)

    print(f"    CE OI: {data['ce_oi']:>15,.0f}  "
          f"PE OI: {data['pe_oi']:>15,.0f}")
    print(f"    PCR (OI):  {data['pcr_oi']:.3f}  "
          f"PCR (Volume): {data.get('pcr_vol', 0):.3f}")
    print(f"    Signal: {signal}")

    return {**data, 'signal': signal}


# ═════════════════════════════════════════════════════════════
# SECTION 5 — PRINT FULL REPORT
# ═════════════════════════════════════════════════════════════

def _is_extreme(signal):
    """
    True only for genuinely extreme / alert-worthy readings.
    Mild signals are excluded to avoid noise.
    """
    if signal is None:
        return False
    s = signal.upper()
    # Contrarian extremes in COT or PCR
    if 'CONTRARIAN_' in s:
        return True
    # FII streaks of 5+ consecutive days (the threshold)
    import re
    streak_match = re.search(r'(\d+)D CONSECUTIVE', s)
    if streak_match and int(streak_match.group(1)) >= FII_STREAK_DAYS:
        return True
    # Heavy single-day FII flow (already labelled WEAK_BEARISH/BULLISH)
    # Don't treat these as extreme - they need context from streak
    return False


def print_flows_report(fii_result, cot_results, pcr_result):
    print("\n" + "="*70)
    print("INSTITUTIONAL FLOWS — LIVE SUMMARY")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*70)

    # ── FII/DII ──────────────────────────────────────────────
    print("\n📊 FII / DII EQUITY FLOWS (NSE India)")
    print("-"*50)
    if fii_result:
        fii_net = fii_result['fii_net']
        dii_net = fii_result['dii_net']
        f_arrow = '↑' if fii_net > 0 else '↓'
        d_arrow = '↑' if dii_net > 0 else '↓'
        print(f"  FII Net:  {f_arrow} {fii_net:>+12,.2f} Cr  "
              f"({fii_result['direction']} × "
              f"{fii_result['streak']} days)")
        print(f"  DII Net:  {d_arrow} {dii_net:>+12,.2f} Cr")
        if _is_extreme(fii_result['signal']):
            print(f"  ⚡ Signal: {fii_result['signal']}")
        else:
            print(f"  Signal:   {fii_result['signal']}")
    else:
        print("  ⚠️  FII/DII data unavailable")

    # ── COT ──────────────────────────────────────────────────
    print("\n📈 CFTC COMMITMENT OF TRADERS")
    print("-"*50)
    if cot_results:
        for asset, r in cot_results.items():
            if r is None:
                print(f"  {asset}: data unavailable")
                continue
            bars = '█' * int(r['mm_net_pct'] / 10)
            extreme = _is_extreme(r['signal'])
            prefix  = '⚡' if extreme else '  '
            print(f"  {asset:<8} MM Net: {r['mm_net']:>+10,.0f}"
                  f"  Pct: {r['mm_net_pct']:>5.1f}%  "
                  f"|{bars:<10}|")
            print(f"  {prefix} Signal: {r['signal']}")
            print(f"    (Report: {r['report_date']}  "
                  f"Long: {r['mm_long']:,.0f}  "
                  f"Short: {r['mm_short']:,.0f})")
    else:
        print("  ⚠️  COT data unavailable")

    # ── PCR ──────────────────────────────────────────────────
    print("\n⚖️  NIFTY PUT/CALL RATIO (OI-based)")
    print("-"*50)
    if pcr_result:
        pcr   = pcr_result.get('pcr_oi', 0) or 0
        pvol  = pcr_result.get('pcr_vol', 0) or 0
        if pcr > PCR_FEAR_LEVEL:
            bar_label = '🔴 EXTREME FEAR'
        elif pcr < PCR_GREED_LEVEL:
            bar_label = '🟢 EXTREME GREED'
        elif pcr > 1.0:
            bar_label = '🟡 MILD FEAR'
        else:
            bar_label = '⬜ NORMAL'
        print(f"  PCR (OI):     {pcr:>6.3f}  {bar_label}")
        print(f"  PCR (Volume): {pvol:>6.3f}")
        print(f"  CE OI: {pcr_result['ce_oi']:>15,.0f}  "
              f"PE OI: {pcr_result['pe_oi']:>15,.0f}")
        if _is_extreme(pcr_result['signal']):
            print(f"  ⚡ Signal: {pcr_result['signal']}")
        else:
            print(f"  Signal:   {pcr_result['signal']}")
    else:
        print("  ⚠️  PCR data unavailable")

    print("\n" + "="*70)

    # ── Combined extremes ────────────────────────────────────
    extremes = []
    if fii_result and _is_extreme(fii_result['signal']):
        extremes.append(('FII/DII', fii_result['signal']))
    if cot_results:
        for a, r in cot_results.items():
            if r and _is_extreme(r['signal']):
                extremes.append((f'COT {a}', r['signal']))
    if pcr_result and _is_extreme(pcr_result['signal']):
        extremes.append(('PCR', pcr_result['signal']))

    if extremes:
        print("\n  ⚡ EXTREME READINGS — CONTRARIAN ALERTS:")
        for label, sig in extremes:
            print(f"     [{label}] {sig}")
    else:
        print("\n  No extreme institutional readings today")

    print("="*70)


# ═════════════════════════════════════════════════════════════
# SECTION 6 — TELEGRAM
# ═════════════════════════════════════════════════════════════

async def _send_telegram(message):
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID,
            text=message,
            parse_mode='HTML'
        )
        print("  ✅ Telegram sent")
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")


def build_telegram_message(fii_result, cot_results,
                           pcr_result):
    date  = datetime.now().strftime('%d %b %Y %H:%M')
    lines = [
        f"🏦 <b>GMIS INSTITUTIONAL FLOWS</b>",
        f"{date}",
        f"{'─' * 30}",
        "",
    ]

    # FII/DII
    if fii_result:
        fii_net = fii_result['fii_net']
        dii_net = fii_result['dii_net']
        f_e  = '🟢' if fii_net > 0 else '🔴'
        d_e  = '🟢' if dii_net > 0 else '🔴'
        extreme = _is_extreme(fii_result['signal'])
        prefix  = '⚡ ' if extreme else ''
        lines.append(
            f"<b>FII/DII Flows</b>"
        )
        lines.append(
            f"  {f_e} FII: {fii_net:+,.0f} Cr "
            f"({fii_result['direction']} ×"
            f" {fii_result['streak']}d)"
        )
        lines.append(f"  {d_e} DII: {dii_net:+,.0f} Cr")
        if extreme:
            lines.append(
                f"  ⚡ <b>{fii_result['signal']}</b>"
            )
        else:
            lines.append(f"  → {fii_result['signal']}")
        lines.append("")

    # COT
    if cot_results:
        lines.append(f"<b>CFTC COT (Managed Money)</b>")
        for asset, r in cot_results.items():
            if r is None:
                continue
            extreme = _is_extreme(r['signal'])
            prefix  = '⚡ ' if extreme else ''
            lines.append(
                f"  {asset}: Net {r['mm_net']:+,.0f} "
                f"| {r['mm_net_pct']:.0f}th pct"
            )
            lines.append(
                f"  {prefix}<b>{r['signal']}</b>"
                if extreme else f"  → {r['signal']}"
            )
        lines.append("")

    # PCR
    if pcr_result:
        pcr     = pcr_result.get('pcr_oi') or 0
        extreme = _is_extreme(pcr_result['signal'])
        prefix  = '⚡ ' if extreme else ''
        lines.append(f"<b>NIFTY PCR (OI-based)</b>")
        lines.append(f"  PCR: {pcr:.3f}")
        lines.append(
            f"  {prefix}<b>{pcr_result['signal']}</b>"
            if extreme else f"  → {pcr_result['signal']}"
        )
        lines.append("")

    lines.append("<i>GMIS Institutional Flows Engine</i>")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_institutional_flows(send_telegram_flag=True):
    print("\n" + "="*65)
    print("GMIS MODULE 26 — LIVE INSTITUTIONAL FLOWS")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    # Run all three components (each fails gracefully)
    fii_result  = run_fii_dii(conn)
    cot_results = run_cot(conn)
    pcr_result  = run_pcr(conn)

    conn.close()

    print_flows_report(fii_result, cot_results, pcr_result)

    # Telegram: fire when any extreme reading detected
    if send_telegram_flag and BOT_TOKEN:
        has_extreme = (
            (fii_result  and _is_extreme(fii_result['signal'])) or
            (cot_results and any(
                r and _is_extreme(r['signal'])
                for r in cot_results.values()
            )) or
            (pcr_result  and _is_extreme(pcr_result['signal']))
        )

        if '--force-send' in sys.argv:
            has_extreme = True

        if has_extreme:
            print("\nExtreme reading detected — "
                  "sending Telegram alert...")
            msg = build_telegram_message(
                fii_result, cot_results, pcr_result
            )
            asyncio.run(_send_telegram(msg))
        else:
            print("\n  No extreme readings — no Telegram alert")

    elif not send_telegram_flag:
        print("\n  Telegram skipped (--no-telegram)")

    return {
        'fii':  fii_result,
        'cot':  cot_results,
        'pcr':  pcr_result,
    }


if __name__ == "__main__":
    no_telegram = '--no-telegram' in sys.argv
    run_institutional_flows(send_telegram_flag=not no_telegram)
