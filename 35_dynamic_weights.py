# ============================================================
# GMIS MODULE 35 — DYNAMIC WEIGHT ADJUSTMENT
# Tracks component performance over 60 days and adjusts
# the decision engine weights on a monthly schedule.
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import asyncio
import os
import sys
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────────────
BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_PATH, 'data', 'macro_system.db')
NO_TELEGRAM = '--no-telegram' in sys.argv

ASSETS = ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']

# Price table → close column index (0=Open,1=High,2=Low,3=Close,4=Vol)
PRICE_TABLES = {
    'NIFTY':  'NIFTY50',
    'SP500':  'SP500',
    'Gold':   'GOLD',
    'Silver': 'SILVER',
    'Crude':  'CRUDE_WTI',
}

# Base weights (from Module 19 Decision Engine)
BASE_WEIGHTS = {
    'signal':    0.35,
    'analog':    0.25,
    'sentiment': 0.15,
    'macro':     0.15,
    'yield':     0.05,
    'vix':       0.05,
}

COMPONENTS = list(BASE_WEIGHTS.keys())

# IC thresholds
IC_HIGH   =  0.10   # boost weight
IC_LOW    = -0.05   # reduce weight
MAX_ADJ   =  0.07   # max ± change from base
MIN_FLOOR =  0.02   # no component below this

# Monthly update gate
UPDATE_INTERVAL_DAYS = 30
MIN_HISTORY_ROWS     = 30   # need at least this many rows for IC

# ── Telegram ────────────────────────────────────────────────
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


# ── DB bootstrap ────────────────────────────────────────────
def setup_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS DYNAMIC_WEIGHTS (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT,
            asset           TEXT,
            component       TEXT,
            base_weight     REAL,
            adjusted_weight REAL,
            ic_60d          REAL,
            hit_rate_60d    REAL,
            rolling_sharpe  REAL,
            adjustment_reason TEXT,
            last_updated    TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS WEIGHT_UPDATE_LOG (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            update_date TEXT,
            asset       TEXT,
            component   TEXT,
            old_weight  REAL,
            new_weight  REAL,
            ic_value    REAL,
            reason      TEXT
        )
    """)
    conn.commit()


# ── Price loader ─────────────────────────────────────────────
def load_prices(conn, table: str) -> pd.Series:
    """Return a date-indexed close price series."""
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Close']).sort_values('Date')
    df = df[df['Close'] > 0]
    return df.set_index('Date')['Close']


# ── Next-day return series ────────────────────────────────────
def next_day_returns(prices: pd.Series) -> pd.Series:
    """Forward 1-day pct return (return on day t = price[t+1]/price[t] - 1)."""
    ret = prices.pct_change().shift(-1)
    return ret.dropna()


# ── IC calculation ────────────────────────────────────────────
def compute_ic(scores: pd.Series, fwd_returns: pd.Series,
               window: int = 60) -> float:
    """Pearson IC between scores and forward returns over trailing window."""
    idx = scores.index.intersection(fwd_returns.index)
    if len(idx) < MIN_HISTORY_ROWS:
        return np.nan
    s = scores.loc[idx].tail(window)
    r = fwd_returns.loc[idx].tail(window)
    common = s.index.intersection(r.index)
    if len(common) < MIN_HISTORY_ROWS:
        return np.nan
    return float(np.corrcoef(s.loc[common].values,
                              r.loc[common].values)[0, 1])


def compute_hit_rate(scores: pd.Series, fwd_returns: pd.Series,
                     window: int = 60) -> float:
    idx = scores.index.intersection(fwd_returns.index)
    if len(idx) < MIN_HISTORY_ROWS:
        return np.nan
    s = scores.loc[idx].tail(window)
    r = fwd_returns.loc[idx].tail(window)
    common = s.index.intersection(r.index)
    if len(common) < MIN_HISTORY_ROWS:
        return np.nan
    hits = ((s.loc[common] > 0) == (r.loc[common] > 0)).sum()
    return float(hits / len(common))


def compute_sharpe(scores: pd.Series, fwd_returns: pd.Series,
                   window: int = 60) -> float:
    """Sharpe of a long/short strategy driven by the component sign."""
    idx = scores.index.intersection(fwd_returns.index)
    if len(idx) < MIN_HISTORY_ROWS:
        return np.nan
    s = scores.loc[idx].tail(window)
    r = fwd_returns.loc[idx].tail(window)
    common = s.index.intersection(r.index)
    if len(common) < MIN_HISTORY_ROWS:
        return np.nan
    strat = r.loc[common] * np.sign(s.loc[common])
    if strat.std() == 0:
        return np.nan
    return float(strat.mean() / strat.std() * np.sqrt(252))


# ──────────────────────────────────────────────────────────────
# COMPONENT 1 — Performance tracking
# ──────────────────────────────────────────────────────────────
def compute_component_stats(conn) -> dict:
    """
    Returns: { asset: { component: {ic, hit_rate, sharpe} } }
    Components with insufficient history get NaN.
    """
    print("\nComponent 1 — Performance tracking (60-day window)...")

    # Load all price series
    prices = {}
    for asset, tbl in PRICE_TABLES.items():
        try:
            prices[asset] = load_prices(conn, tbl)
        except Exception as e:
            print(f"  ⚠️  Price load failed for {asset}: {e}")

    # Forward returns per asset
    fwd_rets = {a: next_day_returns(p) for a, p in prices.items()}

    # ── signal scores from SIGNALS_V3 (richest history) ──
    try:
        sv3 = pd.read_sql("SELECT * FROM SIGNALS_V3", conn,
                          parse_dates=['Date'])
        sv3 = sv3.sort_values('Date').set_index('Date')
    except Exception as e:
        print(f"  ⚠️  SIGNALS_V3 load failed: {e}")
        sv3 = pd.DataFrame()

    # ── yield scores: yield change direction (US_10Y_YIELD) ──
    try:
        yld = pd.read_sql("SELECT Date, DGS10 FROM US_10Y_YIELD", conn,
                          parse_dates=['Date'])
        yld = yld.dropna().sort_values('Date').set_index('Date')
        # Score: -yield_change (rising yields → bearish for equities)
        yield_score = -yld['DGS10'].diff().dropna()
    except Exception as e:
        print(f"  ⚠️  Yield load failed: {e}")
        yield_score = pd.Series(dtype=float)

    # ── vix scores: negative VIX change (falling VIX → bullish) ──
    try:
        vix_df = pd.read_sql("SELECT * FROM VIX_US", conn)
        vix_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        vix_df['Date'] = pd.to_datetime(vix_df['Date'])
        vix_df = vix_df.dropna(subset=['Close']).sort_values('Date')
        vix_score = (-vix_df.set_index('Date')['Close'].diff()).dropna()
    except Exception as e:
        print(f"  ⚠️  VIX load failed: {e}")
        vix_score = pd.Series(dtype=float)

    # ── DECISIONS layer scores (limited history) ──
    try:
        dec = pd.read_sql(
            "SELECT date, asset, layer_analog, layer_sentiment, layer_macro "
            "FROM DECISIONS", conn, parse_dates=['date'])
        dec = dec.sort_values('date').set_index('date')
    except Exception as e:
        print(f"  ⚠️  DECISIONS load failed: {e}")
        dec = pd.DataFrame()

    # ── Compute stats ──
    stats = {}
    for asset in ASSETS:
        stats[asset] = {}
        ret = fwd_rets.get(asset, pd.Series(dtype=float))

        # signal
        if not sv3.empty and f'{asset}_score' in sv3.columns:
            sc = sv3[f'{asset}_score'].dropna()
            stats[asset]['signal'] = {
                'ic':         compute_ic(sc, ret),
                'hit_rate':   compute_hit_rate(sc, ret),
                'sharpe':     compute_sharpe(sc, ret),
                'n_obs':      len(sc.tail(60)),
            }
        else:
            stats[asset]['signal'] = {
                'ic': np.nan, 'hit_rate': np.nan, 'sharpe': np.nan, 'n_obs': 0}

        # yield — use same score for equity assets, inverted for Gold/Silver
        if len(yield_score) >= MIN_HISTORY_ROWS:
            ysc = yield_score if asset not in ('Gold', 'Silver') else -yield_score
            stats[asset]['yield'] = {
                'ic':       compute_ic(ysc, ret),
                'hit_rate': compute_hit_rate(ysc, ret),
                'sharpe':   compute_sharpe(ysc, ret),
                'n_obs':    len(ysc.tail(60)),
            }
        else:
            stats[asset]['yield'] = {
                'ic': np.nan, 'hit_rate': np.nan, 'sharpe': np.nan, 'n_obs': 0}

        # vix — inverted for all (high VIX → risk-off → bearish equities,
        #         but for safe-haven Gold/Silver, high VIX → bullish)
        if len(vix_score) >= MIN_HISTORY_ROWS:
            vsc = vix_score if asset not in ('Gold', 'Silver') else -vix_score
            stats[asset]['vix'] = {
                'ic':       compute_ic(vsc, ret),
                'hit_rate': compute_hit_rate(vsc, ret),
                'sharpe':   compute_sharpe(vsc, ret),
                'n_obs':    len(vsc.tail(60)),
            }
        else:
            stats[asset]['vix'] = {
                'ic': np.nan, 'hit_rate': np.nan, 'sharpe': np.nan, 'n_obs': 0}

        # analog, sentiment, macro — from DECISIONS (may have insufficient history)
        for comp in ('analog', 'sentiment', 'macro'):
            col = f'layer_{comp}'
            if not dec.empty and col in dec.columns:
                asset_dec = dec[dec['asset'] == asset][col].dropna() \
                    if 'asset' in dec.columns \
                    else dec[col].dropna()
                if len(asset_dec) >= MIN_HISTORY_ROWS:
                    stats[asset][comp] = {
                        'ic':       compute_ic(asset_dec, ret),
                        'hit_rate': compute_hit_rate(asset_dec, ret),
                        'sharpe':   compute_sharpe(asset_dec, ret),
                        'n_obs':    len(asset_dec),
                    }
                else:
                    stats[asset][comp] = {
                        'ic': np.nan, 'hit_rate': np.nan, 'sharpe': np.nan,
                        'n_obs': len(asset_dec)}
            else:
                stats[asset][comp] = {
                    'ic': np.nan, 'hit_rate': np.nan, 'sharpe': np.nan, 'n_obs': 0}

    # Print summary
    print(f"\n  {'Asset':<8} {'Component':<12} {'IC':>7} {'HitRate':>8} "
          f"{'Sharpe':>8} {'N':>5} {'Status'}")
    print(f"  {'-'*60}")
    for asset in ASSETS:
        for comp in COMPONENTS:
            s = stats[asset][comp]
            ic    = s['ic']
            hr    = s['hit_rate']
            sh    = s['sharpe']
            n     = s['n_obs']
            if np.isnan(ic):
                status = f"INSUFFICIENT ({n} obs)"
            elif ic > IC_HIGH:
                status = "▲ BOOST"
            elif ic < IC_LOW:
                status = "▼ REDUCE"
            else:
                status = "= HOLD"
            ic_str = f"{ic:+.3f}" if not np.isnan(ic) else "  N/A "
            hr_str = f"{hr:.1%}"  if not np.isnan(hr) else "  N/A "
            sh_str = f"{sh:+.2f}" if not np.isnan(sh) else "  N/A "
            print(f"  {asset:<8} {comp:<12} {ic_str:>7} {hr_str:>8} "
                  f"{sh_str:>8} {n:>5}  {status}")
    return stats


# ──────────────────────────────────────────────────────────────
# COMPONENT 2 — Weight adjustment (global)
# ──────────────────────────────────────────────────────────────
def apply_adjustment_rules(stats_for_asset: dict) -> dict:
    """
    Given per-component stats for one asset, return adjusted weights.
    Returns dict: { component: adjusted_weight }
    """
    raw = {}
    reasons = {}
    for comp in COMPONENTS:
        base = BASE_WEIGHTS[comp]
        ic   = stats_for_asset[comp]['ic']
        if np.isnan(ic):
            raw[comp]     = base
            reasons[comp] = "INSUFFICIENT_HISTORY → base"
        elif ic > IC_HIGH:
            delta = min(base * 0.20, MAX_ADJ)
            raw[comp]     = base + delta
            reasons[comp] = f"IC={ic:+.3f} > {IC_HIGH} → +{delta:.3f}"
        elif ic < IC_LOW:
            delta = min(base * 0.20, MAX_ADJ)
            raw[comp]     = base - delta
            reasons[comp] = f"IC={ic:+.3f} < {IC_LOW} → -{delta:.3f}"
        else:
            raw[comp]     = base
            reasons[comp] = f"IC={ic:+.3f} in band → no change"

    # Enforce floor
    for comp in COMPONENTS:
        if raw[comp] < MIN_FLOOR:
            raw[comp] = MIN_FLOOR

    # Renormalise to sum=1
    total = sum(raw.values())
    adjusted = {c: round(v / total, 6) for c, v in raw.items()}

    return adjusted, reasons


# ──────────────────────────────────────────────────────────────
# COMPONENT 3 — Asset-specific weights
# ──────────────────────────────────────────────────────────────
def compute_all_asset_weights(stats: dict) -> dict:
    """Returns { asset: {component: adjusted_weight} }"""
    print("\nComponent 3 — Asset-specific weight adjustment...")
    all_weights  = {}
    all_reasons  = {}
    for asset in ASSETS:
        adj, rsn = apply_adjustment_rules(stats[asset])
        all_weights[asset] = adj
        all_reasons[asset] = rsn
        total = sum(adj.values())
        print(f"\n  {asset}:  (sum={total:.4f})")
        for comp in COMPONENTS:
            base = BASE_WEIGHTS[comp]
            w    = adj[comp]
            diff = w - base
            diff_str = f"{diff:+.4f}" if abs(diff) > 1e-5 else "  same"
            print(f"    {comp:<12}  base={base:.4f}  adj={w:.4f}  "
                  f"({diff_str})  | {rsn[comp]}")
    return all_weights, all_reasons


# ──────────────────────────────────────────────────────────────
# COMPONENT 4 — Monthly update gate
# ──────────────────────────────────────────────────────────────
def get_last_update_date(conn) -> datetime | None:
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT MAX(last_updated) FROM DYNAMIC_WEIGHTS")
        row = cur.fetchone()
        if row and row[0]:
            return datetime.strptime(row[0][:10], '%Y-%m-%d')
    except Exception:
        pass
    return None


def load_current_weights(conn, today_str: str) -> dict | None:
    """Load today's weights if they exist (already updated today)."""
    try:
        df = pd.read_sql(
            f"SELECT asset, component, adjusted_weight FROM DYNAMIC_WEIGHTS "
            f"WHERE date = '{today_str}'",
            conn)
        if df.empty:
            return None
        result = {}
        for _, row in df.iterrows():
            result.setdefault(row['asset'], {})[row['component']] = \
                row['adjusted_weight']
        return result
    except Exception:
        return None


def load_previous_weights(conn) -> dict | None:
    """Load last saved weights for comparison."""
    try:
        df = pd.read_sql(
            "SELECT asset, component, adjusted_weight FROM DYNAMIC_WEIGHTS "
            "WHERE date = (SELECT MAX(date) FROM DYNAMIC_WEIGHTS)",
            conn)
        if df.empty:
            return None
        result = {}
        for _, row in df.iterrows():
            result.setdefault(row['asset'], {})[row['component']] = \
                row['adjusted_weight']
        return result
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# Save to DB
# ──────────────────────────────────────────────────────────────
def save_weights(conn, today_str: str, all_weights: dict,
                 all_reasons: dict, stats: dict):
    cur = conn.cursor()
    # Remove today's existing rows
    cur.execute("DELETE FROM DYNAMIC_WEIGHTS WHERE date = ?", (today_str,))

    rows = []
    for asset in ASSETS:
        for comp in COMPONENTS:
            w    = all_weights[asset][comp]
            base = BASE_WEIGHTS[comp]
            s    = stats[asset][comp]
            rows.append((
                today_str,
                asset,
                comp,
                base,
                w,
                s['ic']       if not np.isnan(s['ic'])       else None,
                s['hit_rate'] if not np.isnan(s['hit_rate'])  else None,
                s['sharpe']   if not np.isnan(s['sharpe'])    else None,
                all_reasons[asset][comp],
                today_str,
            ))
    cur.executemany("""
        INSERT INTO DYNAMIC_WEIGHTS
          (date, asset, component, base_weight, adjusted_weight,
           ic_60d, hit_rate_60d, rolling_sharpe, adjustment_reason,
           last_updated)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    print(f"\n  ✅ Saved {len(rows)} weight records to DYNAMIC_WEIGHTS")


def log_weight_changes(conn, today_str: str, prev_weights: dict | None,
                       new_weights: dict, stats: dict):
    if prev_weights is None:
        return []
    changes = []
    cur = conn.cursor()
    for asset in ASSETS:
        for comp in COMPONENTS:
            old = prev_weights.get(asset, {}).get(comp, BASE_WEIGHTS[comp])
            new = new_weights[asset][comp]
            delta = abs(new - old)
            if delta > 1e-6:
                reason = (f"IC={stats[asset][comp]['ic']:+.3f}"
                          if not np.isnan(stats[asset][comp]['ic'])
                          else "INIT")
                cur.execute("""
                    INSERT INTO WEIGHT_UPDATE_LOG
                      (update_date, asset, component, old_weight,
                       new_weight, ic_value, reason)
                    VALUES (?,?,?,?,?,?,?)
                """, (today_str, asset, comp, old, new,
                      stats[asset][comp]['ic']
                      if not np.isnan(stats[asset][comp]['ic']) else None,
                      reason))
                if delta / max(old, 1e-9) > 0.10:   # >10% relative change
                    changes.append((asset, comp, old, new, delta))
    conn.commit()
    return changes


# ──────────────────────────────────────────────────────────────
# Telegram message
# ──────────────────────────────────────────────────────────────
def build_telegram_message(today_str: str, changes: list,
                            all_weights: dict) -> str:
    lines = [
        "⚖️ <b>GMIS MODULE 35 — WEIGHT UPDATE</b>",
        f"📅 {today_str}",
        "",
        "⚠️ <b>Significant Weight Changes (>10%):</b>",
    ]
    for asset, comp, old, new, delta in changes:
        pct = (new - old) / old * 100
        arrow = "▲" if new > old else "▼"
        lines.append(
            f"  {arrow} <b>{asset} / {comp}</b>: "
            f"{old:.4f} → {new:.4f} ({pct:+.1f}%)")
    lines += [
        "",
        "<b>Current Adjusted Weights (SP500):</b>",
    ]
    for comp in COMPONENTS:
        w    = all_weights['SP500'][comp]
        base = BASE_WEIGHTS[comp]
        diff = w - base
        marker = " ▲" if diff > 0.001 else (" ▼" if diff < -0.001 else "")
        lines.append(f"  {comp:<12}: {w:.4f}{marker}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Print report
# ──────────────────────────────────────────────────────────────
def print_report(today_str: str, all_weights: dict,
                 prev_weights: dict | None, stats: dict,
                 update_triggered: bool, last_update: datetime | None):
    print(f"\n{'='*70}")
    print(f"DYNAMIC WEIGHT ADJUSTMENT — DAILY REPORT")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*70}")

    if not update_triggered and last_update:
        days_since = (datetime.now() - last_update).days
        next_update = UPDATE_INTERVAL_DAYS - days_since
        print(f"\n  ⏳ Weight update NOT triggered "
              f"({days_since}d since last update; next in ~{next_update}d)")
        print(f"     Using weights from {last_update.strftime('%Y-%m-%d')}")
    else:
        print(f"\n  ✅ Weights UPDATED today ({today_str})")

    print(f"\n{'─'*70}")
    print(f"COMPONENT 1+2 — PERFORMANCE STATS & ADJUSTED WEIGHTS")
    print(f"{'─'*70}")
    hdr = (f"  {'Asset':<8} {'Component':<12} {'Base':>6} {'Adj':>7} "
           f"{'Δ':>7} {'IC':>7} {'Hit%':>6} {'Sharpe':>7}")
    print(hdr)
    print(f"  {'-'*65}")

    for asset in ASSETS:
        for comp in COMPONENTS:
            base = BASE_WEIGHTS[comp]
            w    = all_weights[asset][comp]
            diff = w - base
            s    = stats[asset][comp]
            ic_s = f"{s['ic']:+.3f}" if not np.isnan(s['ic']) else "  N/A"
            hr_s = (f"{s['hit_rate']:.0%}"
                    if not np.isnan(s['hit_rate']) else " N/A")
            sh_s = (f"{s['sharpe']:+.2f}"
                    if not np.isnan(s['sharpe']) else "  N/A")
            d_s  = f"{diff:+.4f}" if abs(diff) > 1e-5 else "  —"
            arrow = ("▲" if diff > 1e-4 else ("▼" if diff < -1e-4 else " "))
            print(f"  {asset:<8} {comp:<12} {base:.4f} {w:.5f} "
                  f"{arrow}{d_s:>6} {ic_s:>7} {hr_s:>6} {sh_s:>7}")

    print(f"\n{'─'*70}")
    print(f"COMPONENT 3 — ASSET-SPECIFIC WEIGHT SUMMARY (adjusted only)")
    print(f"{'─'*70}")
    print(f"  {'Component':<12}", end="")
    for a in ASSETS:
        print(f"  {a:>8}", end="")
    print()
    print(f"  {'-'*60}")
    for comp in COMPONENTS:
        base = BASE_WEIGHTS[comp]
        print(f"  {comp:<12}", end="")
        for a in ASSETS:
            w    = all_weights[a][comp]
            diff = w - base
            mark = "▲" if diff > 1e-4 else ("▼" if diff < -1e-4 else " ")
            print(f"  {mark}{w:.4f}", end="")
        print()

    print(f"\n{'─'*70}")
    print(f"COMPONENT 4 — WEIGHT UPDATE SCHEDULE")
    print(f"{'─'*70}")
    if last_update:
        print(f"  Last update : {last_update.strftime('%Y-%m-%d')}")
        print(f"  Next due    : "
              f"{(last_update + timedelta(days=UPDATE_INTERVAL_DAYS)).strftime('%Y-%m-%d')}")
    else:
        print(f"  First run — weights initialised today")
    print(f"  Update freq : every {UPDATE_INTERVAL_DAYS} days")
    print(f"  Min history : {MIN_HISTORY_ROWS} rows required for IC")

    print(f"\n{'='*70}\n")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*65}")
    print(f"GMIS MODULE 35 — DYNAMIC WEIGHT ADJUSTMENT")
    print(f"{datetime.now().strftime('%A %d %B %Y — %H:%M')}")
    print(f"{'='*65}")

    conn = get_conn()
    setup_tables(conn)

    # ── Check monthly gate ──────────────────────────────────
    print("\nComponent 4 — Checking update schedule...")
    last_update     = get_last_update_date(conn)
    already_today   = load_current_weights(conn, today_str) is not None
    update_triggered = False

    if already_today:
        print(f"  Weights already updated today ({today_str}) — reusing.")
        all_weights = load_current_weights(conn, today_str)
        # Still compute stats for the report
        stats = compute_component_stats(conn)
        _, all_reasons = compute_all_asset_weights(stats)
        prev_weights = None
        changes = []
    else:
        days_since = (datetime.now() - last_update).days if last_update else 999
        if days_since >= UPDATE_INTERVAL_DAYS or last_update is None:
            print(f"  ✅ Update due "
                  f"({'first run' if last_update is None else f'{days_since}d since last'}) "
                  f"— recalculating weights.")
            update_triggered = True
            # Component 1 — stats
            stats = compute_component_stats(conn)
            # Component 2+3 — adjusted weights
            all_weights, all_reasons = compute_all_asset_weights(stats)
            # Component 4 — log & save
            prev_weights = load_previous_weights(conn)
            save_weights(conn, today_str, all_weights, all_reasons, stats)
            changes = log_weight_changes(
                conn, today_str, prev_weights, all_weights, stats)
        else:
            next_due = UPDATE_INTERVAL_DAYS - days_since
            print(f"  ⏳ No update yet — {days_since}d since last, "
                  f"next in ~{next_due}d.  Loading saved weights.")
            # Stats still computed for display; weights loaded from DB
            stats = compute_component_stats(conn)
            all_weights, all_reasons = compute_all_asset_weights(stats)
            prev_weights = None
            changes = []

    # Print full report
    print_report(today_str, all_weights, prev_weights if update_triggered else None,
                 stats, update_triggered, last_update)

    # Telegram — only for significant changes
    if changes and not NO_TELEGRAM:
        msg = build_telegram_message(today_str, changes, all_weights)
        asyncio.run(send_telegram(msg))
        print(f"  📱 Telegram sent ({len(changes)} significant changes)")
    elif NO_TELEGRAM:
        print("  Telegram skipped (--no-telegram)")
    else:
        print("  No significant weight changes — Telegram not sent")

    conn.close()


if __name__ == '__main__':
    main()
