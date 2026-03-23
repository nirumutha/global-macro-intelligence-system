# ============================================================
# GMIS 2.0 — MODULE 22 — MODEL HEALTH MONITOR
# Tracks rolling performance and detects silent failure
# Alerts via Telegram when system is underperforming
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import asyncio
import telegram
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID   = int(os.getenv('TELEGRAM_CHAT_ID'))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')
DATA_PATH = os.path.join(BASE_PATH, 'data')

# ── Health thresholds ─────────────────────────────────────────
# Below these values = alert user to reduce risk
SHARPE_WARNING    = 0.20   # rolling 90d Sharpe below this
SHARPE_CRITICAL   = 0.00   # rolling 90d Sharpe below this
HIT_RATE_WARNING  = 0.45   # recent hit rate below this
HIT_RATE_CRITICAL = 0.35   # recent hit rate below this
DRAWDOWN_WARNING  = -15.0  # current drawdown below this %
DRAWDOWN_CRITICAL = -25.0  # current drawdown below this %

ROLLING_WINDOW = 90   # days for rolling metrics
ASSETS = ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']

ASSET_FILES = {
    'NIFTY':  'NIFTY50.csv',
    'SP500':  'SP500.csv',
    'Gold':   'GOLD.csv',
    'Silver': 'SILVER.csv',
    'Crude':  'CRUDE_WTI.csv',
}

# ═════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ═════════════════════════════════════════════════════════════

def load_price_series(asset):
    """Load clean daily close prices."""
    filepath = os.path.join(DATA_PATH, ASSET_FILES[asset])
    df = pd.read_csv(filepath)
    df = df[~df['Price'].astype(str).str.match(
        r'^[A-Za-z]', na=False)]
    df = df.rename(columns={'Price': 'Date'})
    df['Date']  = pd.to_datetime(df['Date'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Date', 'Close'])
    df = df.sort_values('Date').set_index('Date')
    return df['Close']

def load_signals():
    """Load signal history from database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM SIGNALS", conn)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        conn.close()
        return df
    except Exception as e:
        conn.close()
        return pd.DataFrame()

def load_signal_outcomes():
    """Load tracked signal outcomes."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT * FROM SIGNAL_OUTCOMES", conn)
        conn.close()
        return df
    except:
        conn.close()
        return pd.DataFrame()

def load_signal_log():
    """Load signal log."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT * FROM SIGNAL_LOG", conn)
        conn.close()
        return df
    except:
        conn.close()
        return pd.DataFrame()

# ═════════════════════════════════════════════════════════════
# SECTION 2 — ROLLING BACKTEST METRICS
# ═════════════════════════════════════════════════════════════

def calculate_rolling_strategy_returns(
        asset, price_series, signals_df,
        window=ROLLING_WINDOW,
        threshold=0.15,
        transaction_cost=0.001):
    """
    Calculate strategy returns for the last N days.
    Uses existing signal scores to simulate positions.
    """
    score_col = f'{asset}_score'
    if score_col not in signals_df.columns:
        return None

    # Align to last N days
    cutoff = price_series.index[-1] - \
             pd.Timedelta(days=window * 1.5)

    prices  = price_series[price_series.index >= cutoff]
    scores  = signals_df[score_col][
        signals_df.index >= cutoff]

    if len(prices) < 20:
        return None

    # Align
    sig = scores.reindex(prices.index).ffill().bfill()

    daily_ret = prices.pct_change().fillna(0)
    position  = pd.Series(0.0, index=prices.index)
    position[sig >= threshold]  = 1.0
    position[sig <= -threshold] = 0.0

    pos_change = position.diff().abs().fillna(0)
    strat_ret  = position.shift(1) * daily_ret
    strat_ret -= pos_change * transaction_cost

    return strat_ret

def calculate_sharpe(returns, annualise=True):
    """Calculate Sharpe ratio from return series."""
    if returns is None or len(returns) < 10:
        return None
    mean = returns.mean()
    std  = returns.std()
    if std == 0:
        return 0
    sharpe = mean / std
    if annualise:
        sharpe *= np.sqrt(252)
    return round(float(sharpe), 3)

def calculate_max_drawdown(returns):
    """Calculate max drawdown from return series."""
    if returns is None or len(returns) < 2:
        return None
    equity   = (1 + returns).cumprod() * 100
    roll_max = equity.cummax()
    drawdown = ((equity - roll_max) / roll_max * 100)
    return round(float(drawdown.min()), 2)

def calculate_current_drawdown(returns):
    """Calculate current drawdown from peak."""
    if returns is None or len(returns) < 2:
        return None
    equity   = (1 + returns).cumprod() * 100
    peak     = equity.cummax().iloc[-1]
    current  = equity.iloc[-1]
    dd       = (current - peak) / peak * 100
    return round(float(dd), 2)

# ═════════════════════════════════════════════════════════════
# SECTION 3 — HIT RATE FROM TRACKED OUTCOMES
# ═════════════════════════════════════════════════════════════

def calculate_recent_hit_rate(outcomes_df,
                               lookback_days=60):
    """
    Calculate hit rate from recently resolved signals.
    Uses Signal Outcomes table from Module 21.
    """
    if outcomes_df.empty:
        return None

    cutoff = (datetime.now() -
              timedelta(days=lookback_days)).strftime(
        '%Y-%m-%d')

    recent = outcomes_df[
        outcomes_df['signal_date'] >= cutoff]

    if recent.empty:
        return None

    wins  = (recent['return_pct'] > 0).sum()
    total = len(recent)

    return round(float(wins / total * 100), 1) \
        if total > 0 else None

# ═════════════════════════════════════════════════════════════
# SECTION 4 — HEALTH ASSESSMENT PER ASSET
# ═════════════════════════════════════════════════════════════

def assess_asset_health(asset, price_series,
                         signals_df, outcomes_df):
    """
    Full health assessment for one asset.
    Returns health status and metrics.
    """
    # Rolling strategy returns
    strat_ret = calculate_rolling_strategy_returns(
        asset, price_series, signals_df
    )

    rolling_sharpe  = calculate_sharpe(strat_ret)
    max_dd          = calculate_max_drawdown(strat_ret)
    current_dd      = calculate_current_drawdown(strat_ret)

    # Hit rate from tracked outcomes
    asset_outcomes = outcomes_df[
        outcomes_df['asset'] == asset
    ] if not outcomes_df.empty else pd.DataFrame()

    hit_rate = calculate_recent_hit_rate(asset_outcomes)

    # Signal V3 hit rate from database
    conn = sqlite3.connect(DB_PATH)
    try:
        v3 = pd.read_sql("SELECT * FROM SIGNALS_V3", conn)
        v3['Date'] = pd.to_datetime(v3['Date'])
        v3 = v3.sort_values('Date')
        hr_col = f'{asset}_hit_rate'
        if hr_col in v3.columns and len(v3) > 0:
            v3_hit_rate = float(v3[hr_col].iloc[-1] * 100)
        else:
            v3_hit_rate = None
    except:
        v3_hit_rate = None
    finally:
        conn.close()

    # Use tracked hit rate if available,
    # otherwise fall back to V3 hit rate
    effective_hit_rate = hit_rate \
        if hit_rate is not None else v3_hit_rate

    # Determine health status
    alerts  = []
    status  = 'HEALTHY'

    # Sharpe check
    if rolling_sharpe is not None:
        if rolling_sharpe < SHARPE_CRITICAL:
            status = 'CRITICAL'
            alerts.append(
                f"Rolling Sharpe {rolling_sharpe:.2f} "
                f"below critical threshold "
                f"({SHARPE_CRITICAL})"
            )
        elif rolling_sharpe < SHARPE_WARNING:
            if status != 'CRITICAL':
                status = 'WARNING'
            alerts.append(
                f"Rolling Sharpe {rolling_sharpe:.2f} "
                f"below warning threshold "
                f"({SHARPE_WARNING})"
            )

    # Hit rate check
    if effective_hit_rate is not None:
        if effective_hit_rate < HIT_RATE_CRITICAL * 100:
            status = 'CRITICAL'
            alerts.append(
                f"Hit rate {effective_hit_rate:.1f}% "
                f"critically low "
                f"(threshold: "
                f"{HIT_RATE_CRITICAL*100:.0f}%)"
            )
        elif effective_hit_rate < HIT_RATE_WARNING * 100:
            if status != 'CRITICAL':
                status = 'WARNING'
            alerts.append(
                f"Hit rate {effective_hit_rate:.1f}% "
                f"below warning "
                f"({HIT_RATE_WARNING*100:.0f}%)"
            )

    # Drawdown check
    if current_dd is not None:
        if current_dd < DRAWDOWN_CRITICAL:
            status = 'CRITICAL'
            alerts.append(
                f"Current drawdown {current_dd:.1f}% "
                f"exceeds critical level "
                f"({DRAWDOWN_CRITICAL}%)"
            )
        elif current_dd < DRAWDOWN_WARNING:
            if status != 'CRITICAL':
                status = 'WARNING'
            alerts.append(
                f"Current drawdown {current_dd:.1f}% "
                f"exceeds warning level "
                f"({DRAWDOWN_WARNING}%)"
            )

    return {
        'asset':           asset,
        'status':          status,
        'rolling_sharpe':  rolling_sharpe,
        'max_dd':          max_dd,
        'current_dd':      current_dd,
        'hit_rate':        effective_hit_rate,
        'alerts':          alerts,
    }

# ═════════════════════════════════════════════════════════════
# SECTION 5 — PORTFOLIO HEALTH
# ═════════════════════════════════════════════════════════════

def assess_portfolio_health(price_data,
                             signals_df):
    """
    Assess combined portfolio health.
    Equal weight across all assets.
    """
    all_returns = []

    for asset in ASSETS:
        if asset not in price_data:
            continue
        ret = calculate_rolling_strategy_returns(
            asset, price_data[asset], signals_df
        )
        if ret is not None:
            all_returns.append(ret)

    if not all_returns:
        return None

    # Equal weight portfolio
    port_ret  = pd.concat(
        all_returns, axis=1).dropna().mean(axis=1)

    port_sharpe = calculate_sharpe(port_ret)
    port_maxdd  = calculate_max_drawdown(port_ret)
    port_currdd = calculate_current_drawdown(port_ret)

    # Portfolio status
    status = 'HEALTHY'
    if port_sharpe is not None:
        if port_sharpe < SHARPE_CRITICAL:
            status = 'CRITICAL'
        elif port_sharpe < SHARPE_WARNING:
            status = 'WARNING'

    if port_currdd is not None:
        if port_currdd < DRAWDOWN_CRITICAL:
            status = 'CRITICAL'
        elif port_currdd < DRAWDOWN_WARNING \
                and status != 'CRITICAL':
            status = 'WARNING'

    return {
        'status':          status,
        'rolling_sharpe':  port_sharpe,
        'max_dd':          port_maxdd,
        'current_dd':      port_currdd,
    }

# ═════════════════════════════════════════════════════════════
# SECTION 6 — SAVE HEALTH REPORT
# ═════════════════════════════════════════════════════════════

def save_health_report(asset_health, portfolio_health):
    """Save health metrics to database."""
    conn  = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime('%Y-%m-%d')
    rows  = []

    for asset, h in asset_health.items():
        rows.append({
            'date':           today,
            'asset':          asset,
            'status':         h['status'],
            'rolling_sharpe': h['rolling_sharpe'],
            'max_dd':         h['max_dd'],
            'current_dd':     h['current_dd'],
            'hit_rate':       h['hit_rate'],
            'alert_count':    len(h['alerts']),
        })

    if portfolio_health:
        rows.append({
            'date':           today,
            'asset':          'PORTFOLIO',
            'status':         portfolio_health['status'],
            'rolling_sharpe': portfolio_health[
                'rolling_sharpe'],
            'max_dd':         portfolio_health['max_dd'],
            'current_dd':     portfolio_health['current_dd'],
            'hit_rate':       None,
            'alert_count':    0,
        })

    try:
        df = pd.DataFrame(rows)
        conn.execute(
            "DELETE FROM MODEL_HEALTH WHERE date = ?",
            (today,)
        )
        df.to_sql('MODEL_HEALTH', conn,
                  if_exists='append', index=False)
        conn.commit()
        print(f"  ✅ Health report saved")
    except:
        try:
            df = pd.DataFrame(rows)
            df.to_sql('MODEL_HEALTH', conn,
                      if_exists='replace', index=False)
            conn.commit()
            print(f"  ✅ MODEL_HEALTH table created")
        except Exception as e:
            print(f"  ❌ Save failed: {e}")
    finally:
        conn.close()

# ═════════════════════════════════════════════════════════════
# SECTION 7 — PRINT REPORT
# ═════════════════════════════════════════════════════════════

def print_health_report(asset_health, portfolio_health):
    """Print health report."""
    print("\n" + "="*60)
    print("MODEL HEALTH MONITOR — SYSTEM STATUS")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*60)

    # Portfolio summary
    if portfolio_health:
        ph     = portfolio_health
        emoji  = '✅' if ph['status'] == 'HEALTHY' \
             else '⚠️' if ph['status'] == 'WARNING' \
             else '🚨'
        sharpe = f"{ph['rolling_sharpe']:.3f}" \
            if ph['rolling_sharpe'] is not None \
            else 'N/A'
        curr_dd = f"{ph['current_dd']:.1f}%" \
            if ph['current_dd'] is not None \
            else 'N/A'

        print(f"\n{emoji} PORTFOLIO: {ph['status']}")
        print(f"   Rolling Sharpe ({ROLLING_WINDOW}d): "
              f"{sharpe}")
        print(f"   Current Drawdown: {curr_dd}")

    # Per asset
    print(f"\n{'Asset':<8} {'Status':<10} "
          f"{'Sharpe':>8} {'Curr DD':>9} "
          f"{'Hit Rate':>10}")
    print("-"*55)

    for asset in ASSETS:
        h      = asset_health.get(asset, {})
        status = h.get('status', 'N/A')
        emoji  = '✅' if status == 'HEALTHY' \
             else '⚠️' if status == 'WARNING' \
             else '🚨' if status == 'CRITICAL' \
             else '❓'

        sharpe  = f"{h['rolling_sharpe']:.3f}" \
            if h.get('rolling_sharpe') is not None \
            else 'N/A'
        curr_dd = f"{h['current_dd']:.1f}%" \
            if h.get('current_dd') is not None \
            else 'N/A'
        hit_r   = f"{h['hit_rate']:.1f}%" \
            if h.get('hit_rate') is not None \
            else 'N/A'

        print(f"{asset:<8} {emoji} {status:<8} "
              f"{sharpe:>8} {curr_dd:>9} {hit_r:>10}")

        # Print alerts
        for alert in h.get('alerts', []):
            print(f"         ⚠️  {alert}")

    # Overall recommendation
    critical_assets = [
        a for a, h in asset_health.items()
        if h.get('status') == 'CRITICAL'
    ]
    warning_assets = [
        a for a, h in asset_health.items()
        if h.get('status') == 'WARNING'
    ]
    port_status = portfolio_health.get('status', 'HEALTHY') \
        if portfolio_health else 'HEALTHY'

    print("\n" + "="*60)
    print("RECOMMENDATION:")

    if port_status == 'CRITICAL' or \
            len(critical_assets) >= 3:
        print("🚨 REDUCE RISK — Portfolio in critical state")
        print("   Action: Reduce position sizes by 50%")
        print("   Action: Avoid new signals until recovery")
    elif port_status == 'WARNING' or \
            len(warning_assets) >= 2:
        print("⚠️  CAUTION — System underperforming recently")
        print("   Action: Reduce position sizes by 25%")
        print("   Action: Only trade HIGH confidence signals")
    else:
        print("✅ HEALTHY — System performing within")
        print("   expected parameters")
        print("   Action: Normal position sizing")

    print("="*60)

# ═════════════════════════════════════════════════════════════
# SECTION 8 — TELEGRAM ALERT
# ═════════════════════════════════════════════════════════════

async def send_telegram(message):
    try:
        bot = telegram.Bot(token=BOT_TOKEN)
        await bot.send_message(
            chat_id=CHAT_ID,
            text=message,
            parse_mode='HTML'
        )
        print("  ✅ Telegram alert sent")
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")

def build_telegram_message(asset_health,
                            portfolio_health):
    """Build health alert for Telegram."""
    date = datetime.now().strftime('%d %b %Y %H:%M')

    # Determine overall status
    critical = [
        a for a, h in asset_health.items()
        if h.get('status') == 'CRITICAL']
    warning  = [
        a for a, h in asset_health.items()
        if h.get('status') == 'WARNING']
    port_st  = portfolio_health.get('status', 'HEALTHY') \
        if portfolio_health else 'HEALTHY'

    if port_st == 'CRITICAL' or len(critical) >= 2:
        header_emoji = '🚨'
        header_text  = 'CRITICAL — REDUCE RISK'
    elif port_st == 'WARNING' or len(warning) >= 2:
        header_emoji = '⚠️'
        header_text  = 'WARNING — CAUTION ADVISED'
    else:
        header_emoji = '✅'
        header_text  = 'HEALTHY'

    lines = [
        f"{header_emoji} <b>GMIS MODEL HEALTH: "
        f"{header_text}</b>",
        f"{date}",
        f"{'─' * 30}",
        f"",
    ]

    # Portfolio line
    if portfolio_health:
        ph     = portfolio_health
        sh_str = f"{ph['rolling_sharpe']:.2f}" \
            if ph['rolling_sharpe'] is not None \
            else 'N/A'
        dd_str = f"{ph['current_dd']:.1f}%" \
            if ph['current_dd'] is not None \
            else 'N/A'
        p_emoji = '✅' if ph['status'] == 'HEALTHY' \
              else '⚠️' if ph['status'] == 'WARNING' \
              else '🚨'
        lines.append(
            f"{p_emoji} <b>Portfolio</b>: "
            f"Sharpe {sh_str} | DD {dd_str}"
        )
        lines.append("")

    # Per asset
    lines.append("<b>Asset Status:</b>")
    for asset in ASSETS:
        h      = asset_health.get(asset, {})
        status = h.get('status', 'N/A')
        a_emoji = '✅' if status == 'HEALTHY' \
              else '⚠️' if status == 'WARNING' \
              else '🚨'
        sh_str  = f"{h['rolling_sharpe']:.2f}" \
            if h.get('rolling_sharpe') is not None \
            else 'N/A'
        dd_str  = f"{h['current_dd']:.1f}%" \
            if h.get('current_dd') is not None \
            else 'N/A'
        lines.append(
            f"  {a_emoji} {asset}: "
            f"Sharpe {sh_str} | DD {dd_str}"
        )

    lines.append("")

    # Recommendation
    if port_st == 'CRITICAL' or len(critical) >= 2:
        lines.append(
            "🚨 <b>ACTION: Reduce positions 50%</b>"
        )
        lines.append(
            "Avoid new signals until system recovers"
        )
    elif port_st == 'WARNING' or len(warning) >= 2:
        lines.append(
            "⚠️ <b>ACTION: Reduce positions 25%</b>"
        )
        lines.append(
            "Only trade HIGH confidence signals"
        )
    else:
        lines.append(
            "✅ <b>Normal position sizing</b>"
        )

    lines.append("")
    lines.append("<i>GMIS Model Health Monitor</i>")

    return "\n".join(lines)

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_model_health(send_telegram_flag=True):
    print("\n" + "="*60)
    print("GMIS MODULE 22 — MODEL HEALTH MONITOR")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*60)

    # Load data
    print("\nLoading data...")
    signals_df  = load_signals()
    outcomes_df = load_signal_outcomes()

    print(f"  Signals loaded: {len(signals_df)} rows")
    print(f"  Outcomes loaded: {len(outcomes_df)} rows")

    # Load price data for all assets
    price_data = {}
    for asset in ASSETS:
        try:
            price_data[asset] = load_price_series(asset)
            print(f"  {asset}: {len(price_data[asset])} "
                  f"price rows")
        except Exception as e:
            print(f"  ❌ {asset} price load failed: {e}")

    if signals_df.empty:
        print("\n  ⚠️ No signal data. "
              "Run 12_signal_engine.py first.")
        return

    # Assess each asset
    print("\nAssessing asset health...")
    asset_health = {}
    for asset in ASSETS:
        if asset not in price_data:
            continue
        health = assess_asset_health(
            asset,
            price_data[asset],
            signals_df,
            outcomes_df
        )
        asset_health[asset] = health
        status = health['status']
        sharpe = f"{health['rolling_sharpe']:.3f}" \
            if health['rolling_sharpe'] is not None \
            else 'N/A'
        print(f"  {asset}: {status} "
              f"(Sharpe: {sharpe})")

    # Assess portfolio
    print("\nAssessing portfolio health...")
    portfolio_health = assess_portfolio_health(
        price_data, signals_df
    )
    if portfolio_health:
        p_sh = f"{portfolio_health['rolling_sharpe']:.3f}" \
            if portfolio_health['rolling_sharpe'] \
               is not None else 'N/A'
        print(f"  Portfolio: {portfolio_health['status']} "
              f"(Sharpe: {p_sh})")

    # Save report
    print("\nSaving health report...")
    save_health_report(asset_health, portfolio_health)

    # Print full report
    print_health_report(asset_health, portfolio_health)

    # Send Telegram
    if send_telegram_flag and BOT_TOKEN:
        # Only send alert if WARNING or CRITICAL
        port_st  = portfolio_health.get('status', 'HEALTHY') \
            if portfolio_health else 'HEALTHY'
        critical = [
            a for a, h in asset_health.items()
            if h.get('status') == 'CRITICAL']
        warning  = [
            a for a, h in asset_health.items()
            if h.get('status') == 'WARNING']

        should_alert = (
            port_st in ['WARNING', 'CRITICAL'] or
            len(critical) >= 1 or
            len(warning) >= 2
        )

        # Always send if --force-send flag
        import sys
        if '--force-send' in sys.argv:
            should_alert = True

        if should_alert:
            print("\nSending health alert to Telegram...")
            msg = build_telegram_message(
                asset_health, portfolio_health)
            asyncio.run(send_telegram(msg))
        else:
            print("\n  System healthy — "
                  "no Telegram alert needed")

    return asset_health, portfolio_health

if __name__ == "__main__":
    import sys
    no_telegram = '--no-telegram' in sys.argv
    run_model_health(
        send_telegram_flag=not no_telegram
    )
