# ============================================================
# GMIS 2.0 — MODULE 19 — DECISION ENGINE
# The brain that combines all layers into one final verdict
# Inputs: Signal V3, Analogs, Sentiment, Macro, Yields, VIX
# Output: LONG / SHORT / NO TRADE + confidence + reasoning
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import telegram

load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID   = int(os.getenv('TELEGRAM_CHAT_ID'))

BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_PATH, 'data', 'macro_system.db')
OUTPUT_FILE = os.path.join(BASE_PATH, 'data', 'decisions.json')

ASSETS = ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']

# ── Layer weights ─────────────────────────────────────────────
# These sum to 1.0. Signal is primary, others are context.
WEIGHTS = {
    'signal':    0.35,   # Signal V3 score
    'analog':    0.25,   # Historical analog probability
    'sentiment': 0.15,   # FinBERT sentiment
    'macro':     0.15,   # Macro regime
    'yield':     0.05,   # Yield curve shape
    'vix':       0.05,   # VIX percentile (risk filter)
}

# ── Decision thresholds ───────────────────────────────────────
LONG_THRESHOLD  =  0.20   # combined score above this = LONG
SHORT_THRESHOLD = -0.20   # combined score below this = SHORT
MIN_AGREEMENT   =  50     # % of layers that must agree

# ═════════════════════════════════════════════════════════════
# SECTION 1 — LOAD ALL LAYER DATA
# ═════════════════════════════════════════════════════════════

def load_all_layers():
    """Load latest data from all 6 input layers."""
    conn = sqlite3.connect(DB_PATH)
    layers = {}

    # ── Layer 1: Signal V3 ────────────────────────────────────
    try:
        sig_df = pd.read_sql("SELECT * FROM SIGNALS_V3", conn)
        sig_df['Date'] = pd.to_datetime(sig_df['Date'])
        sig_df = sig_df.sort_values('Date')
        latest_sig = sig_df.iloc[-1]
        layers['signal'] = {
            asset: {
                'score':      float(latest_sig.get(
                               f'{asset}_score', 0)),
                'signal':     latest_sig.get(
                               f'{asset}_signal', 'Neutral'),
                'confidence': latest_sig.get(
                               f'{asset}_confidence', 'NONE'),
                'stable':     latest_sig.get(
                               f'{asset}_stable', 'Neutral'),
                'hit_rate':   float(latest_sig.get(
                               f'{asset}_hit_rate', 0.5)),
            }
            for asset in ASSETS
        }
        layers['signal_date'] = str(
            sig_df['Date'].iloc[-1].date())
    except Exception as e:
        print(f"  ⚠️ Signal layer error: {e}")
        layers['signal'] = {a: {'score': 0, 'signal': 'Neutral',
            'confidence': 'NONE', 'stable': 'Neutral',
            'hit_rate': 0.5} for a in ASSETS}

    # ── Layer 2: Historical Analogs ───────────────────────────
    try:
        analog_df = pd.read_sql(
            "SELECT * FROM ANALOG_OUTCOMES", conn)
        layers['analog'] = {}
        for asset in ASSETS:
            asset_data = analog_df[analog_df['asset'] == asset]
            # Use 30-day probability as primary analog signal
            row_30 = asset_data[
                asset_data['forward_days'] == 30]
            if not row_30.empty:
                prob = float(row_30.iloc[0]['prob_positive'])
                # Convert probability to -1 to +1 score
                # 80% positive → +0.60
                # 50% positive → 0.00
                # 30% positive → -0.40
                analog_score = (prob - 50) / 50
                layers['analog'][asset] = {
                    'prob_positive': prob,
                    'score':         round(analog_score, 3),
                    'median_return': float(
                        row_30.iloc[0]['median_return']),
                    'n_analogs':     int(
                        row_30.iloc[0]['n_analogs']),
                }
            else:
                layers['analog'][asset] = {
                    'prob_positive': 50,
                    'score':         0,
                    'median_return': 0,
                    'n_analogs':     0,
                }
    except Exception as e:
        print(f"  ⚠️ Analog layer error: {e}")
        layers['analog'] = {a: {'prob_positive': 50,
            'score': 0, 'median_return': 0,
            'n_analogs': 0} for a in ASSETS}

    # ── Layer 3: Sentiment ────────────────────────────────────
    try:
        sent_df = pd.read_sql(
            "SELECT * FROM SENTIMENT_DAILY", conn)
        asset_map = {
            'NIFTY': 'NIFTY', 'SP500': 'SP500',
            'Gold': 'Gold', 'Silver': 'Silver',
            'Crude': 'Crude'
        }
        layers['sentiment'] = {}
        overall_sent = float(sent_df['score'].mean())

        for asset in ASSETS:
            keyword = asset_map[asset]
            mask = sent_df['markets'].str.contains(
                keyword, na=False)
            if mask.sum() > 0:
                asset_sent = float(
                    sent_df[mask]['score'].mean())
            else:
                asset_sent = overall_sent
            layers['sentiment'][asset] = {
                'score': round(asset_sent, 4),
                'overall': round(overall_sent, 4),
            }
    except Exception as e:
        print(f"  ⚠️ Sentiment layer error: {e}")
        layers['sentiment'] = {
            a: {'score': 0, 'overall': 0} for a in ASSETS}

    # ── Layer 4: Macro Regime ─────────────────────────────────
    try:
        # Load VIX for regime
        vix_df = pd.read_sql("SELECT * FROM VIX_US", conn)
        vix_df['Date'] = pd.to_datetime(vix_df['Date'])
        vix_df = vix_df.sort_values('Date')
        close_col = [c for c in vix_df.columns
                     if 'Close' in c or 'close' in c]
        vix_col = close_col[0] if close_col else \
                  vix_df.columns[1]
        curr_vix = float(vix_df[vix_col].iloc[-1])

        # VIX percentile over last 3 years
        vix_3y = vix_df[vix_col].tail(756)
        vix_pct = float(
            (vix_3y < curr_vix).mean() * 100)

        # Load SP500 for regime
        sp_df = pd.read_sql("SELECT * FROM SP500", conn)
        sp_df['Date'] = pd.to_datetime(sp_df['Date'])
        sp_df = sp_df.sort_values('Date')
        sp_col = [c for c in sp_df.columns
                  if 'Close' in c or 'close' in c]
        sp_col = sp_col[0] if sp_col else sp_df.columns[1]
        curr_sp = float(sp_df[sp_col].iloc[-1])
        ma200   = float(sp_df[sp_col].tail(200).mean())

        # Regime classification
        sp_above_ma = curr_sp > ma200
        if curr_vix > 30:
            regime = 'Crisis'
            regime_score = -0.8
        elif curr_vix > 20 and not sp_above_ma:
            regime = 'Bear Market'
            regime_score = -0.5
        elif curr_vix <= 20 and sp_above_ma:
            regime = 'Bull Market'
            regime_score = +0.5
        else:
            regime = 'Sideways'
            regime_score = 0.0

        # Load CPI and GDP for macro regime
        cpi_df = pd.read_sql(
            "SELECT * FROM US_CPI ORDER BY Date DESC LIMIT 6",
            conn)
        gdp_df = pd.read_sql(
            "SELECT * FROM US_GDP ORDER BY Date DESC LIMIT 4",
            conn)

        # CPI trend (rising or falling)
        if len(cpi_df) >= 3:
            cpi_recent  = float(cpi_df.iloc[0, 1])
            cpi_3m_ago  = float(cpi_df.iloc[2, 1])
            cpi_trend   = 'Rising' \
                if cpi_recent > cpi_3m_ago else 'Falling'
        else:
            cpi_trend = 'Unknown'

        layers['macro'] = {
            'regime':       regime,
            'regime_score': regime_score,
            'vix':          curr_vix,
            'vix_pct':      round(vix_pct, 1),
            'sp_above_ma':  sp_above_ma,
            'cpi_trend':    cpi_trend,
        }

    except Exception as e:
        print(f"  ⚠️ Macro layer error: {e}")
        layers['macro'] = {
            'regime': 'Unknown', 'regime_score': 0,
            'vix': 20, 'vix_pct': 50,
            'sp_above_ma': True, 'cpi_trend': 'Unknown',
        }

    # ── Layer 5: Yield Curve ──────────────────────────────────
    try:
        y10 = pd.read_sql(
            "SELECT * FROM US_10Y_YIELD "
            "ORDER BY Date DESC LIMIT 30", conn)
        y2  = pd.read_sql(
            "SELECT * FROM US_2Y_YIELD "
            "ORDER BY Date DESC LIMIT 30", conn)

        spread = float(y10.iloc[0, 1]) - float(y2.iloc[0, 1])
        spread_1m_ago = float(y10.iloc[-1, 1]) - \
                        float(y2.iloc[-1, 1])
        spread_trend = 'Steepening' \
            if spread > spread_1m_ago else 'Flattening'

        if spread < 0:
            yield_score = -0.4   # inverted = bearish
        elif spread > 1.0:
            yield_score = +0.3   # steep = bullish growth
        else:
            yield_score = spread * 0.3  # proportional

        layers['yield'] = {
            'spread':       round(spread, 3),
            'spread_trend': spread_trend,
            'inverted':     spread < 0,
            'score':        round(yield_score, 3),
        }
    except Exception as e:
        print(f"  ⚠️ Yield layer error: {e}")
        layers['yield'] = {
            'spread': 0.5, 'spread_trend': 'Unknown',
            'inverted': False, 'score': 0.1,
        }

    # ── Layer 6: VIX Risk Filter ──────────────────────────────
    # Already loaded in macro layer
    vix_pct = layers['macro']['vix_pct']
    # High VIX = risk-off = negative for longs
    # Convert percentile to score: 50th pct = 0, 90th = -0.5
    vix_score = -((vix_pct - 50) / 100)
    layers['vix'] = {
        'percentile': vix_pct,
        'score':      round(vix_score, 3),
        'level':      layers['macro']['vix'],
    }

    conn.close()
    return layers

# ═════════════════════════════════════════════════════════════
# SECTION 2 — COMBINE LAYERS FOR ONE ASSET
# ═════════════════════════════════════════════════════════════

def combine_layers(asset, layers):
    """
    Combine all 6 layers into one final decision.
    Returns combined score, agreement, confidence, reasoning.
    """

    # ── Extract individual layer scores ───────────────────────
    sig_data  = layers['signal'][asset]
    ana_data  = layers['analog'][asset]
    sent_data = layers['sentiment'][asset]
    mac_data  = layers['macro']
    yld_data  = layers['yield']
    vix_data  = layers['vix']

    # Raw scores from each layer (-1 to +1)
    signal_score    = sig_data['score']
    analog_score    = ana_data['score']
    sentiment_score = min(max(sent_data['score'], -1), 1)

    # Macro regime score (asset-specific adjustment)
    # Gold is bullish in Crisis/Stagflation, bearish in Bull
    # Equities are bullish in Bull/Goldilocks, bearish in Crisis
    regime = mac_data['regime']
    if asset in ['Gold', 'Silver']:
        if regime == 'Crisis':
            macro_score = +0.6
        elif regime == 'Bear Market':
            macro_score = +0.3
        elif regime == 'Bull Market':
            macro_score = -0.2
        else:
            macro_score = 0.0
    elif asset == 'Crude':
        if regime == 'Crisis':
            macro_score = +0.2   # geopolitical helps crude
        elif regime == 'Bear Market':
            macro_score = -0.3
        elif regime == 'Bull Market':
            macro_score = +0.3
        else:
            macro_score = 0.0
    else:  # NIFTY, SP500
        if regime == 'Bull Market':
            macro_score = +0.4
        elif regime == 'Bear Market':
            macro_score = -0.5
        elif regime == 'Crisis':
            macro_score = -0.7
        else:
            macro_score = 0.0

    yield_score = yld_data['score']
    vix_score   = vix_data['score']

    # ── Stability adjustment ──────────────────────────────────
    # If signal is not yet stable, reduce its weight
    is_stable = sig_data['stable'] == sig_data['signal']
    if not is_stable:
        signal_score *= 0.5   # halve unstable signal weight

    # ── Hit rate adjustment ───────────────────────────────────
    # Scale signal by recent hit rate
    # 60% hit rate = scale by 1.2, 40% = scale by 0.8
    hit_rate     = sig_data['hit_rate']
    hit_multiplier = (hit_rate - 0.5) * 2 + 1
    hit_multiplier = max(0.5, min(1.5, hit_multiplier))
    signal_score  *= hit_multiplier

    # ── Weighted combination ──────────────────────────────────
    combined = (
        signal_score    * WEIGHTS['signal']    +
        analog_score    * WEIGHTS['analog']    +
        sentiment_score * WEIGHTS['sentiment'] +
        macro_score     * WEIGHTS['macro']     +
        yield_score     * WEIGHTS['yield']     +
        vix_score       * WEIGHTS['vix']
    )

    # ── Agreement score ───────────────────────────────────────
    # Count how many layers point in the same direction
    all_scores = [
        signal_score, analog_score, sentiment_score,
        macro_score, yield_score, vix_score
    ]
    if combined > 0:
        agreeing = sum(1 for s in all_scores if s > 0)
    elif combined < 0:
        agreeing = sum(1 for s in all_scores if s < 0)
    else:
        agreeing = 0

    agreement_pct = (agreeing / len(all_scores)) * 100

    # ── Final decision ────────────────────────────────────────
    if combined >= LONG_THRESHOLD and \
       agreement_pct >= MIN_AGREEMENT:
        decision = 'LONG'
    elif combined <= SHORT_THRESHOLD and \
         agreement_pct >= MIN_AGREEMENT:
        decision = 'SHORT'
    else:
        decision = 'NO TRADE'

    # ── Confidence level ──────────────────────────────────────
    abs_score = abs(combined)
    if abs_score >= 0.40 and agreement_pct >= 70:
        confidence = 'HIGH'
    elif abs_score >= 0.25 and agreement_pct >= 60:
        confidence = 'MEDIUM'
    elif abs_score >= 0.15 and agreement_pct >= 50:
        confidence = 'LOW'
    else:
        confidence = 'NONE'

    # ── Build reasoning ───────────────────────────────────────
    reasons = []

    # Signal reason
    sig_label = sig_data['signal']
    sig_conf  = sig_data['confidence']
    stab_txt  = "stable" if is_stable else "not yet stable"
    reasons.append(
        f"Signal {sig_label} ({sig_conf}, {stab_txt})"
    )

    # Analog reason
    prob = ana_data['prob_positive']
    med  = ana_data['median_return']
    if prob >= 60:
        reasons.append(
            f"Analogs bullish ({prob:.0f}% positive, "
            f"median +{med:.1f}% in 30d)"
        )
    elif prob <= 40:
        reasons.append(
            f"Analogs bearish ({prob:.0f}% positive, "
            f"median {med:.1f}% in 30d)"
        )
    else:
        reasons.append(f"Analogs neutral ({prob:.0f}% positive)")

    # Sentiment reason
    s = sent_data['score']
    if abs(s) > 0.3:
        sent_dir = "positive" if s > 0 else "negative"
        reasons.append(
            f"Sentiment {sent_dir} ({s:+.3f})"
        )

    # Macro reason
    reasons.append(f"Regime: {regime}")

    # Yield reason
    if yld_data['inverted']:
        reasons.append("Yield curve inverted ⚠️")
    else:
        reasons.append(
            f"Yield spread +{yld_data['spread']:.2f}% "
            f"({yld_data['spread_trend']})"
        )

    # VIX reason
    vp = vix_data['percentile']
    if vp > 75:
        reasons.append(f"VIX elevated ({vp:.0f}th pct) ⚠️")
    elif vp < 30:
        reasons.append(f"VIX calm ({vp:.0f}th pct) ✅")

    return {
        'asset':         asset,
        'decision':      decision,
        'combined_score': round(combined, 4),
        'agreement_pct': round(agreement_pct, 1),
        'confidence':    confidence,
        'is_stable':     is_stable,
        'layer_scores': {
            'signal':    round(signal_score, 4),
            'analog':    round(analog_score, 4),
            'sentiment': round(sentiment_score, 4),
            'macro':     round(macro_score, 4),
            'yield':     round(yield_score, 4),
            'vix':       round(vix_score, 4),
        },
        'reasoning': reasons,
        'analog_prob':    prob,
        'analog_median':  med,
        'regime':         regime,
        'timestamp':      datetime.now().strftime(
                           '%Y-%m-%d %H:%M'),
    }

# ═════════════════════════════════════════════════════════════
# SECTION 3 — RUN ALL ASSETS
# ═════════════════════════════════════════════════════════════

def run_decision_engine():
    """Run Decision Engine for all 5 assets."""

    print("\n" + "="*65)
    print("GMIS MODULE 19 — DECISION ENGINE")
    print(datetime.now().strftime('%A %d %B %Y — %H:%M'))
    print("="*65)

    print("\nLoading all layers...")
    layers = load_all_layers()
    print(f"  Signal date:  {layers.get('signal_date', 'N/A')}")
    print(f"  Regime:       {layers['macro']['regime']}")
    print(f"  VIX:          {layers['macro']['vix']:.1f} "
          f"({layers['macro']['vix_pct']:.0f}th pct)")
    print(f"  Yield spread: "
          f"{layers['yield']['spread']:+.2f}%")

    print("\nGenerating decisions...")
    decisions = {}
    for asset in ASSETS:
        result = combine_layers(asset, layers)
        decisions[asset] = result

    return decisions, layers

# ═════════════════════════════════════════════════════════════
# SECTION 4 — SAVE TO DATABASE
# ═════════════════════════════════════════════════════════════

def save_decisions(decisions):
    """Save decisions to database and JSON file."""
    conn  = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime('%Y-%m-%d')

    rows = []
    for asset in decisions:
        d = decisions[asset]
        rows.append({
            'date':            today,
            'asset':           asset,
            'bias':            d['decision'],
            'combined':        d['combined_score'],
            'confidence':      d['confidence'],
            'agreement':       d['agreement_pct'],
            'layer_signal':    d['layer_scores']['signal'],
            'layer_analog':    d['layer_scores']['analog'],
            'layer_sentiment': d['layer_scores']['sentiment'],
            'layer_macro':     d['layer_scores']['macro'],
            'layer_yield':     d['layer_scores']['yield'],
            'layer_vix':       d['layer_scores']['vix'],
        })

    try:
        df = pd.DataFrame(rows)
        # Delete today's existing decisions
        conn.execute(
            "DELETE FROM DECISIONS WHERE date = ?", (today,))
        df.to_sql('DECISIONS', conn,
                  if_exists='append', index=False)
        conn.commit()
        print(f"\n  ✅ Decisions saved to database")
    except Exception as e:
        print(f"\n  ❌ Database save failed: {e}")
    finally:
        conn.close()

    # Also save to JSON for quick access
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(decisions, f, indent=2, default=str)
    print(f"  ✅ Decisions saved to {OUTPUT_FILE}")

# ═════════════════════════════════════════════════════════════
# SECTION 5 — PRINT DECISIONS
# ═════════════════════════════════════════════════════════════

def print_decisions(decisions):
    """Print all decisions in a clean format."""

    print("\n" + "="*65)
    print("FINAL DECISIONS — ALL ASSETS")
    print("="*65)
    print(f"\n{'Asset':<8} {'Decision':<10} {'Score':>7} "
          f"{'Agreement':>10} {'Confidence':<10} {'Stable'}")
    print("-"*65)

    decision_emojis = {
        'LONG': '🟢', 'SHORT': '🔴', 'NO TRADE': '⬜'
    }

    for asset in ASSETS:
        d     = decisions[asset]
        emoji = decision_emojis.get(d['decision'], '⬜')
        stab  = "✅" if d['is_stable'] else "⏳"
        print(f"{asset:<8} "
              f"{emoji} {d['decision']:<8} "
              f"{d['combined_score']:>+7.3f}  "
              f"{d['agreement_pct']:>8.0f}%  "
              f"{d['confidence']:<10} "
              f"{stab}")

    print("\n" + "="*65)
    print("LAYER BREAKDOWN")
    print("="*65)
    print(f"\n{'Asset':<8} {'Signal':>8} {'Analog':>8} "
          f"{'Sentiment':>10} {'Macro':>7} "
          f"{'Yield':>7} {'VIX':>7}")
    print("-"*65)

    for asset in ASSETS:
        d  = decisions[asset]
        ls = d['layer_scores']
        print(f"{asset:<8} "
              f"{ls['signal']:>+8.3f} "
              f"{ls['analog']:>+8.3f} "
              f"{ls['sentiment']:>+10.3f} "
              f"{ls['macro']:>+7.3f} "
              f"{ls['yield']:>+7.3f} "
              f"{ls['vix']:>+7.3f}")

    print("\n" + "="*65)
    print("REASONING")
    print("="*65)

    for asset in ASSETS:
        d     = decisions[asset]
        emoji = decision_emojis.get(d['decision'], '⬜')
        print(f"\n{emoji} {asset} — "
              f"{d['decision']} "
              f"(Confidence: {d['confidence']})")
        for reason in d['reasoning']:
            print(f"   • {reason}")

    # Actionable summary
    print("\n" + "="*65)
    print("📡 ACTIONABLE DECISIONS "
          "(Decision confirmed + Confidence ≥ MEDIUM)")
    print("="*65)

    actionable = [
        (a, d) for a, d in decisions.items()
        if d['decision'] != 'NO TRADE' and
        d['confidence'] in ['HIGH', 'MEDIUM']
    ]

    if actionable:
        for asset, d in actionable:
            emoji = decision_emojis[d['decision']]
            print(f"\n  {emoji} {asset}: {d['decision']}")
            print(f"     Score:     {d['combined_score']:+.3f}")
            print(f"     Agreement: {d['agreement_pct']:.0f}%")
            print(f"     Analog:    "
                  f"{d['analog_prob']:.0f}% probability, "
                  f"median {d['analog_median']:+.1f}% (30d)")
            for r in d['reasoning'][:3]:
                print(f"     • {r}")
    else:
        print("\n  No high-confidence actionable decisions today.")
        print("  Patience — wait for better alignment.")

    print("\n" + "="*65)

# ═════════════════════════════════════════════════════════════
# SECTION 6 — TELEGRAM DELIVERY
# ═════════════════════════════════════════════════════════════

async def send_decisions_telegram(decisions):
    """Send decision summary to Telegram."""
    bot = telegram.Bot(token=BOT_TOKEN)

    decision_emojis = {
        'LONG': '🟢', 'SHORT': '🔴', 'NO TRADE': '⬜'
    }

    lines = [
        f"🧠 <b>GMIS DECISION ENGINE</b>",
        f"{datetime.now().strftime('%d %b %Y — %H:%M IST')}",
        f"{'─' * 30}",
        "",
    ]

    for asset in ASSETS:
        d     = decisions[asset]
        emoji = decision_emojis.get(d['decision'], '⬜')
        stab  = "✅" if d['is_stable'] else "⏳"
        lines.append(
            f"{emoji} <b>{asset}</b>: {d['decision']} "
            f"| {d['confidence']} "
            f"| {d['agreement_pct']:.0f}% agree "
            f"| {stab}"
        )

    # Actionable section
    actionable = [
        (a, d) for a, d in decisions.items()
        if d['decision'] != 'NO TRADE' and
        d['confidence'] in ['HIGH', 'MEDIUM']
    ]

    if actionable:
        lines.append("")
        lines.append("📡 <b>ACTIONABLE:</b>")
        for asset, d in actionable:
            emoji = decision_emojis[d['decision']]
            lines.append(
                f"  {emoji} {asset}: {d['decision']} "
                f"({d['combined_score']:+.3f})"
            )
            lines.append(
                f"     Analog: {d['analog_prob']:.0f}% "
                f"probability | "
                f"Median {d['analog_median']:+.1f}% (30d)"
            )
    else:
        lines.append("")
        lines.append("⬜ No actionable decisions today.")
        lines.append("Patience — wait for better alignment.")

    message = "\n".join(lines)
    await bot.send_message(
        chat_id=CHAT_ID,
        text=message,
        parse_mode='HTML'
    )
    print("  ✅ Decisions sent to Telegram")

def send_telegram_decisions(decisions):
    asyncio.run(send_decisions_telegram(decisions))

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Run decision engine
    decisions, layers = run_decision_engine()

    # Print results
    print_decisions(decisions)

    # Save to database
    save_decisions(decisions)

    # Send to Telegram
    print("\nSending to Telegram...")
    send_telegram_decisions(decisions)

    print("\n✅ Decision Engine complete.")
