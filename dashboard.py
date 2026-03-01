# ============================================================
# GLOBAL MACRO INTELLIGENCE SYSTEM
# Three-Layer Professional Streamlit Dashboard
# Layer 1: Headline signals (60-second test)
# Layer 2: Interactive charts (context)
# Layer 3: Deep dive tables (expanders)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Global Macro Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour system (strict palette) ───────────────────────────
GREEN   = "#1E6B3C"
RED     = "#C00000"
BLUE    = "#1F3864"
MID_BLUE= "#2E75B6"
ORANGE  = "#C55A11"
GRAY    = "#595959"
LIGHT   = "#F4F4F4"

# ── CSS styling ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 5px solid #2E75B6;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 8px;
    }
    .metric-positive { border-left-color: #1E6B3C !important; }
    .metric-negative { border-left-color: #C00000 !important; }
    .metric-neutral  { border-left-color: #C55A11 !important; }
    .signal-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 13px;
    }
    .badge-long    { background: #D5E8D4; color: #1E6B3C; }
    .badge-short   { background: #FFCCCC; color: #C00000; }
    .badge-neutral { background: #FFF3CD; color: #856404; }
    .commentary-box {
        background: #EDE7F6;
        border-left: 4px solid #5C2D91;
        padding: 10px 16px;
        border-radius: 4px;
        font-size: 13px;
        color: #333;
        margin: 8px 0;
    }
    .delta-positive { color: #1E6B3C; font-weight: bold; }
    .delta-negative { color: #C00000; font-weight: bold; }
    h1, h2, h3 { color: #1F3864; }
</style>
""", unsafe_allow_html=True)

# ── Database connection ───────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')

@st.cache_data(ttl=3600)
def load_close(table):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    close_col = [c for c in df.columns if 'Close' in c or 'close' in c]
    return df[close_col[0]].dropna() if close_col else df.iloc[:, 0].dropna()

@st.cache_data(ttl=3600)
def load_macro(table):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    return df.iloc[:, 0].dropna()

@st.cache_data(ttl=3600)
def load_sentiment():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM SENTIMENT_DAILY", conn)
        conn.close()
        return df
    except:
        conn.close()
        return pd.DataFrame()

# ── Load all data ─────────────────────────────────────────────
nifty     = load_close('NIFTY50')
sp500     = load_close('SP500')
gold      = load_close('GOLD')
silver    = load_close('SILVER')
crude     = load_close('CRUDE_WTI')
vix_us    = load_close('VIX_US')
vix_india = load_close('VIX_INDIA')
usd_inr   = load_close('USD_INR')
dxy       = load_close('DXY')
yield_10y = load_macro('US_10Y_YIELD')
yield_2y  = load_macro('US_2Y_YIELD')
sentiment_df = load_sentiment()

# ── Helper functions ──────────────────────────────────────────
def get_regime(vix_val, sp_above_ma):
    if vix_val > 30:
        return "Crisis", RED
    elif vix_val > 20 and not sp_above_ma:
        return "Bear Market", "#8B0000"
    elif vix_val <= 20 and sp_above_ma:
        return "Bull Market", GREEN
    else:
        return "Sideways", ORANGE

def pct_change_label(current, prev):
    if prev == 0:
        return 0
    return ((current - prev) / abs(prev)) * 100

def delta_arrow(val):
    if val > 0:
        return f"<span class='delta-positive'>▲ {val:+.2f}%</span>"
    elif val < 0:
        return f"<span class='delta-negative'>▼ {val:.2f}%</span>"
    return f"<span style='color:{GRAY}'>→ {val:.2f}%</span>"

def generate_signal(asset_name, price_series, vix_series):
    """Simple rule-based signal for demonstration."""
    if len(price_series) < 60:
        return "Neutral", ORANGE
    ma20  = price_series.rolling(20).mean().iloc[-1]
    ma60  = price_series.rolling(60).mean().iloc[-1]
    curr  = price_series.iloc[-1]
    vix   = vix_series.iloc[-1]
    if curr > ma20 > ma60 and vix < 25:
        return "Long", GREEN
    elif curr < ma20 < ma60 or vix > 30:
        return "Short", RED
    else:
        return "Neutral", ORANGE

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/combo-chart.png", width=48)
    st.title("GMIS Controls")
    st.markdown("---")

    # Date range filter
    st.subheader("📅 Date Range")
    min_date = nifty.index.min().date()
    max_date = nifty.index.max().date()
    start_date = st.date_input("From", value=datetime(2015, 1, 1).date(),
                                min_value=min_date, max_value=max_date)
    end_date   = st.date_input("To",   value=max_date,
                                min_value=min_date, max_value=max_date)

    st.markdown("---")
    st.subheader("🎯 Market Focus")
    selected_markets = st.multiselect(
        "Select markets to display",
        ["NIFTY 50", "S&P 500", "Gold", "Silver", "Crude WTI"],
        default=["NIFTY 50", "S&P 500", "Gold"]
    )

    st.markdown("---")
    st.subheader("⚙️ Signal Settings")
    vix_crisis = st.slider("VIX Crisis Level", 20, 50, 30)
    vix_caution= st.slider("VIX Caution Level", 10, 30, 20)

    st.markdown("---")
    st.caption(f"Data updated: {max_date}")
    st.caption("Global Macro Intelligence System")
    st.caption("Built by Niraj Mutha — 2026")

# ── Filter data by date ───────────────────────────────────────
def filter_series(s):
    return s[(s.index.date >= start_date) & (s.index.date <= end_date)]

nifty_f   = filter_series(nifty)
sp500_f   = filter_series(sp500)
gold_f    = filter_series(gold)
silver_f  = filter_series(silver)
crude_f   = filter_series(crude)
vix_f     = filter_series(vix_us)
usd_inr_f = filter_series(usd_inr)
dxy_f     = filter_series(dxy)

# ── Current values & deltas ───────────────────────────────────
def safe_last(s, n=1):
    return s.iloc[-n] if len(s) >= n else np.nan

curr_nifty  = safe_last(nifty_f)
prev_nifty  = safe_last(nifty_f, 2)
curr_sp500  = safe_last(sp500_f)
prev_sp500  = safe_last(sp500_f, 2)
curr_gold   = safe_last(gold_f)
prev_gold   = safe_last(gold_f, 2)
curr_vix    = safe_last(vix_f)
curr_inr    = safe_last(usd_inr_f)
prev_inr    = safe_last(usd_inr_f, 2)

# Regime
sp_ma200      = sp500_f.rolling(200).mean()
sp_above_ma   = curr_sp500 > sp_ma200.iloc[-1] if len(sp_ma200) > 0 else False
regime_label, regime_color = get_regime(curr_vix, sp_above_ma)

# Yield curve
yields_aligned = pd.DataFrame({'10Y': yield_10y, '2Y': yield_2y}).dropna()
curr_spread    = yields_aligned['10Y'].iloc[-1] - yields_aligned['2Y'].iloc[-1] if len(yields_aligned) > 0 else 0
curve_shape    = "Inverted ⚠️" if curr_spread < 0 else "Normal ✓"

# Signals
sig_nifty,  col_nifty  = generate_signal("NIFTY",  nifty_f,  vix_f)
sig_sp500,  col_sp500  = generate_signal("SP500",  sp500_f,  vix_f)
sig_gold,   col_gold   = generate_signal("Gold",   gold_f,   vix_f)
sig_crude,  col_crude  = generate_signal("Crude",  crude_f,  vix_f)

# Sentiment
overall_sentiment = "N/A"
sentiment_score   = 0.0
if not sentiment_df.empty:
    sentiment_score   = sentiment_df['score'].mean()
    overall_sentiment = "Positive" if sentiment_score > 0.05 else \
                        "Negative" if sentiment_score < -0.05 else "Neutral"

# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background: linear-gradient(135deg, #1F3864, #2E75B6);
     padding: 20px 28px; border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: white; margin: 0; font-size: 26px;'>
        📊 Global Macro Intelligence System
    </h1>
    <p style='color: #D6E4F0; margin: 4px 0 0 0; font-size: 14px;'>
        Multi-Asset Signal Framework &nbsp;|&nbsp;
        {start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')} &nbsp;|&nbsp;
        Built by Niraj Mutha
    </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# LAYER 1 — HEADLINE SIGNALS (60-second test)
# ════════════════════════════════════════════════════════════════════
st.markdown("### 🎯 Layer 1 — Market Intelligence Snapshot")
st.caption("The 60-second view — understand the full market state at a glance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    color_cls = "metric-positive" if regime_label == "Bull Market" else \
                "metric-negative" if regime_label in ["Bear Market","Crisis"] else \
                "metric-neutral"
    st.markdown(f"""
    <div class='metric-card {color_cls}'>
        <div style='font-size:11px; color:{GRAY}; text-transform:uppercase;'>Market Regime</div>
        <div style='font-size:20px; font-weight:bold; color:#1F3864; color:{regime_color};'>{regime_label}</div>
        <div style='font-size:11px; color:{GRAY};'>VIX: {curr_vix:.1f}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    d_nifty = pct_change_label(curr_nifty, prev_nifty)
    st.markdown(f"""
    <div class='metric-card {"metric-positive" if d_nifty >= 0 else "metric-negative"}'>
        <div style='font-size:11px; color:{GRAY}; text-transform:uppercase;'>NIFTY 50</div>
        <div style='font-size:20px; font-weight:bold; color:#1F3864;'>{curr_nifty:,.0f}</div>
        <div>{delta_arrow(d_nifty)}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    d_sp = pct_change_label(curr_sp500, prev_sp500)
    st.markdown(f"""
    <div class='metric-card {"metric-positive" if d_sp >= 0 else "metric-negative"}'>
        <div style='font-size:11px; color:{GRAY}; text-transform:uppercase;'>S&P 500</div>
        <div style='font-size:20px; font-weight:bold; color:#1F3864;'>{curr_sp500:,.0f}</div>
        <div>{delta_arrow(d_sp)}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    d_gold = pct_change_label(curr_gold, prev_gold)
    st.markdown(f"""
    <div class='metric-card {"metric-positive" if d_gold >= 0 else "metric-negative"}'>
        <div style='font-size:11px; color:{GRAY}; text-transform:uppercase;'>Gold (USD)</div>
        <div style='font-size:20px; font-weight:bold; color:#1F3864;'>${curr_gold:,.0f}</div>
        <div>{delta_arrow(d_gold)}</div>
    </div>""", unsafe_allow_html=True)

with col5:
    sent_color = "metric-positive" if overall_sentiment == "Positive" else \
                 "metric-negative" if overall_sentiment == "Negative" else \
                 "metric-neutral"
    st.markdown(f"""
    <div class='metric-card {sent_color}'>
        <div style='font-size:11px; color:{GRAY}; text-transform:uppercase;'>Sentiment</div>
        <div style='font-size:20px; font-weight:bold; color:#1F3864;'>{overall_sentiment}</div>
        <div style='font-size:11px; color:{GRAY};'>Score: {sentiment_score:+.3f}</div>
    </div>""", unsafe_allow_html=True)

# Signal row
st.markdown("#### 📡 Active Trading Signals")
sc1, sc2, sc3, sc4, sc5 = st.columns(5)

def signal_badge(label, signal, color):
    badge_class = f"badge-{'long' if signal=='Long' else 'short' if signal=='Short' else 'neutral'}"
    return f"""
    <div style='text-align:center; padding: 10px; background:white;
         border-radius:8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
        <div style='font-size:11px; color:{GRAY}; margin-bottom:4px;'>{label}</div>
        <span class='signal-badge {badge_class}'>{signal}</span>
    </div>"""

with sc1: st.markdown(signal_badge("NIFTY 50",  sig_nifty, col_nifty),  unsafe_allow_html=True)
with sc2: st.markdown(signal_badge("S&P 500",   sig_sp500, col_sp500),  unsafe_allow_html=True)
with sc3: st.markdown(signal_badge("Gold",      sig_gold,  col_gold),   unsafe_allow_html=True)
with sc4: st.markdown(signal_badge("Crude WTI", sig_crude, col_crude),  unsafe_allow_html=True)
with sc5: st.markdown(f"""
    <div style='text-align:center; padding:10px; background:white;
         border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1);'>
        <div style='font-size:11px; color:{GRAY}; margin-bottom:4px;'>Yield Curve</div>
        <span class='signal-badge {"badge-short" if "Inverted" in curve_shape else "badge-long"}'>{curve_shape}</span>
    </div>""", unsafe_allow_html=True)

# Dynamic commentary
vix_pct = (vix_f.rank(pct=True).iloc[-1] * 100) if len(vix_f) > 0 else 50
commentary = f"""
📌 <strong>System Commentary ({end_date.strftime('%d %b %Y')}):</strong>
Market is in <strong>{regime_label}</strong> regime with VIX at {curr_vix:.1f}
({vix_pct:.0f}th percentile of selected history).
Yield curve is <strong>{curve_shape}</strong> (spread: {curr_spread:+.2f}%).
Sentiment reads <strong>{overall_sentiment}</strong> ({sentiment_score:+.3f}).
NIFTY signal is <strong>{sig_nifty}</strong> | S&P 500 signal is <strong>{sig_sp500}</strong> |
Gold signal is <strong>{sig_gold}</strong>.
"""
st.markdown(f"<div class='commentary-box'>{commentary}</div>", unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════
# LAYER 2 — INTERACTIVE CHARTS (context)
# ════════════════════════════════════════════════════════════════════
st.markdown("### 📈 Layer 2 — Interactive Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌍 Macro", "📊 Volatility", "💱 FX & Bonds", "😐 Sentiment", "📉 Backtest"])

# ── TAB 1: MACRO ──────────────────────────────────────────────
with tab1:
    st.subheader("Cross-Asset Price Performance")

    # Normalise selected markets
    market_map = {
        "NIFTY 50":   nifty_f,
        "S&P 500":    sp500_f,
        "Gold":       gold_f,
        "Silver":     silver_f,
        "Crude WTI":  crude_f,
    }
    color_map = {
        "NIFTY 50":  BLUE,
        "S&P 500":   MID_BLUE,
        "Gold":      ORANGE,
        "Silver":    GRAY,
        "Crude WTI": "#7030A0",
    }

    fig = go.Figure()
    for market in selected_markets:
        if market in market_map:
            s = market_map[market].dropna()
            if len(s) > 0:
                s_norm = s / s.iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=s_norm.index, y=s_norm.values,
                    name=market, line=dict(color=color_map[market], width=1.8),
                    hovertemplate=f"{market}: %{{y:.1f}}<extra></extra>"
                ))

    fig.add_hline(y=100, line_dash="dash", line_color="gray", line_width=0.8,
                   annotation_text="Start (100)")
    fig.update_layout(
        title="Indexed Performance (Base = 100 at start of selected period)",
        yaxis_title="Indexed Price (Base 100)",
        xaxis_title="",
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Rolling Correlation Matrix")
    corr_window = st.slider("Correlation window (days)", 30, 252, 60, key="corr_slider")

    all_returns = pd.DataFrame({
        'NIFTY':  nifty_f.pct_change(),
        'SP500':  sp500_f.pct_change(),
        'Gold':   gold_f.pct_change(),
        'Silver': silver_f.pct_change(),
        'Crude':  crude_f.pct_change(),
    }).dropna()

    corr_matrix = all_returns.tail(corr_window).corr()
    fig_corr = px.imshow(corr_matrix, color_continuous_scale='RdYlGn',
                          zmin=-1, zmax=1, text_auto='.2f',
                          title=f"Correlation Matrix — Last {corr_window} Trading Days")
    fig_corr.update_layout(height=380, template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)

# ── TAB 2: VOLATILITY ─────────────────────────────────────────
with tab2:
    st.subheader("VIX & Volatility Regime")

    fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("VIX — Fear Index", "VIX vs India VIX"),
                             vertical_spacing=0.12)

    fig_vix.add_trace(go.Scatter(x=vix_f.index, y=vix_f.values, name='VIX US',
                                  line=dict(color=RED, width=1.5)), row=1, col=1)
    fig_vix.add_hline(y=vix_caution, line_dash="dash", line_color="orange",
                       annotation_text=f"Caution ({vix_caution})", row=1, col=1)
    fig_vix.add_hline(y=vix_crisis,  line_dash="dash", line_color=RED,
                       annotation_text=f"Crisis ({vix_crisis})", row=1, col=1)

    vix_mean = vix_f.mean()
    fig_vix.add_hline(y=vix_mean, line_dash="dot", line_color=GRAY,
                       annotation_text=f"Mean ({vix_mean:.1f})", row=1, col=1)

    vix_india_f = filter_series(vix_india)
    fig_vix.add_trace(go.Scatter(x=vix_india_f.index, y=vix_india_f.values,
                                  name='India VIX', line=dict(color=BLUE, width=1.5)), row=2, col=1)
    fig_vix.add_trace(go.Scatter(x=vix_f.index, y=vix_f.values, name='US VIX',
                                  line=dict(color=RED, width=1.2, dash='dot')), row=2, col=1)

    fig_vix.update_layout(height=520, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig_vix, use_container_width=True)

    # VIX stats
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Current VIX",      f"{curr_vix:.1f}")
    v2.metric("VIX Mean",         f"{vix_f.mean():.1f}")
    v3.metric("VIX Max (period)", f"{vix_f.max():.1f}")
    v4.metric("Days above crisis",f"{(vix_f > vix_crisis).sum()}")

    # Commentary
    nifty_ret_aligned = nifty_f.pct_change()
    vix_aligned       = vix_f.reindex(nifty_ret_aligned.index).ffill()
    hist_ret_high_vix = nifty_ret_aligned[vix_aligned > vix_crisis].mean() * 100
    hist_ret_low_vix  = nifty_ret_aligned[vix_aligned < vix_caution].mean() * 100
    st.markdown(f"""<div class='commentary-box'>
    📌 <strong>Volatility Commentary:</strong>
    When VIX exceeds {vix_crisis} (crisis), NIFTY averages
    <strong>{hist_ret_high_vix:.3f}%</strong> next day (sell pressure).
    When VIX is below {vix_caution} (calm), NIFTY averages
    <strong>{hist_ret_low_vix:.3f}%</strong> next day (drift higher).
    Current VIX is at the <strong>{vix_pct:.0f}th percentile</strong> of history.
    </div>""", unsafe_allow_html=True)

# ── TAB 3: FX & BONDS ─────────────────────────────────────────
with tab3:
    st.subheader("Yield Curve & Currency Analysis")

    # Yield curve
    yields_f = pd.DataFrame({
        '10Y': filter_series(yield_10y),
        '2Y':  filter_series(yield_2y),
    }).dropna()
    yields_f['Spread'] = yields_f['10Y'] - yields_f['2Y']

    fig_yc = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Treasury Yields", "Yield Spread (10Y − 2Y)"),
                            vertical_spacing=0.12)
    fig_yc.add_trace(go.Scatter(x=yields_f.index, y=yields_f['10Y'],
                                 name='10Y', line=dict(color=BLUE, width=1.5)), row=1, col=1)
    fig_yc.add_trace(go.Scatter(x=yields_f.index, y=yields_f['2Y'],
                                 name='2Y', line=dict(color=ORANGE, width=1.5)), row=1, col=1)
    fig_yc.add_trace(go.Scatter(x=yields_f.index, y=yields_f['Spread'],
                                 name='Spread', line=dict(color=MID_BLUE, width=1.5),
                                 fill='tozeroy',
                                 fillcolor='rgba(30,107,60,0.15)'), row=2, col=1)
    fig_yc.add_hline(y=0, line_dash="dash", line_color=RED,
                      annotation_text="Inversion level", row=2, col=1)
    fig_yc.update_layout(height=480, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig_yc, use_container_width=True)

    # USD/INR
    st.subheader("USD/INR Exchange Rate")
    fig_inr = go.Figure()
    fig_inr.add_trace(go.Scatter(x=usd_inr_f.index, y=usd_inr_f.values,
                                  name='USD/INR', line=dict(color=RED, width=1.5),
                                  fill='tozeroy', fillcolor='rgba(192,0,0,0.05)'))
    inr_mean = usd_inr_f.mean()
    fig_inr.add_hline(y=inr_mean, line_dash="dot", line_color=GRAY,
                       annotation_text=f"Mean ({inr_mean:.1f})")
    fig_inr.update_layout(height=320, template="plotly_white",
                           yaxis_title="USD/INR Rate",
                           title="Rising = Rupee Weakening")
    st.plotly_chart(fig_inr, use_container_width=True)

    # INR depreciation stat
    if len(usd_inr_f) > 1:
        depr = pct_change_label(usd_inr_f.iloc[-1], usd_inr_f.iloc[0])
        st.markdown(f"""<div class='commentary-box'>
        📌 <strong>FX Commentary:</strong>
        USD/INR moved from <strong>{usd_inr_f.iloc[0]:.1f}</strong> to
        <strong>{usd_inr_f.iloc[-1]:.1f}</strong> over the selected period
        ({depr:+.1f}% Rupee depreciation).
        Yield curve is currently <strong>{curve_shape}</strong> with a
        {curr_spread:+.2f}% spread — inversions historically precede recessions
        by 12–18 months.
        </div>""", unsafe_allow_html=True)

# ── TAB 4: SENTIMENT ──────────────────────────────────────────
with tab4:
    st.subheader("Live News Sentiment Analysis")

    if sentiment_df.empty:
        st.warning("No sentiment data found. Run 09_sentiment_engine.py first.")
    else:
        s1, s2, s3 = st.columns(3)
        s1.metric("Overall Score",       f"{sentiment_df['score'].mean():+.4f}")
        s2.metric("Total Headlines",     f"{len(sentiment_df)}")
        s3.metric("Positive %",          f"{(sentiment_df['sentiment']=='Positive').mean()*100:.1f}%")

        # Pie chart
        sent_counts = sentiment_df['sentiment'].value_counts()
        fig_pie = px.pie(values=sent_counts.values, names=sent_counts.index,
                          color=sent_counts.index,
                          color_discrete_map={'Positive': GREEN, 'Neutral': MID_BLUE, 'Negative': RED},
                          title="Sentiment Distribution")
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Sentiment by market bar chart
        if 'markets' in sentiment_df.columns:
            market_sent = {}
            for mkt in ['NIFTY','SP500','Gold','Crude','General']:
                mask = sentiment_df['markets'].str.contains(mkt, na=False)
                if mask.sum() > 0:
                    market_sent[mkt] = sentiment_df[mask]['score'].mean()

            if market_sent:
                fig_mkt = go.Figure(go.Bar(
                    x=list(market_sent.keys()),
                    y=list(market_sent.values()),
                    marker_color=[GREEN if v >= 0.05 else RED if v <= -0.05
                                   else ORANGE for v in market_sent.values()],
                ))
                fig_mkt.add_hline(y=0.05,  line_dash="dot", line_color=GREEN,
                                   annotation_text="Positive threshold")
                fig_mkt.add_hline(y=-0.05, line_dash="dot", line_color=RED,
                                   annotation_text="Negative threshold")
                fig_mkt.update_layout(title="Sentiment Score by Market",
                                       yaxis_title="Avg Score",
                                       template="plotly_white", height=350)
                st.plotly_chart(fig_mkt, use_container_width=True)

        # Top headlines table
        with st.expander("📰 View All Headlines"):
            st.dataframe(
                sentiment_df[['headline','score','sentiment','markets','source']]
                .sort_values('score', ascending=False),
                use_container_width=True
            )
# ── TAB 5: BACKTEST ───────────────────────────────────────────
with tab5:
    st.subheader("Signal Strategy vs Buy-and-Hold Backtest")
    st.caption("15 years of real data | Transaction cost: 0.1% per trade | No leverage")

    # ── Load signals from database ────────────────────────────
    @st.cache_data(ttl=3600)
    def load_signals():
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql("SELECT * FROM SIGNALS", conn)
            conn.close()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            return df
        except:
            conn.close()
            return pd.DataFrame()

    signals = load_signals()

    if signals.empty:
        st.warning("No signal data found. Run 12_signal_engine.py first.")
    else:
        # ── Backtest engine ───────────────────────────────────
        def run_backtest_dash(price, signal_scores,
                               transaction_cost=0.001,
                               threshold=0.15):
            sig      = signal_scores.reindex(price.index).ffill().bfill()
            daily_ret= price.pct_change().fillna(0)
            position = pd.Series(0.0, index=price.index)
            position[sig >= threshold]  = 1.0
            position[sig <= -threshold] = 0.0
            pos_change = position.diff().abs().fillna(0)
            strat_ret  = position.shift(1) * daily_ret
            strat_ret -= pos_change * transaction_cost
            bnh_ret    = daily_ret.copy()
            strat_eq   = (1 + strat_ret).cumprod() * 100
            bnh_eq     = (1 + bnh_ret).cumprod()   * 100

            def metrics(ret, eq):
                n_years  = len(ret) / 252
                total    = eq.iloc[-1] / 100
                cagr     = (total ** (1/n_years) - 1) * 100 if n_years > 0 and total > 0 else 0
                sharpe   = (ret.mean() / (ret.std() + 1e-10)) * np.sqrt(252)
                roll_max = eq.cummax()
                maxdd    = ((eq - roll_max) / roll_max).min() * 100
                sortino  = (ret.mean() / (ret[ret<0].std() + 1e-10)) * np.sqrt(252)
                winrate  = (ret > 0).sum() / (ret != 0).sum() * 100
                return {
                    'CAGR': cagr, 'Sharpe': sharpe,
                    'MaxDD': maxdd, 'Sortino': sortino,
                    'WinRate': winrate,
                }

            return {
                'strat_eq':  strat_eq,
                'bnh_eq':    bnh_eq,
                'strat_ret': strat_ret,
                'bnh_ret':   bnh_ret,
                'position':  position,
                'strat_m':   metrics(strat_ret, strat_eq),
                'bnh_m':     metrics(bnh_ret,   bnh_eq),
                'drawdown':  ((strat_eq - strat_eq.cummax()) /
                               strat_eq.cummax() * 100),
            }

        # ── Asset selector ────────────────────────────────────
        bt_asset = st.selectbox(
            "Select asset to backtest",
            ["NIFTY 50", "S&P 500", "Gold", "Silver", "Crude WTI"],
            key="bt_asset"
        )

        asset_map = {
            "NIFTY 50":  (nifty,  'NIFTY_score'),
            "S&P 500":   (sp500,  'SP500_score'),
            "Gold":      (gold,   'Gold_score'),
            "Silver":    (silver, 'Silver_score'),
            "Crude WTI": (crude,  'Crude_score'),
        }
        color_map_bt = {
            "NIFTY 50": BLUE, "S&P 500": MID_BLUE,
            "Gold": ORANGE, "Silver": "#7030A0", "Crude WTI": "#1E6B3C",
        }

        price_series, score_col = asset_map[bt_asset]
        color_bt = color_map_bt[bt_asset]

        if score_col in signals.columns:
            bt = run_backtest_dash(price_series, signals[score_col])

            # ── Metrics row ───────────────────────────────────
            st.markdown("#### Performance Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)

            sm = bt['strat_m']
            bm = bt['bnh_m']

            m1.metric(
                "Strategy CAGR",
                f"{sm['CAGR']:+.1f}%",
                f"{sm['CAGR']-bm['CAGR']:+.1f}% vs B&H"
            )
            m2.metric(
                "Sharpe Ratio",
                f"{sm['Sharpe']:.2f}",
                f"{sm['Sharpe']-bm['Sharpe']:+.2f} vs B&H"
            )
            m3.metric(
                "Max Drawdown",
                f"{sm['MaxDD']:.1f}%",
                f"{sm['MaxDD']-bm['MaxDD']:+.1f}% vs B&H",
                delta_color="inverse"
            )
            m4.metric(
                "Sortino Ratio",
                f"{sm['Sortino']:.2f}",
                f"{sm['Sortino']-bm['Sortino']:+.2f} vs B&H"
            )
            m5.metric(
                "Win Rate",
                f"{sm['WinRate']:.1f}%",
                None
            )

            # ── Equity curve ──────────────────────────────────
            st.markdown("#### Equity Curve")

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=bt['strat_eq'].index,
                y=bt['strat_eq'].values,
                name=f'Signal Strategy',
                line=dict(color=color_bt, width=2),
                hovertemplate='Strategy: %{y:.1f}<extra></extra>'
            ))
            fig_eq.add_trace(go.Scatter(
                x=bt['bnh_eq'].index,
                y=bt['bnh_eq'].values,
                name='Buy & Hold',
                line=dict(color='gray', width=1.5, dash='dash'),
                hovertemplate='B&H: %{y:.1f}<extra></extra>'
            ))

            # Shade outperformance
            bnh_r = bt['bnh_eq'].reindex(bt['strat_eq'].index)
            fig_eq.add_trace(go.Scatter(
                x=bt['strat_eq'].index.tolist() +
                   bt['strat_eq'].index.tolist()[::-1],
                y=bt['strat_eq'].values.tolist() +
                   bnh_r.values.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(30,107,60,0.08)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Outperformance',
                showlegend=True
            ))

            fig_eq.add_hline(y=100, line_dash="dot",
                              line_color="gray", line_width=0.8,
                              annotation_text="Starting value")
            fig_eq.update_layout(
                height=420,
                template="plotly_white",
                yaxis_title="Portfolio Value (Base 100)",
                hovermode="x unified",
                legend=dict(orientation="h",
                             yanchor="bottom", y=1.02,
                             xanchor="right", x=1)
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # ── Drawdown chart ────────────────────────────────
            st.markdown("#### Drawdown Analysis")

            bnh_dd = ((bt['bnh_eq'] - bt['bnh_eq'].cummax()) /
                       bt['bnh_eq'].cummax() * 100)

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=bt['drawdown'].index,
                y=bt['drawdown'].values,
                fill='tozeroy',
                fillcolor=f'rgba(30,107,60,0.3)',
                line=dict(color=color_bt, width=1),
                name='Strategy Drawdown'
            ))
            fig_dd.add_trace(go.Scatter(
                x=bnh_dd.index,
                y=bnh_dd.values,
                line=dict(color='gray', width=1, dash='dash'),
                name='B&H Drawdown'
            ))
            fig_dd.update_layout(
                height=300,
                template="plotly_white",
                yaxis_title="Drawdown (%)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            # ── Position chart ────────────────────────────────
            st.markdown("#### Signal Position Over Time")

            fig_pos = go.Figure()
            fig_pos.add_trace(go.Scatter(
                x=bt['position'].index,
                y=bt['position'].values,
                fill='tozeroy',
                fillcolor='rgba(30,107,60,0.2)',
                line=dict(color=color_bt, width=0.8),
                name='Position (1=Long, 0=Cash)'
            ))
            fig_pos.update_layout(
                height=200,
                template="plotly_white",
                yaxis_title="Position",
                yaxis=dict(tickvals=[0, 1],
                            ticktext=['Cash', 'Long']),
                hovermode="x unified"
            )
            st.plotly_chart(fig_pos, use_container_width=True)

            # ── Commentary ────────────────────────────────────
            beats = sm['CAGR'] > bm['CAGR']
            dd_saved = sm['MaxDD'] - bm['MaxDD']
            sharpe_better = sm['Sharpe'] > bm['Sharpe']

            st.markdown(f"""<div class='commentary-box'>
            📌 <strong>Backtest Commentary — {bt_asset}:</strong>
            The signal strategy achieved a CAGR of <strong>{sm['CAGR']:+.1f}%</strong>
            vs <strong>{bm['CAGR']:+.1f}%</strong> for buy-and-hold.
            Sharpe ratio: <strong>{sm['Sharpe']:.2f}</strong>
            vs <strong>{bm['Sharpe']:.2f}</strong> for B&H
            ({'✅ better' if sharpe_better else '⚠️ lower'} risk-adjusted returns).
            Maximum drawdown was <strong>{sm['MaxDD']:.1f}%</strong>
            vs <strong>{bm['MaxDD']:.1f}%</strong> for B&H —
            the strategy saved <strong>{abs(dd_saved):.1f} percentage points</strong>
            of drawdown by moving to cash during bad regimes.
            </div>""", unsafe_allow_html=True)

        else:
            st.warning(f"Signal scores not found for {bt_asset}. "
                        f"Run 12_signal_engine.py first.")

        # ── Combined portfolio summary ────────────────────────
        st.markdown("---")
        st.markdown("#### Combined Portfolio — All 5 Assets Equal Weight")

        try:
            all_strat_rets = []
            all_bnh_rets   = []

            for asset_name, (price_s, score_c) in asset_map.items():
                if score_c in signals.columns:
                    bt_temp = run_backtest_dash(price_s, signals[score_c])
                    all_strat_rets.append(bt_temp['strat_ret'])
                    all_bnh_rets.append(bt_temp['bnh_ret'])

            if all_strat_rets:
                port_ret  = pd.concat(all_strat_rets, axis=1).dropna().mean(axis=1)
                port_eq   = (1 + port_ret).cumprod() * 100
                bnh_port  = pd.concat(all_bnh_rets,   axis=1).dropna().mean(axis=1)
                bnh_port_eq = (1 + bnh_port).cumprod() * 100

                n_years    = len(port_ret) / 252
                port_cagr  = ((port_eq.iloc[-1]/100)**(1/n_years)-1)*100
                port_sharpe= (port_ret.mean()/(port_ret.std()+1e-10))*np.sqrt(252)
                port_dd    = ((port_eq-port_eq.cummax())/port_eq.cummax()*100).min()
                bnh_cagr   = ((bnh_port_eq.iloc[-1]/100)**(1/n_years)-1)*100
                bnh_sharpe = (bnh_port.mean()/(bnh_port.std()+1e-10))*np.sqrt(252)
                bnh_dd     = ((bnh_port_eq-bnh_port_eq.cummax())/
                               bnh_port_eq.cummax()*100).min()

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Portfolio CAGR",
                           f"{port_cagr:+.1f}%",
                           f"{port_cagr-bnh_cagr:+.1f}% vs B&H")
                p2.metric("Portfolio Sharpe",
                           f"{port_sharpe:.2f}",
                           f"{port_sharpe-bnh_sharpe:+.2f} vs B&H")
                p3.metric("Portfolio MaxDD",
                           f"{port_dd:.1f}%",
                           f"{port_dd-bnh_dd:+.1f}% vs B&H",
                           delta_color="inverse")
                p4.metric("B&H Sharpe",
                           f"{bnh_sharpe:.2f}",
                           None)

                fig_port = go.Figure()
                fig_port.add_trace(go.Scatter(
                    x=port_eq.index, y=port_eq.values,
                    name=f'Signal Portfolio (Sharpe: {port_sharpe:.2f})',
                    line=dict(color=BLUE, width=2)
                ))
                fig_port.add_trace(go.Scatter(
                    x=bnh_port_eq.index, y=bnh_port_eq.values,
                    name=f'B&H Portfolio (Sharpe: {bnh_sharpe:.2f})',
                    line=dict(color='gray', width=1.5, dash='dash')
                ))
                fig_port.add_hline(y=100, line_dash="dot",
                                    line_color="gray", line_width=0.8)
                fig_port.update_layout(
                    height=380,
                    template="plotly_white",
                    yaxis_title="Portfolio Value (Base 100)",
                    hovermode="x unified",
                    title=f"Combined Portfolio: Sharpe {port_sharpe:.2f} "
                           f"vs B&H {bnh_sharpe:.2f}"
                )
                st.plotly_chart(fig_port, use_container_width=True)

                st.markdown(f"""<div class='commentary-box'>
                📌 <strong>Portfolio Summary:</strong>
                The combined equal-weight signal portfolio achieved a Sharpe ratio of
                <strong>{port_sharpe:.2f}</strong> vs <strong>{bnh_sharpe:.2f}</strong>
                for buy-and-hold — a <strong>{port_sharpe/bnh_sharpe:.1f}x</strong>
                improvement in risk-adjusted returns.
                CAGR: <strong>{port_cagr:+.1f}%</strong> vs
                <strong>{bnh_cagr:+.1f}%</strong> B&H.
                Maximum drawdown: <strong>{port_dd:.1f}%</strong> vs
                <strong>{bnh_dd:.1f}%</strong> B&H.
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Portfolio calculation error: {e}")
            
st.markdown("---")

# ════════════════════════════════════════════════════════════════════
# LAYER 3 — DEEP DIVE (expanders)
# ════════════════════════════════════════════════════════════════════
st.markdown("### 🔬 Layer 3 — Deep Dive Analytics")

with st.expander("📊 Annual Returns by Asset"):
    returns_all = pd.DataFrame({
        'NIFTY':  nifty_f.pct_change(),
        'SP500':  sp500_f.pct_change(),
        'Gold':   gold_f.pct_change(),
        'Crude':  crude_f.pct_change(),
    }).dropna()

    annual = (1 + returns_all).resample('YE').prod() - 1
    annual.index = annual.index.year
    annual_pct = annual * 100

    fig_ann = px.bar(annual_pct, barmode='group',
                      color_discrete_map={'NIFTY': BLUE, 'SP500': MID_BLUE,
                                          'Gold': ORANGE, 'Crude': "#7030A0"},
                      title="Annual Returns (%) by Asset Class")
    fig_ann.add_hline(y=0, line_color="black", line_width=0.8)
    fig_ann.update_layout(height=420, template="plotly_white",
                           yaxis_title="Annual Return (%)")
    st.plotly_chart(fig_ann, use_container_width=True)

with st.expander("📋 Raw Price Data Table"):
    raw_df = pd.DataFrame({
        'NIFTY':   nifty_f,
        'SP500':   sp500_f,
        'Gold':    gold_f,
        'Silver':  silver_f,
        'Crude':   crude_f,
        'VIX':     vix_f,
        'USD_INR': usd_inr_f,
    }).tail(30)
    st.dataframe(raw_df.style.format("{:.2f}"), use_container_width=True)

with st.expander("📈 Rolling Statistics"):
    stat_asset = st.selectbox("Select asset", ["NIFTY","SP500","Gold","Crude"])
    stat_map   = {"NIFTY": nifty_f, "SP500": sp500_f, "Gold": gold_f, "Crude": crude_f}
    s = stat_map[stat_asset].pct_change().dropna() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Daily Return", f"{s.mean():.4f}%")
    c2.metric("Volatility (Ann.)", f"{s.std()*np.sqrt(252):.1f}%")
    c3.metric("Best Day",          f"{s.max():+.2f}%")
    c4.metric("Worst Day",         f"{s.min():+.2f}%")

    sharpe = (s.mean() / s.std()) * np.sqrt(252) if s.std() > 0 else 0
    st.metric("Sharpe Ratio (approx)", f"{sharpe:.3f}")

# Footer
st.markdown(f"""
<div style='text-align:center; padding:20px; color:{GRAY}; font-size:12px;'>
    Global Macro Intelligence System &nbsp;|&nbsp;
    Built by Niraj Mutha &nbsp;|&nbsp;
    Data: Yahoo Finance, FRED &nbsp;|&nbsp;
    Last updated: {datetime.now().strftime('%d %b %Y %H:%M')}
</div>
""", unsafe_allow_html=True)
