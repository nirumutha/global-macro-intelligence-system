# ============================================================
# GLOBAL MACRO INTELLIGENCE SYSTEM — DASHBOARD v2
# GMIS 1.0 + GMIS 2.0 integrated
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

# ── Colour system ─────────────────────────────────────────────
GREEN    = "#1E6B3C"
RED      = "#C00000"
BLUE     = "#1F3864"
MID_BLUE = "#2E75B6"
ORANGE   = "#C55A11"
GRAY     = "#CCCCCC"
LIGHT    = "#F4F4F4"

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    .metric-card {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 5px solid #2E75B6;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 8px;
        color: #FFFFFF;
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
        background: rgba(92,45,145,0.3);
        border-left: 4px solid #5C2D91;
        padding: 10px 16px;
        border-radius: 4px;
        font-size: 13px;
        color: #FFFFFF;
        margin: 8px 0;
    }
    .delta-positive { color: #1E6B3C; font-weight: bold; }
    .delta-negative { color: #C00000; font-weight: bold; }
    h1, h2, h3 { color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

# ── Database path ─────────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_PATH, 'data', 'macro_system.db')

# ═════════════════════════════════════════════════════════════
# DATA LOADERS
# ═════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_close(table):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    close_col = [c for c in df.columns
                 if 'Close' in c or 'close' in c]
    return df[close_col[0]].dropna() \
           if close_col else df.iloc[:, 0].dropna()

@st.cache_data(ttl=300)
def load_macro(table):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    return df.iloc[:, 0].dropna()

@st.cache_data(ttl=300)
def load_sentiment():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM SENTIMENT_DAILY", conn)
        conn.close()
        return df
    except:
        conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_signals_v3():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM SIGNALS_V3", conn)
        conn.close()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        return df
    except:
        conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_analog_outcomes():
    conn = sqlite3.connect(DB_PATH)
    try:
        analogs  = pd.read_sql("SELECT * FROM ANALOG_DATES", conn)
        outcomes = pd.read_sql("SELECT * FROM ANALOG_OUTCOMES", conn)
        conn.close()
        return analogs, outcomes
    except:
        conn.close()
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=300)
def load_walkforward():
    conn = sqlite3.connect(DB_PATH)
    try:
        results   = pd.read_sql(
            "SELECT * FROM WALKFORWARD_RESULTS", conn)
        portfolio = pd.read_sql(
            "SELECT * FROM WALKFORWARD_PORTFOLIO", conn)
        conn.close()
        return results, portfolio
    except:
        conn.close()
        return pd.DataFrame(), pd.DataFrame()

# ── Load all data ─────────────────────────────────────────────
nifty        = load_close('NIFTY50')
sp500        = load_close('SP500')
gold         = load_close('GOLD')
silver       = load_close('SILVER')
crude        = load_close('CRUDE_WTI')
vix_us       = load_close('VIX_US')
vix_india    = load_close('VIX_INDIA')
usd_inr      = load_close('USD_INR')
dxy          = load_close('DXY')
yield_10y    = load_macro('US_10Y_YIELD')
yield_2y     = load_macro('US_2Y_YIELD')
sentiment_df = load_sentiment()
signals_v3_df            = load_signals_v3()
analog_dates, analog_outcomes = load_analog_outcomes()
wf_results, wf_portfolio = load_walkforward()

# ── Generic safe loader for new module tables ─────────────────
@st.cache_data(ttl=300)
def safe_load(table: str, order_col: str = 'date', limit: int = 0) -> pd.DataFrame:
    """Load any DB table gracefully — returns empty DataFrame on error."""
    try:
        conn = sqlite3.connect(DB_PATH)
        q = f'SELECT * FROM "{table}"'
        if order_col:
            q += f' ORDER BY {order_col} DESC'
        if limit > 0:
            q += f' LIMIT {limit}'
        df = pd.read_sql(q, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def sfloat(val, default: float = 0.0) -> float:
    """Safe float — handles None, NaN, empty strings."""
    try:
        v = float(val)
        return default if (v != v) else v   # NaN check
    except Exception:
        return default

# Pre-load tables needed for Layer 1 and banner
decisions_df  = safe_load('DECISIONS',        limit=20)
entry_exit_df = safe_load('ENTRY_EXIT',        limit=10)
calendar_df   = safe_load('ECONOMIC_CALENDAR', limit=60)

# ═════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════

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
    if len(price_series) < 60:
        return "Neutral", ORANGE
    ma20 = price_series.rolling(20).mean().iloc[-1]
    ma60 = price_series.rolling(60).mean().iloc[-1]
    curr = price_series.iloc[-1]
    vix  = vix_series.iloc[-1]
    if curr > ma20 > ma60 and vix < 25:
        return "Long", GREEN
    elif curr < ma20 < ma60 or vix > 30:
        return "Short", RED
    else:
        return "Neutral", ORANGE

def get_v3_signal(asset):
    if signals_v3_df.empty:
        return 'Neutral', 0.0, 'N/A', 'Neutral'
    try:
        latest     = signals_v3_df.iloc[-1]
        signal     = latest.get(f'{asset}_signal',     'Neutral')
        score      = latest.get(f'{asset}_score',      0.0)
        confidence = latest.get(f'{asset}_confidence', 'NONE')
        stable     = latest.get(f'{asset}_stable',     signal)
        return signal, float(score), confidence, stable
    except:
        return 'Neutral', 0.0, 'N/A', 'Neutral'

# ═════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/48/combo-chart.png",
        width=48
    )
    st.title("GMIS Controls")
    if st.sidebar.button("🔄 Clear Cache & Refresh"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")

    st.subheader("📅 Date Range")
    min_date   = nifty.index.min().date()
    max_date   = nifty.index.max().date()
    start_date = st.date_input(
        "From", value=datetime(2015, 1, 1).date(),
        min_value=min_date, max_value=max_date)
    end_date   = st.date_input(
        "To", value=max_date,
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
    vix_crisis  = st.slider("VIX Crisis Level",  20, 50, 30)
    vix_caution = st.slider("VIX Caution Level", 10, 30, 20)

    st.markdown("---")
    st.caption(f"Data updated: {max_date}")
    st.caption("Global Macro Intelligence System")
    st.caption("Built by Niraj Mutha — 2026")

# ── Filter data by date ───────────────────────────────────────
def filter_series(s):
    return s[
        (s.index.date >= start_date) &
        (s.index.date <= end_date)
    ]

nifty_f   = filter_series(nifty)
sp500_f   = filter_series(sp500)
gold_f    = filter_series(gold)
silver_f  = filter_series(silver)
crude_f   = filter_series(crude)
vix_f     = filter_series(vix_us)
usd_inr_f = filter_series(usd_inr)
dxy_f     = filter_series(dxy)

# ── Current values ────────────────────────────────────────────
def safe_last(s, n=1):
    return s.iloc[-n] if len(s) >= n else np.nan

curr_nifty = safe_last(nifty_f)
prev_nifty = safe_last(nifty_f, 2)
curr_sp500 = safe_last(sp500_f)
prev_sp500 = safe_last(sp500_f, 2)
curr_gold  = safe_last(gold_f)
prev_gold  = safe_last(gold_f, 2)
curr_vix   = safe_last(vix_f)
curr_inr   = safe_last(usd_inr_f)
prev_inr   = safe_last(usd_inr_f, 2)

# Regime
sp_ma200    = sp500_f.rolling(200).mean()
sp_above_ma = curr_sp500 > sp_ma200.iloc[-1] \
              if len(sp_ma200) > 0 else False
regime_label, regime_color = get_regime(curr_vix, sp_above_ma)

# Yield curve
yields_aligned = pd.DataFrame(
    {'10Y': yield_10y, '2Y': yield_2y}
).dropna()
curr_spread = yields_aligned['10Y'].iloc[-1] - \
              yields_aligned['2Y'].iloc[-1] \
              if len(yields_aligned) > 0 else 0
curve_shape = "Inverted ⚠️" if curr_spread < 0 else "Normal ✓"

# Legacy signals (fallback)
sig_nifty, col_nifty = generate_signal("NIFTY", nifty_f, vix_f)
sig_sp500, col_sp500 = generate_signal("SP500", sp500_f, vix_f)
sig_gold,  col_gold  = generate_signal("Gold",  gold_f,  vix_f)
sig_crude, col_crude = generate_signal("Crude", crude_f, vix_f)

# Sentiment
overall_sentiment = "N/A"
sentiment_score   = 0.0
if not sentiment_df.empty:
    sentiment_score   = sentiment_df['score'].mean()
    overall_sentiment = "Positive" if sentiment_score > 0.05 \
                   else "Negative" if sentiment_score < -0.05 \
                   else "Neutral"

# V3 signals
nifty_sig, nifty_score, nifty_conf, nifty_stable = \
    get_v3_signal('NIFTY')
sp500_sig, sp500_score, sp500_conf, sp500_stable = \
    get_v3_signal('SP500')
gold_sig,  gold_score,  gold_conf,  gold_stable  = \
    get_v3_signal('Gold')
crude_sig, crude_score, crude_conf, crude_stable = \
    get_v3_signal('Crude')

# ═════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════

st.markdown(f"""
<div style='background: linear-gradient(135deg, #1F3864, #2E75B6);
     padding: 20px 28px; border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: white; margin: 0; font-size: 26px;'>
        📊 Global Macro Intelligence System
    </h1>
    <p style='color: #D6E4F0; margin: 4px 0 0 0; font-size: 14px;'>
        Multi-Asset Signal Framework &nbsp;|&nbsp;
        {start_date.strftime('%d %b %Y')} →
        {end_date.strftime('%d %b %Y')} &nbsp;|&nbsp;
        Built by Niraj Mutha
    </p>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# LAYER 1 — MARKET INTELLIGENCE SNAPSHOT
# ═════════════════════════════════════════════════════════════

# ── Economic Calendar — High-Impact Warning Banner ────────────
try:
    if not calendar_df.empty and 'date' in calendar_df.columns:
        _cal = calendar_df.copy()
        _cal['date'] = pd.to_datetime(_cal['date'], errors='coerce')
        _today = pd.Timestamp.now().normalize()
        _h48   = _today + pd.Timedelta(days=2)
        _h7d   = _today + pd.Timedelta(days=7)
        _impact_col = [c for c in _cal.columns
                       if c.lower() == 'impact']
        if _impact_col:
            _high = _cal[
                _cal[_impact_col[0]].str.upper().str.strip() == 'HIGH'
            ]
            _soon = _high[
                (_high['date'] >= _today) & (_high['date'] <= _h48)
            ]
            _week = _high[
                (_high['date'] > _h48) & (_high['date'] <= _h7d)
            ]
            if len(_soon) > 0 or len(_week) > 0:
                _parts = []
                for _, _ev in _soon.iterrows():
                    _t = str(_ev.get('time', '')).strip()
                    _t = '' if _t in ('', 'nan', 'None') else f' {_t}'
                    _parts.append(
                        f"🔴 <strong>{_ev.get('event','?')}</strong>"
                        f" ({_ev.get('country','?')}){_t}"
                    )
                for _, _ev in _week.iterrows():
                    _diff = max(1, (_ev['date'] - _today).days)
                    _parts.append(
                        f"🟡 <strong>{_ev.get('event','?')}</strong>"
                        f" in {_diff}d ({_ev.get('country','?')})"
                    )
                _border = RED if len(_soon) > 0 else ORANGE
                _bg     = 'rgba(192,0,0,0.25)' if len(_soon) > 0 else 'rgba(197,90,17,0.2)'
                st.markdown(
                    f"<div style='background:{_bg};border-left:4px solid "
                    f"{_border};padding:8px 16px;border-radius:4px;"
                    f"margin-bottom:12px;font-size:13px;color:#FFFFFF;'>"
                    f"⚠️ <strong>High-Impact Events:</strong> &nbsp;"
                    f"{'&nbsp;&nbsp;|&nbsp;&nbsp;'.join(_parts)}</div>",
                    unsafe_allow_html=True
                )
except Exception:
    pass

st.markdown("### 🎯 Layer 1 — Market Intelligence Snapshot")
st.caption("The 60-second view — understand the full market state at a glance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    color_cls = "metric-positive" \
        if regime_label == "Bull Market" else \
        "metric-negative" \
        if regime_label in ["Bear Market", "Crisis"] else \
        "metric-neutral"
    st.markdown(f"""
    <div class='metric-card {color_cls}'>
        <div style='font-size:11px; color:{GRAY};
             text-transform:uppercase;'>Market Regime</div>
        <div style='font-size:20px; font-weight:bold;
             color:{regime_color};'>{regime_label}</div>
        <div style='font-size:11px; color:{GRAY};'>
             VIX: {curr_vix:.1f}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    d_nifty = pct_change_label(curr_nifty, prev_nifty)
    st.markdown(f"""
    <div class='metric-card {"metric-positive" if d_nifty >= 0 else "metric-negative"}'>
        <div style='font-size:11px; color:{GRAY};
             text-transform:uppercase;'>NIFTY 50</div>
        <div style='font-size:20px; font-weight:bold;
             color:#FFFFFF;'>{curr_nifty:,.0f}</div>
        <div>{delta_arrow(d_nifty)}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    d_sp = pct_change_label(curr_sp500, prev_sp500)
    st.markdown(f"""
    <div class='metric-card {"metric-positive" if d_sp >= 0 else "metric-negative"}'>
        <div style='font-size:11px; color:{GRAY};
             text-transform:uppercase;'>S&P 500</div>
        <div style='font-size:20px; font-weight:bold;
             color:#FFFFFF;'>{curr_sp500:,.0f}</div>
        <div>{delta_arrow(d_sp)}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    d_gold = pct_change_label(curr_gold, prev_gold)
    st.markdown(f"""
    <div class='metric-card {"metric-positive" if d_gold >= 0 else "metric-negative"}'>
        <div style='font-size:11px; color:{GRAY};
             text-transform:uppercase;'>Gold (USD)</div>
        <div style='font-size:20px; font-weight:bold;
             color:#FFFFFF;'>${curr_gold:,.0f}</div>
        <div>{delta_arrow(d_gold)}</div>
    </div>""", unsafe_allow_html=True)

with col5:
    sent_color = "metric-positive" \
        if overall_sentiment == "Positive" else \
        "metric-negative" \
        if overall_sentiment == "Negative" else \
        "metric-neutral"
    st.markdown(f"""
    <div class='metric-card {sent_color}'>
        <div style='font-size:11px; color:{GRAY};
             text-transform:uppercase;'>Sentiment</div>
        <div style='font-size:20px; font-weight:bold;
             color:#FFFFFF;'>{overall_sentiment}</div>
        <div style='font-size:11px; color:{GRAY};'>
             Score: {sentiment_score:+.3f}</div>
    </div>""", unsafe_allow_html=True)

# ── Decision Engine signal badges ────────────────────────────
st.markdown("#### 🎯 Decision Engine — Active Signals")

_DE_ASSETS = [
    ('NIFTY',  'NIFTY 50'),
    ('SP500',  'S&P 500'),
    ('Gold',   'Gold'),
    ('Silver', 'Silver'),
    ('Crude',  'Crude WTI'),
]

def _get_dec(asset_key):
    if decisions_df.empty:
        return None
    rows = decisions_df[
        decisions_df['asset'].str.upper() == asset_key.upper()
    ]
    return rows.iloc[0] if len(rows) > 0 else None

def _get_ee(asset_key):
    if entry_exit_df.empty:
        return None
    rows = entry_exit_df[
        entry_exit_df['asset'].str.upper() == asset_key.upper()
    ]
    return rows.iloc[0] if len(rows) > 0 else None

def decision_badge_html(label, asset_key):
    d  = _get_dec(asset_key)
    ee = _get_ee(asset_key)
    if d is not None:
        bias     = str(d.get('bias', 'NEUTRAL')).upper()
        combined = sfloat(d.get('combined', 0))
        conf     = str(d.get('confidence', 'LOW'))
        agr      = sfloat(d.get('agreement', 0))
    else:
        sig, score, conf_v3, _ = get_v3_signal(asset_key)
        bias = sig.upper(); combined = score
        conf = conf_v3;     agr      = 0.0
    if 'LONG' in bias:
        badge_cls, txt_col = 'badge-long',    GREEN
    elif 'SHORT' in bias:
        badge_cls, txt_col = 'badge-short',   RED
    else:
        badge_cls, txt_col = 'badge-neutral', ORANGE
    conf_col = GREEN if conf == 'HIGH' else \
               ORANGE if conf == 'MEDIUM' else GRAY
    entry_stop = ''
    if ee is not None:
        try:
            el = sfloat(ee.get('entry_low'))
            eh = sfloat(ee.get('entry_high'))
            sl = sfloat(ee.get('stop_level'))
            if el > 0 and eh > 0:
                entry_stop += (
                    f"<div style='font-size:10px;color:{GRAY};"
                    f"margin-top:3px;'>Entry "
                    f"{el:,.0f}–{eh:,.0f}</div>"
                )
            if sl > 0:
                entry_stop += (
                    f"<div style='font-size:10px;color:{RED};'>"
                    f"Stop {sl:,.0f}</div>"
                )
        except Exception:
            pass
    return (
        f"<div style='text-align:center;padding:10px;background:rgba(255,255,255,0.08);"
        f"border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);'>"
        f"<div style='font-size:11px;color:{GRAY};margin-bottom:4px;'>"
        f"{label}</div>"
        f"<span class='signal-badge {badge_cls}'>{bias}</span>"
        f"<div style='font-size:11px;font-weight:bold;color:{conf_col};"
        f"margin-top:4px;'>{conf} | {combined:+.2f}</div>"
        f"<div style='font-size:10px;color:{GRAY};'>Agr: {agr:.0f}%</div>"
        f"{entry_stop}</div>"
    )

dc1, dc2, dc3, dc4, dc5 = st.columns(5)
for _col, (_akey, _alabel) in zip(
        [dc1, dc2, dc3, dc4, dc5], _DE_ASSETS):
    with _col:
        st.markdown(decision_badge_html(_alabel, _akey),
                    unsafe_allow_html=True)

# vix_pct — also used in Tab 2 volatility commentary
vix_pct = (vix_f.rank(pct=True).iloc[-1] * 100) \
           if len(vix_f) > 0 else 50

_nd = _get_dec('NIFTY')
_sd = _get_dec('SP500')
_gd = _get_dec('Gold')
_nbias = str(_nd.get('bias', nifty_sig)) if _nd is not None else nifty_sig
_sbias = str(_sd.get('bias', sp500_sig)) if _sd is not None else sp500_sig
_gbias = str(_gd.get('bias', gold_sig))  if _gd is not None else gold_sig
_nconf = str(_nd.get('confidence', nifty_conf)) if _nd is not None else nifty_conf

commentary = (
    f"📌 <strong>System Commentary "
    f"({end_date.strftime('%d %b %Y')}):</strong> "
    f"Market is in <strong>{regime_label}</strong> regime. "
    f"VIX: {curr_vix:.1f} ({vix_pct:.0f}th pct). "
    f"Yield curve: <strong>{curve_shape}</strong> "
    f"({curr_spread:+.2f}%). "
    f"Sentiment: <strong>{overall_sentiment}</strong> "
    f"({sentiment_score:+.3f}). "
    f"Decision Engine — NIFTY: <strong>{_nbias}</strong> "
    f"(conf: {_nconf}) | "
    f"S&P 500: <strong>{_sbias}</strong> | "
    f"Gold: <strong>{_gbias}</strong>."
)
st.markdown(
    f"<div class='commentary-box'>{commentary}</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════
# LAYER 2 — INTERACTIVE ANALYSIS
# ═════════════════════════════════════════════════════════════

st.markdown("### 📈 Layer 2 — Interactive Analysis")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, \
tab8, tab9, tab10, tab11 = st.tabs([
    "🌍 Macro", "📊 Volatility", "💱 FX & Bonds",
    "😐 Sentiment", "📉 Backtest",
    "🔮 Analogs", "📋 Walk-Forward",
    "📊 Decision Engine", "🇮🇳 India Intelligence",
    "⚠️ Risk Monitor", "🌊 Intelligence",
])

# ── TAB 1: MACRO ──────────────────────────────────────────────
with tab1:
    st.subheader("Cross-Asset Price Performance")
    market_map = {
        "NIFTY 50":  nifty_f,
        "S&P 500":   sp500_f,
        "Gold":      gold_f,
        "Silver":    silver_f,
        "Crude WTI": crude_f,
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
                    name=market,
                    line=dict(color=color_map[market], width=1.8),
                    hovertemplate=f"{market}: %{{y:.1f}}<extra></extra>"
                ))
    fig.add_hline(y=100, line_dash="dash",
                   line_color="gray", line_width=0.8,
                   annotation_text="Start (100)")
    fig.update_layout(
        title="Indexed Performance (Base = 100)",
        yaxis_title="Indexed Price (Base 100)",
        height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rolling Correlation Matrix")
    corr_window = st.slider(
        "Correlation window (days)", 30, 252, 60,
        key="corr_slider"
    )
    all_returns = pd.DataFrame({
        'NIFTY':  nifty_f.pct_change(),
        'SP500':  sp500_f.pct_change(),
        'Gold':   gold_f.pct_change(),
        'Silver': silver_f.pct_change(),
        'Crude':  crude_f.pct_change(),
    }).dropna()
    corr_matrix = all_returns.tail(corr_window).corr()
    fig_corr = px.imshow(
        corr_matrix, color_continuous_scale='RdYlGn',
        zmin=-1, zmax=1, text_auto='.2f',
        title=f"Correlation Matrix — Last {corr_window} Days"
    )
    fig_corr.update_layout(height=380, template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)

# ── TAB 2: VOLATILITY ─────────────────────────────────────────
with tab2:
    st.subheader("VIX & Volatility Regime")
    fig_vix = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("VIX — Fear Index", "VIX vs India VIX"),
        vertical_spacing=0.12
    )
    fig_vix.add_trace(go.Scatter(
        x=vix_f.index, y=vix_f.values,
        name='VIX US', line=dict(color=RED, width=1.5)
    ), row=1, col=1)
    fig_vix.add_hline(
        y=vix_caution, line_dash="dash",
        line_color="orange",
        annotation_text=f"Caution ({vix_caution})",
        row=1, col=1
    )
    fig_vix.add_hline(
        y=vix_crisis, line_dash="dash",
        line_color=RED,
        annotation_text=f"Crisis ({vix_crisis})",
        row=1, col=1
    )
    vix_mean = vix_f.mean()
    fig_vix.add_hline(
        y=vix_mean, line_dash="dot", line_color=GRAY,
        annotation_text=f"Mean ({vix_mean:.1f})",
        row=1, col=1
    )
    vix_india_f = filter_series(vix_india)
    fig_vix.add_trace(go.Scatter(
        x=vix_india_f.index, y=vix_india_f.values,
        name='India VIX', line=dict(color=BLUE, width=1.5)
    ), row=2, col=1)
    fig_vix.add_trace(go.Scatter(
        x=vix_f.index, y=vix_f.values,
        name='US VIX', line=dict(color=RED, width=1.2, dash='dot')
    ), row=2, col=1)
    fig_vix.update_layout(
        height=520, template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_vix, use_container_width=True)

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Current VIX",       f"{curr_vix:.1f}")
    v2.metric("VIX Mean",          f"{vix_f.mean():.1f}")
    v3.metric("VIX Max (period)",  f"{vix_f.max():.1f}")
    v4.metric("Days above crisis", f"{(vix_f > vix_crisis).sum()}")

    nifty_ret_aligned = nifty_f.pct_change()
    vix_aligned       = vix_f.reindex(
        nifty_ret_aligned.index).ffill()
    hist_ret_high_vix = nifty_ret_aligned[
        vix_aligned > vix_crisis].mean() * 100
    hist_ret_low_vix  = nifty_ret_aligned[
        vix_aligned < vix_caution].mean() * 100
    st.markdown(f"""<div class='commentary-box'>
    📌 <strong>Volatility Commentary:</strong>
    When VIX exceeds {vix_crisis} (crisis), NIFTY averages
    <strong>{hist_ret_high_vix:.3f}%</strong> next day.
    When VIX is below {vix_caution} (calm), NIFTY averages
    <strong>{hist_ret_low_vix:.3f}%</strong> next day.
    Current VIX is at the
    <strong>{vix_pct:.0f}th percentile</strong> of history.
    </div>""", unsafe_allow_html=True)

# ── TAB 3: FX & BONDS ─────────────────────────────────────────
with tab3:
    st.subheader("Yield Curve & Currency Analysis")
    yields_f = pd.DataFrame({
        '10Y': filter_series(yield_10y),
        '2Y':  filter_series(yield_2y),
    }).dropna()
    yields_f['Spread'] = yields_f['10Y'] - yields_f['2Y']

    fig_yc = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Treasury Yields",
                        "Yield Spread (10Y − 2Y)"),
        vertical_spacing=0.12
    )
    fig_yc.add_trace(go.Scatter(
        x=yields_f.index, y=yields_f['10Y'],
        name='10Y', line=dict(color=BLUE, width=1.5)
    ), row=1, col=1)
    fig_yc.add_trace(go.Scatter(
        x=yields_f.index, y=yields_f['2Y'],
        name='2Y', line=dict(color=ORANGE, width=1.5)
    ), row=1, col=1)
    fig_yc.add_trace(go.Scatter(
        x=yields_f.index, y=yields_f['Spread'],
        name='Spread', line=dict(color=MID_BLUE, width=1.5),
        fill='tozeroy', fillcolor='rgba(30,107,60,0.15)'
    ), row=2, col=1)
    fig_yc.add_hline(
        y=0, line_dash="dash", line_color=RED,
        annotation_text="Inversion level", row=2, col=1
    )
    fig_yc.update_layout(
        height=480, template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_yc, use_container_width=True)

    st.subheader("USD/INR Exchange Rate")
    fig_inr = go.Figure()
    fig_inr.add_trace(go.Scatter(
        x=usd_inr_f.index, y=usd_inr_f.values,
        name='USD/INR', line=dict(color=RED, width=1.5),
        fill='tozeroy', fillcolor='rgba(192,0,0,0.05)'
    ))
    inr_mean = usd_inr_f.mean()
    fig_inr.add_hline(
        y=inr_mean, line_dash="dot", line_color=GRAY,
        annotation_text=f"Mean ({inr_mean:.1f})"
    )
    fig_inr.update_layout(
        height=320, template="plotly_white",
        yaxis_title="USD/INR Rate",
        title="Rising = Rupee Weakening"
    )
    st.plotly_chart(fig_inr, use_container_width=True)

    if len(usd_inr_f) > 1:
        depr = pct_change_label(
            usd_inr_f.iloc[-1], usd_inr_f.iloc[0])
        st.markdown(f"""<div class='commentary-box'>
        📌 <strong>FX Commentary:</strong>
        USD/INR moved from
        <strong>{usd_inr_f.iloc[0]:.1f}</strong> to
        <strong>{usd_inr_f.iloc[-1]:.1f}</strong>
        ({depr:+.1f}% Rupee depreciation).
        Yield curve is currently <strong>{curve_shape}</strong>
        with a {curr_spread:+.2f}% spread.
        </div>""", unsafe_allow_html=True)

# ── TAB 4: SENTIMENT ──────────────────────────────────────────
with tab4:
    st.subheader("Live News Sentiment Analysis (FinBERT)")
    if sentiment_df.empty:
        st.warning(
            "No sentiment data. Run 15_finbert_sentiment.py first."
        )
    else:
        s1, s2, s3 = st.columns(3)
        s1.metric("Overall Score",
                   f"{sentiment_df['score'].mean():+.4f}")
        s2.metric("Total Headlines", f"{len(sentiment_df)}")
        s3.metric("Positive %",
                   f"{(sentiment_df['sentiment']=='Positive').mean()*100:.1f}%")

        sent_counts = sentiment_df['sentiment'].value_counts()
        fig_pie = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            color=sent_counts.index,
            color_discrete_map={
                'Positive': GREEN,
                'Neutral':  MID_BLUE,
                'Negative': RED
            },
            title="Sentiment Distribution"
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

        if 'markets' in sentiment_df.columns:
            market_sent = {}
            for mkt in ['NIFTY','SP500','Gold','Crude','General']:
                mask = sentiment_df['markets'].str.contains(
                    mkt, na=False)
                if mask.sum() > 0:
                    market_sent[mkt] = \
                        sentiment_df[mask]['score'].mean()
            if market_sent:
                fig_mkt = go.Figure(go.Bar(
                    x=list(market_sent.keys()),
                    y=list(market_sent.values()),
                    marker_color=[
                        GREEN  if v >= 0.05 else
                        RED    if v <= -0.05 else
                        ORANGE for v in market_sent.values()
                    ],
                ))
                fig_mkt.add_hline(
                    y=0.05, line_dash="dot", line_color=GREEN,
                    annotation_text="Positive threshold"
                )
                fig_mkt.add_hline(
                    y=-0.05, line_dash="dot", line_color=RED,
                    annotation_text="Negative threshold"
                )
                fig_mkt.update_layout(
                    title="Sentiment Score by Market",
                    yaxis_title="Avg Score",
                    template="plotly_white", height=350
                )
                st.plotly_chart(fig_mkt, use_container_width=True)

        with st.expander("📰 View All Headlines"):
            st.dataframe(
                sentiment_df[['headline','score',
                               'sentiment','markets','source']]
                .sort_values('score', ascending=False),
                use_container_width=True
            )

# ── TAB 5: BACKTEST ───────────────────────────────────────────
with tab5:
    st.subheader("Signal Strategy vs Buy-and-Hold Backtest")
    st.caption(
        "16 years of real data | "
        "Transaction cost: 0.1% per trade | No leverage"
    )

    @st.cache_data(ttl=300)
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
        st.warning(
            "No signal data. Run 12_signal_engine.py first."
        )
    else:
        def run_backtest_dash(price, signal_scores,
                               transaction_cost=0.001,
                               threshold=0.15):
            sig       = signal_scores.reindex(
                price.index).ffill().bfill()
            daily_ret = price.pct_change().fillna(0)
            position  = pd.Series(0.0, index=price.index)
            position[sig >= threshold]  = 1.0
            position[sig <= -threshold] = 0.0
            pos_change = position.diff().abs().fillna(0)
            strat_ret  = position.shift(1) * daily_ret
            strat_ret -= pos_change * transaction_cost
            bnh_ret    = daily_ret.copy()
            strat_eq   = (1 + strat_ret).cumprod() * 100
            bnh_eq     = (1 + bnh_ret).cumprod()   * 100

            def metrics(ret, eq):
                n_years = len(ret) / 252
                total   = eq.iloc[-1] / 100
                cagr    = (total**(1/n_years)-1)*100 \
                           if n_years > 0 and total > 0 else 0
                sharpe  = (ret.mean() /
                           (ret.std()+1e-10)) * np.sqrt(252)
                roll_max = eq.cummax()
                maxdd    = ((eq-roll_max)/roll_max).min()*100
                sortino  = (ret.mean() /
                            (ret[ret<0].std()+1e-10)) * np.sqrt(252)
                winrate  = (ret > 0).sum() / \
                            (ret != 0).sum() * 100
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

        bt_asset = st.selectbox(
            "Select asset to backtest",
            ["NIFTY 50","S&P 500","Gold","Silver","Crude WTI"],
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
            "Gold": ORANGE, "Silver": "#7030A0",
            "Crude WTI": "#1E6B3C",
        }

        price_series, score_col = asset_map[bt_asset]
        color_bt = color_map_bt[bt_asset]

        if score_col in signals.columns:
            bt = run_backtest_dash(price_series,
                                    signals[score_col])
            st.markdown("#### Performance Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            sm = bt['strat_m']
            bm = bt['bnh_m']
            m1.metric("Strategy CAGR",
                       f"{sm['CAGR']:+.1f}%",
                       f"{sm['CAGR']-bm['CAGR']:+.1f}% vs B&H")
            m2.metric("Sharpe Ratio",
                       f"{sm['Sharpe']:.2f}",
                       f"{sm['Sharpe']-bm['Sharpe']:+.2f} vs B&H")
            m3.metric("Max Drawdown",
                       f"{sm['MaxDD']:.1f}%",
                       f"{sm['MaxDD']-bm['MaxDD']:+.1f}% vs B&H",
                       delta_color="inverse")
            m4.metric("Sortino Ratio",
                       f"{sm['Sortino']:.2f}",
                       f"{sm['Sortino']-bm['Sortino']:+.2f} vs B&H")
            m5.metric("Win Rate", f"{sm['WinRate']:.1f}%", None)

            st.markdown("#### Equity Curve")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=bt['strat_eq'].index, y=bt['strat_eq'].values,
                name='Signal Strategy',
                line=dict(color=color_bt, width=2),
                hovertemplate='Strategy: %{y:.1f}<extra></extra>'
            ))
            fig_eq.add_trace(go.Scatter(
                x=bt['bnh_eq'].index, y=bt['bnh_eq'].values,
                name='Buy & Hold',
                line=dict(color='gray', width=1.5, dash='dash'),
                hovertemplate='B&H: %{y:.1f}<extra></extra>'
            ))
            bnh_r = bt['bnh_eq'].reindex(bt['strat_eq'].index)
            fig_eq.add_trace(go.Scatter(
                x=bt['strat_eq'].index.tolist() +
                   bt['strat_eq'].index.tolist()[::-1],
                y=bt['strat_eq'].values.tolist() +
                   bnh_r.values.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(30,107,60,0.08)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Outperformance', showlegend=True
            ))
            fig_eq.add_hline(y=100, line_dash="dot",
                              line_color="gray", line_width=0.8,
                              annotation_text="Starting value")
            fig_eq.update_layout(
                height=420, template="plotly_white",
                yaxis_title="Portfolio Value (Base 100)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom",
                             y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            st.markdown("#### Drawdown Analysis")
            bnh_dd = ((bt['bnh_eq'] - bt['bnh_eq'].cummax()) /
                       bt['bnh_eq'].cummax() * 100)
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=bt['drawdown'].index, y=bt['drawdown'].values,
                fill='tozeroy', fillcolor='rgba(30,107,60,0.3)',
                line=dict(color=color_bt, width=1),
                name='Strategy Drawdown'
            ))
            fig_dd.add_trace(go.Scatter(
                x=bnh_dd.index, y=bnh_dd.values,
                line=dict(color='gray', width=1, dash='dash'),
                name='B&H Drawdown'
            ))
            fig_dd.update_layout(
                height=300, template="plotly_white",
                yaxis_title="Drawdown (%)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            st.markdown("#### Signal Position Over Time")
            fig_pos = go.Figure()
            fig_pos.add_trace(go.Scatter(
                x=bt['position'].index, y=bt['position'].values,
                fill='tozeroy', fillcolor='rgba(30,107,60,0.2)',
                line=dict(color=color_bt, width=0.8),
                name='Position (1=Long, 0=Cash)'
            ))
            fig_pos.update_layout(
                height=200, template="plotly_white",
                yaxis_title="Position",
                yaxis=dict(tickvals=[0,1],
                            ticktext=['Cash','Long']),
                hovermode="x unified"
            )
            st.plotly_chart(fig_pos, use_container_width=True)

            dd_saved      = sm['MaxDD'] - bm['MaxDD']
            sharpe_better = sm['Sharpe'] > bm['Sharpe']
            st.markdown(f"""<div class='commentary-box'>
            📌 <strong>Backtest Commentary — {bt_asset}:</strong>
            Strategy CAGR <strong>{sm['CAGR']:+.1f}%</strong>
            vs <strong>{bm['CAGR']:+.1f}%</strong> B&H.
            Sharpe: <strong>{sm['Sharpe']:.2f}</strong>
            vs <strong>{bm['Sharpe']:.2f}</strong>
            ({'✅ better' if sharpe_better else '⚠️ lower'}).
            Max drawdown <strong>{sm['MaxDD']:.1f}%</strong>
            vs <strong>{bm['MaxDD']:.1f}%</strong> B&H —
            saved <strong>{abs(dd_saved):.1f}pp</strong>.
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "#### Combined Portfolio — All 5 Assets Equal Weight"
        )
        try:
            all_strat_rets = []
            all_bnh_rets   = []
            for asset_name, (price_s, score_c) in \
                    asset_map.items():
                if score_c in signals.columns:
                    bt_temp = run_backtest_dash(
                        price_s, signals[score_c])
                    all_strat_rets.append(bt_temp['strat_ret'])
                    all_bnh_rets.append(bt_temp['bnh_ret'])

            if all_strat_rets:
                port_ret = pd.concat(
                    all_strat_rets, axis=1).dropna().mean(axis=1)
                port_eq  = (1 + port_ret).cumprod() * 100
                bnh_port = pd.concat(
                    all_bnh_rets, axis=1).dropna().mean(axis=1)
                bnh_port_eq = (1 + bnh_port).cumprod() * 100

                n_years     = len(port_ret) / 252
                port_cagr   = ((port_eq.iloc[-1]/100)**(
                    1/n_years)-1)*100
                port_sharpe = (port_ret.mean() /
                    (port_ret.std()+1e-10)) * np.sqrt(252)
                port_dd     = ((port_eq - port_eq.cummax()) /
                    port_eq.cummax() * 100).min()
                bnh_cagr    = ((bnh_port_eq.iloc[-1]/100)**(
                    1/n_years)-1)*100
                bnh_sharpe  = (bnh_port.mean() /
                    (bnh_port.std()+1e-10)) * np.sqrt(252)
                bnh_dd      = ((bnh_port_eq -
                    bnh_port_eq.cummax()) /
                    bnh_port_eq.cummax() * 100).min()

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
                           f"{bnh_sharpe:.2f}", None)

                fig_port = go.Figure()
                fig_port.add_trace(go.Scatter(
                    x=port_eq.index, y=port_eq.values,
                    name=f'Signal Portfolio '
                          f'(Sharpe: {port_sharpe:.2f})',
                    line=dict(color=BLUE, width=2)
                ))
                fig_port.add_trace(go.Scatter(
                    x=bnh_port_eq.index, y=bnh_port_eq.values,
                    name=f'B&H Portfolio '
                          f'(Sharpe: {bnh_sharpe:.2f})',
                    line=dict(color='gray', width=1.5,
                               dash='dash')
                ))
                fig_port.add_hline(
                    y=100, line_dash="dot",
                    line_color="gray", line_width=0.8
                )
                fig_port.update_layout(
                    height=380, template="plotly_white",
                    yaxis_title="Portfolio Value (Base 100)",
                    hovermode="x unified",
                    title=f"Combined Portfolio: "
                           f"Sharpe {port_sharpe:.2f} "
                           f"vs B&H {bnh_sharpe:.2f}"
                )
                st.plotly_chart(
                    fig_port, use_container_width=True)

                st.markdown(f"""<div class='commentary-box'>
                📌 <strong>Portfolio Summary:</strong>
                Combined Sharpe
                <strong>{port_sharpe:.2f}</strong> vs
                <strong>{bnh_sharpe:.2f}</strong> B&H —
                <strong>{port_sharpe/bnh_sharpe:.1f}x</strong>
                improvement.
                CAGR: <strong>{port_cagr:+.1f}%</strong> vs
                <strong>{bnh_cagr:+.1f}%</strong> B&H.
                Max DD: <strong>{port_dd:.1f}%</strong> vs
                <strong>{bnh_dd:.1f}%</strong> B&H.
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Portfolio calculation error: {e}")

# ── TAB 6: HISTORICAL ANALOGS ─────────────────────────────────
with tab6:
    st.subheader("Historical Analog Engine")
    st.caption(
        "Current conditions matched against 16 years of history"
    )

    if analog_dates.empty or analog_outcomes.empty:
        st.warning(
            "No analog data. Run 17_historical_analog.py first."
        )
    else:
        st.markdown("#### 📅 Most Similar Historical Periods")
        st.caption(
            "Periods where market conditions most closely "
            "resembled today"
        )

        cols = st.columns(len(analog_dates))
        for i, (_, row) in enumerate(analog_dates.iterrows()):
            with cols[i]:
                sim  = float(row['similarity']) * 100
                date = pd.Timestamp(row['analog_date'])
                st.markdown(f"""
                <div class='metric-card metric-neutral'>
                    <div style='font-size:11px; color:{GRAY};
                         text-transform:uppercase;'>
                         Analog {i+1}</div>
                    <div style='font-size:16px;
                         font-weight:bold; color:#FFFFFF;'>
                         {date.strftime('%b %Y')}</div>
                    <div style='font-size:12px; color:{GRAY};'>
                         Similarity: {sim:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("#### 📊 Forward Return Expectations")
        st.caption(
            "What typically happened after similar conditions"
        )

        assets_list = ['NIFTY', 'SP500', 'Gold', 'Silver', 'Crude']
        horizons    = [10, 30, 60]

        for horizon in horizons:
            st.markdown(f"**{horizon}-Day Outlook:**")
            h_data = analog_outcomes[
                analog_outcomes['forward_days'] == horizon
            ]
            if h_data.empty:
                continue

            cols = st.columns(len(assets_list))
            for i, asset in enumerate(assets_list):
                row = h_data[h_data['asset'] == asset]
                if row.empty:
                    continue
                row    = row.iloc[0]
                prob   = float(row['prob_positive'])
                median = float(row['median_return'])

                if prob >= 60:
                    color_cls = "metric-positive"
                    arrow = "↑"
                elif prob <= 40:
                    color_cls = "metric-negative"
                    arrow = "↓"
                else:
                    color_cls = "metric-neutral"
                    arrow = "→"

                with cols[i]:
                    st.markdown(f"""
                    <div class='metric-card {color_cls}'>
                        <div style='font-size:11px; color:{GRAY};
                             text-transform:uppercase;'>
                             {asset}</div>
                        <div style='font-size:18px;
                             font-weight:bold; color:#FFFFFF;'>
                             {arrow} {prob:.0f}%</div>
                        <div style='font-size:11px; color:{GRAY};'>
                             Median: {median:+.1f}%</div>
                        <div style='font-size:10px; color:{GRAY};'>
                             P25: {float(row["p25_return"]):+.1f}%
                             / P75:
                             {float(row["p75_return"]):+.1f}%
                             </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("")

        if not analog_dates.empty:
            top_date = pd.Timestamp(
                analog_dates.iloc[0]['analog_date'])
            top_sim  = float(
                analog_dates.iloc[0]['similarity']) * 100
            st.markdown(f"""<div class='commentary-box'>
            📌 <strong>Analog Commentary:</strong>
            Current macro conditions most closely resemble
            <strong>{top_date.strftime('%B %Y')}</strong>
            ({top_sim:.1f}% similarity).
            Based on VIX level, yield curve shape, momentum
            across 4 asset classes, and macro regime.
            Forward return expectations above reflect what
            actually happened after these
            {len(analog_dates)} historical periods.
            These are tendencies, not predictions.
            </div>""", unsafe_allow_html=True)

# ── TAB 7: WALK-FORWARD ───────────────────────────────────────
with tab7:
    st.subheader("Walk-Forward Backtest")
    st.caption(
        "Out-of-sample validation — each year tested on "
        "data the model never saw"
    )

    if wf_portfolio.empty:
        st.warning(
            "No walk-forward data. "
            "Run 18_walkforward_backtest.py first."
        )
    else:
        avg_port_sharpe = wf_portfolio['port_sharpe'].mean()
        avg_bnh_sharpe  = wf_portfolio['bnh_sharpe'].mean()
        years_beat      = wf_portfolio['beat_bnh'].sum()
        total_years     = len(wf_portfolio)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Avg Out-of-Sample Sharpe",
            f"{avg_port_sharpe:.3f}",
            f"{avg_port_sharpe-avg_bnh_sharpe:+.3f} vs B&H"
        )
        m2.metric("Avg B&H Sharpe", f"{avg_bnh_sharpe:.3f}")
        m3.metric(
            "Years Beating B&H",
            f"{years_beat}/{total_years}",
            f"{years_beat/total_years*100:.0f}%"
        )
        m4.metric(
            "Verdict",
            "ROBUST" if avg_port_sharpe > 0.4 else "MARGINAL"
        )

        st.markdown("#### Portfolio Walk-Forward Performance")
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(
            x=wf_portfolio['year'],
            y=wf_portfolio['port_sharpe'],
            name='Strategy Sharpe',
            marker_color=[
                GREEN if v > 0 else RED
                for v in wf_portfolio['port_sharpe']
            ]
        ))
        fig_wf.add_trace(go.Scatter(
            x=wf_portfolio['year'],
            y=wf_portfolio['bnh_sharpe'],
            name='B&H Sharpe',
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig_wf.add_hline(
            y=0, line_color='black', line_width=0.8)
        fig_wf.update_layout(
            height=380, template='plotly_white',
            yaxis_title='Sharpe Ratio',
            xaxis_title='Year (out-of-sample)',
            hovermode='x unified',
            title='Annual Out-of-Sample Sharpe — '
                   'Strategy vs B&H'
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        fig_cagr = go.Figure()
        fig_cagr.add_trace(go.Bar(
            x=wf_portfolio['year'],
            y=wf_portfolio['port_cagr'],
            name='Strategy CAGR',
            marker_color=[
                GREEN if v > 0 else RED
                for v in wf_portfolio['port_cagr']
            ]
        ))
        fig_cagr.add_trace(go.Scatter(
            x=wf_portfolio['year'],
            y=wf_portfolio['bnh_cagr'],
            name='B&H CAGR',
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig_cagr.add_hline(
            y=0, line_color='black', line_width=0.8)
        fig_cagr.update_layout(
            height=350, template='plotly_white',
            yaxis_title='CAGR (%)',
            xaxis_title='Year (out-of-sample)',
            hovermode='x unified',
            title='Annual Out-of-Sample CAGR — Strategy vs B&H'
        )
        st.plotly_chart(fig_cagr, use_container_width=True)

        st.markdown("#### Per-Asset Walk-Forward Summary")
        if not wf_results.empty:
            asset_summary = wf_results.groupby('asset').agg(
                avg_sharpe    =('strat_sharpe', 'mean'),
                avg_bnh_sharpe=('bnh_sharpe',   'mean'),
                avg_maxdd     =('strat_maxdd',   'mean'),
                avg_bnh_maxdd =('bnh_maxdd',     'mean'),
                years_positive=('strat_sharpe',
                                 lambda x: (x > 0).sum()),
                total_years   =('strat_sharpe', 'count'),
            ).reset_index()

            fig_asset = go.Figure()
            fig_asset.add_trace(go.Bar(
                x=asset_summary['asset'],
                y=asset_summary['avg_sharpe'],
                name='Strategy Sharpe',
                marker_color=MID_BLUE
            ))
            fig_asset.add_trace(go.Bar(
                x=asset_summary['asset'],
                y=asset_summary['avg_bnh_sharpe'],
                name='B&H Sharpe',
                marker_color=GRAY
            ))
            fig_asset.update_layout(
                height=350, template='plotly_white',
                barmode='group',
                yaxis_title='Avg Sharpe Ratio',
                title='Average Out-of-Sample Sharpe by Asset'
            )
            st.plotly_chart(
                fig_asset, use_container_width=True)

        st.markdown(f"""<div class='commentary-box'>
        📌 <strong>Walk-Forward Commentary:</strong>
        Combined portfolio achieved average out-of-sample
        Sharpe of <strong>{avg_port_sharpe:.3f}</strong> vs
        <strong>{avg_bnh_sharpe:.3f}</strong> B&H across
        <strong>{total_years} years</strong> of unseen data.
        Beat buy-and-hold in
        <strong>{years_beat} out of {total_years} years</strong>.
        Each year was tested on data the model had never seen.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 8 — DECISION ENGINE
# ══════════════════════════════════════════════════════════════
with tab8:
    st.subheader("Decision Engine — Full Layer Breakdown")
    st.caption("IC-weighted signal aggregation across 6 layers for every asset")

    # ── 1. Layer scores ───────────────────────────────────────
    decs = safe_load('DECISIONS', limit=50)
    if decs.empty:
        st.warning("No DECISIONS data. Run 19_decision_engine.py first.")
    else:
        _ld = decs['date'].max()
        latest_decs = decs[decs['date'] == _ld].copy()
        st.markdown(f"**Latest run: {_ld}**")

        _layer_cols = ['layer_signal', 'layer_analog', 'layer_sentiment',
                       'layer_macro',  'layer_yield',  'layer_vix']
        _dcols = st.columns(min(len(latest_decs), 5))
        for _i, (_, _row) in enumerate(latest_decs.iterrows()):
            if _i >= 5:
                break
            with _dcols[_i]:
                _bias = str(_row.get('bias', 'N/A')).upper()
                _comb = sfloat(_row.get('combined', 0))
                _conf = str(_row.get('confidence', '?'))
                _agr  = sfloat(_row.get('agreement', 0))
                _cc   = ('metric-positive' if 'LONG'  in _bias else
                         'metric-negative' if 'SHORT' in _bias else
                         'metric-neutral')
                _tc   = (GREEN  if 'LONG'  in _bias else
                         RED    if 'SHORT' in _bias else ORANGE)
                _lhtml = ''
                for _lc in _layer_cols:
                    _lv = _row.get(_lc)
                    if _lv is not None:
                        _lname = _lc.replace('layer_', '')
                        _lhtml += (
                            f"<div style='font-size:10px;color:{GRAY};'>"
                            f"{_lname}: {sfloat(_lv):+.3f}</div>"
                        )
                st.markdown(f"""
                <div class='metric-card {_cc}'>
                    <div style='font-size:11px;color:{GRAY};
                         text-transform:uppercase;'>
                         {_row.get('asset','?')}</div>
                    <div style='font-size:22px;font-weight:bold;
                         color:{_tc};'>{_bias}</div>
                    <div style='font-size:11px;color:{GRAY};'>
                         Score: {_comb:+.3f}</div>
                    <div style='font-size:11px;color:{GRAY};'>
                         {_conf} | Agr: {_agr:.0f}%</div>
                    <hr style='margin:4px 0;border-color:#eee;'>
                    {_lhtml}
                </div>""", unsafe_allow_html=True)

        with st.expander("📋 Full Decisions Table"):
            st.dataframe(decs, use_container_width=True)

    st.markdown("---")

    # ── 2. Dynamic weights ────────────────────────────────────
    st.subheader("Dynamic IC Weights")
    weights_df = safe_load('DYNAMIC_WEIGHTS', limit=200)
    if weights_df.empty:
        st.warning("No DYNAMIC_WEIGHTS data. Run 35_dynamic_weights.py first.")
    else:
        _ld_w = weights_df['date'].max()
        _latest_w = weights_df[weights_df['date'] == _ld_w].copy()
        try:
            _pivot_w = _latest_w.pivot_table(
                index='asset', columns='component',
                values='adjusted_weight', aggfunc='first'
            ).round(3)
            st.dataframe(
                _pivot_w.style.background_gradient(
                    cmap='RdYlGn', axis=None),
                use_container_width=True
            )
        except Exception:
            _show = [c for c in ['asset', 'component', 'base_weight',
                                  'adjusted_weight', 'ic_60d']
                     if c in _latest_w.columns]
            st.dataframe(_latest_w[_show], use_container_width=True)

        if 'ic_60d' in _latest_w.columns:
            _ic = _latest_w[['asset', 'component', 'ic_60d']].dropna()
            if len(_ic) > 0:
                _ic['label'] = _ic['component'] + ' / ' + _ic['asset']
                _fig_ic = go.Figure(go.Bar(
                    x=_ic['label'], y=_ic['ic_60d'],
                    marker_color=[
                        GREEN  if v >  0.10 else
                        RED    if v < -0.05 else ORANGE
                        for v in _ic['ic_60d']
                    ],
                ))
                _fig_ic.add_hline(y=0.10,  line_dash='dot',
                                   line_color=GREEN,
                                   annotation_text='Boost (>0.10)')
                _fig_ic.add_hline(y=-0.05, line_dash='dot',
                                   line_color=RED,
                                   annotation_text='Cut (<-0.05)')
                _fig_ic.update_layout(
                    height=320, template='plotly_white',
                    title='Rolling 60-Day IC by Component / Asset',
                    yaxis_title='IC', xaxis_tickangle=-45
                )
                st.plotly_chart(_fig_ic, use_container_width=True)

    st.markdown("---")

    # ── 3. Physical basis ─────────────────────────────────────
    st.subheader("Physical Basis Monitor")
    basis_df = safe_load('PHYSICAL_BASIS', limit=20)
    if basis_df.empty:
        st.warning("No PHYSICAL_BASIS data. Run 23_physical_basis.py first.")
    else:
        _ld_b   = basis_df['date'].max()
        _latest_b = basis_df[basis_df['date'] == _ld_b].copy()
        _bc = st.columns(min(len(_latest_b), 5))
        for _i, (_, _row) in enumerate(_latest_b.iterrows()):
            if _i >= 5:
                break
            with _bc[_i]:
                _st  = str(_row.get('status', 'N/A'))
                _bp  = sfloat(_row.get('basis_pct', 0))
                _bcc = ('metric-positive' if 'CONTANGO' in _st.upper() else
                        'metric-negative' if 'BACK'     in _st.upper() else
                        'metric-neutral')
                st.markdown(f"""
                <div class='metric-card {_bcc}'>
                    <div style='font-size:11px;color:{GRAY};
                         text-transform:uppercase;'>
                         {_row.get('asset','?')}</div>
                    <div style='font-size:16px;font-weight:bold;'>
                         {_st}</div>
                    <div style='font-size:11px;color:{GRAY};'>
                         Basis: {_bp:+.3f}%</div>
                    <div style='font-size:11px;color:{GRAY};'>
                         Trend: {_row.get('trend','?')}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 4. Execution friction ─────────────────────────────────
    st.subheader("Execution Friction")
    friction_df = safe_load('EXECUTION_FRICTION', limit=10)
    if friction_df.empty:
        st.warning(
            "No EXECUTION_FRICTION data. Run 36_execution_friction.py first."
        )
    else:
        _ld_f   = friction_df['date'].max()
        _latest_f = friction_df[friction_df['date'] == _ld_f].copy()
        _fc = st.columns(min(len(_latest_f), 5))
        for _i, (_, _row) in enumerate(_latest_f.iterrows()):
            if _i >= 5:
                break
            with _fc[_i]:
                _tq  = str(_row.get('timing_quality', 'N/A'))
                _fp  = sfloat(_row.get('total_friction_pct', 0))
                _fcc = ('metric-positive' if 'GOOD' in _tq.upper() else
                        'metric-negative' if 'POOR' in _tq.upper() else
                        'metric-neutral')
                st.markdown(f"""
                <div class='metric-card {_fcc}'>
                    <div style='font-size:11px;color:{GRAY};
                         text-transform:uppercase;'>
                         {_row.get('asset','?')}</div>
                    <div style='font-size:16px;font-weight:bold;'>
                         {_tq}</div>
                    <div style='font-size:11px;color:{GRAY};'>
                         Friction: {_fp:.3f}%</div>
                    <div style='font-size:10px;color:{GRAY};'>
                         {str(_row.get('timing_note',''))[:50]}</div>
                </div>""", unsafe_allow_html=True)
        with st.expander("📋 Full Friction Detail"):
            _show_f = [c for c in ['date', 'asset', 'vix_regime',
                                    'slippage_pct', 'spread_pct',
                                    'total_friction_pct',
                                    'timing_quality', 'timing_note']
                       if c in _latest_f.columns]
            st.dataframe(_latest_f[_show_f], use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 9 — INDIA INTELLIGENCE
# ══════════════════════════════════════════════════════════════
with tab9:
    st.subheader("India Intelligence Dashboard")

    # ── 1. Macro regime ───────────────────────────────────────
    india_m = safe_load('INDIA_MACRO',   limit=5)
    macro_r = safe_load('MACRO_REGIME',  limit=5)

    _im1, _im2 = st.columns(2)
    with _im1:
        st.markdown("#### India Macro Regime")
        if india_m.empty:
            st.warning("No INDIA_MACRO data. Run 37_india_macro.py first.")
        else:
            _r = india_m.iloc[0]
            _reg = str(_r.get('macro_regime', 'N/A'))
            _rcc = ('metric-positive' if 'GOLDILOCKS' in _reg else
                    'metric-negative' if 'STAGFLATION' in _reg else
                    'metric-neutral')
            st.markdown(f"""
            <div class='metric-card {_rcc}'>
                <div style='font-size:11px;color:{GRAY};'>
                     India Macro Regime</div>
                <div style='font-size:20px;font-weight:bold;'>
                     {_reg}</div>
                <div style='font-size:12px;color:{GRAY};'>
                     CPI YoY: {sfloat(_r.get('cpi_yoy')):.2f}%</div>
                <div style='font-size:12px;color:{GRAY};'>
                     GDP: {sfloat(_r.get('gdp_growth')):.2f}%</div>
                <div style='font-size:12px;color:{GRAY};'>
                     Real Rate: {sfloat(_r.get('real_rate')):+.2f}%</div>
                <div style='font-size:12px;color:{GRAY};'>
                     FII Signal: {_r.get('fii_macro_signal','?')}</div>
                <div style='font-size:12px;color:{GRAY};'>
                     NIFTY Adj: {sfloat(_r.get('nifty_adjustment')):+.2f}</div>
            </div>""", unsafe_allow_html=True)

    with _im2:
        st.markdown("#### US vs India Regime")
        if macro_r.empty:
            st.warning("No MACRO_REGIME data. Run 41_macro_regime.py first.")
        else:
            _r = macro_r.iloc[0]
            _us = str(_r.get('us_regime', 'N/A'))
            _in = str(_r.get('india_regime', 'N/A'))
            _dn = str(_r.get('divergence_note', ''))[:140]
            st.markdown(f"""
            <div class='metric-card metric-neutral'>
                <div style='font-size:13px;font-weight:bold;'>
                     🇺🇸 US: {_us}</div>
                <div style='font-size:13px;font-weight:bold;
                     margin-top:4px;'>🇮🇳 India: {_in}</div>
                <div style='font-size:11px;color:{GRAY};margin-top:8px;'>
                     {_dn}</div>
                <hr style='margin:6px 0;border-color:#eee;'>
                <div style='font-size:11px;color:{GRAY};'>
                     SP500: {sfloat(_r.get('adj_sp500')):+.2f} |
                     NIFTY: {sfloat(_r.get('adj_nifty')):+.2f} |
                     Gold: {sfloat(_r.get('adj_gold')):+.2f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. FII / DII flows ────────────────────────────────────
    st.markdown("#### FII / DII Flows — Last 60 Days")
    fii_df = safe_load('FII_DII_FLOWS', limit=60)
    if fii_df.empty:
        st.warning(
            "No FII_DII_FLOWS data. Run 26_institutional_flows.py first."
        )
    else:
        fii_df['date'] = pd.to_datetime(fii_df['date'], errors='coerce')
        fii_df = fii_df.sort_values('date')
        _fig_fii = go.Figure()
        _fig_fii.add_trace(go.Bar(
            x=fii_df['date'], y=fii_df['fii_net'],
            name='FII Net',
            marker_color=[
                GREEN if v >= 0 else RED
                for v in fii_df['fii_net'].fillna(0)
            ],
        ))
        if 'dii_net' in fii_df.columns:
            _fig_fii.add_trace(go.Scatter(
                x=fii_df['date'], y=fii_df['dii_net'],
                name='DII Net',
                line=dict(color=BLUE, width=1.5, dash='dot'),
            ))
        _fig_fii.update_layout(
            height=320, template='plotly_white',
            title='FII / DII Net Flows (₹ Cr)',
            hovermode='x unified',
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(_fig_fii, use_container_width=True)

    st.markdown("---")

    # ── 3. Options Intelligence ───────────────────────────────
    st.markdown("#### NIFTY Options Dashboard")
    opts_df = safe_load('OPTIONS_INTELLIGENCE', limit=5)
    if opts_df.empty:
        st.warning(
            "No OPTIONS_INTELLIGENCE data. "
            "Run 43_options_intelligence.py first."
        )
    else:
        _or = opts_df.iloc[0]
        _oc1, _oc2, _oc3, _oc4, _oc5 = st.columns(5)
        with _oc1:
            _pcr = sfloat(_or.get('pcr_oi'))
            st.markdown(f"""
            <div class='metric-card metric-neutral'>
                <div style='font-size:11px;color:{GRAY};'>PCR OI</div>
                <div style='font-size:22px;font-weight:bold;'>
                     {_pcr:.3f}</div>
                <div style='font-size:11px;color:{GRAY};'>
                     {_or.get('pcr_signal','?')}</div>
            </div>""", unsafe_allow_html=True)
        with _oc2:
            _mp    = sfloat(_or.get('max_pain_strike'))
            _mpdev = sfloat(_or.get('max_pain_deviation'))
            _mpcc  = ('metric-positive' if _mpdev >  1 else
                      'metric-negative' if _mpdev < -1 else
                      'metric-neutral')
            st.markdown(f"""
            <div class='metric-card {_mpcc}'>
                <div style='font-size:11px;color:{GRAY};'>Max Pain</div>
                <div style='font-size:22px;font-weight:bold;'>
                     {_mp:,.0f}</div>
                <div style='font-size:11px;color:{GRAY};'>
                     {_mpdev:+.2f}% from spot</div>
                <div style='font-size:11px;color:{GRAY};'>
                     {_or.get('max_pain_bias','?')}</div>
            </div>""", unsafe_allow_html=True)
        with _oc3:
            _iv    = sfloat(_or.get('near_iv'))
            _ivreg = str(_or.get('iv_regime', 'N/A'))
            _ivcc  = ('metric-negative' if _ivreg in ['CRISIS', 'ELEVATED'] else
                      'metric-positive' if _ivreg == 'COMPRESSED' else
                      'metric-neutral')
            st.markdown(f"""
            <div class='metric-card {_ivcc}'>
                <div style='font-size:11px;color:{GRAY};'>
                     India VIX (Near IV)</div>
                <div style='font-size:22px;font-weight:bold;'>
                     {_iv:.1f}</div>
                <div style='font-size:11px;color:{GRAY};'>
                     {_ivreg} | TS: {_or.get('iv_ts_signal','?')}</div>
            </div>""", unsafe_allow_html=True)
        with _oc4:
            _sk   = sfloat(_or.get('skew_ratio'))
            _sksig = str(_or.get('skew_signal', 'N/A'))
            st.markdown(f"""
            <div class='metric-card metric-neutral'>
                <div style='font-size:11px;color:{GRAY};'>
                     Skew (PE/CE)</div>
                <div style='font-size:22px;font-weight:bold;'>
                     {_sk:.2f}</div>
                <div style='font-size:11px;color:{GRAY};'>
                     {_sksig}</div>
            </div>""", unsafe_allow_html=True)
        with _oc5:
            _cs    = sfloat(_or.get('composite_score'))
            _csig  = str(_or.get('composite_signal', 'N/A'))
            _cscc  = ('metric-positive' if _cs >  0.15 else
                      'metric-negative' if _cs < -0.15 else
                      'metric-neutral')
            st.markdown(f"""
            <div class='metric-card {_cscc}'>
                <div style='font-size:11px;color:{GRAY};'>Composite</div>
                <div style='font-size:22px;font-weight:bold;'>
                     {_cs:+.2f}</div>
                <div style='font-size:11px;color:{GRAY};'>{_csig}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class='commentary-box'>
        📌 <strong>Options Commentary:</strong>
        PCR OI: <strong>{sfloat(_or.get('pcr_oi')):.3f}</strong>
        ({_or.get('pcr_signal','?')}) |
        Max Pain: <strong>{sfloat(_or.get('max_pain_strike')):,.0f}</strong>
        ({sfloat(_or.get('max_pain_deviation')):+.2f}%) |
        IV Regime: <strong>{_or.get('iv_regime','?')}</strong> |
        Skew: <strong>{_or.get('skew_signal','?')}</strong> |
        Composite: <strong>{_or.get('composite_signal','?')}</strong>.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 4. GIFT-Gap history ───────────────────────────────────
    st.markdown("#### GIFT-City Gap History — Last 10 Gaps")
    gaps_df = safe_load('GIFT_GAPS', limit=10)
    if gaps_df.empty:
        st.warning(
            "No GIFT_GAPS data. Run 39_gift_gap_engine.py first."
        )
    else:
        gaps_df = gaps_df.sort_values('date', ascending=False)
        _gcols = [c for c in ['date', 'gap_pct', 'gap_direction',
                               'gap_size_class', 'classification',
                               'recommendation', 'hist_fill_rate',
                               'breakout_score', 'trap_score']
                  if c in gaps_df.columns]
        _disp = gaps_df[_gcols].copy()
        if 'gap_pct' in _disp.columns:
            _disp['gap_pct'] = _disp['gap_pct'].apply(
                lambda v: f"{sfloat(v):+.2f}%"
            )
        if 'hist_fill_rate' in _disp.columns:
            _disp['hist_fill_rate'] = _disp['hist_fill_rate'].apply(
                lambda v: f"{sfloat(v):.0%}"
            )
        st.dataframe(_disp, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 10 — RISK MONITOR
# ══════════════════════════════════════════════════════════════
with tab10:
    st.subheader("Risk Monitor — Portfolio & Market Risk")

    # ── 1. Model health ───────────────────────────────────────
    st.markdown("#### Model Health (Rolling Sharpe per Asset)")
    health_df = safe_load('MODEL_HEALTH', limit=60)
    if health_df.empty:
        st.warning("No MODEL_HEALTH data. Run 22_model_health.py first.")
    else:
        _ld_h     = health_df['date'].max()
        _latest_h = health_df[health_df['date'] == _ld_h].copy()
        _hc = st.columns(min(len(_latest_h), 5))
        for _i, (_, _row) in enumerate(_latest_h.iterrows()):
            if _i >= 5:
                break
            with _hc[_i]:
                _sh  = sfloat(_row.get('rolling_sharpe'))
                _hst = str(_row.get('status', 'N/A'))
                _hcc = ('metric-positive'
                        if any(w in _hst.upper()
                               for w in ['GOOD', 'HEALTHY']) else
                        'metric-negative'
                        if any(w in _hst.upper()
                               for w in ['POOR', 'CRITICAL']) else
                        'metric-neutral')
                st.markdown(f"""
                <div class='metric-card {_hcc}'>
                    <div style='font-size:11px;color:{GRAY};
                         text-transform:uppercase;'>
                         {_row.get('asset','?')}</div>
                    <div style='font-size:20px;font-weight:bold;'>
                         {_sh:+.2f}</div>
                    <div style='font-size:11px;color:{GRAY};'>
                         {_hst}</div>
                    <div style='font-size:18px;color:#FFFFFF;font-weight:bold;'>
                         Hit: {sfloat(_row.get("hit_rate")):.1%}
                         | DD: {sfloat(_row.get("current_dd")):.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)

        if len(health_df) > 5:
            health_df['date'] = pd.to_datetime(
                health_df['date'], errors='coerce')
            _fig_h = go.Figure()
            for _ast in health_df['asset'].unique():
                _ah = health_df[
                    health_df['asset'] == _ast
                ].sort_values('date')
                if 'rolling_sharpe' in _ah.columns:
                    _fig_h.add_trace(go.Scatter(
                        x=_ah['date'], y=_ah['rolling_sharpe'],
                        name=_ast, line=dict(width=1.5)
                    ))
            _fig_h.add_hline(y=0,   line_color='black', line_width=0.8)
            _fig_h.add_hline(y=0.5, line_dash='dot', line_color=GREEN,
                              annotation_text='Good (0.5)')
            _fig_h.update_layout(
                height=300, template='plotly_white',
                title='Rolling Sharpe by Asset',
                hovermode='x unified'
            )
            st.plotly_chart(_fig_h, use_container_width=True)

    st.markdown("---")

    # ── 2. Portfolio risk ─────────────────────────────────────
    st.markdown("#### Portfolio Risk & Kelly Sizing")
    risk_df = safe_load('PORTFOLIO_RISK', limit=20)
    if risk_df.empty:
        st.warning("No PORTFOLIO_RISK data. Run 27_portfolio_risk.py first.")
    else:
        _ld_r   = risk_df['date'].max()
        _latest_r = risk_df[risk_df['date'] == _ld_r].copy()
        _rc = st.columns(min(len(_latest_r), 5))
        for _i, (_, _row) in enumerate(_latest_r.iterrows()):
            if _i >= 5:
                break
            with _rc[_i]:
                _kh   = sfloat(_row.get('kelly_half'))
                _rbias= str(_row.get('bias', 'N/A'))
                _cf   = str(_row.get('corr_flag', '') or '')
                _rcc  = ('metric-positive'  if 'LONG'  in _rbias.upper() else
                         'metric-negative'  if 'SHORT' in _rbias.upper() else
                         'metric-neutral')
                st.markdown(f"""
                <div class='metric-card {_rcc}'>
                    <div style='font-size:11px;color:{GRAY};
                         text-transform:uppercase;'>
                         {_row.get('asset','?')}</div>
                    <div style='font-size:20px;font-weight:bold;'>
                         {_kh:.1%} Kelly</div>
                    <div style='font-size:11px;color:{GRAY};'>
                         {_rbias}</div>
                    <div style='font-size:10px;
                         color:{"#C00000" if _cf else GRAY};'>
                         {_cf if _cf else '✓ No corr flag'}</div>
                </div>""", unsafe_allow_html=True)
        with st.expander("📋 Full Risk Detail"):
            st.dataframe(_latest_r, use_container_width=True)

    st.markdown("---")

    # ── 3. Correlation regime ─────────────────────────────────
    st.markdown("#### Correlation Regime Monitor")
    corr_reg = safe_load('CORRELATION_REGIMES', limit=60)
    if corr_reg.empty:
        st.warning(
            "No CORRELATION_REGIMES data. "
            "Run 28_correlation_regime.py first."
        )
    else:
        _ld_cr    = corr_reg['date'].max()
        _latest_cr = corr_reg[corr_reg['date'] == _ld_cr].copy()
        try:
            _pivot_c = _latest_cr.pivot_table(
                index='asset1', columns='asset2',
                values='corr_20d', aggfunc='first'
            )
            if len(_pivot_c) > 0:
                _fig_cr = px.imshow(
                    _pivot_c, color_continuous_scale='RdYlGn',
                    zmin=-1, zmax=1, text_auto='.2f',
                    title='Current 20-Day Correlation Regime'
                )
                _fig_cr.update_layout(
                    height=320, template='plotly_white')
                st.plotly_chart(_fig_cr, use_container_width=True)
        except Exception:
            st.dataframe(_latest_cr, use_container_width=True)

        if 'shift' in _latest_cr.columns:
            _shifts = _latest_cr[
                _latest_cr['shift'].apply(
                    lambda v: abs(sfloat(v)) > 0.15
                )
            ]
            for _, _sr in _shifts.iterrows():
                st.warning(
                    f"⚠️ Correlation shift detected: "
                    f"{_sr.get('asset1','?')} / {_sr.get('asset2','?')} "
                    f"— 20d: {sfloat(_sr.get('corr_20d')):+.2f}, "
                    f"60d: {sfloat(_sr.get('corr_60d')):+.2f}"
                )

    st.markdown("---")

    # ── 4. Composite Liquidity ────────────────────────────────
    st.markdown("#### Composite Liquidity Score")
    liq_comp = safe_load('COMPOSITE_LIQUIDITY', limit=60)
    if liq_comp.empty:
        st.warning(
            "No COMPOSITE_LIQUIDITY data. "
            "Run 40_composite_liquidity.py first."
        )
    else:
        _lrow = liq_comp.iloc[0]
        _lq1, _lq2, _lq3 = st.columns(3)
        with _lq1:
            _lscore = sfloat(_lrow.get('composite_score'))
            _lreg   = str(_lrow.get('regime', 'N/A'))
            _lcc    = 'metric-positive' if _lscore > 0 else 'metric-negative'
            st.markdown(f"""
            <div class='metric-card {_lcc}'>
                <div style='font-size:11px;color:{GRAY};'>
                     Composite Liquidity</div>
                <div style='font-size:20px;font-weight:bold;color:#FFFFFF;'>
                     {_lscore:+.3f}</div>
                <div style='font-size:12px;color:#FFFFFF;font-weight:bold;'>
                     {_lreg}</div>
            </div>""", unsafe_allow_html=True)
        with _lq2:
            st.markdown(f"""
            <div class='metric-card metric-neutral'>
                <div style='font-size:11px;color:{GRAY};'>
                     Liquidity Adjustments</div>
                <div style='font-size:12px;'>
                     SP500: {sfloat(_lrow.get('adj_sp500')):+.3f}</div>
                <div style='font-size:12px;'>
                     NIFTY: {sfloat(_lrow.get('adj_nifty')):+.3f}</div>
                <div style='font-size:12px;'>
                     Gold: {sfloat(_lrow.get('adj_gold')):+.3f}</div>
                <div style='font-size:12px;'>
                     Silver: {sfloat(_lrow.get('adj_silver')):+.3f}</div>
            </div>""", unsafe_allow_html=True)
        with _lq3:
            st.markdown(f"""
            <div class='metric-card metric-neutral'>
                <div style='font-size:11px;color:{GRAY};'>
                     Component Scores</div>
                <div style='font-size:11px;'>
                     TGA: {sfloat(_lrow.get('tga_score')):+.2f}</div>
                <div style='font-size:11px;'>
                     Fed BS: {sfloat(_lrow.get('fed_bs_score')):+.2f}</div>
                <div style='font-size:11px;'>
                     Term Prem: {sfloat(_lrow.get('term_premium_score')):+.2f}
                </div>
                <div style='font-size:11px;'>
                     Credit: {sfloat(_lrow.get('credit_score')):+.2f}</div>
            </div>""", unsafe_allow_html=True)

        if len(liq_comp) > 2:
            _lhist = liq_comp.sort_values('date')
            _lhist['date'] = pd.to_datetime(_lhist['date'], errors='coerce')
            _fig_lq = go.Figure(go.Scatter(
                x=_lhist['date'], y=_lhist['composite_score'],
                fill='tozeroy',
                fillcolor='rgba(30,107,60,0.12)',
                line=dict(color=BLUE, width=1.5)
            ))
            _fig_lq.add_hline(y=0, line_color='black', line_width=0.8)
            _fig_lq.update_layout(
                height=260, template='plotly_white',
                title='Composite Liquidity History',
                yaxis_title='Score'
            )
            st.plotly_chart(_fig_lq, use_container_width=True)

    st.markdown("---")

    # ── 5. Geopolitical premium ───────────────────────────────
    st.markdown("#### Geopolitical Risk Premium")
    geo_df = safe_load('GEOPOLITICAL_PREMIUM', limit=30)
    if geo_df.empty:
        st.warning(
            "No GEOPOLITICAL_PREMIUM data. "
            "Run 33_geopolitical_premium.py first."
        )
    else:
        _gr = geo_df.iloc[0]
        _gp1, _gp2 = st.columns(2)
        with _gp1:
            _cprem = sfloat(_gr.get('crude_premium_pct'))
            _ccls  = str(_gr.get('crude_classification', 'N/A'))
            _cgcc  = ('metric-negative' if _cprem > 10 else
                      'metric-neutral'  if _cprem > 0  else
                      'metric-positive')
            st.markdown(f"""
            <div class='metric-card {_cgcc}'>
                <div style='font-size:11px;color:{GRAY};'>
                     Crude Geo Premium</div>
                <div style='font-size:22px;font-weight:bold;'>
                     ${sfloat(_gr.get('crude_premium_usd')):+.1f}</div>
                <div style='font-size:12px;color:#FFFFFF;font-weight:bold;'>
                     {_cprem:+.1f}% | {_ccls}</div>
                <div style='font-size:11px;color:{GRAY};'>
                     Spot: ${sfloat(_gr.get('crude_spot')):.1f} |
                     Fair: ${sfloat(_gr.get('crude_fair_value')):.1f}</div>
                <div style='font-size:10px;color:{GRAY};'>
                     Rank: {sfloat(_gr.get('crude_pct_rank')):.0f}th pct</div>
            </div>""", unsafe_allow_html=True)
        with _gp2:
            _gprem = sfloat(_gr.get('gold_premium_pct'))
            _gcls  = str(_gr.get('gold_classification', 'N/A'))
            _ggcc  = ('metric-negative' if _gprem > 10 else
                      'metric-neutral'  if _gprem > 0  else
                      'metric-positive')
            st.markdown(f"""
            <div class='metric-card {_ggcc}'>
                <div style='font-size:11px;color:{GRAY};'>
                     Gold Geo Premium</div>
                <div style='font-size:22px;font-weight:bold;'>
                     ${sfloat(_gr.get('gold_premium_usd')):+.1f}</div>
                <div style='font-size:12px;color:#FFFFFF;font-weight:bold;'>
                     {_gprem:+.1f}% | {_gcls}</div>
                <div style='font-size:11px;color:{GRAY};'>
                     Spot: ${sfloat(_gr.get('gold_spot')):.1f} |
                     Fair: ${sfloat(_gr.get('gold_fair_value')):.1f}</div>
                <div style='font-size:10px;color:{GRAY};'>
                     Rank: {sfloat(_gr.get('gold_pct_rank')):.0f}th pct</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 11 — INTELLIGENCE
# ══════════════════════════════════════════════════════════════
with tab11:
    st.subheader("Market Intelligence — Narrative & Relative Value")

    # ── 1. Narrative engine ───────────────────────────────────
    st.markdown("#### Narrative Engine — Active Themes")
    narr_df = safe_load('NARRATIVE_SCORES', limit=60)
    if narr_df.empty:
        st.warning(
            "No NARRATIVE_SCORES data. Run 34_narrative_engine.py first."
        )
    else:
        _ld_n    = narr_df['date'].max()
        _latest_n = narr_df[narr_df['date'] == _ld_n].copy()
        _latest_n = _latest_n.sort_values('normalised', ascending=False)
        _top5 = _latest_n.head(5)
        _fig_narr = go.Figure(go.Bar(
            x=_top5['narrative'],
            y=_top5['normalised'],
            marker_color=[
                GREEN  if sfloat(v) >= 70 else
                ORANGE if sfloat(v) >= 40 else
                GRAY   for v in _top5['normalised']
            ],
            text=_top5.get('status', pd.Series([''] * len(_top5))),
            textposition='outside',
        ))
        _fig_narr.update_layout(
            height=340, template='plotly_white',
            title='Top 5 Active Market Narratives (0–100)',
            yaxis_title='Score', yaxis_range=[0, 115]
        )
        st.plotly_chart(_fig_narr, use_container_width=True)

        _ncols = [c for c in ['narrative', 'score_7d', 'score_14d',
                               'score_30d', 'normalised',
                               'status', 'momentum_7d']
                  if c in _latest_n.columns]
        if 'narrative' in _ncols:
            st.dataframe(
                _latest_n[_ncols].set_index('narrative'),
                use_container_width=True
            )

    st.markdown("---")

    # ── 2. RV Ratios ──────────────────────────────────────────
    st.markdown("#### Relative Value Ratios")
    rv_df = safe_load('RV_RATIOS', limit=60)
    if rv_df.empty:
        st.warning(
            "No RV_RATIOS data. Run 38_rv_ratio_overlay.py first."
        )
    else:
        _rrow  = rv_df.iloc[0]
        _rv1, _rv2, _rv3 = st.columns(3)

        def _rv_card(label, ratio, pct, signal, rotation):
            _cc = ('metric-negative' if 'EXTREME_HIGH' in str(signal) else
                   'metric-positive' if 'EXTREME_LOW'  in str(signal) else
                   'metric-neutral')
            return (
                f"<div class='metric-card {_cc}'>"
                f"<div style='font-size:11px;color:{GRAY};'>{label}</div>"
                f"<div style='font-size:22px;font-weight:bold;'>"
                f"{sfloat(ratio):.2f}</div>"
                f"<div style='font-size:11px;color:{GRAY};'>"
                f"{sfloat(pct):.0f}th pct | {signal}</div>"
                f"<div style='font-size:11px;color:{GRAY};'>"
                f"→ {rotation}</div></div>"
            )

        with _rv1:
            st.markdown(_rv_card(
                'Gold / Silver',
                _rrow.get('gs_ratio'),  _rrow.get('gs_pct_5y'),
                _rrow.get('gs_signal'), _rrow.get('gs_rotation')
            ), unsafe_allow_html=True)
        with _rv2:
            st.markdown(_rv_card(
                'Gold / SP500',
                _rrow.get('gsp_ratio'),  _rrow.get('gsp_pct_5y'),
                _rrow.get('gsp_signal'), _rrow.get('gsp_rotation')
            ), unsafe_allow_html=True)
        with _rv3:
            st.markdown(_rv_card(
                'Crude / Gold',
                _rrow.get('cg_ratio'),  _rrow.get('cg_pct_5y'),
                _rrow.get('cg_signal'), _rrow.get('cg_rotation')
            ), unsafe_allow_html=True)

        if len(rv_df) > 3:
            _rvh = rv_df.sort_values('date')
            _rvh['date'] = pd.to_datetime(_rvh['date'], errors='coerce')
            _fig_rv = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                subplot_titles=(
                    'Gold/Silver', 'Gold/SP500', 'Crude/Gold'),
                vertical_spacing=0.08
            )
            _fig_rv.add_trace(go.Scatter(
                x=_rvh['date'], y=_rvh['gs_ratio'],
                name='G/S', line=dict(color=ORANGE, width=1.4)
            ), row=1, col=1)
            _fig_rv.add_trace(go.Scatter(
                x=_rvh['date'], y=_rvh['gsp_ratio'],
                name='G/SP', line=dict(color=BLUE, width=1.4)
            ), row=2, col=1)
            _fig_rv.add_trace(go.Scatter(
                x=_rvh['date'], y=_rvh['cg_ratio'],
                name='C/G', line=dict(color=RED, width=1.4)
            ), row=3, col=1)
            _fig_rv.update_layout(
                height=440, template='plotly_white',
                hovermode='x unified'
            )
            st.plotly_chart(_fig_rv, use_container_width=True)

    st.markdown("---")

    # ── 3. Polarization ───────────────────────────────────────
    st.markdown("#### Market Polarization (Mag-7 vs Breadth)")
    polar_df = safe_load('POLARIZATION_DATA', limit=30)
    if polar_df.empty:
        st.warning(
            "No POLARIZATION_DATA data. "
            "Run 32_polarization_monitor.py first."
        )
    else:
        _pr = polar_df.iloc[0]
        _pp1, _pp2 = st.columns(2)
        with _pp1:
            _m7r = sfloat(_pr.get('mag7_return'))
            _pcc = 'metric-positive' if _m7r > 0 else 'metric-negative'
            st.markdown(f"""
            <div class='metric-card {_pcc}'>
                <div style='font-size:11px;color:{GRAY};'>
                     Mag-7 Return (latest)</div>
                <div style='font-size:22px;font-weight:bold;'>
                     {_m7r:+.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with _pp2:
            _ret_cols = [c for c in polar_df.columns
                         if c.startswith('ret_')][:5]
            _rhtml = ''.join([
                f"<div style='font-size:11px;'>"
                f"{c.replace('ret_','')}: "
                f"{sfloat(_pr.get(c)):+.2f}%</div>"
                for c in _ret_cols
            ])
            st.markdown(f"""
            <div class='metric-card metric-neutral'>
                <div style='font-size:11px;color:{GRAY};'>
                     Individual Returns</div>
                {_rhtml}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 4. Liquidity Pulse ────────────────────────────────────
    st.markdown("#### Liquidity Pulse Monitor")
    pulse_df = safe_load('LIQUIDITY_PULSE', limit=30)
    if pulse_df.empty:
        st.warning(
            "No LIQUIDITY_PULSE data. Run 30_liquidity_pulse.py first."
        )
    else:
        _plr = pulse_df.iloc[0]
        _tga_chg = sfloat(_plr.get('tga_4wk_change'))
        _tga_sig = str(_plr.get('tga_signal', 'N/A'))
        _tp      = sfloat(_plr.get('term_premium'))
        _tp_sig  = str(_plr.get('term_premium_signal', 'N/A'))

        _lp1, _lp2 = st.columns(2)
        with _lp1:
            _tcc = 'metric-positive' if _tga_chg < 0 else 'metric-negative'
            st.markdown(f"""
            <div class='metric-card {_tcc}'>
                <div style='font-size:11px;color:{GRAY};'>
                     TGA 4-Week Change</div>
                <div style='font-size:22px;font-weight:bold;'>
                     ${_tga_chg:+.0f}B</div>
                <div style='font-size:11px;color:{GRAY};'>
                     {_tga_sig}</div>
            </div>""", unsafe_allow_html=True)
        with _lp2:
            st.markdown(f"""
            <div class='metric-card metric-neutral'>
                <div style='font-size:11px;color:{GRAY};'>
                     Term Premium</div>
                <div style='font-size:22px;font-weight:bold;'>
                     {_tp:+.3f}%</div>
                <div style='font-size:11px;color:{GRAY};'>
                     {_tp_sig}</div>
            </div>""", unsafe_allow_html=True)

        # All signal columns as a summary table
        _sig_cols = [c for c in pulse_df.columns if '_signal' in c.lower()]
        if _sig_cols:
            _sig_summary = {
                c.replace('_signal', '').replace('_', ' ').title():
                str(_plr.get(c, 'N/A'))
                for c in _sig_cols
            }
            st.dataframe(
                pd.DataFrame.from_dict(
                    _sig_summary, orient='index',
                    columns=['Signal']
                ),
                use_container_width=True
            )

st.markdown("---")

# ═════════════════════════════════════════════════════════════
# LAYER 3 — DEEP DIVE
# ═════════════════════════════════════════════════════════════

st.markdown("### 🔬 Layer 3 — Deep Dive Analytics")

with st.expander("📊 Annual Returns by Asset"):
    returns_all = pd.DataFrame({
        'NIFTY': nifty_f.pct_change(),
        'SP500': sp500_f.pct_change(),
        'Gold':  gold_f.pct_change(),
        'Crude': crude_f.pct_change(),
    }).dropna()
    annual     = (1 + returns_all).resample('YE').prod() - 1
    annual.index = annual.index.year
    annual_pct = annual * 100
    fig_ann = px.bar(
        annual_pct, barmode='group',
        color_discrete_map={
            'NIFTY': BLUE, 'SP500': MID_BLUE,
            'Gold': ORANGE, 'Crude': "#7030A0"
        },
        title="Annual Returns (%) by Asset Class"
    )
    fig_ann.add_hline(y=0, line_color="black", line_width=0.8)
    fig_ann.update_layout(
        height=420, template="plotly_white",
        yaxis_title="Annual Return (%)"
    )
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
    st.dataframe(
        raw_df.style.format("{:.2f}"),
        use_container_width=True
    )

with st.expander("📈 Rolling Statistics"):
    stat_asset = st.selectbox(
        "Select asset",
        ["NIFTY","SP500","Gold","Crude"]
    )
    stat_map = {
        "NIFTY": nifty_f, "SP500": sp500_f,
        "Gold":  gold_f,  "Crude": crude_f
    }
    s = stat_map[stat_asset].pct_change().dropna() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Daily Return", f"{s.mean():.4f}%")
    c2.metric("Volatility (Ann.)", f"{s.std()*np.sqrt(252):.1f}%")
    c3.metric("Best Day",          f"{s.max():+.2f}%")
    c4.metric("Worst Day",         f"{s.min():+.2f}%")
    sharpe = (s.mean()/s.std())*np.sqrt(252) \
              if s.std() > 0 else 0
    st.metric("Sharpe Ratio (approx)", f"{sharpe:.3f}")

# Footer
st.markdown(f"""
<div style='text-align:center; padding:20px;
     color:{GRAY}; font-size:12px;'>
    Global Macro Intelligence System &nbsp;|&nbsp;
    Built by Niraj Mutha &nbsp;|&nbsp;
    Data: Yahoo Finance, FRED &nbsp;|&nbsp;
    Last updated: {datetime.now().strftime('%d %b %Y %H:%M')}
</div>
""", unsafe_allow_html=True)
