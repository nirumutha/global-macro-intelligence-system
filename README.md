# Global Macro Intelligence System (GMIS)
### A Professional Multi-Asset Signal Framework | Built by Niraj Mutha

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_Dashboard-red)
![Assets](https://img.shields.io/badge/Assets-5_Markets-green)
![Data](https://img.shields.io/badge/Data-11_Years-orange)

---

## 🎯 What This System Does

A fully automated macro intelligence system that monitors five global 
asset classes in real time, generates daily Long/Short/Neutral signals, 
and presents everything in a professional three-layer Streamlit dashboard.

Built entirely in Python using free data sources — no Bloomberg terminal, 
no paid APIs.

---

## 📊 Live Dashboard

The dashboard updates every morning with:
- Fresh market prices (Yahoo Finance)
- Live news sentiment from 8 RSS feeds (NLP-scored)
- Recalculated signals for all 5 assets
- Macro regime classification (Goldilocks / Overheating / Stagflation / Recession)

**Markets covered:** NIFTY 50 | S&P 500 | Gold | Silver | Crude WTI

---

## 🏆 Verified Backtest Results (2015–2026)

| Metric | Signal Strategy | Buy & Hold | Improvement |
|--------|----------------|------------|-------------|
| Sharpe Ratio (NIFTY) | 0.76 | 0.69 | +0.07 |
| Max Drawdown (NIFTY) | -17.1% | -38.4% | **+21.4pp saved** |
| Portfolio Sharpe | 0.59 | 0.24 | **2.5x better** |
| Portfolio CAGR | +5.3% | +1.9% | +3.4% |
| Portfolio Max DD | -22.4% | -81.2% | **+58.8pp saved** |

*Transaction cost: 0.1% per trade. No leverage. Conservative shorting = cash only.*

---

## 🧠 System Architecture — 10 Analytical Modules

| Module | What It Does |
|--------|-------------|
| 01 Data Collection | Downloads daily prices for 10 assets via Yahoo Finance |
| 02 Macro Data | Pulls CPI, GDP, Fed Rate, Payrolls, Yields from FRED API |
| 03 Database | SQLite database — 18 tables, 46,000+ rows |
| 04 Correlation Engine | Rolling cross-asset correlation matrix |
| 05 Regime Detection | Bull/Bear/Crisis/Sideways classification |
| 06 FX & Bonds | Yield curve analysis, USD/INR, DXY signals |
| 07 Institutional Flows | Volume analysis and flow indicators |
| 08 Volatility | VIX regime classification and crisis detection |
| 09 Sentiment Engine | Context-aware NLP scoring of live headlines |
| 10 Signal Engine | Composite Long/Short/Neutral signal generator |
| 11 Forecasting | Prophet + ARIMA 30/90-day price forecasts |
| 12 Backtesting | Full walk-forward backtest with Sharpe, Sortino, drawdown |

---

## 📈 Signal Engine — How Signals Are Generated

Each signal combines 7 components with weighted scoring:
```
Signal Score = 
  MA Momentum     (0.20) — Triple moving average trend confirmation
  RSI Counter     (0.15) — Mean reversion, prevents chasing overbought markets  
  ROC Short-term  (0.10) — Rate of change, reduces signal lag
  VIX Filter      (0.20) — Volatility regime overlay
  Macro Regime    (0.20) — CPI + GDP regime classification
  Yield Curve     (0.10) — 10Y-2Y spread signal
  FX Signal       (0.05) — USD/INR and DXY momentum
  Sentiment       (0.05) — Live NLP score adjustment
```

Score >= +0.15 → **Long** | Score <= -0.15 → **Short** | Between → **Neutral**

---

## 🔴 Real-Time Event Detection

On **28 February 2026**, the system detected the US-Israel strikes on 
Iran in real time through live RSS sentiment scoring — flagging the 
following signals before markets opened:

- 🟢 Gold: **LONG** (safe haven demand)
- 🟢 Crude WTI: **LONG** (Iran supply disruption)
- 🔴 NIFTY 50: **SHORT** (India imports 85% of oil)
- ⚠️ S&P 500: **NEUTRAL** (uncertainty)

All signals were directionally correct when markets opened.

---

## 📁 Project Structure
```
GlobalMacroSystem/
│
├── 01_data_collection.py      # Market price downloads
├── 02_macro_data.py           # FRED macro data
├── 03_database_setup.py       # SQLite database builder
├── 04_correlation_analysis.py # Correlation engine
├── 05_regime_detection.py     # Market regime classifier
├── 06_fx_bonds_analysis.py    # FX and yield curve
├── 07_institutional_flows.py  # Flow analysis
├── 08_volatility_analysis.py  # VIX and volatility
├── 09_sentiment_engine.py     # NLP sentiment scoring
├── 10_macro_scorecard.py      # Macro regime scorecard
├── 11_forecasting.py          # Prophet + ARIMA models
├── 12_signal_engine.py        # Signal generation
├── 13_backtesting.py          # Full backtest suite
├── refresh_daily.py           # Daily data refresh
├── dashboard.py               # Streamlit dashboard
│
├── data/                      # SQLite database + CSVs
├── outputs/                   # 40 professional charts
└── venv/                      # Python virtual environment
```

---

## 🚀 How to Run

**Daily refresh (run every morning):**
```bash
python refresh_daily.py
streamlit run dashboard.py
```

**First time setup:**
```bash
git clone https://github.com/nirumutha/global-macro-intelligence-system.git
cd global-macro-intelligence-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python 01_data_collection.py
python 02_macro_data.py
python 03_database_setup.py
streamlit run dashboard.py
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Core language |
| pandas / numpy | Data manipulation |
| yfinance | Market price data |
| fredapi | Macro economic data |
| SQLite | Local database |
| Prophet / ARIMA | Forecasting models |
| VADER + feedparser | NLP sentiment engine |
| Streamlit | Live dashboard |
| Plotly | Interactive charts |
| matplotlib | Static chart outputs |

---

## 📊 Output Gallery

40 professional charts across 13 modules including:
- Cross-asset correlation heatmaps
- Regime detection overlays
- Yield curve analysis
- Sentiment dashboards
- Prophet and ARIMA forecasts
- Backtest equity curves and drawdown analysis

---

## 👤 About

**Niraj Mutha**
Built as part of a structured 15-week self-directed learning programme
in quantitative finance and macro analysis.

[GitHub](https://github.com/nirumutha) | [https://www.linkedin.com/in/nirajmutha/](#)

---

*Data sources: Yahoo Finance, FRED (Federal Reserve Economic Data)*
*Disclaimer: This system is for educational purposes only. 
Not financial advice.*