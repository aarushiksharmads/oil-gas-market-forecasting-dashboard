# 🛢️ Oil & Gas Market Analysis & Geopolitical Risk Dashboard

An interactive Python dashboard that analyses **current oil & gas market trends**,
puts them in the context of **world geopolitical risk**, and produces
**machine-learning price forecasts** for WTI crude, Brent crude, and Henry Hub
natural gas.

Built with **Python 3.14**, `pandas`, `scikit-learn`, `XGBoost`, `statsmodels`,
`Streamlit`, and `Plotly`.

---

## What it does

| Tab | Content |
|-----|---------|
| 📈 **Market Overview** | Live prices, moving averages, daily returns, rebased performance, volatility & trend stats, Brent–WTI spread. |
| 🌍 **Geopolitical Risk** | The Caldara–Iacoviello **Geopolitical Risk (GPR) Index** overlaid on prices, a risk-regime gauge, lead/lag correlation analysis, and automatic detection of risk spikes. |
| 🤖 **ML Forecast** | Forward price path (1–63 trading days) with uncertainty bands, an out-of-sample **back-test** (MAE / RMSE / MAPE / R² / directional accuracy), a SARIMAX baseline, and feature-importance ranking. |
| 🗄️ **Data & Sources** | Full provenance of every data series + CSV downloads. |

---

## Verified data sources

All series come from **public, citable, authoritative** providers. Nothing is scraped from anonymous sites.

| # | Source | What it provides | Access | Link |
|---|--------|------------------|--------|------|
| 1 | **FRED — Federal Reserve Bank of St. Louis** | Spot prices: WTI (`DCOILWTICO`), Brent (`DCOILBRENTEU`), Henry Hub natural gas (`DHHNGSP`) | Public, no key | https://fred.stlouisfed.org/ |
| 2 | **Yahoo Finance** (`yfinance`) | WTI/Brent/NatGas futures (`CL=F`,`BZ=F`,`NG=F`), energy funds `XLE` & `USO` | Public | https://finance.yahoo.com/ |
| 3 | **Geopolitical Risk (GPR) Index — Caldara & Iacoviello (Federal Reserve Board)** | Daily geopolitical-risk index + *threat* and *act* sub-indices | Public | https://www.matteoiacoviello.com/gpr.htm |
| 4 | **U.S. Energy Information Administration (EIA) API v2** | Official U.S. production, inventories, refinery utilisation, prices | Free API key (optional) | https://www.eia.gov/opendata/ |
| 5 | **World Bank Commodity Markets ("Pink Sheet")** | Monthly global commodity benchmarks | Public | https://www.worldbank.org/en/research/commodity-markets |
| 6 | **OPEC Monthly Oil Market Report** | Supply/demand balances, OPEC+ policy | Public | https://www.opec.org/ |

**GPR methodology citation:** Caldara, Dario and Matteo Iacoviello (2022),
*"Measuring Geopolitical Risk,"* **American Economic Review**, 112(4), 1194–1225.
The GPR index counts the share of newspaper articles discussing adverse
geopolitical events; it is widely used by central banks and the IMF.

> The app fetches **live** data on launch and caches it in `./data`. If a source
> is temporarily unreachable it falls back to a clearly-labelled synthetic series
> so the dashboard always runs. The active source is shown in the sidebar and the
> *Data & Sources* tab.

---

## Quick start (Windows / PowerShell)

```powershell
cd oil-gas-dashboard

# 1. Create & activate a virtual environment (Python 3.11–3.14)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

On macOS / Linux:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

### Optional: enable the official EIA data feed

Get a free key at https://www.eia.gov/opendata/register.php, then:

```powershell
$env:EIA_API_KEY = "your_key_here"   # PowerShell
# export EIA_API_KEY=your_key_here   # bash
```

---

## How the forecasting works

1. **Feature engineering** (`src/features.py`) — autoregressive lags, rolling means,
   momentum, realised volatility, the Brent–WTI spread, and **lagged
   geopolitical-risk** features (GPR level, 7/30-day averages, changes, lags).
2. **Models** (`src/models.py`):
   - **XGBoost** (default), **Random Forest**, **Gradient Boosting**, **Ridge**.
   - **Direct multi-horizon** strategy — a separate model is trained per horizon, so
     a forward path can be produced without needing *future* geopolitical values.
   - **SARIMAX** (statsmodels) classical baseline for comparison.
   - **Time-series-aware** train/test split (no shuffling → no look-ahead leakage).
3. **Evaluation** — MAE, RMSE, MAPE, R², and **directional accuracy** on a held-out
   tail, plus feature-importance ranking to show how much signal the
   geopolitical-risk features carry.

---

## Project structure

```
oil-gas-dashboard/
├── app.py                  # Streamlit dashboard (UI)
├── requirements.txt
├── README.md
├── .streamlit/config.toml  # dark theme
├── data/                   # auto-created cache (parquet/csv)
└── src/
    ├── data_sources.py     # FRED / Yahoo / GPR / EIA fetching + caching + fallback
    ├── features.py         # feature engineering
    ├── models.py           # ML + SARIMAX forecasting & back-testing
    └── geopolitical.py     # risk regime, correlation, lead/lag, spike detection
```

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. Forecasts are
statistical estimates derived from historical patterns and the geopolitical-risk
index. Energy markets are driven by shocks (OPEC+ decisions, conflicts, weather,
policy) that models cannot foresee. **This is not investment advice.**
