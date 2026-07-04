"""
Oil & Gas Market Analysis Dashboard
===================================
Current market trends + geopolitical-risk context + ML-driven price forecasts.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import data_sources as ds
from src import geopolitical as geo
from src import models as ml

st.set_page_config(
    page_title="Oil & Gas Market & Geopolitical Risk Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .main { background-color: #0e1117; }
      .block-container { padding-top: 1.5rem; }
      div[data-testid="stMetric"] {
          background: #161b26; border: 1px solid #232a38;
          border-radius: 12px; padding: 14px 16px;
      }
      div[data-testid="stMetricLabel"] { color: #9aa4b2; }
      h1, h2, h3 { color: #f5f6fa; }
      .src-badge {
          display:inline-block; padding:3px 10px; border-radius:8px;
          background:#1f2937; color:#9ad; font-size:0.8rem; margin-right:6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=60 * 60 * 6, show_spinner="Fetching market & geopolitical data…")
def load_bundle(force: bool = False) -> dict:
    return ds.load_all(force_refresh=force)


@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def cached_eval(prices, gpr, target, horizon, model_name):
    res = ml.train_evaluate(prices, gpr, target=target, horizon=horizon, model_name=model_name)
    return res


@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def cached_forecast(prices, gpr, target, model_name):
    return ml.forecast_path(prices, gpr, target=target, model_name=model_name)


@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def cached_sarimax(prices, target, steps):
    return ml.sarimax_forecast(prices, target=target, steps=steps)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🛢️ Controls")

if st.sidebar.button("🔄 Refresh live data", width="stretch"):
    load_bundle.clear()
    cached_eval.clear()
    cached_forecast.clear()
    cached_sarimax.clear()
    st.rerun()

bundle = load_bundle()
prices: pd.DataFrame = bundle["prices"]
gpr: pd.DataFrame = bundle["gpr"]

commodity_opts = [c for c in ["WTI", "Brent", "NatGas"] if c in prices.columns]
target = st.sidebar.selectbox("Commodity", commodity_opts, index=0)
unit = "$/MMBtu" if target == "NatGas" else "$/bbl"

model_name = st.sidebar.selectbox("ML model", ml.available_models(), index=0)
horizon = st.sidebar.slider("Back-test forecast horizon (trading days)", 1, 42, 5)

lookback = st.sidebar.select_slider(
    "Chart lookback",
    options=["6M", "1Y", "2Y", "5Y", "Max"],
    value="2Y",
)
_LB = {"6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "Max": len(prices)}
n_lb = min(_LB[lookback], len(prices))

st.sidebar.markdown("---")
st.sidebar.markdown("**Data provenance**")
st.sidebar.markdown(f"Prices: `{bundle['prices_source']}`")
st.sidebar.markdown(f"Geopolitical: `{bundle['gpr_source']}`")
st.sidebar.caption(
    "Sources: FRED (St. Louis Fed), Yahoo Finance, Caldara & Iacoviello GPR "
    "Index (Federal Reserve Board), U.S. EIA."
)


# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("Oil & Gas Market Analysis & Geopolitical Risk Dashboard")
st.caption(
    "Current trends • geopolitical-risk context • machine-learning price forecasts. "
    f"Latest data point: **{pd.to_datetime(prices.index[-1]).date()}**"
)

snap = geo.current_risk_snapshot(gpr)

k1, k2, k3, k4, k5 = st.columns(5)


def _delta(series: pd.Series, periods: int = 1) -> float:
    if len(series) <= periods:
        return 0.0
    return (series.iloc[-1] / series.iloc[-1 - periods] - 1) * 100


for col, label in zip((k1, k2, k3), commodity_opts):
    s = prices[label].dropna()
    u = "$/MMBtu" if label == "NatGas" else "$/bbl"
    col.metric(
        f"{label} ({u})",
        f"{s.iloc[-1]:,.2f}",
        f"{_delta(s, 1):+.2f}% d/d",
    )

k4.metric(
    "Geopolitical Risk (30d avg)",
    f"{snap['ma30']:.0f}",
    f"{snap['change_30d_%']:+.1f}% (30d)",
    delta_color="inverse",
)
k5.metric("Risk Regime", snap["regime"], f"{snap['percentile']:.0f}th pct")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_overview, tab_geo, tab_ml, tab_data = st.tabs(
    ["📈 Market Overview", "🌍 Geopolitical Risk", "🤖 ML Forecast", "🗄️ Data & Sources"]
)

# =========================== MARKET OVERVIEW =============================== #
with tab_overview:
    px_lb = prices.iloc[-n_lb:]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{target} price ({unit})", "Daily return %"),
    )
    fig.add_trace(
        go.Scatter(
            x=px_lb.index, y=px_lb[target], name=target,
            line=dict(color="#f39c12", width=2),
        ),
        row=1, col=1,
    )
    for ma, color in ((21, "#3498db"), (63, "#9b59b6")):
        fig.add_trace(
            go.Scatter(
                x=px_lb.index, y=px_lb[target].rolling(ma).mean(),
                name=f"MA{ma}", line=dict(width=1, color=color),
            ),
            row=1, col=1,
        )
    ret = px_lb[target].pct_change() * 100
    fig.add_trace(
        go.Bar(x=px_lb.index, y=ret, name="Return %",
               marker_color=np.where(ret >= 0, "#2ecc71", "#e74c3c")),
        row=2, col=1,
    )
    fig.update_layout(
        template="plotly_dark", height=560, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.08), showlegend=True,
    )
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Normalized performance (rebased to 100)")
        norm = px_lb / px_lb.iloc[0] * 100
        nfig = go.Figure()
        palette = ["#f39c12", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
        for i, c in enumerate(norm.columns):
            nfig.add_trace(go.Scatter(x=norm.index, y=norm[c], name=c,
                                      line=dict(color=palette[i % len(palette)])))
        nfig.update_layout(template="plotly_dark", height=340,
                           margin=dict(l=10, r=10, t=10, b=10),
                           legend=dict(orientation="h", y=1.12))
        st.plotly_chart(nfig, width="stretch")

    with c2:
        st.subheader("Trend & volatility snapshot")
        s = prices[target].dropna()
        stats = {
            "Spot": f"{s.iloc[-1]:,.2f} {unit}",
            "1M change": f"{_delta(s, 21):+.2f}%",
            "3M change": f"{_delta(s, 63):+.2f}%",
            "1Y change": f"{_delta(s, 252):+.2f}%",
            "Ann. volatility": f"{s.pct_change().std() * np.sqrt(252) * 100:.1f}%",
            "52-wk high": f"{s.tail(252).max():,.2f}",
            "52-wk low": f"{s.tail(252).min():,.2f}",
        }
        st.table(pd.DataFrame(stats.items(), columns=["Metric", "Value"]))
        if "WTI" in prices and "Brent" in prices:
            spread = (prices["Brent"] - prices["WTI"]).iloc[-1]
            st.info(f"Brent–WTI spread: **{spread:+.2f} $/bbl**")

# =========================== GEOPOLITICAL RISK ============================= #
with tab_geo:
    corr = geo.correlation_analysis(prices, gpr, target=target)

    st.markdown(geo.risk_adjusted_outlook(snap, corr))

    g_lb = gpr.iloc[-min(n_lb, len(gpr)):]
    gfig = make_subplots(specs=[[{"secondary_y": True}]])
    gfig.add_trace(
        go.Scatter(x=g_lb.index, y=g_lb["GPR"], name="GPR Index",
                   line=dict(color="#e74c3c", width=1.5), fill="tozeroy",
                   fillcolor="rgba(231,76,60,0.12)"),
        secondary_y=False,
    )
    px_al = prices[target].reindex(g_lb.index).ffill()
    gfig.add_trace(
        go.Scatter(x=g_lb.index, y=px_al, name=f"{target} price",
                   line=dict(color="#f39c12", width=1.8)),
        secondary_y=True,
    )
    # risk band shading
    for lo, hi, label, color in geo.RISK_BANDS:
        if np.isfinite(hi):
            gfig.add_hrect(y0=lo, y1=hi, line_width=0, fillcolor=color, opacity=0.05,
                           secondary_y=False)
    gfig.update_layout(template="plotly_dark", height=440,
                       margin=dict(l=10, r=10, t=30, b=10),
                       title="Geopolitical Risk vs. price",
                       legend=dict(orientation="h", y=1.12))
    gfig.update_yaxes(title_text="GPR index", secondary_y=False)
    gfig.update_yaxes(title_text=f"{target} ({unit})", secondary_y=True)
    st.plotly_chart(gfig, width="stretch")

    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.subheader("Lead/lag correlation")
        st.caption(
            "Correlation of GPR change with future price returns. "
            "Positive lag = GPR leads price."
        )
        ll = corr["lead_lag"]
        lfig = go.Figure(go.Bar(
            x=[f"{k:+d}d" for k in ll.index], y=ll.values,
            marker_color=np.where(ll.values >= 0, "#3498db", "#e67e22"),
        ))
        lfig.update_layout(template="plotly_dark", height=320,
                           margin=dict(l=10, r=10, t=10, b=10),
                           yaxis_title="correlation")
        st.plotly_chart(lfig, width="stretch")

    with c2:
        st.subheader("Current risk gauge")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=snap["ma30"],
            delta={"reference": snap["mean_all"]},
            gauge={
                "axis": {"range": [0, 300]},
                "bar": {"color": snap["color"]},
                "steps": [
                    {"range": [0, 50], "color": "#16331f"},
                    {"range": [50, 100], "color": "#3a3413"},
                    {"range": [100, 150], "color": "#3d2a12"},
                    {"range": [150, 250], "color": "#3d1715"},
                    {"range": [250, 300], "color": "#2c163d"},
                ],
            },
            title={"text": f"GPR regime: {snap['regime']}"},
        ))
        gauge.update_layout(template="plotly_dark", height=320,
                            margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(gauge, width="stretch")

    st.subheader("Recent geopolitical risk spikes (rolling z-score > 2)")
    spikes = geo.detect_spikes(gpr).tail(15)
    if spikes.empty:
        st.write("No significant spikes detected in the available window.")
    else:
        st.dataframe(
            spikes.assign(date=spikes.index.date).round(2),
            width="stretch", hide_index=True,
        )

# =============================== ML FORECAST =============================== #
with tab_ml:
    st.subheader(f"{model_name} — forward price path for {target}")
    st.caption(
        "Direct multi-horizon forecasting using engineered price + geopolitical-risk "
        "features. Shaded band = 95% interval from back-tested residuals."
    )

    try:
        fc = cached_forecast(prices, gpr, target, model_name)
        hist = prices[target].iloc[-180:]

        ffig = go.Figure()
        ffig.add_trace(go.Scatter(x=hist.index, y=hist, name="History",
                                  line=dict(color="#7f8c8d")))
        ffig.add_trace(go.Scatter(
            x=fc.index, y=fc["upper"], name="Upper 95%",
            line=dict(width=0), showlegend=False))
        ffig.add_trace(go.Scatter(
            x=fc.index, y=fc["lower"], name="95% interval",
            fill="tonexty", fillcolor="rgba(243,156,18,0.18)",
            line=dict(width=0)))
        ffig.add_trace(go.Scatter(
            x=fc.index, y=fc["forecast"], name="ML forecast",
            line=dict(color="#f39c12", width=2.5, dash="dot"),
            mode="lines+markers"))

        # optional SARIMAX baseline
        sar = cached_sarimax(prices, target, 63)
        if sar is not None:
            ffig.add_trace(go.Scatter(
                x=sar.index, y=sar["forecast"], name="SARIMAX baseline",
                line=dict(color="#3498db", width=1.5, dash="dash")))

        ffig.update_layout(template="plotly_dark", height=460,
                           margin=dict(l=10, r=10, t=10, b=10),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(ffig, width="stretch")

        if len(fc) > 1:
            spot = fc["forecast"].iloc[0]
            cols = st.columns(min(5, len(fc) - 1))
            for i, (idx, row) in enumerate(fc.iloc[1:].iterrows()):
                if i >= len(cols):
                    break
                chg = (row["forecast"] / spot - 1) * 100
                cols[i].metric(
                    f"+{int(row['horizon_days'])}d ({idx.date()})",
                    f"{row['forecast']:,.2f}",
                    f"{chg:+.1f}%",
                )
    except Exception as e:
        st.error(f"Forecast unavailable: {e}")

    st.markdown("---")
    st.subheader(f"Back-test ({horizon}-day horizon)")
    try:
        res = cached_eval(prices, gpr, target, horizon, model_name)
        m = res.metrics
        mc = st.columns(5)
        mc[0].metric("MAE", f"{m['MAE']:.2f}")
        mc[1].metric("RMSE", f"{m['RMSE']:.2f}")
        mc[2].metric("MAPE", f"{m['MAPE_%']:.2f}%")
        mc[3].metric("R²", f"{m['R2']:.3f}")
        mc[4].metric("Directional acc.", f"{m['DirAcc_%']:.1f}%")

        bt = go.Figure()
        bt.add_trace(go.Scatter(x=res.test_index, y=res.y_test.values,
                                name="Actual", line=dict(color="#2ecc71")))
        bt.add_trace(go.Scatter(x=res.test_index, y=res.y_pred,
                                name="Predicted", line=dict(color="#f39c12", dash="dot")))
        bt.update_layout(template="plotly_dark", height=340,
                         margin=dict(l=10, r=10, t=10, b=10),
                         title="Out-of-sample: actual vs. predicted",
                         legend=dict(orientation="h", y=1.12))
        st.plotly_chart(bt, width="stretch")

        if not res.feature_importance.empty:
            st.subheader("Top predictive features")
            top = res.feature_importance.head(15)[::-1]
            ifig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                    marker_color="#3498db"))
            ifig.update_layout(template="plotly_dark", height=420,
                               margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(ifig, width="stretch")
            st.caption(
                "Note: feature importance is associative, not causal. Geopolitical-risk "
                "(GPR*) features appearing high indicates risk carries predictive signal."
            )
    except Exception as e:
        st.error(f"Back-test unavailable: {e}")

    st.warning(
        "⚠️ Forecasts are statistical estimates from historical patterns and the "
        "geopolitical-risk index. Energy markets are driven by shocks (OPEC+ decisions, "
        "conflicts, weather) that models cannot foresee. **Not investment advice.**"
    )

# ============================ DATA & SOURCES =============================== #
with tab_data:
    st.subheader("Verified data sources")
    st.markdown(
        """
| # | Source | What it provides | Access | Link |
|---|--------|------------------|--------|------|
| 1 | **FRED — Federal Reserve Bank of St. Louis** | WTI (`DCOILWTICO`), Brent (`DCOILBRENTEU`), Henry Hub gas (`DHHNGSP`) spot prices | Public, no key | https://fred.stlouisfed.org/ |
| 2 | **Yahoo Finance** (`yfinance`) | WTI/Brent/NatGas futures, XLE & USO energy funds | Public | https://finance.yahoo.com/ |
| 3 | **Geopolitical Risk (GPR) Index** — Caldara & Iacoviello, Federal Reserve Board | Daily geopolitical-risk index + threat/act components | Public | https://www.matteoiacoviello.com/gpr.htm |
| 4 | **U.S. Energy Information Administration (EIA) API v2** | Official production, inventories, refinery & price data | Free API key | https://www.eia.gov/opendata/ |
| 5 | **World Bank Commodity Markets ("Pink Sheet")** | Monthly global commodity price benchmarks | Public | https://www.worldbank.org/en/research/commodity-markets |
| 6 | **OPEC Monthly Oil Market Report** | Supply/demand balances, OPEC+ policy | Public | https://www.opec.org/ |
        """
    )
    st.markdown(
        f"<span class='src-badge'>Prices: {bundle['prices_source']}</span>"
        f"<span class='src-badge'>GPR: {bundle['gpr_source']}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "GPR methodology: Caldara, D. & Iacoviello, M. (2022), "
        "“Measuring Geopolitical Risk,” American Economic Review 112(4)."
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Price data (tail)")
        st.dataframe(prices.tail(12).round(3), width="stretch")
        st.download_button("⬇️ Download prices CSV", prices.to_csv(),
                           "oil_gas_prices.csv", "text/csv")
    with c2:
        st.subheader("Geopolitical risk (tail)")
        st.dataframe(gpr.tail(12).round(2), width="stretch")
        st.download_button("⬇️ Download GPR CSV", gpr.to_csv(),
                           "geopolitical_risk.csv", "text/csv")

st.markdown("---")
st.caption(
    "Built with Python · pandas · scikit-learn · XGBoost · statsmodels · Streamlit · Plotly. "
    "For research/educational use — not financial advice."
)
