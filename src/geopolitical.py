"""
Geopolitical risk analysis: quantify how world-event risk (GPR index) relates
to oil & gas prices, classify the current risk regime, and measure lead/lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RISK_BANDS = [
    (0, 50, "Low", "#2ecc71"),
    (50, 100, "Normal", "#f1c40f"),
    (100, 150, "Elevated", "#e67e22"),
    (150, 250, "High", "#e74c3c"),
    (250, np.inf, "Extreme", "#8e44ad"),
]


def classify_regime(value: float) -> tuple[str, str]:
    for lo, hi, label, color in RISK_BANDS:
        if lo <= value < hi:
            return label, color
    return "Extreme", "#8e44ad"


def current_risk_snapshot(gpr: pd.DataFrame) -> dict:
    s = gpr["GPR"].dropna()
    latest = float(s.iloc[-1])
    ma30 = float(s.tail(30).mean())
    pctile = float((s <= latest).mean() * 100)
    label, color = classify_regime(ma30)

    # 30-day change
    if len(s) > 31:
        chg30 = float((s.iloc[-1] - s.iloc[-31]) / max(abs(s.iloc[-31]), 1e-9) * 100)
    else:
        chg30 = float("nan")

    return {
        "latest": latest,
        "ma30": ma30,
        "percentile": pctile,
        "regime": label,
        "color": color,
        "change_30d_%": chg30,
        "mean_all": float(s.mean()),
    }


def correlation_analysis(
    prices: pd.DataFrame, gpr: pd.DataFrame, target: str = "WTI"
) -> dict:
    """Contemporaneous correlation of GPR with price returns + lead/lag profile."""
    px = prices[target].astype(float)
    g = gpr["GPR"].astype(float)

    df = pd.concat([px.rename("price"), g.rename("gpr")], axis=1).ffill().dropna()
    ret = df["price"].pct_change().replace([np.inf, -np.inf], np.nan)
    # Use absolute change in GPR (robust to near-zero values vs. pct_change).
    grisk_chg = df["gpr"].diff().replace([np.inf, -np.inf], np.nan)

    def _safe_corr(a: pd.Series, b: pd.Series) -> float:
        pair = pd.concat([a, b], axis=1).dropna()
        if len(pair) < 10 or pair.iloc[:, 0].std() == 0 or pair.iloc[:, 1].std() == 0:
            return float("nan")
        return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))

    contemp = _safe_corr(ret, grisk_chg)

    # lead/lag: does a GPR move today predict price returns k days later?
    leadlag = {k: _safe_corr(ret, grisk_chg.shift(k))
               for k in (-10, -5, -3, -1, 0, 1, 3, 5, 10)}

    ll = pd.Series(leadlag)
    best_lag = int(ll.abs().idxmax()) if ll.abs().notna().any() else 0

    return {
        "contemporaneous_corr": contemp,
        "lead_lag": ll,
        "best_lag": best_lag,
        "best_lag_corr": float(ll.loc[best_lag]),
        "level_corr": float(df["price"].corr(df["gpr"])),
    }


def detect_spikes(gpr: pd.DataFrame, window: int = 30, z_thresh: float = 2.0) -> pd.DataFrame:
    """Flag dates where GPR exceeds rolling mean by z_thresh standard deviations."""
    s = gpr["GPR"].astype(float)
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std()
    z = (s - roll_mean) / roll_std
    spikes = gpr.loc[z > z_thresh, ["GPR"]].copy()
    spikes["z_score"] = z[z > z_thresh]
    return spikes.sort_index()


def risk_adjusted_outlook(snapshot: dict, corr: dict) -> str:
    """A short human-readable interpretation of the current risk picture."""
    regime = snapshot["regime"]
    pctile = snapshot["percentile"]
    bias = "upward" if corr["level_corr"] > 0 else "downward"
    strength = abs(corr["level_corr"])
    link = "strong" if strength > 0.5 else "moderate" if strength > 0.25 else "weak"

    return (
        f"Geopolitical risk is currently **{regime}** "
        f"(~{pctile:.0f}th historical percentile). "
        f"The historical link between geopolitical risk and {('crude' )} prices is "
        f"**{link}** (level corr {corr['level_corr']:+.2f}), implying a {bias} price "
        f"bias when risk is elevated. Best predictive lag observed: "
        f"{corr['best_lag']} day(s) (corr {corr['best_lag_corr']:+.2f})."
    )
