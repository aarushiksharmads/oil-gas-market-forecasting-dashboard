"""
Feature engineering for oil & gas price forecasting.

Builds a supervised-learning matrix from price history and the geopolitical
risk (GPR) index: calendar features, autoregressive lags, rolling statistics,
returns/volatility, and lagged geopolitical-risk signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_frame(
    prices: pd.DataFrame,
    gpr: pd.DataFrame,
    target: str = "WTI",
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Construct features X and target y = target price `horizon` trading days ahead.

    Returns (X, y, feature_names).
    """
    df = prices.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Align GPR (often lower frequency / different calendar) onto price dates.
    g = gpr.copy()
    g.index = pd.to_datetime(g.index)
    g = g.sort_index().reindex(df.index.union(g.index)).ffill().reindex(df.index)

    feat = pd.DataFrame(index=df.index)

    # --- price-based features for every available commodity ---------------- #
    for col in df.columns:
        s = df[col]
        feat[f"{col}_ret1"] = s.pct_change()
        feat[f"{col}_ret5"] = s.pct_change(5)
        feat[f"{col}_vol10"] = s.pct_change().rolling(10).std()

    # --- richer features for the target series ----------------------------- #
    s = df[target]
    feat[f"{target}_level"] = s
    for lag in (1, 2, 3, 5, 10, 21):
        feat[f"{target}_lag{lag}"] = s.shift(lag)
    feat[f"{target}_ma5"] = s.rolling(5).mean()
    feat[f"{target}_ma21"] = s.rolling(21).mean()
    feat[f"{target}_ma63"] = s.rolling(63).mean()
    feat[f"{target}_ma_ratio"] = feat[f"{target}_ma5"] / feat[f"{target}_ma21"]
    feat[f"{target}_mom21"] = s.pct_change(21)

    # crack/spread features if both crudes present
    if "WTI" in df.columns and "Brent" in df.columns:
        feat["WTI_Brent_spread"] = df["Brent"] - df["WTI"]

    # --- geopolitical-risk features ---------------------------------------- #
    for col in g.columns:
        feat[f"{col}"] = g[col]
        feat[f"{col}_ma7"] = g[col].rolling(7).mean()
        feat[f"{col}_ma30"] = g[col].rolling(30).mean()
        feat[f"{col}_chg5"] = g[col].diff(5)
        for lag in (1, 5, 10):
            feat[f"{col}_lag{lag}"] = g[col].shift(lag)

    # --- calendar features ------------------------------------------------- #
    feat["dow"] = feat.index.dayofweek
    feat["month"] = feat.index.month
    feat["quarter"] = feat.index.quarter

    # --- target: future level -------------------------------------------- #
    y = s.shift(-horizon)
    y.name = f"{target}_t+{horizon}"

    data = feat.join(y).replace([np.inf, -np.inf], np.nan).dropna()
    feature_names = [c for c in data.columns if c != y.name]
    X = data[feature_names]
    y_clean = data[y.name]
    return X, y_clean, feature_names
