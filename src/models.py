"""
Machine-learning forecasting models for oil & gas prices.

Strategy
--------
* Supervised regression on engineered features (see features.py).
* Time-series-aware train/test split (NO shuffling -> no look-ahead leakage).
* Multiple algorithms: XGBoost, Random Forest, Gradient Boosting, Ridge.
* Classical SARIMAX baseline (statsmodels) on the target series alone.
* Direct multi-horizon forecasting: a separate model per horizon is trained so
  a forward price path can be produced without needing future GPR values.

Metrics reported: MAE, RMSE, MAPE, R^2, and directional accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_feature_frame

# Cap training history to a recent, regime-relevant window. FRED ships ~40y of
# daily data; using the last ~12 years keeps models fast and current.
MAX_HISTORY = 3000

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
def get_model(name: str):
    name = name.lower()
    if name in ("xgboost", "xgb") and _HAS_XGB:
        return XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
        )
    if name in ("randomforest", "rf"):
        return RandomForestRegressor(
            n_estimators=200, max_depth=16, min_samples_leaf=3,
            n_jobs=-1, random_state=42,
        )
    if name in ("gradientboosting", "gbr"):
        return GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    # default
    if _HAS_XGB:
        return get_model("xgboost")
    return get_model("randomforest")


def available_models() -> list[str]:
    models = ["RandomForest", "GradientBoosting", "Ridge"]
    if _HAS_XGB:
        models.insert(0, "XGBoost")
    return models


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-9
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _directional_accuracy(last_known, y_true, y_pred) -> float:
    """Share of forecasts that get the up/down direction vs. today's price right."""
    true_dir = np.sign(np.asarray(y_true) - last_known)
    pred_dir = np.sign(np.asarray(y_pred) - last_known)
    if len(true_dir) == 0:
        return float("nan")
    return float(np.mean(true_dir == pred_dir) * 100)


@dataclass
class EvalResult:
    model_name: str
    horizon: int
    metrics: dict
    test_index: pd.DatetimeIndex
    y_test: pd.Series
    y_pred: np.ndarray
    feature_importance: pd.Series = field(default_factory=pd.Series)


# --------------------------------------------------------------------------- #
# Train + evaluate (back-test on a held-out tail)
# --------------------------------------------------------------------------- #
def train_evaluate(
    prices: pd.DataFrame,
    gpr: pd.DataFrame,
    target: str = "WTI",
    horizon: int = 5,
    model_name: str = "XGBoost",
    test_frac: float = 0.2,
) -> EvalResult:
    prices = prices.iloc[-MAX_HISTORY:]
    X, y, feat_names = build_feature_frame(prices, gpr, target=target, horizon=horizon)
    if len(X) < 60:
        raise ValueError("Not enough data to train a model.")

    split = int(len(X) * (1 - test_frac))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = get_model(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # "today's" price aligned to each test row (the level feature)
    level_col = f"{target}_level"
    last_known = X_test[level_col].values if level_col in X_test else y_test.values

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": rmse,
        "MAPE_%": _mape(y_test, y_pred),
        "R2": float(r2_score(y_test, y_pred)),
        "DirAcc_%": _directional_accuracy(last_known, y_test.values, y_pred),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    importance = pd.Series(dtype=float)
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feat_names).sort_values(
            ascending=False
        )

    return EvalResult(
        model_name=model_name,
        horizon=horizon,
        metrics=metrics,
        test_index=X_test.index,
        y_test=y_test,
        y_pred=y_pred,
        feature_importance=importance,
    )


# --------------------------------------------------------------------------- #
# Forward forecast path (direct multi-horizon)
# --------------------------------------------------------------------------- #
def forecast_path(
    prices: pd.DataFrame,
    gpr: pd.DataFrame,
    target: str = "WTI",
    model_name: str = "XGBoost",
    horizons: tuple[int, ...] = (1, 5, 10, 21, 42, 63),
) -> pd.DataFrame:
    """
    Train one model per horizon and predict the target price at each horizon
    from the most recent feature row. Returns a DataFrame indexed by future
    business date with columns [forecast, lower, upper].
    """
    # The forecast trains many models (2 fits x len(horizons)); use a tighter
    # window than the back-test for responsiveness.
    prices = prices.iloc[-min(MAX_HISTORY, 1800):]
    last_date = pd.to_datetime(prices.index[-1])
    spot = float(prices[target].iloc[-1])

    rows = []
    for h in horizons:
        X, y, _ = build_feature_frame(prices, gpr, target=target, horizon=h)
        if len(X) < 60:
            continue
        # back-test residual std for a rough uncertainty band
        split = int(len(X) * 0.85)
        model = get_model(model_name)
        model.fit(X.iloc[:split], y.iloc[:split])
        resid = y.iloc[split:].values - model.predict(X.iloc[split:])
        sigma = float(np.std(resid)) if len(resid) > 5 else spot * 0.05

        # refit on all data, predict from the latest feature row
        model.fit(X, y)
        latest = X.iloc[[-1]]
        pred = float(model.predict(latest)[0])

        future_date = last_date + pd.tseries.offsets.BDay(h)
        rows.append(
            {
                "horizon_days": h,
                "date": future_date,
                "forecast": pred,
                "lower": pred - 1.96 * sigma,
                "upper": pred + 1.96 * sigma,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["forecast", "lower", "upper"])

    out = pd.DataFrame(rows).set_index("date")
    # prepend spot as t0 for a continuous line
    t0 = pd.DataFrame(
        {"horizon_days": 0, "forecast": spot, "lower": spot, "upper": spot},
        index=[last_date],
    )
    return pd.concat([t0, out]).sort_index()


# --------------------------------------------------------------------------- #
# SARIMAX classical baseline
# --------------------------------------------------------------------------- #
def sarimax_forecast(
    prices: pd.DataFrame, target: str = "WTI", steps: int = 63
) -> pd.DataFrame | None:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        return None
    try:
        s = prices[target].iloc[-MAX_HISTORY:].astype(float).asfreq("B").ffill().dropna()
        s = s.clip(lower=1e-3)  # guard log of non-positive
        # work in log space for positivity & stability
        ls = np.log(s)
        model = SARIMAX(
            ls, order=(2, 1, 2), trend="c",
            enforce_stationarity=False, enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=steps)
        mean = np.exp(fc.predicted_mean)
        ci = np.exp(fc.conf_int(alpha=0.05))
        out = pd.DataFrame(
            {
                "forecast": mean.values,
                "lower": ci.iloc[:, 0].values,
                "upper": ci.iloc[:, 1].values,
            },
            index=mean.index,
        )
        return out
    except Exception:
        return None
