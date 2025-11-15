"""Forecasting utilities for Algorzen Helix.

Contains simple baseline (rolling mean), and a scikit-learn GradientBoosting option.
Prophet support is optional and attempted if installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

try:
    from prophet import Prophet  # type: ignore
except Exception:
    Prophet = None


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    metrics: Dict
    model: object


def baseline_moving_average(train: pd.Series, horizon: int = 30) -> pd.DataFrame:
    last_avg = train.rolling(window=7, min_periods=1).mean().iloc[-1]
    idx = pd.date_range(start=train.index[-1] + pd.Timedelta(1, unit="D"), periods=horizon, freq="D")
    yhat = np.full(horizon, last_avg)
    df = pd.DataFrame({"ds": idx, "yhat": yhat})
    df["yhat_lower"] = df["yhat"] * 0.9
    df["yhat_upper"] = df["yhat"] * 1.1
    return df


def train_prophet(df: pd.DataFrame, kpi: str, horizon: int = 30) -> ForecastResult:
    if Prophet is None:
        raise ImportError("Prophet is not available")
    dfp = df[[kpi]].reset_index().rename(columns={df.index.name or "index": "ds", kpi: "y"})
    dfp.columns = ["ds", "y"]
    m = Prophet()
    m.fit(dfp)
    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    # compute simple metrics on historical holdout
    metrics = {"method": "prophet"}
    return ForecastResult(forecast=forecast.tail(horizon), metrics=metrics, model=m)


def train_gbm(df: pd.DataFrame, kpi: str, horizon: int = 30) -> ForecastResult:
    # simple lag features
    series = df[kpi].copy()
    X = []
    y = []
    max_lag = 14
    for i in range(max_lag, len(series)):
        lag_vals = [series.iloc[i - l] for l in range(1, max_lag + 1)]
        X.append(lag_vals)
        y.append(series.iloc[i])
    X = np.array(X)
    y = np.array(y)
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = HistGradientBoostingRegressor(max_iter=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mape = float(mean_absolute_percentage_error(y_test, preds) * 100)
    r2 = float(r2_score(y_test, preds))
    # Directional accuracy
    y_test_diff = np.diff(y_test)
    preds_diff = np.diff(preds)
    directional_acc = float(np.mean((y_test_diff * preds_diff) > 0) * 100)
    # produce naive recursive forecast from last known lags
    last_window = series.iloc[-max_lag :].tolist()
    forecasts = []
    win = last_window.copy()
    for h in range(horizon):
        x = np.array(win[-max_lag:][::-1]).reshape(1, -1)
        p = model.predict(x)[0]
        forecasts.append(p)
        win.append(p)
    idx = pd.date_range(start=series.index[-1] + pd.Timedelta(1, unit="D"), periods=horizon, freq="D")
    df_fore = pd.DataFrame({"ds": idx, "yhat": forecasts})
    df_fore["yhat_lower"] = df_fore["yhat"] - mae
    df_fore["yhat_upper"] = df_fore["yhat"] + mae
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "directional_accuracy": directional_acc,
        "method": "gbm"
    }
    return ForecastResult(forecast=df_fore, metrics=metrics, model=model)


def forecast(df: pd.DataFrame, kpi: str = "revenue", model_name: str = "baseline", horizon: int = 30) -> ForecastResult:
    if model_name == "prophet":
        try:
            return train_prophet(df, kpi, horizon=horizon)
        except Exception:
            model_name = "baseline"
    if model_name == "gbm":
        return train_gbm(df, kpi, horizon=horizon)
    # default baseline
    fr = baseline_moving_average(df[kpi], horizon=horizon)
    metrics = {"method": "baseline", "note": "7-day moving average"}
    return ForecastResult(forecast=fr, metrics=metrics, model=None)


if __name__ == "__main__":
    import sys
    from ingest import ingest

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_kpi_history.csv"
    df, rpt = ingest(path)
    res = forecast(df, kpi="revenue", model_name="baseline", horizon=14)
    print(res.metrics)