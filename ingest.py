"""Ingest and validation utilities for Algorzen Helix.

Functions:
- load_data: load CSV or URL
- validate_schema: ensure timestamp + KPI present
- detect_frequency: infer data frequency
- resample_align: resample to regular frequency
"""
from __future__ import annotations

import io
import json
from typing import Dict, Tuple

import pandas as pd


def load_data(path_or_url: str) -> pd.DataFrame:
    """Load CSV from file path or URL into a DataFrame.

    Accepts local path or http(s) URL. Returns raw DataFrame.
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        df = pd.read_csv(path_or_url)
    else:
        df = pd.read_csv(path_or_url)
    return df


def validate_schema(df: pd.DataFrame, timestamp_col: str, kpi_col: str) -> Dict:
    """Validate presence and types of required columns.

    Returns a schema report dict.
    """
    report = {"ok": True, "errors": []}
    if timestamp_col not in df.columns:
        report["ok"] = False
        report["errors"].append(f"Missing timestamp column: {timestamp_col}")
    if kpi_col not in df.columns:
        report["ok"] = False
        report["errors"].append(f"Missing KPI column: {kpi_col}")
    # attempt to parse timestamp
    try:
        pd.to_datetime(df[timestamp_col])
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"Timestamp parse error: {e}")
    return report


def detect_frequency(df: pd.DataFrame, timestamp_col: str = "date") -> str:
    """Detect a reasonable frequency string (daily, hourly, monthly).

    Returns pandas offset alias string.
    """
    s = pd.to_datetime(df[timestamp_col]).sort_values()
    diffs = s.diff().dropna().dt.total_seconds()
    median = diffs.median()
    # daily ~ 86400s, hourly ~3600s
    if median < 4000:
        return "H"
    if median < 86400 * 1.5:
        return "D"
    return "M"


def resample_align(df: pd.DataFrame, timestamp_col: str, freq: str = "D", kpi_col: str = None) -> Tuple[pd.DataFrame, Dict]:
    """Return DataFrame indexed by timestamp and resampled to `freq`.

    NaNs are forward-filled where appropriate. Returns (df, report)
    """
    dfc = df.copy()
    dfc[timestamp_col] = pd.to_datetime(dfc[timestamp_col])
    dfc = dfc.set_index(timestamp_col).sort_index()
    resampled = dfc.resample(freq).mean()
    resampled = resampled.ffill()
    report = {
        "freq": freq,
        "start": str(resampled.index.min()),
        "end": str(resampled.index.max()),
        "n_points": len(resampled),
    }
    return resampled, report


def ingest(path_or_url: str, timestamp_col: str = "date", kpi_col: str = "revenue") -> Tuple[pd.DataFrame, Dict]:
    """Full ingest pipeline: load, validate, detect frequency, align.

    Returns cleaned DataFrame and a combined report.
    """
    df = load_data(path_or_url)
    schema = validate_schema(df, timestamp_col, kpi_col)
    if not schema.get("ok", False):
        return df, {"schema": schema}
    freq = detect_frequency(df, timestamp_col)
    resampled, resample_report = resample_align(df, timestamp_col, freq=freq, kpi_col=kpi_col)
    report = {"schema": schema, "resample": resample_report}
    return resampled, report


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_kpi_history.csv"
    df, rpt = ingest(path)
    print("Ingest report:")
    print(rpt)