"""Driver analysis utilities: feature engineering, importance, and SHAP (optional).
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import shap
except Exception:
    shap = None


def engineer_features(df: pd.DataFrame, kpi: str, max_lag: int = 14) -> pd.DataFrame:
    dfc = df.copy()
    for lag in range(1, max_lag + 1):
        dfc[f"lag_{lag}"] = dfc[kpi].shift(lag)
    dfc["rolling_7"] = dfc[kpi].rolling(7, min_periods=1).mean()
    dfc["roc_7"] = dfc[kpi].pct_change(7)
    dfc = dfc.dropna()
    return dfc


def compute_importance(df: pd.DataFrame, kpi: str, features: List[str] = None) -> Dict:
    if features is None:
        features = [c for c in df.columns if c != kpi]
    X = df[features]
    y = df[kpi]
    model = HistGradientBoostingRegressor(max_iter=200)
    model.fit(X, y)
    res = permutation_importance(model, X, y, n_repeats=10, random_state=0)
    imp = sorted(list(zip(features, res.importances_mean)), key=lambda x: -abs(x[1]))
    importance = [{"feature": f, "importance": float(s)} for f, s in imp]
    return {"importance": importance, "model": model}


def save_shap_plot(model, X: pd.DataFrame, outpath: str) -> Dict:
    if shap is None:
        return {"ok": False, "reason": "shap not installed"}
    explainer = shap.Explainer(model.predict, X)
    sv = explainer(X)
    plt.figure(figsize=(8, 6))
    shap.plots.bar(sv, show=False)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return {"ok": True, "path": outpath}


def temporal_precedence_check(df: pd.DataFrame, feature: str, kpi: str) -> float:
    # simple lead-lag correlation: correlate feature shifted forward with KPI
    series_f = df[feature].dropna()
    series_k = df[kpi].dropna()
    min_len = min(len(series_f), len(series_k))
    corr = series_f.tail(min_len).corr(series_k.tail(min_len))
    return float(corr)


def generate_dataset_summary(df: pd.DataFrame, kpi: str) -> Dict:
    """Generate comprehensive statistical summary of the dataset."""
    summary = {
        "shape": df.shape,
        "date_range": {
            "start": str(df.index.min()) if hasattr(df.index, 'min') else None,
            "end": str(df.index.max()) if hasattr(df.index, 'max') else None,
            "days": len(df)
        },
        "kpi_stats": {
            "mean": float(df[kpi].mean()),
            "median": float(df[kpi].median()),
            "std": float(df[kpi].std()),
            "min": float(df[kpi].min()),
            "max": float(df[kpi].max()),
            "growth": float((df[kpi].iloc[-1] / df[kpi].iloc[0] - 1) * 100) if len(df) > 0 else 0
        },
        "correlations": {}
    }
    
    # Add correlations with KPI for all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != kpi:
            summary["correlations"][col] = float(df[col].corr(df[kpi]))
    
    return summary


def save_correlation_heatmap(df: pd.DataFrame, kpi: str, outpath: str, top_n: int = 15) -> Dict:
    """Create correlation heatmap for top features."""
    try:
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if kpi not in numeric_cols:
            return {"ok": False, "reason": "KPI not in numeric columns"}
        
        # Get correlation with KPI and select top features
        corr_with_kpi = df[numeric_cols].corr()[kpi].abs().sort_values(ascending=False)
        top_features = corr_with_kpi.head(min(top_n, len(corr_with_kpi))).index.tolist()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df[top_features].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, 
                    vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Feature Correlation Matrix (Top {len(top_features)} vs {kpi})\n" +
                     f"Strongest correlation: {corr_with_kpi.iloc[1]:.3f} ({corr_with_kpi.index[1]})",
                     fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        return {"ok": True, "path": outpath, "top_corr": float(corr_with_kpi.iloc[1])}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def save_feature_importance_plot(importance_list: List[Dict], outpath: str, top_n: int = 15) -> Dict:
    """Create horizontal bar chart of feature importance."""
    try:
        top_features = importance_list[:top_n]
        features = [f["feature"] for f in top_features]
        importances = [f["importance"] for f in top_features]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances, color="steelblue")
        plt.yticks(range(len(features)), features)
        plt.xlabel("Importance Score")
        plt.title(f"Top {len(features)} Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
        return {"ok": True, "path": outpath}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


if __name__ == "__main__":
    from ingest import ingest

    df, rpt = ingest("data/sample_kpi_history.csv")
    df2 = engineer_features(df, "revenue")
    imp = compute_importance(df2, "revenue")
    print(imp["importance"][:5])