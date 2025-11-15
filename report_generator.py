"""Report generator: create a branded PDF with forecast charts and driver visuals.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def _plot_forecast(forecast: pd.DataFrame, historical: pd.DataFrame, kpi: str, outpath: str):
    """Plot forecast with historical context."""
    plt.figure(figsize=(14, 6))
    
    # Plot last 90 days of historical data
    hist_tail = historical.tail(90)
    plt.plot(hist_tail.index, hist_tail[kpi], label="Historical", color="black", linewidth=2)
    
    # Plot forecast
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="steelblue", linewidth=2, linestyle="--")
    
    if "yhat_lower" in forecast.columns:
        plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], 
                        alpha=0.3, color="steelblue", label="Confidence Interval")
    
    plt.axvline(x=hist_tail.index[-1], color="red", linestyle=":", linewidth=1.5, label="Forecast Start")
    plt.legend(loc="best")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel(kpi.title(), fontsize=11)
    plt.title(f"{kpi.title()} Forecast with Historical Context", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _plot_metrics_comparison(metrics: Dict, outpath: str):
    """Create visual comparison of model metrics."""
    metric_names = []
    metric_values = []
    
    for key, value in metrics.items():
        if key not in ["method", "note"] and isinstance(value, (int, float)):
            metric_names.append(key.upper())
            metric_values.append(value)
    
    if not metric_names:
        return
    
    plt.figure(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metric_names)))
    plt.bar(metric_names, metric_values, color=colors, edgecolor="black", linewidth=1.5)
    plt.ylabel("Value", fontsize=11)
    plt.title("Model Performance Metrics", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def build_pdf(report_path: str, kpi: str, summary: str, forecast: pd.DataFrame, driver_image_path: str, metrics: Dict, historical: pd.DataFrame = None, correlation_path: str = None, feature_imp_path: str = None, dataset_summary: Dict = None):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    tmp_fig = os.path.join(os.path.dirname(report_path), "forecast_plot.png")
    tmp_metrics = os.path.join(os.path.dirname(report_path), "metrics_plot.png")
    
    if historical is not None:
        _plot_forecast(forecast, historical, kpi, tmp_fig)
    else:
        # Fallback to simple plot
        plt.figure(figsize=(10, 4))
        plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
        if "yhat_lower" in forecast.columns:
            plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2)
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("KPI")
        plt.tight_layout()
        plt.savefig(tmp_fig)
        plt.close()
    
    _plot_metrics_comparison(metrics, tmp_metrics)
    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4
    # Cover
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 40 * mm, "Algorzen Helix — Forecast & Driver Report")
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, height - 50 * mm, f"Author: Rishi Singh")
    c.drawString(20 * mm, height - 55 * mm, f"Generated: {datetime.now().isoformat()} UTC")
    c.showPage()
    # Executive summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, height - 20 * mm, "Executive Summary")
    text = c.beginText(20 * mm, height - 30 * mm)
    text.setFont("Helvetica", 10)
    for line in summary.splitlines():
        text.textLine(line)
    c.drawText(text)
    c.showPage()
    # Dataset Statistics page (NEW)
    if dataset_summary:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(20 * mm, height - 20 * mm, "Dataset Analysis")
        
        y_pos = height - 35 * mm
        c.setFont("Helvetica-Bold", 11)
        c.drawString(20 * mm, y_pos, "Overview")
        y_pos -= 7 * mm
        
        c.setFont("Helvetica", 10)
        c.drawString(25 * mm, y_pos, f"Shape: {dataset_summary.get('shape', 'N/A')}")
        y_pos -= 5 * mm
        date_range = dataset_summary.get('date_range', {})
        c.drawString(25 * mm, y_pos, f"Period: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")
        y_pos -= 5 * mm
        c.drawString(25 * mm, y_pos, f"Total Days: {date_range.get('days', 'N/A')}")
        y_pos -= 10 * mm
        
        c.setFont("Helvetica-Bold", 11)
        c.drawString(20 * mm, y_pos, f"{kpi.upper()} Statistics")
        y_pos -= 7 * mm
        
        c.setFont("Helvetica", 10)
        kpi_stats = dataset_summary.get('kpi_stats', {})
        c.drawString(25 * mm, y_pos, f"Mean: ${kpi_stats.get('mean', 0):.2f}")
        y_pos -= 5 * mm
        c.drawString(25 * mm, y_pos, f"Median: ${kpi_stats.get('median', 0):.2f}")
        y_pos -= 5 * mm
        c.drawString(25 * mm, y_pos, f"Std Dev: ${kpi_stats.get('std', 0):.2f}")
        y_pos -= 5 * mm
        c.drawString(25 * mm, y_pos, f"Min: ${kpi_stats.get('min', 0):.2f}")
        y_pos -= 5 * mm
        c.drawString(25 * mm, y_pos, f"Max: ${kpi_stats.get('max', 0):.2f}")
        y_pos -= 5 * mm
        c.setFont("Helvetica-Bold", 10)
        c.drawString(25 * mm, y_pos, f"Total Growth: {kpi_stats.get('growth', 0):.1f}%")
        y_pos -= 10 * mm
        
        c.setFont("Helvetica-Bold", 11)
        c.drawString(20 * mm, y_pos, "Correlations with " + kpi.upper())
        y_pos -= 7 * mm
        
        c.setFont("Helvetica", 10)
        correlations = dataset_summary.get('correlations', {})
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, corr_val in sorted_corr:
            c.drawString(25 * mm, y_pos, f"{feature}: {corr_val:.4f}")
            y_pos -= 5 * mm
            if y_pos < 30 * mm:
                break
        
        c.showPage()
    # Forecast page
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, height - 20 * mm, "Forecast")
    c.drawImage(tmp_fig, 20 * mm, height - 120 * mm, width=160 * mm, preserveAspectRatio=True, mask='auto')
    c.showPage()
    # Metrics visualization page
    if os.path.exists(tmp_metrics):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(20 * mm, height - 20 * mm, "Model Performance Metrics")
        c.drawImage(tmp_metrics, 20 * mm, height - 120 * mm, width=160 * mm, preserveAspectRatio=True, mask='auto')
        c.showPage()
    # Feature Importance
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, height - 20 * mm, "Feature Importance")
    if feature_imp_path and os.path.exists(feature_imp_path):
        c.drawImage(feature_imp_path, 20 * mm, height - 120 * mm, width=160 * mm, preserveAspectRatio=True, mask='auto')
    c.showPage()
    # Correlation Heatmap
    if correlation_path and os.path.exists(correlation_path):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(20 * mm, height - 20 * mm, "Feature Correlations")
        c.drawImage(correlation_path, 20 * mm, height - 120 * mm, width=160 * mm, preserveAspectRatio=True, mask='auto')
        c.showPage()
    # SHAP Drivers
    if driver_image_path and os.path.exists(driver_image_path):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(20 * mm, height - 20 * mm, "SHAP Analysis")
        c.drawImage(driver_image_path, 20 * mm, height - 120 * mm, width=160 * mm, preserveAspectRatio=True, mask='auto')
        c.showPage()
    # Metrics
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, height - 20 * mm, "Model Performance")
    text = c.beginText(20 * mm, height - 30 * mm)
    text.setFont("Helvetica", 10)
    for k, v in metrics.items():
        text.textLine(f"{k}: {v}")
    c.drawText(text)
    # Footer
    c.setFont("Helvetica", 8)
    c.drawString(20 * mm, 10 * mm, "Algorzen Research Division © 2025 — Author: Rishi Singh")
    c.save()
    # cleanup tmp
    for tmp_file in [tmp_fig, tmp_metrics]:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def save_metadata(report_path: str, kpi: str, model_name: str, openai_used: bool = False):
    meta = {
        "project": "Algorzen Helix",
        "report_id": f"HELIX-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "generated_by": "Rishi Singh",
        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
        "kpi": kpi,
        "model_used": model_name,
        "openai_used": openai_used,
    }
    meta_path = os.path.join(os.path.dirname(report_path), "report_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


if __name__ == "__main__":
    # small demo when run directly
    import ingest
    from ai_summary import generate_summary
    from drivers import engineer_features, compute_importance, save_shap_plot

    df, rpt = ingest.ingest("data/sample_kpi_history.csv")
    fr = pd.DataFrame({"ds": pd.date_range(start=df.index[-1] + pd.Timedelta(1, unit='D'), periods=14, freq='D'), "yhat": [100]*14})
    summary = generate_summary("revenue", "Revenue steady.", [], [], use_openai=False)
    out = "reports/Helix_Forecast_Report_sample.pdf"
    build_pdf(out, "revenue", summary, fr, None, {"mae": 0})
    print("Wrote", out)