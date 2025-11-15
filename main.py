"""CLI orchestrator for Algorzen Helix pipeline.

Usage example:
python main.py --input data/sample_kpi_history.csv --kpi revenue --horizon 30 --model baseline --output reports/Helix_Forecast_Report_20251115.pdf
"""
from __future__ import annotations

import argparse
import os
import json
from datetime import datetime, timezone

import pandas as pd

from ingest import ingest
from forecasting import forecast
from drivers import engineer_features, compute_importance, save_shap_plot, save_correlation_heatmap, save_feature_importance_plot, generate_dataset_summary
from ai_summary import generate_summary
from report_generator import build_pdf, save_metadata


def run_pipeline(input_path: str, kpi: str, horizon: int, model_name: str, use_openai: bool, output_path: str):
    print("1) Ingesting data...")
    df, rpt = ingest(input_path, timestamp_col="date", kpi_col=kpi)
    print("Ingest report:", rpt)
    
    # Generate dataset summary
    dataset_summary = generate_dataset_summary(df, kpi)
    print(f"\nDataset Summary:")
    print(f"  Period: {dataset_summary['date_range']['start']} to {dataset_summary['date_range']['end']}")
    print(f"  {kpi.upper()} — Mean: {dataset_summary['kpi_stats']['mean']:.2f}, Growth: {dataset_summary['kpi_stats']['growth']:.1f}%")
    print(f"  Top correlations with {kpi}:")
    sorted_corr = sorted(dataset_summary['correlations'].items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr[:3]:
        print(f"    {feat}: {corr:.3f}")
    
    print("\n2) Forecasting...")
    fr = forecast(df, kpi=kpi, model_name=model_name, horizon=horizon)
    forecast_df = fr.forecast
    print("Forecast metrics:", fr.metrics)
    print("3) Engineering features and driver analysis...")
    feat = engineer_features(df, kpi)
    imp = compute_importance(feat, kpi)
    drivers = imp.get("importance", [])
    
    print(f"  Engineered {feat.shape[1]} total features from {df.shape[1]} original columns")
    print(f"  Top 5 drivers by importance:")
    for i, d in enumerate(drivers[:5], 1):
        print(f"    {i}. {d['feature']}: {d['importance']:.4f}")
    
    assets_dir = os.path.join(os.path.dirname(output_path), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Generate correlation heatmap on ENGINEERED features
    correlation_img = os.path.join(assets_dir, "correlation_heatmap.png")
    corr_result = save_correlation_heatmap(feat, kpi, correlation_img, top_n=15)
    print(f"  Correlation heatmap: {corr_result.get('ok', False)} - Top corr: {corr_result.get('top_corr', 0):.3f}")
    
    # Generate feature importance plot
    feature_imp_img = os.path.join(assets_dir, "feature_importance.png")
    save_feature_importance_plot(drivers, feature_imp_img, top_n=15)
    
    # Try to save a SHAP bar plot (best-effort)
    driver_img = None
    try:
        driver_img = os.path.join(assets_dir, "shap_bar.png")
        res = save_shap_plot(imp.get("model"), feat[[x["feature"] for x in drivers if x["feature"].startswith("lag_")][:20]], driver_img)
        if not res.get("ok"):
            driver_img = None
    except Exception:
        driver_img = None
    print("4) Generating narrative...")
    headline = f"Forecast: {float(forecast_df['yhat'].pct_change().mean() * 100):.1f}% avg change"
    recommendations = ["Review marketing spend allocation.", "Monitor price elasticity.", "Prepare inventory for expected demand."]
    summary = generate_summary(kpi, headline, drivers, recommendations, use_openai=use_openai)
    print("5) Building report...")
    build_pdf(output_path, kpi, summary, forecast_df, driver_img, fr.metrics, 
              historical=df, correlation_path=correlation_img, feature_imp_path=feature_imp_img,
              dataset_summary=dataset_summary)
    meta_path = save_metadata(output_path, kpi, model_name, openai_used=use_openai)
    print("Report written:", output_path)
    print("Metadata written:", meta_path)


def main():
    parser = argparse.ArgumentParser(description="Algorzen Helix CLI")
    parser.add_argument("--input", required=False, default="data/sample_kpi_history.csv")
    parser.add_argument("--kpi", required=False, default="revenue")
    parser.add_argument("--horizon", required=False, type=int, default=30)
    parser.add_argument("--model", required=False, default="baseline", choices=["baseline", "prophet", "gbm"])
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--output", required=False, default=f"reports/Helix_Forecast_Report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.pdf")
    args = parser.parse_args()
    run_pipeline(args.input, args.kpi, args.horizon, args.model, args.use_openai, args.output)


if __name__ == "__main__":
    main()
