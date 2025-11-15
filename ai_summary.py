"""Narrative generator: uses OpenAI when available, otherwise fallback template.
"""
from __future__ import annotations

import os
from typing import Dict, List

import json

try:
    import openai
except Exception:
    openai = None


def _fallback_summary(context: Dict) -> str:
    kpi = context.get("kpi", "KPI")
    period = context.get("period", "period")
    headline = context.get("headline", "Forecast shows stable trend")
    drivers = context.get("drivers", [])
    recs = context.get("recommendations", [])
    s = []
    s.append(f"Executive Summary — {kpi}")
    s.append(f"Period: {period}")
    s.append("")
    s.append(headline)
    s.append("")
    s.append("Top drivers:")
    for d in drivers[:3]:
        s.append(f"- {d.get('feature')}: {d.get('importance'):.3f}")
    s.append("")
    s.append("Recommendations:")
    for r in recs[:3]:
        s.append(f"- {r}")
    return "\n".join(s)


def generate_summary(kpi: str, forecast_headline: str, drivers: List[Dict], recommendations: List[str], use_openai: bool = False) -> str:
    context = {
        "kpi": kpi,
        "period": "next period",
        "headline": forecast_headline,
        "drivers": drivers,
        "recommendations": recommendations,
    }
    if use_openai and openai is not None and os.environ.get("OPENAI_API_KEY"):
        try:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            prompt = (
                "You are an executive business analyst. Write a concise executive summary:\n"
                f"KPI: {kpi}\nHeadline: {forecast_headline}\nDrivers: {json.dumps(drivers[:5])}\nRecommendations: {json.dumps(recommendations[:5])}\nKeep it brief and action-oriented."
            )
            resp = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], max_tokens=300)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return _fallback_summary(context)
    else:
        return _fallback_summary(context)


if __name__ == "__main__":
    print(generate_summary("revenue", "Revenue expected to increase 8% next month.", [{"feature": "marketing_spend", "importance": 0.12}], ["Increase budget on high-performing channels"], use_openai=False))