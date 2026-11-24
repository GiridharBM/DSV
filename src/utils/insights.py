"""Automatically generated insights from real computed aggregates."""
from __future__ import annotations

import pandas as pd

from src import config
from src.prioritization.ips import DEFAULT_WEIGHTS, compute_condition_scores


def build_insights(clean: pd.DataFrame, features: pd.DataFrame, cat_imp: pd.DataFrame,
                   glob_imp: pd.DataFrame, ips: pd.DataFrame) -> list[dict]:
    """Each insight includes its denominator so frequency and rates are distinct."""
    insights: list[dict] = []
    if clean.empty:
        return [{"title": "No data", "detail": "Load collision data to generate insights.", "kind": "info"}]

    counts = clean[config.TARGET].value_counts()
    n = len(clean)
    for label in config.SEVERITY_ORDER:
        count = int(counts.get(label, 0))
        insights.append({
            "title": f"{label} collisions",
            "detail": f"{count:,} of {n:,} recorded collisions ({100 * count / n:.2f}%).",
            "kind": label.lower(),
        })

    years = clean.groupby("collision_year").size()
    if len(years) > 1:
        first, last = int(years.index.min()), int(years.index.max())
        change = 100 * (years.iloc[-1] - years.iloc[0]) / years.iloc[0] if years.iloc[0] else 0
        insights.append({
            "title": "Annual trend",
            "detail": f"Recorded collisions moved from {int(years.iloc[0]):,} in {first} to "
                      f"{int(years.iloc[-1]):,} in {last} ({change:+.1f}%).",
            "kind": "trend",
        })

    if not ips.empty:
        top_freq = ips.nlargest(1, "collisions").iloc[0]
        insights.append({
            "title": "Most frequent condition",
            "detail": f"{top_freq['condition_type']}: '{top_freq['condition']}' appears in "
                      f"{int(top_freq['collisions']):,} collisions "
                      f"({100 * top_freq['collisions'] / n:.1f}% of all records).",
            "kind": "frequency",
        })
        eligible = ips[ips["collisions"] >= 1000]
        if not eligible.empty:
            worst_rate = eligible.nlargest(1, "fatal_serious_rate").iloc[0]
            insights.append({
                "title": "Highest serious/fatal proportion",
                "detail": f"{worst_rate['condition_type']}: '{worst_rate['condition']}' has "
                          f"{worst_rate['fatal_serious_pct']:.1f}% fatal-or-serious outcomes "
                          f"among {int(worst_rate['collisions']):,} collisions "
                          f"({int(worst_rate['fatal'] + worst_rate['serious']):,} such outcomes).",
                "kind": "severity",
            })
        top_ips = ips.iloc[0]
        insights.append({
            "title": "Highest-priority condition (current weights)",
            "detail": f"IPS {top_ips['IPS']:.3f} for {top_ips['condition_type']}: '{top_ips['condition']}' "
                      f"(frequency {top_ips['frequency']:.3f}, severity {top_ips['severity']:.3f}, "
                      f"SHAP {top_ips['shap_importance']:.3f}; n={int(top_ips['collisions']):,}). "
                      "This is a ranking signal, not a causal claim.",
            "kind": "priority",
        })

    if glob_imp is not None and not glob_imp.empty:
        ranked = glob_imp.head(3)["feature"].str.split("__").str[-1].tolist()
        insights.append({
            "title": "Most influential model features",
            "detail": "Top mean-|SHAP| features: " + ", ".join(ranked) + ".",
            "kind": "model",
        })
    return insights
