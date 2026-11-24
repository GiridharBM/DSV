"""Persist evaluation tables and a multi-page PDF summary built from real results."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.evaluation.metrics import metrics_to_frames


def write_tables(models_bundle: dict, ips: pd.DataFrame, stability: pd.DataFrame,
                 sensitivity: pd.DataFrame, insights: list[dict], cleaning: dict, provenance: dict) -> dict:
    frames = {}
    for name, metrics in models_bundle["results"].items():
        parts = metrics_to_frames(metrics, name)
        for key, frame in parts.items():
            mode = "a" if key in frames else "w"
            header = not (key in frames)
            if key not in frames:
                frames[key] = frame
            else:
                frames[key] = pd.concat([frames[key], frame], ignore_index=True)
    for key, frame in frames.items():
        frame.to_csv(config.REPORTS_DIR / f"model_{key}.csv", index=False)

    for name, frame in [
        ("ips_ranking.csv", ips), ("ranking_stability.csv", stability),
        ("weight_sensitivity.csv", sensitivity),
    ]:
        frame.to_csv(config.REPORTS_DIR / name, index=False)

    with (config.REPORTS_DIR / "insights.json").open("w", encoding="utf-8") as handle:
        json.dump(insights, handle, indent=2, default=str)

    summary = {
        "provenance": provenance,
        "cleaning": cleaning,
        "best_model": models_bundle["best_name"],
        "selection_rule": "macro_f1 + 0.05*(fatal_recall + serious_recall)",
        "train_rows": models_bundle["train_rows"],
        "test_rows": models_bundle["test_rows"],
        "results": models_bundle["results"],
    }
    with (config.REPORTS_DIR / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return frames


def _wrap(text: str, width: int = 95) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width=width))


def write_pdf(models_bundle: dict, frames: dict, ips: pd.DataFrame, insights: list[dict],
              global_imp: pd.DataFrame, provenance: dict) -> bool:
    """Create a concise PDF report. Returns False if matplotlib is unavailable."""
    try:
        path = config.REPORTS_DIR / "summary_report.pdf"
        with plt.rc_context({"figure.facecolor": "#0b1220", "axes.facecolor": "#0b1220",
                             "text.color": "#e2e8f0", "axes.labelcolor": "#e2e8f0",
                             "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
                             "axes.edgecolor": "#334155"}):
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.06, 0.95, "Road-Accident Severity & Intervention Prioritizer", fontsize=20, color="#22d3ee", weight="bold")
            fig.text(0.06, 0.915, "Automated summary — all numbers computed from the loaded dataset and trained models", fontsize=10, color="#94a3b8")
            lines = [
                f"Data source: {provenance.get('source', 'unknown')}",
                f"Geographic scope: {provenance.get('geography', 'Great Britain')}",
                f"Years: {provenance.get('years', 'n/a')}    Rows analysed: {provenance.get('clean_rows', 'n/a')}",
                f"Best model: {models_bundle['best_name']}",
                f"Chronological holdout: train through {config.TRAIN_THROUGH_YEAR}, test = later years",
                f"Training rows used: {models_bundle['train_rows']:,}   Test rows: {models_bundle['test_rows']:,}",
                "",
                "Model comparison (chronological holdout):",
            ]
            comp = frames.get("comparison", pd.DataFrame())
            if not comp.empty:
                for r in comp.itertuples():
                    lines.append(f"  {r.model:<22} acc={r.accuracy:.3f}  macro-F1={r.macro_f1:.3f}  "
                                 f"weighted-F1={r.weighted_f1:.3f}  fatal-recall={r.fatal_recall:.3f}  "
                                 f"serious-recall={r.serious_recall:.3f}")
            lines += ["", "Key insights:"]
            for item in insights[:8]:
                lines.append("  • " + _wrap(item["detail"], 100).replace("\n", "\n    "))
            if not ips.empty:
                lines += ["", "Top 10 intervention-priority conditions (current weights):"]
                for r in ips.head(10).itertuples():
                    lines.append(f"  {r.rank:>2}. {str(r.condition_type):<18} {str(r.condition):<26} "
                                 f"IPS={r.IPS:.3f}  n={int(r.collisions):,}  "
                                 f"fatal+serious={100*r.fatal_serious_rate:.1f}%")
            if not global_imp.empty:
                lines += ["", "Top SHAP features: " + ", ".join(global_imp.head(8)["feature"].tolist())]
            lines += ["", "Method note: IPS = wF*frequency + wS*severity + wE*SHAP importance. "
                      "It is a prioritisation aid based on associations; it makes no causal claim "
                      "and does not guarantee that any intervention will reduce collisions."]
            y = 0.885
            for line in lines:
                fig.text(0.06, y, line, fontsize=9, color="#cbd5e1", va="top")
                y -= 0.0185
                if y < 0.04:
                    break
            fig.savefig(path, dpi=130)
            plt.close(fig)
        log_ok = True
    except Exception as exc:
        from src.utils.logging import get_logger
        get_logger(__name__).warning("PDF report generation failed: %s", exc)
        log_ok = False
    return log_ok
