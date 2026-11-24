"""Intervention Priority Score (IPS).

IPS_i = w_f * F_i + w_s * S_i + w_e * E_i

F = normalised collision frequency, S = normalised severity impact,
E = normalised SHAP-based feature/category importance. Weights are non-negative
and are normalised to sum to 1. Scores are associative signals for review, NOT
causal estimates or guarantees of intervention effectiveness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config

DEFAULT_WEIGHTS = {"frequency": 0.4, "severity": 0.35, "shap": 0.25}
MIN_CONDITION_ROWS = 150


def normalise_weights(frequency: float, severity: float, shap: float) -> dict[str, float]:
    values = {"frequency": float(frequency), "severity": float(severity), "shap": float(shap)}
    if any(v < 0 for v in values.values()):
        raise ValueError("Weights must be non-negative.")
    total = sum(values.values())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("At least one weight must be positive.")
    return {k: v / total for k, v in values.items()}


def _speed_label(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.select(
        [values <= 20, values <= 30, values <= 40, values <= 50, values <= 60],
        ["≤20 mph", "30 mph", "40 mph", "50 mph", "60 mph"], default="70+ mph",
    ), index=series.index)


def _condition_values(features: pd.DataFrame, source: str) -> pd.Series:
    if source == "speed_limit":
        return _speed_label(features["speed_limit"])
    return features[source].astype(str)


def compute_condition_scores(
    clean: pd.DataFrame,
    features: pd.DataFrame,
    category_importance: pd.DataFrame,
    global_importance: pd.DataFrame,
    weights: dict[str, float] | None = None,
    min_rows: int = MIN_CONDITION_ROWS,
) -> pd.DataFrame:
    """Build one row per (feature, category) condition with F, S, E and IPS."""
    weights = normalise_weights(**(weights or DEFAULT_WEIGHTS))
    if clean.empty or features.empty:
        return pd.DataFrame()

    cat = category_importance.copy() if category_importance is not None else pd.DataFrame()
    if not cat.empty:
        cat_map = {(r.source_feature, str(r.category)): float(r.e_raw) for r in cat.itertuples()}
    else:
        cat_map = {}

    feature_e: dict[str, float] = {}
    if global_importance is not None and not global_importance.empty:
        for r in global_importance.itertuples():
            source = str(r.feature).split("__", 1)[-1]
            match = next((s for s, _ in config.CONDITION_SPECS if source == s or source.startswith(s + "_")), None)
            if match:
                feature_e[match] = feature_e.get(match, 0.0) + float(r.mean_abs_shap)
    if not feature_e and cat_map:
        for (source, _), value in cat_map.items():
            feature_e[source] = feature_e.get(source, 0.0) + value

    target = clean[config.TARGET].to_numpy()
    weight = clean["severity_impact_weight"].to_numpy(dtype=float)
    rows = []
    for source, display in config.CONDITION_SPECS:
        if source not in features.columns:
            continue
        values = _condition_values(features, source)
        for category, idx in values.groupby(values).groups.items():
            pos = np.asarray(list(idx), dtype=int)
            n = len(pos)
            if n < min_rows:
                continue
            classes = target[pos]
            fatal = int((classes == "Fatal").sum())
            serious = int((classes == "Serious").sum())
            slight = int((classes == "Slight").sum())
            mean_impact = float(weight[pos].mean())
            e_raw = cat_map.get((source, str(category)))
            basis = "shap-category"
            if e_raw is None:
                root = "speed_limit" if source == "speed_limit" else source
                e_raw = feature_e.get(root, 0.0)
                basis = "shap-feature-fallback"
            rows.append({
                "feature": source, "condition_type": display, "condition": str(category),
                "collisions": n, "fatal": fatal, "serious": serious, "slight": slight,
                "fatal_serious_rate": (fatal + serious) / n,
                "mean_severity_impact": mean_impact,
                "e_raw": float(e_raw), "e_basis": basis,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["frequency"] = df["collisions"] / df["collisions"].max()
    max_impact = df["mean_severity_impact"].max()
    df["severity"] = df["mean_severity_impact"] / max_impact if max_impact > 0 else 0.0
    max_e = df["e_raw"].max()
    df["shap_importance"] = df["e_raw"] / max_e if max_e > 0 else 0.0
    df["IPS"] = (
        weights["frequency"] * df["frequency"]
        + weights["severity"] * df["severity"]
        + weights["shap"] * df["shap_importance"]
    )
    df["rank"] = df["IPS"].rank(ascending=False, method="min").astype(int)
    df["fatal_serious_pct"] = 100 * df["fatal_serious_rate"]
    for col in ["frequency", "severity", "shap_importance", "IPS"]:
        df[f"{col}_score"] = 100 * df[col]
    return df.sort_values("IPS", ascending=False).reset_index(drop=True)


def ranking_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[["condition_type", "condition", "collisions", "IPS"]].copy()
    out["rank_frequency"] = df["frequency"].rank(ascending=False, method="min").astype(int)
    out["rank_severity"] = df["severity"].rank(ascending=False, method="min").astype(int)
    out["rank_shap"] = df["shap_importance"].rank(ascending=False, method="min").astype(int)
    out["rank_IPS"] = df["rank"]
    return out.sort_values("rank_IPS").reset_index(drop=True)


def ranking_stability(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    ranks = ranking_columns(df)
    rows = []
    for alternative in ["rank_frequency", "rank_severity", "rank_shap"]:
        rho, pvalue = spearmanr(ranks["rank_IPS"], ranks[alternative])
        if not np.isfinite(rho):
            rho, pvalue = 0.0, 1.0
        top_n = 10
        overlap = len(set(ranks.nsmallest(top_n, "rank_IPS")["condition"]) & set(ranks.nsmallest(top_n, alternative)["condition"]))
        rows.append({
            "comparison": alternative.replace("rank_", "").title() + "-only",
            "spearman_rho": float(rho), "p_value": float(pvalue),
            "top10_overlap": overlap, "top10_of": top_n,
        })
    return pd.DataFrame(rows)


def weight_sensitivity(clean, features, cat_imp, glob_imp, weights, min_rows: int = MIN_CONDITION_ROWS) -> pd.DataFrame:
    """Vary frequency weight 0..1 on a grid; split the remainder in the current S:E ratio."""
    base = normalise_weights(**weights)
    remaining = base["severity"] + base["shap"]
    s_share = base["severity"] / remaining if remaining > 0 else 0.5
    rows = []
    baseline = compute_condition_scores(clean, features, cat_imp, glob_imp, base, min_rows=min_rows)
    if baseline.empty:
        return pd.DataFrame()
    base_order = dict(zip(baseline["condition"], baseline["rank"]))
    for f in np.linspace(0, 1, 11):
        w = {"frequency": float(f), "severity": float((1 - f) * s_share), "shap": float((1 - f) * (1 - s_share))}
        scores = compute_condition_scores(clean, features, cat_imp, glob_imp, w, min_rows=min_rows)
        if scores.empty:
            continue
        order = dict(zip(scores["condition"], scores["rank"]))
        common = list(base_order)
        rho = spearmanr([base_order[c] for c in common], [order.get(c, len(order) + 1) for c in common])[0]
        top = set(scores.nsmallest(10, "rank")["condition"])
        rows.append({
            "w_frequency": f, "w_severity": w["severity"], "w_shap": w["shap"],
            "spearman_vs_baseline": float(rho),
            "top10_overlap_with_baseline": len(top & set(baseline.nsmallest(10, "rank")["condition"])),
        })
    return pd.DataFrame(rows)
