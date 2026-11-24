"""SHAP explanations computed from the actual trained model.

Outputs global feature importance, category-level importance used by the IPS engine,
and a reusable per-prediction explanation function.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap

from src import config
from src.utils.logging import get_logger

log = get_logger(__name__)


def _as_multiclass(values, n_samples: int, n_classes: int, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        values = np.stack([np.asarray(v) for v in values], axis=-1)
    arr = np.asarray(values)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected SHAP shape {arr.shape}")
    if arr.shape[0] == n_classes and arr.shape[1] == n_samples:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[0] != n_samples or arr.shape[1] != n_features:
        raise ValueError(f"Unexpected SHAP layout {arr.shape} for {n_samples}x{n_features}")
    if arr.shape[2] == 1 and n_classes > 1:
        arr = np.repeat(arr, n_classes, axis=2)
    return arr


def make_explainer(model, background=None):
    clf = model.named_steps["clf"]
    name = clf.__class__.__name__
    try:
        if name in ("XGBClassifier", "RandomForestClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier"):
            kwargs = {"model_output": "raw"} if name == "XGBClassifier" else {}
            return shap.TreeExplainer(clf, **kwargs), "TreeExplainer"
        if name == "LogisticRegression":
            return shap.LinearExplainer(clf, background), "LinearExplainer"
    except Exception as exc:
        log.warning("Primary SHAP explainer failed (%s); trying generic Explainer", exc)
    try:
        return shap.Explainer(clf, background), "Explainer"
    except Exception as exc:
        log.warning("Generic SHAP explainer failed: %s", exc)
        return None, "unavailable"


def transformed_matrix(model, frame: pd.DataFrame):
    prep = model.named_steps["prep"]
    matrix = prep.transform(frame)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64), list(prep.get_feature_names_out())


def shap_values_for(model, frame: pd.DataFrame, background=None):
    """Return (values[n, features, classes], feature_names, explainer_type)."""
    matrix, names = transformed_matrix(model, frame)
    n_classes = len(model.named_steps["clf"].classes_)
    explainer, kind = make_explainer(model, background=matrix[: min(100, len(matrix))])
    if explainer is None:
        return None, names, kind
    try:
        values = explainer.shap_values(matrix, check_additivity=False) if kind == "TreeExplainer" else explainer(matrix).values
        values = _as_multiclass(values, len(matrix), n_classes, len(names))
        return values, names, kind
    except TypeError:
        values = explainer.shap_values(matrix)
        return _as_multiclass(values, len(matrix), n_classes, len(names)), names, kind


def _clean_name(name: str) -> str:
    return name.split("__", 1)[-1]


def category_importance(values: np.ndarray, names: list[str], X_sample: pd.DataFrame, min_count: int = 30) -> pd.DataFrame:
    """Mean absolute SHAP per encoded category, used as the E component of IPS."""
    abs_sv = np.abs(values).mean(axis=2)
    feature_imp = {n: float(abs_sv[:, j].mean()) for j, n in enumerate(names)}
    rows = []
    for j, full_name in enumerate(names):
        name = _clean_name(full_name)
        if full_name.startswith("cat__"):
            matched = [p for p in config.CONDITION_SPECS if name.startswith(p[0] + "_")]
            if not matched:
                continue
            source = matched[0][0]
            category = name[len(source) + 1:]
            mask = X_sample[source].astype(str).to_numpy() == category
            if mask.sum() >= min_count:
                score = float(abs_sv[mask, j].mean())
            else:
                score = feature_imp[full_name] * 0.5  # small-sample shrinkage
            rows.append({"source_feature": source, "category": category, "e_raw": score,
                         "n_explained": int(mask.sum()), "basis": "category"})
        elif name == "speed_limit":
            limits = X_sample["speed_limit"].to_numpy(dtype=float)
            for label, mask in _speed_bins(limits):
                if mask.sum() >= min_count:
                    score = float(abs_sv[mask, j].mean())
                else:
                    score = feature_imp[full_name] * 0.5
                rows.append({"source_feature": "speed_limit", "category": label, "e_raw": score,
                             "n_explained": int(mask.sum()), "basis": "numeric-bin"})
    return pd.DataFrame(rows)


def _speed_bins(values: np.ndarray):
    specs = [
        ("≤20 mph", values <= 20), ("30 mph", (values > 20) & (values <= 30)),
        ("40 mph", (values > 30) & (values <= 40)), ("50 mph", (values > 40) & (values <= 50)),
        ("60 mph", (values > 50) & (values <= 60)), ("70+ mph", values > 60),
    ]
    return specs


def global_importance(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    abs_sv = np.abs(values)
    frame = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": abs_sv.mean(axis=(0, 2)),
        "mean_abs_shap_fatal": abs_sv[:, :, 0].mean(axis=0),
        "mean_abs_shap_serious": abs_sv[:, :, 1].mean(axis=0),
        "mean_abs_shap_slight": abs_sv[:, :, 2].mean(axis=0),
    })
    return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_instance(model, feature_frame: pd.DataFrame):
    """SHAP contributions for one or more feature rows: (n, features, classes)."""
    values, names, _ = shap_values_for(model, feature_frame)
    return values, names


def generate_explainability(bundle: dict, X_test: pd.DataFrame) -> dict:
    model = bundle["best_model"]
    n = min(config.SHAP_EXPLAIN_ROWS, len(X_test))
    X_sample = X_test.sample(n=n, random_state=config.RANDOM_SEED)
    values, names, kind = shap_values_for(model, X_sample)
    if values is None:
        log.error("SHAP unavailable for %s", bundle["best_name"])
        return {"available": False, "explainer": kind, "model": bundle["best_name"]}

    global_df = global_importance(values, names)
    category_df = category_importance(values, names, X_sample)
    global_df.to_csv(config.REPORTS_DIR / "shap_global_importance.csv", index=False)
    category_df.to_csv(config.REPORTS_DIR / "shap_category_importance.csv", index=False)
    sample_out = X_sample.copy()
    y_test = bundle.get("y_test")
    if y_test is not None:
        sample_out["actual_class_index"] = y_test.loc[X_sample.index].to_numpy()
    sample_out.to_csv(config.PROCESSED_DIR / "shap_explain_sample.csv", index=False)
    np.save(config.MODELS_DIR / "shap_values.npy", values.astype(np.float32))
    meta = {
        "available": True,
        "explainer": kind,
        "model": bundle["best_name"],
        "n_explained": int(n),
        "n_features": len(names),
        "feature_names": names,
        "class_order": config.SEVERITY_ORDER,
    }
    with (config.MODELS_DIR / "shap_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    log.info("SHAP (%s) computed on %s rows; %s condition scores", kind, n, len(category_df))
    return {**meta, "global": global_df, "category": category_df}
