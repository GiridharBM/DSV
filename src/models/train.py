"""Train, compare and persist severity classifiers with a chronological holdout."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src import config
from src.evaluation.metrics import evaluate, selection_score
from src.preprocessing.encoder import make_preprocessor
from src.utils.logging import get_logger

log = get_logger(__name__)

LABEL_TO_INT = {label: i for i, label in enumerate(config.SEVERITY_ORDER)}
INT_TO_LABEL = {i: label for label, i in LABEL_TO_INT.items()}


def build_models(seed: int = config.RANDOM_SEED) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, C=0.8, class_weight="balanced", random_state=seed, solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180, max_depth=14, min_samples_leaf=12, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350, max_depth=6, learning_rate=0.07, subsample=0.85,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.5,
            tree_method="hist", eval_metric="mlogloss", n_jobs=-1,
            random_state=seed, objective="multi:softprob", verbosity=0,
        ),
    }


def chronological_split(features: pd.DataFrame, labels: pd.Series, years: pd.Series):
    train_mask = years <= config.TRAIN_THROUGH_YEAR
    test_mask = years > config.TRAIN_THROUGH_YEAR
    if not test_mask.any():
        raise ValueError("Chronological split produced an empty test set.")
    return features[train_mask], features[test_mask], labels[train_mask], labels[test_mask]


def cap_training_rows(X, y, limit: int = config.MAX_MODEL_ROWS, seed: int = config.RANDOM_SEED):
    """Uniform, seeded cap on training rows; keeps the full chronological test set."""
    if len(X) <= limit:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=limit, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def train_all(features: pd.DataFrame, labels: pd.Series, years: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = chronological_split(features, labels, years)
    X_train, y_train = cap_training_rows(X_train, y_train)
    log.info("Training rows=%s, test rows=%s (test year(s)=%s)", len(X_train), len(X_test), sorted(years[years > config.TRAIN_THROUGH_YEAR].unique()))

    results, fitted = {}, {}
    for name, estimator in build_models().items():
        started = time.time()
        try:
            pipeline = Pipeline([("prep", make_preprocessor()), ("clf", estimator)])
            fit_kwargs = {}
            if name == "XGBoost":
                weights = compute_sample_weight("balanced", y_train)
                fit_kwargs["clf__sample_weight"] = weights
            pipeline.fit(X_train, y_train, **fit_kwargs)
            proba = pipeline.predict_proba(X_test)
            pred = pipeline.predict(X_test)
            metrics = evaluate(y_test, pred, proba)
            results[name] = metrics
            fitted[name] = pipeline
            log.info("%s: macro-F1=%.4f accuracy=%.4f fatal-recall=%.4f (%.1fs)",
                     name, metrics["macro_f1"], metrics["accuracy"], metrics["recall"]["Fatal"], time.time() - started)
        except Exception as exc:
            log.exception("Training failed for %s: %s", name, exc)

    if not results:
        raise RuntimeError("No model trained successfully.")

    best_name = max(results, key=lambda n: selection_score(results[n]))
    best = fitted[best_name]
    log.info("Selected model: %s (selection score %.4f)", best_name, selection_score(results[best_name]))
    return {
        "results": results, "fitted": fitted, "best_name": best_name, "best_model": best,
        "X_test": X_test, "y_test": y_test, "X_train": X_train, "y_train": y_train,
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
    }


def save_artifacts(bundle: dict) -> dict:
    best = bundle["best_model"]
    joblib.dump(best, config.MODELS_DIR / "best_model.joblib")
    for name, model in bundle["fitted"].items():
        safe = name.lower().replace(" ", "_")
        joblib.dump(model, config.MODELS_DIR / f"model_{safe}.joblib")

    meta = {
        "best_model": bundle["best_name"],
        "selection_rule": "macro_f1 + 0.05*(fatal_recall + serious_recall)",
        "label_mapping": INT_TO_LABEL,
        "feature_columns": config.FEATURE_COLUMNS,
        "train_rows": bundle["train_rows"],
        "test_rows": bundle["test_rows"],
        "train_through_year": config.TRAIN_THROUGH_YEAR,
        "random_seed": config.RANDOM_SEED,
        "results": bundle["results"],
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    with (config.MODELS_DIR / "model_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)
    return meta


def load_best_model():
    path = config.MODELS_DIR / "best_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)
