"""Evaluation metrics, class-wise reporting and confusion matrices."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import config


def selection_score(metrics: dict) -> float:
    """Macro-F1 plus a small explicit bonus for Fatal/Serious recall."""
    return metrics["macro_f1"] + 0.05 * (metrics["recall"]["Fatal"] + metrics["recall"]["Serious"])


def evaluate(y_true, y_pred, y_proba=None, labels=None) -> dict:
    labels = list(labels or config.SEVERITY_ORDER)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision = precision_score(y_true, y_pred, labels=range(len(labels)), average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=range(len(labels)), average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=range(len(labels)), average=None, zero_division=0)
    support = np.bincount(y_true, minlength=len(labels))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": {labels[i]: float(precision[i]) for i in range(len(labels))},
        "recall": {labels[i]: float(recall[i]) for i in range(len(labels))},
        "f1": {labels[i]: float(f1[i]) for i in range(len(labels))},
        "support": {labels[i]: int(support[i]) for i in range(len(labels))},
        "macro_f1": float(f1_score(y_true, y_pred, labels=range(len(labels)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=range(len(labels)), average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=range(len(labels)), target_names=labels, output_dict=True, zero_division=0
        ),
        "n_test": int(len(y_true)),
    }


def metrics_to_frames(metrics: dict, model_name: str) -> dict[str, pd.DataFrame]:
    comparison = pd.DataFrame([{
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "fatal_recall": metrics["recall"]["Fatal"],
        "serious_recall": metrics["recall"]["Serious"],
        "selection_score": selection_score(metrics),
    }])
    classwise = pd.DataFrame([
        {
            "model": model_name, "class": label,
            "precision": metrics["precision"][label], "recall": metrics["recall"][label],
            "f1": metrics["f1"][label], "support": metrics["support"][label],
        }
        for label in config.SEVERITY_ORDER
    ])
    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"actual_{c}" for c in config.SEVERITY_ORDER],
        columns=[f"pred_{c}" for c in config.SEVERITY_ORDER],
    )
    return {"comparison": comparison, "classwise": classwise, "confusion": cm}
