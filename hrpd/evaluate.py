"""Evaluation metrics for HRPD predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, roc_auc_score


def compute_metrics(
    probs: np.ndarray, y_true: np.ndarray, n_classes: int
) -> dict[str, object]:
    """Compute the full metric set from class probabilities.

    Args:
        probs: Class probabilities, shape ``(n_samples, n_classes)``.
        y_true: Integer ground-truth labels, shape ``(n_samples,)``.
        n_classes: Number of classes.

    Returns:
        Dict with ``balanced_accuracy`` (mean per-class recall),
        ``per_class_recall`` (list), ``macro_f1``, ``cohen_kappa`` and
        ``auroc_macro_ovr`` (``None`` when undefined, e.g. a class is
        absent from ``y_true``).
    """
    y_pred = probs.argmax(1)
    recall = recall_score(
        y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0
    )
    try:
        auroc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        if np.isnan(auroc):
            auroc = None
    except ValueError:
        auroc = None
    return {
        "balanced_accuracy": float(np.mean(recall)),
        "per_class_recall": [float(r) for r in recall],
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "auroc_macro_ovr": auroc,
    }


def summarise_seeds(per_seed: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-seed metric dicts into mean/std summaries.

    Args:
        per_seed: List of dicts as returned by :func:`compute_metrics`.

    Returns:
        Dict with ``balanced_accuracy_mean``, ``balanced_accuracy_std``
        (population std, matching the paper) and the per-class recall
        means, plus the raw per-seed balanced accuracies.
    """
    bal = np.array([m["balanced_accuracy"] for m in per_seed])
    recalls = np.array([m["per_class_recall"] for m in per_seed])
    return {
        "balanced_accuracy_mean": float(bal.mean()),
        "balanced_accuracy_std": float(bal.std()),
        "per_class_recall_mean": [float(r) for r in recalls.mean(0)],
        "per_seed_balanced_accuracy": [float(b) for b in bal],
    }
