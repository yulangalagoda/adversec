"""
evaluation.py
=============
Shared metrics, used identically for every model so comparisons are apples to
apples.

evaluate_model reports accuracy, macro-F1, per-class precision/recall/F1 and the
confusion matrix. Overall accuracy is reported but never trusted alone: on a test
set dominated by benign frames it is misleading, so macro-F1 and the per-class
numbers carry the weight.

robust_support_f1 is the macro-F1 over only the classes with enough held-out test
signatures to be a real measurement (>=2 test frames), passed in as label ids. It
prevents single-frame classes from swinging the headline either way.
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_model(y_true, y_pred, class_names, model_name: str = "model") -> dict:
    """Compute and print the full metric set for one model's predictions."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'=' * 60}\n    {model_name}\n{'=' * 60}")
    print(f"    Accuracy    : {acc:.4f}")
    print(f"    Macro-F1    : {macro_f1:.4f}")
    print("\n   Per-class report:")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    ))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print("     Confusion matrix (rows=true, cols=pred):")
    print(cm)

    return {
        "model": model_name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
    }


def robust_support_f1(y_true, y_pred, robust_labels) -> float:
    """Macro-F1 over only the robust-support classes (>=2 test frames), by label id."""
    return f1_score(
        y_true, y_pred,
        labels=list(robust_labels),
        average="macro",
        zero_division=0,
    )
