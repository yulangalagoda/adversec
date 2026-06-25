"""
evaluation.py
=============
Stage 3 shared evaluation. One metrics function used identically for every model so all comparisons are apple-to-apples.

Reports accuracy, macro-F1, per-class precision/recall/F1, and the confusion matrix.
Overall accuracy is reported but never trusted alone: the test set is 709 benign vs 9 attack signatures, so accuracy is dominated by the majority.
"""

# Library import
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# The evaluation function
def evaluate_model(y_true, y_pred, class_names, model_name="model"):
    """
    Compute and print the full metric set for one model's predictions.

    Args:
        y_true: array of true integer labels.
        y_pred: array of predicted integer labels.
        class_names: list of class-names strings in label-integer order (from the saved LabelEncoder.classes_).
        model_name: a label for the printout.

    Returns:
        A dict of the headline metrics, for later comparison.
    """

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)

    # Macro-F1
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"    {model_name}")
    print(f"{'='*60}")
    print(f"    Accuracy    : {acc:.4f}")
    print(f"    Macro-F1    : {macro_f1:.4f}")

    # Per-class precision/recall/F1
    print("\n   Per-class report:")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    ))

    # Confusion matrix: rows = true class, columns = predicted class.
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print("     Confusion matrix (rows=true, cols=pred):")
    print(cm)

    return {
        "model": model_name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
    }