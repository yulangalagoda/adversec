"""
baseline.py
===========
Baseline experiment: train the Random Forest and 1D-CNN on a dataset's prepared
arrays, evaluate on the held-out test set, and save a metrics report. This is the
clean-data reference the adversarial stages are later compared against.

The CNN uses balanced class weights only when the dataset config asks for it
(cnn_class_weights): CICIoV2024 is severely imbalanced and needs it; ROAD is
cleanly separable and does not. The Random Forest always uses class_weight
"balanced" (built into build_random_forest).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import torch

from .. import config
from ..evaluation import evaluate_model
from ..models import CNN1D, build_random_forest, train_cnn


def _load_arrays(name: str):
    proc = config.PROCESSED_DIR
    arrays = np.load(proc / f"{name}_stage2_arrays.npz")
    encoder = joblib.load(proc / f"{name}_label_encoder.joblib")
    return arrays, list(encoder.classes_)


def run_baseline(name: str, device: str = "cpu", cnn_epochs: int = 50,
                 use_class_weights=None, save: bool = True) -> dict:
    """Train + evaluate RF and CNN baselines for a prepared dataset; return/save metrics."""
    cfg = config.load_dataset_config(name)
    if use_class_weights is None:
        use_class_weights = bool(cfg.get("cnn_class_weights", False))

    arrays, class_names = _load_arrays(name)
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]

    # --- Random Forest (class_weight="balanced" is built in) ---
    rf = build_random_forest(random_seed=config.RANDOM_SEED)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(y_test, rf.predict(X_test), class_names,
                                model_name=f"{name} Random Forest")

    # --- 1D-CNN ---
    class_weights = None
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    cnn = CNN1D(n_features=X_train.shape[1], n_classes=len(class_names))
    cnn = train_cnn(cnn, X_train, y_train, n_epochs=cnn_epochs, device=device,
                    class_weights=class_weights, random_seed=config.RANDOM_SEED)
    cnn.eval()
    with torch.no_grad():
        pred = cnn(torch.tensor(X_test, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    cnn_metrics = evaluate_model(y_test, pred, class_names, model_name=f"{name} 1D-CNN")

    report = {
        "dataset": name,
        "device": device,
        "n_classes": len(class_names),
        "class_names": class_names,
        "test_size": int(len(y_test)),
        "random_forest": {k: rf_metrics[k] for k in ("accuracy", "macro_f1", "confusion_matrix")},
        "cnn_1d": {k: cnn_metrics[k] for k in ("accuracy", "macro_f1", "confusion_matrix")},
    }
    if save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.RESULTS_DIR / f"{name}_baseline_metrics.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print("saved baseline metrics ->", path)
    return report
