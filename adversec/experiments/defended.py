"""
defended.py
===========
Defended-model experiment: adversarial training vs baseline vs Random Forest,
cross-validated. The experiment SHAPE is dataset-specific and declared in the
dataset config under `defence`:

  robust_support_cv  (CICIoV2024): robust-support macro-F1 for the base CNN, two
      defended CNNs (PGD-only, multi) and the RF, under PGD at one epsilon.
      Reproduces "AT gives the CNN a robustness advantage the RF cannot match".

  perclass_comparison (ROAD): per-class F1 for baseline vs adversarial training at
      two epsilon ranges, white-box, across the sweep. Reproduces "AT fails / harms
      on ROAD".
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from .. import config
from ..contract import FEATURES
from .crossval import crossval_defence_comparison, crossval_defended


def _proc():
    return config.PROCESSED_DIR


def _mean_std(vals) -> dict:
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "folds": [float(x) for x in a]}


def run_defended(name, device="cpu", cnn_epochs=50, save=True) -> dict:
    cfg = config.load_dataset_config(name)
    dcfg = cfg.get("defence", {})
    mode = dcfg.get("mode")
    proc = _proc()
    strict = pd.read_csv(proc / f"{name}_strict.csv")
    class_names = list(joblib.load(proc / f"{name}_label_encoder.joblib").classes_)

    if mode == "robust_support_cv":
        out = crossval_defended(
            strict, FEATURES,
            robust_support=tuple(dcfg["robust_support"]),
            attack_eps=float(dcfg.get("attack_eps", 0.10)),
            n_splits=int(dcfg.get("n_splits", 2)),
            device=device, cnn_epochs=cnn_epochs,
        )
        report = {
            "dataset": name,
            "mode": mode,
            "robust_support": list(dcfg["robust_support"]),
            "attack_eps": float(dcfg.get("attack_eps", 0.10)),
            "n_splits": int(dcfg.get("n_splits", 2)),
            "results": {k: _mean_std(v) for k, v in out.items()},
        }

    elif mode == "perclass_comparison":
        epsilons = config.FGSM_EPSILONS
        res = crossval_defence_comparison(
            strict, FEATURES, class_names,
            epsilons=epsilons,
            eps_meaningful=list(dcfg["eps_meaningful"]),
            eps_full=list(dcfg["eps_full"]),
            n_splits=int(dcfg.get("n_splits", 5)),
            device=device, cnn_epochs=cnn_epochs,
        )
        eps_keys = ["clean"] + list(epsilons)
        table = {}
        for mkey, by_eps in res.items():
            table[mkey] = {
                str(e): {class_names[c]: _mean_std(by_eps[e][c]) for c in range(len(class_names))}
                for e in eps_keys
            }
        report = {
            "dataset": name,
            "mode": mode,
            "eps_meaningful": list(dcfg["eps_meaningful"]),
            "eps_full": list(dcfg["eps_full"]),
            "n_splits": int(dcfg.get("n_splits", 5)),
            "class_names": class_names,
            "results": table,
        }

    else:
        raise ValueError(f"unknown or missing defence.mode for '{name}': {mode!r}")

    if save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.RESULTS_DIR / f"{name}_defence_results.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print("saved defence results ->", path)
    return report
