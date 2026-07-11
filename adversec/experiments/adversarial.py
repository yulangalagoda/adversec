"""
adversarial.py
==============
Adversarial-robustness experiment for one dataset. Reproduces the numeric content
of <name>_adversarial_results.json:

  1. cross-validated per-class F1 under PGD across the epsilon sweep (the
     robustness gradient),
  2. the distance-to-benign mechanism measurement (per-class L2 distance from
     each attack signature to the nearest benign signature, vs robustness),
  3. physical threat-sizing (integer rounding + per-ID benign-envelope rejection).

Reads the prepared artifacts written by `adversec prep` (or notebook 06):
<name>_strict.csv, <name>_train_dup.csv, <name>_test.csv, <name>_stage2_arrays.npz,
<name>_feature_scaler.joblib, <name>_label_encoder.joblib.

Note: threat-sizing numbers differ from any pre-fix committed run, because the
realism integer-clip is now per-feature -- the arbitration ID is no longer crushed
to a byte range. That change is intentional.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

from .. import config
from ..contract import DATA_COLUMNS, FEATURES, ID_COLUMN, LABEL_COLUMN
from ..models import CNN1D, train_cnn
from . import attack as attacks
from . import realism
from .crossval import crossval_perclass_robustness


def _proc():
    return config.PROCESSED_DIR


def _load(name: str):
    proc = _proc()
    strict = pd.read_csv(proc / f"{name}_strict.csv")
    train = pd.read_csv(proc / f"{name}_train_dup.csv")
    test = pd.read_csv(proc / f"{name}_test.csv")
    arrays = np.load(proc / f"{name}_stage2_arrays.npz")
    scaler = joblib.load(proc / f"{name}_feature_scaler.joblib")
    encoder = joblib.load(proc / f"{name}_label_encoder.joblib")
    return strict, train, test, arrays, scaler, list(encoder.classes_)


def _train_baseline(arrays, class_names, name, device, cnn_epochs):
    cfg = config.load_dataset_config(name)
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    class_weights = None
    if bool(cfg.get("cnn_class_weights", False)):
        from sklearn.utils.class_weight import compute_class_weight
        w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    cnn = CNN1D(n_features=X_train.shape[1], n_classes=len(class_names))
    return train_cnn(cnn, X_train, y_train, n_epochs=cnn_epochs, device=device,
                     class_weights=class_weights, random_seed=config.RANDOM_SEED)


def cv_perclass_pgd(name, epsilons, n_splits, device, cnn_epochs):
    """Cross-validated per-class F1 under PGD; returns (formatted_table, class_names, raw_results, sig_counts)."""
    strict, _, _, _, _, class_names = _load(name)
    results = crossval_perclass_robustness(
        strict, FEATURES, class_names, epsilons, attack="pgd",
        n_splits=n_splits, device=device, cnn_epochs=cnn_epochs, random_seed=config.RANDOM_SEED,
    )
    sig = {str(k): int(v) for k, v in strict[LABEL_COLUMN].value_counts().items()}
    table = {}
    eps_keys = ["clean"] + list(epsilons)
    for i, cname in enumerate(class_names):
        table[cname] = {"signatures": sig.get(cname, 0), "by_epsilon": {}}
        for e in eps_keys:
            arr = np.array(results[e][i])
            table[cname]["by_epsilon"][str(e)] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "folds": [float(x) for x in arr],
            }
    return table, class_names, results, sig


def distance_to_benign(name, cv_results, class_names, sig, benign_label="benign", robust_eps=0.01):
    """Per-class L2 distance to nearest benign signature, lined up against robustness@robust_eps."""
    _, train, test, _, scaler, _ = _load(name)

    def scaled(df):
        return scaler.transform(df[FEATURES].values).astype(np.float32)

    train_benign = train[train[LABEL_COLUMN] == benign_label]
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(scaled(train_benign))
    rob = {class_names[i]: float(np.mean(cv_results[robust_eps][i])) for i in range(len(class_names))}

    per_class, dists, robs = {}, [], []
    for cname in class_names:
        if cname == benign_label:
            continue
        atk = test[test[LABEL_COLUMN] == cname]
        if len(atk) == 0:
            continue
        d, _ = nn.kneighbors(scaled(atk))
        d = d.ravel()
        per_class[cname] = {
            "signatures": sig.get(cname, 0),
            "mean_distance": round(float(d.mean()), 4),
            "median_distance": round(float(np.median(d)), 4),
            "robustness_at_0.01": round(rob[cname], 3),
        }
        dists.append(d.mean())
        robs.append(rob[cname])

    r = float(np.corrcoef(dists, robs)[0, 1]) if len(dists) >= 2 else float("nan")
    return {
        "reference_pool": "training benign signatures",
        "epsilon_for_robustness": robust_eps,
        "per_class": per_class,
        "pearson_r": round(r, 3),
        "n_classes": len(per_class),
    }


def threat_sizing(name, epsilons, device, cnn_epochs, benign_label="benign"):
    """Integer-rounding + per-ID benign-envelope rejection, per epsilon (uses the fixed per-feature clip)."""
    _, train, test, arrays, scaler, class_names = _load(name)
    cnn = _train_baseline(arrays, class_names, name, device, cnn_epochs)
    clf = attacks.wrap_cnn_for_art(cnn, n_features=len(FEATURES), n_classes=len(class_names), device=device)

    X_test = arrays["X_test"].astype(np.float32)
    y_test = arrays["y_test"]

    benign_all = pd.concat(
        [train[train[LABEL_COLUMN] == benign_label], test[test[LABEL_COLUMN] == benign_label]],
        ignore_index=True,
    )
    ranges = realism.learn_observed_ranges(benign_all, ID_COLUMN, DATA_COLUMNS)
    id_idx = FEATURES.index(ID_COLUMN)
    data_idx = [FEATURES.index(c) for c in DATA_COLUMNS]

    def mf1(pred):
        return f1_score(y_test, pred, average="macro", zero_division=0)

    rows = []
    for eps in epsilons:
        X_adv = attacks.generate_pgd(clf, X_test, epsilon=eps)
        f1_raw = mf1(clf.predict(X_adv).argmax(axis=1))
        X_round, X_int = realism.round_to_integer_frames(X_adv, scaler)
        f1_round = mf1(clf.predict(X_round).argmax(axis=1))
        mask = realism.observed_range_mask(X_int, ranges, id_idx, data_idx)
        rows.append({
            "eps": eps,
            "f1_raw_adversarial": round(float(f1_raw), 3),
            "f1_rounded_integer": round(float(f1_round), 3),
            "pct_rejected_by_envelope": round(100.0 * (~mask).sum() / len(mask), 1),
            "n_survivors": int(mask.sum()),
        })

    return {
        "benign_reference_frames": int(len(benign_all)),
        "benign_ids": len(ranges),
        "test_frames": int(len(y_test)),
        "by_epsilon": rows,
        "note": "per-feature integer clip (arbitration ID not crushed to a byte range)",
    }


def run_adversarial(name, device="cpu", cnn_epochs=50, n_splits=5, save=True) -> dict:
    """Full adversarial-robustness experiment; reproduces <name>_adversarial_results.json numerics."""
    epsilons = config.FGSM_EPSILONS
    cfg = config.load_dataset_config(name)
    benign = cfg["benign_label"]

    cv_table, class_names, cv_results, sig = cv_perclass_pgd(name, epsilons, n_splits, device, cnn_epochs)
    dist = distance_to_benign(name, cv_results, class_names, sig, benign_label=benign)
    threat = threat_sizing(name, epsilons, device, cnn_epochs, benign_label=benign)

    report = {
        "dataset": name,
        "device": device,
        "attack_config": {
            "pgd_epsilon_sweep": epsilons,
            "pgd_step_size": config.PGD_STEP_SIZE,
            "pgd_max_iter": config.PGD_MAX_ITER,
            "n_folds": n_splits,
            "seed": config.RANDOM_SEED,
        },
        "cross_validated_per_class_pgd": cv_table,
        "mechanism_distance_to_benign": dist,
        "threat_sizing": threat,
    }
    if save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.RESULTS_DIR / f"{name}_adversarial_results.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print("saved adversarial results ->", path)
    return report
