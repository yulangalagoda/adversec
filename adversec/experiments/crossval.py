"""
crossval.py
===========
Cross-validation schemes.

Scheme B (row-level): stratified K-fold over the duplicated rows -- copies of a
signature leak across folds, so results are clean but inflated (never reported as
truth). Scheme A (signature-level): K-fold over unique signatures with light
duplication applied inside each fold's train portion only -- no signature copy
crosses the fold boundary. The gap between A and B quantifies the accuracy trap.

The defended and per-class functions extend this to attack/defence robustness.
Robust-support classes are passed in (not hard-coded), so this module is
dataset-agnostic.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from .. import config
from ..contract import FEATURES, LABEL_COLUMN
from ..models import CNN1D, build_random_forest, train_cnn
from ..pipeline import duplicate_train_classes, encode_labels, scale_features
from . import attack as attacks
from . import defense


def _macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _predict_cnn(model, X, device):
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        return model(xt).argmax(dim=1).cpu().numpy()


def crossval_rowlevel(X, y, n_splits=5, device="cpu", random_seed=42, cnn_epochs=50, class_weights=None):
    """Scheme B: stratified K-fold over the duplicated rows directly (leaky contrast)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    rf_scores, cnn_scores = [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        rf = build_random_forest(random_seed=random_seed)
        rf.fit(X_tr, y_tr)
        rf_scores.append(_macro_f1(y_te, rf.predict(X_te)))

        cnn = CNN1D(n_features=X.shape[1], n_classes=len(np.unique(y)))
        cnn = train_cnn(cnn, X_tr, y_tr, n_epochs=cnn_epochs, device=device,
                        class_weights=class_weights, random_seed=random_seed)
        cnn_scores.append(_macro_f1(y_te, _predict_cnn(cnn, X_te, device)))
        print(f"    fold {fold}: RF={rf_scores[-1]:.4f}     CNN={cnn_scores[-1]:.4f}")
    return {"rf": rf_scores, "cnn": cnn_scores}


def crossval_signature_level(
    strict_df, feature_columns=FEATURES,
    n_splits=2, dup_target=200, device="cpu",
    random_seed=42, cnn_epochs=50,
    use_duplication=True, use_class_weights=False, use_anova=False, anova_k=5,
):
    """Scheme A: K-fold over unique signatures; duplicate inside train folds only."""
    rf_scores, cnn_scores = [], []
    y_sig = strict_df[LABEL_COLUMN].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(strict_df, y_sig), start=1):
        train_sig = strict_df.iloc[tr_idx].reset_index(drop=True)
        test_sig = strict_df.iloc[te_idx].reset_index(drop=True)

        if use_duplication:
            train_dup = duplicate_train_classes(train_sig, target_per_class=dup_target, random_seed=random_seed)
        else:
            train_dup = train_sig.copy()

        y_tr, y_te, _ = encode_labels(train_dup, test_sig)
        X_tr, X_te, _ = scale_features(train_dup, test_sig, feature_columns)

        if use_anova:
            selector = SelectKBest(score_func=f_classif, k=anova_k)
            X_tr = selector.fit_transform(X_tr, y_tr)
            X_te = selector.transform(X_te)
            if fold == 1:
                selected = [feature_columns[i] for i in selector.get_support(indices=True)]
                print(f"    ANOVA selected (fold 1): {selected}")

        rf = build_random_forest(random_seed=random_seed)
        rf.fit(X_tr, y_tr)
        rf_scores.append(_macro_f1(y_te, rf.predict(X_te)))

        cw = None
        if use_class_weights:
            w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
            cw = torch.tensor(w, dtype=torch.float32, device=device)
        cnn = CNN1D(n_features=X_tr.shape[1], n_classes=len(np.unique(y_tr)))
        cnn = train_cnn(cnn, X_tr, y_tr, n_epochs=cnn_epochs, device=device,
                        class_weights=cw, random_seed=random_seed)
        cnn_scores.append(_macro_f1(y_te, _predict_cnn(cnn, X_te, device)))
        print(f"    fold {fold}: RF={rf_scores[-1]:.4f}     CNN={cnn_scores[-1]:.4f}")
    return {"rf": rf_scores, "cnn": cnn_scores}


def crossval_defended(
    strict_df, feature_columns=FEATURES,
    robust_support=("DoS", "benign", "spoofing-RPM"),
    attack_eps=0.10, n_splits=2, dup_target=200, device="cpu",
    random_seed=42, cnn_epochs=50,
):
    """
    Cross-validate the full attack+defence pipeline on robust-support macro-F1.

    robust_support: class NAMES that have enough held-out test signatures to be a
    real measurement (>=2 test frames). Pass the dataset's own set.
    """
    y_sig = strict_df[LABEL_COLUMN].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    keys = ["base_clean", "base_adv", "pgd_clean", "pgd_adv",
            "multi_clean", "multi_adv", "rf_clean", "rf_adv"]
    out = {k: [] for k in keys}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(strict_df, y_sig), start=1):
        train_sig = strict_df.iloc[tr_idx].reset_index(drop=True)
        test_sig = strict_df.iloc[te_idx].reset_index(drop=True)

        train_dup = duplicate_train_classes(train_sig, target_per_class=dup_target, random_seed=random_seed)
        y_tr, y_te, enc = encode_labels(train_dup, test_sig)
        X_tr, X_te, _ = scale_features(train_dup, test_sig, feature_columns)

        names = list(enc.classes_)
        robust = [i for i, n in enumerate(names) if n in set(robust_support)]

        w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw = torch.tensor(w, dtype=torch.float32, device=device)

        base = CNN1D(n_features=len(feature_columns), n_classes=len(np.unique(y_tr)))
        base = train_cnn(base, X_tr, y_tr, n_epochs=cnn_epochs, device=device,
                         class_weights=cw, random_seed=random_seed)
        def_pgd = defense.adversarial_train_cnn(X_tr, y_tr, strategy="pgd", n_epochs=cnn_epochs,
                                                device=device, class_weights=cw, random_seed=random_seed)
        def_multi = defense.adversarial_train_cnn(X_tr, y_tr, strategy="multi", n_epochs=cnn_epochs,
                                                  device=device, class_weights=cw, random_seed=random_seed)
        rf = build_random_forest(random_seed=random_seed)
        rf.fit(X_tr, y_tr)

        clf = attacks.wrap_cnn_for_art(base, n_features=len(feature_columns),
                                       n_classes=len(np.unique(y_tr)), device=device)
        X_te_adv = attacks.generate_pgd(clf, X_te, epsilon=attack_eps)

        def cnn_rf1(m, X):
            p = _predict_cnn(m, X, device)
            return f1_score(y_te, p, labels=robust, average="macro", zero_division=0)

        def rf_rf1(X):
            return f1_score(y_te, rf.predict(X), labels=robust, average="macro", zero_division=0)

        print(f"  fold {fold}: robust labels = {robust} (names={names})")
        out["base_clean"].append(cnn_rf1(base, X_te));   out["base_adv"].append(cnn_rf1(base, X_te_adv))
        out["pgd_clean"].append(cnn_rf1(def_pgd, X_te)); out["pgd_adv"].append(cnn_rf1(def_pgd, X_te_adv))
        out["multi_clean"].append(cnn_rf1(def_multi, X_te)); out["multi_adv"].append(cnn_rf1(def_multi, X_te_adv))
        out["rf_clean"].append(rf_rf1(X_te));            out["rf_adv"].append(rf_rf1(X_te_adv))
        print(f"  fold {fold}: base_adv={out['base_adv'][-1]:.4f} pgd_adv={out['pgd_adv'][-1]:.4f} "
              f"multi_adv={out['multi_adv'][-1]:.4f} rf_adv={out['rf_adv'][-1]:.4f}")
    return out


def crossval_perclass_robustness(
    strict_df, feature_columns, class_names,
    epsilons, attack="pgd",
    n_splits=5, dup_target=200, device="cpu", random_seed=42, cnn_epochs=50,
    use_duplication=True,
):
    """
    Cross-validate PER-CLASS F1 under attack across a diversity gradient.
    Returns results[epsilon][class_index] = list of per-fold F1s (+ a "clean" key).

    use_duplication=False skips the light-duplication convergence crutch entirely,
    training on the real (unpadded) train-fold signatures only -- for ablating
    whether a result depends on that crutch rather than on the underlying data.
    """
    n_classes = len(class_names)
    results = {"clean": [[] for _ in range(n_classes)]}
    for eps in epsilons:
        results[eps] = [[] for _ in range(n_classes)]

    y_sig = strict_df[LABEL_COLUMN].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(strict_df, y_sig), start=1):
        train_sig = strict_df.iloc[tr_idx].reset_index(drop=True)
        test_sig = strict_df.iloc[te_idx].reset_index(drop=True)

        if use_duplication:
            train_dup = duplicate_train_classes(train_sig, target_per_class=dup_target, random_seed=random_seed)
        else:
            train_dup = train_sig.copy()
        y_tr, y_te, _ = encode_labels(train_dup, test_sig)
        X_tr, X_te, _ = scale_features(train_dup, test_sig, feature_columns)

        w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw = torch.tensor(w, dtype=torch.float32, device=device)

        cnn = CNN1D(n_features=len(feature_columns), n_classes=n_classes)
        cnn = train_cnn(cnn, X_tr, y_tr, n_epochs=cnn_epochs, device=device,
                        class_weights=cw, random_seed=random_seed)
        clf = attacks.wrap_cnn_for_art(cnn, n_features=len(feature_columns), n_classes=n_classes, device=device)
        X_te_f = X_te.astype(np.float32)

        clean_per = f1_score(y_te, clf.predict(X_te_f).argmax(axis=1),
                             average=None, labels=range(n_classes), zero_division=0)
        for c in range(n_classes):
            results["clean"][c].append(clean_per[c])

        for eps in epsilons:
            X_adv = (attacks.generate_pgd(clf, X_te_f, epsilon=eps) if attack == "pgd"
                     else attacks.generate_fgsm(clf, X_te_f, epsilon=eps))
            per = f1_score(y_te, clf.predict(X_adv).argmax(axis=1),
                           average=None, labels=range(n_classes), zero_division=0)
            for c in range(n_classes):
                results[eps][c].append(per[c])
        print(f"  fold {fold}/{n_splits} done")
    return results


def crossval_defence_comparison(
    strict_df, feature_columns, class_names,
    epsilons, eps_meaningful, eps_full,
    n_splits=5, dup_target=200, device="cpu", random_seed=42, cnn_epochs=50,
):
    """
    Cross-validate the per-class defence comparison under PGD: baseline vs
    adversarial training at eps_meaningful vs eps_full, white-box.
    Returns results[model][epsilon][class_index] = list of per-fold F1s.
    """
    n_classes = len(class_names)
    model_keys = ["baseline", "def_meaning", "def_full"]
    eps_keys = ["clean"] + list(epsilons)
    results = {m: {e: [[] for _ in range(n_classes)] for e in eps_keys} for m in model_keys}

    y_sig = strict_df[LABEL_COLUMN].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(strict_df, y_sig), start=1):
        train_sig = strict_df.iloc[tr_idx].reset_index(drop=True)
        test_sig = strict_df.iloc[te_idx].reset_index(drop=True)

        train_dup = duplicate_train_classes(train_sig, target_per_class=dup_target, random_seed=random_seed)
        y_tr, y_te, _ = encode_labels(train_dup, test_sig)
        X_tr, X_te, _ = scale_features(train_dup, test_sig, feature_columns)
        X_te_f = X_te.astype(np.float32)

        w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw = torch.tensor(w, dtype=torch.float32, device=device)

        baseline = CNN1D(n_features=len(feature_columns), n_classes=n_classes)
        baseline = train_cnn(baseline, X_tr, y_tr, n_epochs=cnn_epochs, device=device,
                             class_weights=cw, random_seed=random_seed)
        def_meaning = defense.adversarial_train_cnn(X_tr, y_tr, strategy="pgd", epsilons=eps_meaningful,
                                                    n_epochs=cnn_epochs, device=device, class_weights=cw,
                                                    random_seed=random_seed)
        def_full = defense.adversarial_train_cnn(X_tr, y_tr, strategy="pgd", epsilons=eps_full,
                                                 n_epochs=cnn_epochs, device=device, class_weights=cw,
                                                 random_seed=random_seed)
        fold_models = {"baseline": baseline, "def_meaning": def_meaning, "def_full": def_full}

        for mkey, model in fold_models.items():
            clf = attacks.wrap_cnn_for_art(model, n_features=len(feature_columns), n_classes=n_classes, device=device)
            clean_per = f1_score(y_te, clf.predict(X_te_f).argmax(axis=1),
                                 average=None, labels=range(n_classes), zero_division=0)
            for c in range(n_classes):
                results[mkey]["clean"][c].append(clean_per[c])
            for eps in epsilons:
                X_adv = attacks.generate_pgd(clf, X_te_f, epsilon=eps)
                per = f1_score(y_te, clf.predict(X_adv).argmax(axis=1),
                               average=None, labels=range(n_classes), zero_division=0)
                for c in range(n_classes):
                    results[mkey][eps][c].append(per[c])
        print(f"  fold {fold}/{n_splits} done")
    return results
