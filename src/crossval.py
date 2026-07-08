"""
crossval.py
===========
Stage 3 cross-validation, two schemes:
    Scheme B (row-level): Standard stratified K-fold over the 3838 Duplicated rows. Copies of a signature leak across folds, so results are clean but inflated.
    Scheme A (signature-level): K-fold over the unique signatures, with light duplication applied inside each fold's training portion only.

The gap between A and B is itself a reported finding.
"""

# Library import
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import torch

import config
import models
import cleaning
import preprocessing
import defense
import attacks


# A helper to score one model's predictions
def _macro_f1(y_true, y_pred):
    """
    Macro-F1 for one fold's predictions
    """
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# Scheme B
def crossval_rowlevel(X, y, n_splits=5, device="cpu", random_seed=42, cnn_epochs=50, class_weights=None):
    """
    Scheme B: stratified K-fold over the duplicated rows directly.

    Leaks duplicated copies across folds.
    Returns per-fold macro-F1 lists for RF and CNN.
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    rf_scores = []
    cnn_scores = []

    # skf.split yields train/test index arrays for each fold, keeping class proportions roughly equal in every fold.
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # --- Random Forest ---
        rf = models.build_random_forest(random_seed=random_seed)
        rf.fit(X_tr, y_tr)
        rf_pred = rf.predict(X_te)
        rf_scores.append(_macro_f1(y_te, rf_pred))

        # --- 1D-CNN ---
        cnn = models.CNN1D(n_features=X.shape[1], n_classes=len(np.unique(y)))
        cnn = models.train_cnn(
            cnn, X_tr, y_tr,
            n_epochs=cnn_epochs, device=device,
            class_weights=class_weights, random_seed=random_seed
        )
        cnn.eval()
        with torch.no_grad():
            X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
            cnn_pred = cnn(X_te_t).argmax(dim=1).cpu().numpy()
        cnn_scores.append(_macro_f1(y_te, cnn_pred))

        print(f"    fold {fold}: RF={rf_scores[-1]:.4f}     CNN={cnn_scores[-1]:.4f}")
    
    return {"rf": rf_scores, "cnn": cnn_scores}

# Scheme A
def crossval_signature_level(
        strict_df,
        feature_columns,
        n_splits=2,
        dup_target=200,
        device="cpu",
        random_seed=42,
        cnn_epochs=50,
        use_duplication=True,
        use_class_weights=False,
        use_anova=False,
        anova_k=5,
):
    """
    Scheme A: K-fold over unique signatures, duplicate inside train folds only
    
    No signature copy crosses the fold boundary, so this is the high variance estimate.

    Returns per-fold macro-F1 lists for RN and CNN.
    """

    rf_scores = []
    cnn_scores = []

    # Stratify the signature-level split by class.
    y_sig = strict_df["true_class"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(strict_df, y_sig), start=1):
        # Split the unique signatures into this fold's train and test.
        train_sig = strict_df.iloc[tr_idx].reset_index(drop=True)
        test_sig = strict_df.iloc[te_idx].reset_index(drop=True)

        # Duplicate attack classes in the train signatures only
        if use_duplication:
            train_dup = cleaning.duplicate_train_classes(train_sig, target_per_class=dup_target, random_seed=random_seed)
        else:
            train_dup = train_sig.copy()

        # Encode labels and scale features, fitting on this fold's train only.
        y_tr, y_te, _ = preprocessing.encode_labels(train_dup, test_sig)
        X_tr, X_te, _ = preprocessing.scale_features(train_dup, test_sig, feature_columns)

        # Optional ANOVA F-test feature selection, fitted on train only (per-fold)
        if use_anova:
            selector = SelectKBest(score_func=f_classif, k=anova_k)
            X_tr = selector.fit_transform(X_tr, y_tr)   # fit on train
            X_te = selector.transform(X_te)             # apply same selection to test
            selected = [feature_columns[i] for i in selector.get_support(indices=True)]
            if fold == 1:
                print(f"    ANOVA selected (fold 1): {selected}")

        # --- Random Forest ---
        rf = models.build_random_forest(random_seed=random_seed)
        rf.fit(X_tr, y_tr)
        rf_scores.append(_macro_f1(y_te, rf.predict(X_te)))

        # --- 1D-CNN ---
        cnn = models.CNN1D(n_features=X_tr.shape[1], n_classes=len(np.unique(y_tr)))        
        
        # Optionally compute balance class weights for this fold's training labels.
        cw = None
        if use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            classes_arr = np.unique(y_tr)
            w = compute_class_weight(class_weight="balanced", classes=classes_arr, y=y_tr)
            cw = torch.tensor(w, dtype=torch.float32, device=device)
        cnn = models.train_cnn(cnn, X_tr, y_tr, n_epochs=cnn_epochs, device=device, random_seed=random_seed)
        cnn.eval()
        with torch.no_grad():
            X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
            cnn_pred = cnn(X_te_t).argmax(dim=1).cpu().numpy()
        cnn_scores.append(_macro_f1(y_te, cnn_pred))

        print(f"    fold {fold}: RF={rf_scores[-1]:.4f}     CNN={cnn_scores[-1]:.4f}")
    
    return {"rf": rf_scores, "cnn":cnn_scores}





# The defended-model CV
def crossval_defended(
        strict_df, feature_columns, attack_eps=0.10,
        n_splits=2, dup_target=200, device="cpu",
        random_seed=42, cnn_epochs=50,
):
    """
    Cross validate the full attack+defense pipleline.

    For each fold:
        split signatures
        duplicate trian only
        fit scaler/encoder on train only
        train base CNN
        build defended CNNs
        craft test-side PGD attack at attack_eps from that fold's baseline
        evaluate all 4 models on robust support macro-F1 under that attack

    Returns per-fold robust-support macro-F1 for each model, clean and attacked.
    """
    from sklearn.utils.class_weight import compute_class_weight

    y_sig = strict_df["true_class"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    out = {k: [] for k in ["base_clean","base_adv","pgd_clean","pgd_adv",
                            "multi_clean","multi_adv","rf_clean","rf_adv"]}
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(strict_df, y_sig), start=1):
        train_sig = strict_df.iloc[tr_idx].reset_index(drop=True)
        test_sig = strict_df.iloc[te_idx].reset_index(drop=True)

        # Duplicate train only and then preprocess fit-on-train-only
        train_dup = cleaning.duplicate_train_classes(
            train_sig, target_per_class=dup_target, random_seed=random_seed,
        )
        y_tr, y_te, enc = preprocessing.encode_labels(train_dup, test_sig)
        X_tr, X_te, _ = preprocessing.scale_features(train_dup, test_sig, feature_columns)

        # Resolve robust support classes
        names = list(enc.classes_)
        robust = [i for i, n in enumerate(names)
                  if n in ("DoS", "benign", "spoofing-RPM")]
        
        # per-fold class weights
        w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw = torch.tensor(w, dtype=torch.float32, device=device)

        # Base CNN
        base = models.CNN1D(n_features=len(feature_columns), n_classes=len(np.unique(y_tr)))
        base = models.train_cnn(
            base, X_tr, y_tr, n_epochs=cnn_epochs, device=device, class_weights=cw, random_seed=random_seed,
        )

        # Defended CNNs on both strategies
        def_pgd = defense.adversarial_train_cnn(
            X_tr, y_tr, strategy="pgd", n_epochs=cnn_epochs, device=device, class_weights=cw, random_seed=random_seed,
        )
        def_multi = defense.adversarial_train_cnn(
            X_tr, y_tr, strategy="multi", n_epochs=cnn_epochs, device=device, class_weights=cw, random_seed=random_seed,
        )

        # RF
        rf = models.build_random_forest(random_seed=random_seed)
        rf.fit(X_tr, y_tr)

        # Craft the test side PGD attack from this fold's baseline CNN
        clf = attacks.wrap_cnn_for_art(base, n_features=len(feature_columns), n_classes=len(np.unique(y_tr)), device=device)
        X_te_adv = attacks.generate_pgd(clf, X_te, epsilon=attack_eps)

        # Helper: robust support macro-F1 for a CNN model on given inputs
        def cnn_rf1(m, X):
            m.eval()
            with torch.no_grad():
                p = m(torch.tensor(X, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
            return f1_score(y_te, p, labels=robust, average="macro", zero_division=0)
        
        def rf_rf1(X):
            return f1_score(y_te, rf.predict(X), labels=robust, average="macro", zero_division=0)
            out["base_clean"].append(cnn_rf1(base, X_te));   out["base_adv"].append(cnn_rf1(base, X_te_adv))

        print(f"  fold {fold}: robust labels = {robust} (names={names})")

        out["base_clean"].append(cnn_rf1(base, X_te))
        out["base_adv"].append(cnn_rf1(base, X_te_adv))
        out["pgd_clean"].append(cnn_rf1(def_pgd, X_te))
        out["pgd_adv"].append(cnn_rf1(def_pgd, X_te_adv))
        out["multi_clean"].append(cnn_rf1(def_multi, X_te))
        out["multi_adv"].append(cnn_rf1(def_multi, X_te_adv))
        out["rf_clean"].append(rf_rf1(X_te))
        out["rf_adv"].append(rf_rf1(X_te_adv))

        print(f"  fold {fold}: base_adv={out['base_adv'][-1]:.4f} "
              f"pgd_adv={out['pgd_adv'][-1]:.4f} multi_adv={out['multi_adv'][-1]:.4f} "
              f"rf_adv={out['rf_adv'][-1]:.4f}")
        
    return out


# Per-class robustness cross-validation (new: records per-class F1 under attack)
def crossval_perclass_robustness(
        strict_df, feature_columns, class_names,
        epsilons, attack="pgd",
        n_splits=5, dup_target=200, device="cpu",
        random_seed=42, cnn_epochs=50,
):
    """
    Cross-validate PER-CLASS F1 under adversarial attack across a diversity
    gradient. For each fold: split signatures, duplicate train only, preprocess
    fit-on-train, train the CNN, wrap for ART, craft attack examples on the test
    set at each epsilon, and record per-class F1 for every class at every epsilon.

    Returns a nested dict: results[epsilon][class_index] = list of per-fold F1s.
    This lets us report mean +/- std per class per epsilon, so small-class
    numbers (fuzzing, reverse-off) can be judged as stable or noisy.

    Args:
        strict_df: strict de-duplicated signatures with 'true_class'.
        feature_columns: the 9 CAN features.
        class_names: ordered class-name list (for reporting).
        epsilons: list of attack epsilons to sweep.
        attack: "pgd" or "fgsm".
        n_splits: number of CV folds.
        others: as in the existing CV functions.
    """
    from sklearn.utils.class_weight import compute_class_weight

    n_classes = len(class_names)

    # results[eps] = list (per class) of lists (per fold) of F1 scores
    # Also record clean F1 for reference under key "clean".
    results = {"clean": [[] for _ in range(n_classes)]}
    for eps in epsilons:
        results[eps] = [[] for _ in range(n_classes)]

    y_sig = strict_df["true_class"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(strict_df, y_sig), start=1):
        train_sig = strict_df.iloc[tr_idx].reset_index(drop=True)
        test_sig  = strict_df.iloc[te_idx].reset_index(drop=True)

        # Duplicate train only, then preprocess fit-on-train-only
        train_dup = cleaning.duplicate_train_classes(
            train_sig, target_per_class=dup_target, random_seed=random_seed,
        )
        y_tr, y_te, _ = preprocessing.encode_labels(train_dup, test_sig)
        X_tr, X_te, _ = preprocessing.scale_features(train_dup, test_sig, feature_columns)

        # Per-fold balanced class weights for the CNN
        w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw = torch.tensor(w, dtype=torch.float32, device=device)

        # Train the CNN for this fold
        cnn = models.CNN1D(n_features=len(feature_columns), n_classes=n_classes)
        cnn = models.train_cnn(
            cnn, X_tr, y_tr, n_epochs=cnn_epochs, device=device,
            class_weights=cw, random_seed=random_seed,
        )

        # Wrap for ART
        clf = attacks.wrap_cnn_for_art(
            cnn, n_features=len(feature_columns), n_classes=n_classes, device=device,
        )

        X_te_f = X_te.astype(np.float32)

        # Clean per-class F1 for this fold
        clean_pred = clf.predict(X_te_f).argmax(axis=1)
        clean_per = f1_score(y_te, clean_pred, average=None,
                             labels=range(n_classes), zero_division=0)
        for c in range(n_classes):
            results["clean"][c].append(clean_per[c])

        # Attack at each epsilon, record per-class F1
        for eps in epsilons:
            if attack == "pgd":
                X_adv = attacks.generate_pgd(clf, X_te_f, epsilon=eps)
            else:
                X_adv = attacks.generate_fgsm(clf, X_te_f, epsilon=eps)
            pred = clf.predict(X_adv).argmax(axis=1)
            per = f1_score(y_te, pred, average=None,
                           labels=range(n_classes), zero_division=0)
            for c in range(n_classes):
                results[eps][c].append(per[c])

        print(f"  fold {fold}/{n_splits} done")

    return results

# Per-class defence comparison, cross-validated (baseline vs two defences)
def crossval_defence_comparison(
        strict_df, feature_columns, class_names,
        epsilons, eps_meaningful, eps_full,
        n_splits=5, dup_target=200, device="cpu",
        random_seed=42, cnn_epochs=50,
):
    """
    Cross-validate the defence comparison per class under PGD.

    For each fold, trains three models on that fold's train split:
        baseline    : plain CNN
        def_meaning : adversarial training at eps_meaningful
        def_full    : adversarial training at eps_full
    Then, white-box, crafts PGD against each model and records per-class F1 at
    each evaluation epsilon.

    Returns nested dict: results[model][epsilon][class_index] = list of per-fold F1s.
    """
    from sklearn.utils.class_weight import compute_class_weight

    n_classes = len(class_names)
    model_keys = ["baseline", "def_meaning", "def_full"]
    eps_keys = ["clean"] + list(epsilons)

    # results[model][eps] = list (per class) of lists (per fold)
    results = {m: {e: [[] for _ in range(n_classes)] for e in eps_keys}
               for m in model_keys}

    y_sig = strict_df["true_class"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(strict_df, y_sig), start=1):
        train_sig = strict_df.iloc[tr_idx].reset_index(drop=True)
        test_sig  = strict_df.iloc[te_idx].reset_index(drop=True)

        train_dup = cleaning.duplicate_train_classes(
            train_sig, target_per_class=dup_target, random_seed=random_seed,
        )
        y_tr, y_te, _ = preprocessing.encode_labels(train_dup, test_sig)
        X_tr, X_te, _ = preprocessing.scale_features(train_dup, test_sig, feature_columns)
        X_te_f = X_te.astype(np.float32)

        w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw = torch.tensor(w, dtype=torch.float32, device=device)

        # Train the three models for this fold
        baseline = models.CNN1D(n_features=len(feature_columns), n_classes=n_classes)
        baseline = models.train_cnn(
            baseline, X_tr, y_tr, n_epochs=cnn_epochs, device=device,
            class_weights=cw, random_seed=random_seed,
        )
        def_meaning = defense.adversarial_train_cnn(
            X_tr, y_tr, strategy="pgd", epsilons=eps_meaningful,
            n_epochs=cnn_epochs, device=device, class_weights=cw, random_seed=random_seed,
        )
        def_full = defense.adversarial_train_cnn(
            X_tr, y_tr, strategy="pgd", epsilons=eps_full,
            n_epochs=cnn_epochs, device=device, class_weights=cw, random_seed=random_seed,
        )

        fold_models = {"baseline": baseline, "def_meaning": def_meaning, "def_full": def_full}

        # White-box evaluate each model: attack crafted against itself
        for mkey, model in fold_models.items():
            clf = attacks.wrap_cnn_for_art(
                model, n_features=len(feature_columns), n_classes=n_classes, device=device,
            )
            clean_pred = clf.predict(X_te_f).argmax(axis=1)
            clean_per = f1_score(y_te, clean_pred, average=None,
                                 labels=range(n_classes), zero_division=0)
            for c in range(n_classes):
                results[mkey]["clean"][c].append(clean_per[c])

            for eps in epsilons:
                X_adv = attacks.generate_pgd(clf, X_te_f, epsilon=eps)
                pred = clf.predict(X_adv).argmax(axis=1)
                per = f1_score(y_te, pred, average=None,
                               labels=range(n_classes), zero_division=0)
                for c in range(n_classes):
                    results[mkey][eps][c].append(per[c])

        print(f"  fold {fold}/{n_splits} done")

    return results