"""
defense.py
==========
Adversarial training. Craft adversarial examples from the TRAIN data only, augment
the clean training set with them (augment, not replace, to preserve clean-traffic
accuracy), and retrain the CNN.

Leakage discipline: train-time adversarial examples come from TRAIN signatures
only; test-time adversarial examples are crafted separately and never mix in.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..models import CNN1D, train_cnn
from . import attack as attacks


def build_adversarial_trainset(classifier, X_train, y_train, strategy="pgd", epsilons=None):
    """Craft adversarial examples from the training data and append to the clean set."""
    eps_list = config.FGSM_EPSILONS if epsilons is None else epsilons
    adv_batches = []
    for eps in eps_list:
        adv_batches.append(attacks.generate_pgd(classifier, X_train, epsilon=eps))
        if strategy == "multi":
            adv_batches.append(attacks.generate_fgsm(classifier, X_train, epsilon=eps))
    X_adv = np.vstack(adv_batches)
    y_adv = np.tile(y_train, len(adv_batches))
    X_aug = np.vstack([X_train, X_adv]).astype(np.float32)
    y_aug = np.concatenate([y_train, y_adv])
    return X_aug, y_aug


def adversarial_train_cnn(
    X_train, y_train,
    strategy="pgd", epsilons=None,
    n_epochs=50, device="cpu", class_weights=None, random_seed=42,
):
    """Train a defended CNN via adversarial augmentation. Returns the defended CNN."""
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # 1. Baseline CNN to craft attacks from.
    base = CNN1D(n_features=n_features, n_classes=n_classes)
    base = train_cnn(base, X_train, y_train, n_epochs=n_epochs, device=device,
                     class_weights=class_weights, random_seed=random_seed)

    # 2. Wrap it and craft the training-time adversarial examples.
    classifier = attacks.wrap_cnn_for_art(base, n_features=n_features, n_classes=n_classes, device=device)

    # 3. Build the augmented (clean + adversarial) training set.
    X_aug, y_aug = build_adversarial_trainset(classifier, X_train, y_train, strategy=strategy, epsilons=epsilons)
    print(f"    augmented trainset: {X_train.shape[0]} clean -> {X_aug.shape[0]} total ({strategy} strategy)")

    # 4. Train a fresh CNN on the augmented set -> defended model.
    defended = CNN1D(n_features=n_features, n_classes=n_classes)
    defended = train_cnn(defended, X_aug, y_aug, n_epochs=n_epochs, device=device,
                         class_weights=class_weights, random_seed=random_seed)
    return defended
