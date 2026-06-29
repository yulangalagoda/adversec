"""
defense.py
==========
Stage 5: adversarial training for the AdverSec Pipleline

Generates adversarial examples from the training data only, aduments the clean training set with them, and retrains the CNN.
The result is a defended CNN that should resist attacks the baseline CNN folds to.

Leakage: train-time adversarial examples come from TRAIN signatures only.
Test-time adversarial examples (crafted separately from the test) are what will be evaluated on. the two never mix.
"""

import numpy as np
import torch
import torch.nn as nn

import config
import models
import attacks


# Building the adversarial-augmented training set
def build_adversarial_trainset(classifier, X_train, y_train, strategy="pgd", epsilons=None):
    """
    Craft adversarial examples from training data and augment the clean set.

    Args:
        classifier: ART-wrapped CNN used to craft the adversarial examples.
        X_trian, y_train: the clean training data
        strategy: "pgd" only or "multi" strategy (PGD+FGSM)
        epsilons: list of epsilons to craft at

    Returns:
        (X-aug, y_aug): clean training data with adversarial examples appended
    """

    eps_list = epsilons if epsilons is not None else config.FGSM_EPSILONS

    adv_batches = []

    for eps in eps_list:
        # PGD examples at this epsilon
        adv_batches.append(attacks.generate_pgd(classifier, X_train, epsilon=eps))

        # In multi-strategy add FGSM examples at this epsilon
        if strategy == "multi":
            adv_batches.append(attacks.generate_fgsm(classifier, X_train, epsilon=eps))
    
    # Stack all adversarial examples into one array
    X_adv = np.vstack(adv_batches)
    y_adv = np.tile(y_train, len(adv_batches))

    # Augment: clean data first
    X_aug = np.vstack([X_train, X_adv]).astype(np.float32)
    y_aug = np.concatenate([y_train, y_adv])

    return X_aug, y_aug


def adversarial_train_cnn(
        X_train, y_train,
        strategy="pgd",
        epsilons=None,
        n_epochs=50,
        device="cpu",
        class_weights=None,
        random_seed=42,
):
    """
    Train a defended CNN via adversarial augmentation.

    The procedure:
        1. Train a baseline CNN on the clean data
        2. Wrap it in ART, craft adversarial examples from the train data.
        3. Augment: clean train + adversarial examples
        4. Train a fresh CNN on the augmented set -> defended model

    Returns the defended CNN.
    """

    # STep 1: A baseline CNN to craft attacks from.
    base = models.CNN1D(n_features=X_train.shape[1], n_classes=len(np.unique(y_train)))
    base = models.train_cnn(
        base, X_train, y_train, n_epochs=n_epochs, device=device,
        class_weights=class_weights, random_seed=random_seed,
    )

    # Step2: wrap and craft the training-time adversarial examples
    classifier = attacks.wrap_cnn_for_art(
        base, n_features=X_train.shape[1],
        n_classes=len(np.unique(y_train)), device=device,
    )

    # Step3: Build the augmented training set
    X_aug, y_aug = build_adversarial_trainset(
        classifier, X_train, y_train, strategy=strategy, epsilons=epsilons,
    )
    print(f"    augmented trainset: {X_train.shape[0]} clean -> {X_aug.shape[0]} total "
          f"({strategy} strategy)")
    
    # Step 4: train a fresh cnn on the augmented set.
    defended = models.CNN1D(n_features=X_train.shape[1], n_classes=len(np.unique(y_train)))
    defended = models.train_cnn(
        defended, X_aug, y_aug, n_epochs=n_epochs, device=device,
        class_weights=class_weights, random_seed=random_seed,
    )

    return defended