"""
attacks.py
==========
Stage 4: adversarial attack generation for the AdverSec pipeline.

Wraps the trained 1D-CNN in ART PyTorchClassifier, then generates FGSM and PGD adversarial examples.
The same crafted examples are later fed to both models, so the robustness comparison is fair.

Perturbations are crafted in the scaled [0,1] feature space.
An integer-valid check is applied separately downstream.
"""

# Library imports
import numpy as np
import torch
import torch.nn as nn
# ART
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

import config

# Wrapping the CNN for ART
def wrap_cnn_for_art(model, n_features=9, n_classes=6, device="cpu"):
    """
    Wrap a trianed CNN1D in an ART PyTorchClassifier so ART can attack it.
    """

    # ART requires a loss function
    criterion = nn.CrossEntropyLoss()

    # ART requires an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(n_features,),
        nb_classes=n_classes,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device == "cuda" else "cpu"
    )

    return classifier


# FGSM
def generate_fgsm(classifier, X, epsilon, mask=None):
    """
    Generate FGSM adversarial examples from X at a given epsilon.

    mask: optional array broadcastable to X. Features where mask is 0 are NOT
    perturbed (e.g. freeze the ID feature to attack payload bytes only).
    """
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)

    if mask is not None:
        X_adv = attack.generate(x=X.astype(np.float32), mask=mask)
    else:
        X_adv = attack.generate(x=X.astype(np.float32))
    return X_adv


# PGD
def generate_pgd(classifier, X, epsilon, step_size=None, max_iter=None, mask=None):
    """
    Generate PGD adversarial examples from X.

    mask: optional array broadcastable to X. Features where mask is 0 are NOT
    perturbed (e.g. freeze the ID feature to attack payload bytes only).
    """
    step = step_size if step_size is not None else config.PGD_STEP_SIZE
    iters = max_iter if max_iter is not None else config.PGD_MAX_ITER

    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=epsilon,
        eps_step=step,
        max_iter=iters,
    )

    if mask is not None:
        X_adv = attack.generate(x=X.astype(np.float32), mask=mask)
    else:
        X_adv = attack.generate(x=X.astype(np.float32))
    return X_adv