"""
attack.py
=========
Adversarial attack generation. Wraps the trained CNN in ART, then crafts FGSM and
PGD examples in the scaled [0,1] space. The same crafted examples are fed to both
models downstream, so the robustness comparison is a fair transfer test.

An optional mask (0 = frozen, 1 = perturbable, broadcastable to X) freezes chosen
features, e.g. hold the ID and perturb payload bytes only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod, HopSkipJump, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

from .. import config


def wrap_cnn_for_art(model, n_features: int = 9, n_classes: int = 6, device: str = "cpu"):
    """Wrap a trained CNN1D in an ART PyTorchClassifier so ART can compute gradients."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(n_features,),
        nb_classes=n_classes,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device == "cuda" else "cpu",
    )


def generate_fgsm(classifier, X, epsilon, mask=None):
    """FGSM adversarial examples from X at a given epsilon."""
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    X = X.astype(np.float32)
    return attack.generate(x=X, mask=mask) if mask is not None else attack.generate(x=X)


def generate_pgd(classifier, X, epsilon, step_size=None, max_iter=None, mask=None):
    """PGD adversarial examples from X (step size and iterations default from config)."""
    step = config.PGD_STEP_SIZE if step_size is None else step_size
    iters = config.PGD_MAX_ITER if max_iter is None else max_iter
    attack = ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=step, max_iter=iters)
    X = X.astype(np.float32)
    return attack.generate(x=X, mask=mask) if mask is not None else attack.generate(x=X)


def generate_hopskipjump(classifier, X, mask=None, max_iter=15, max_eval=300,
                         init_eval=30, init_size=30, batch_size=64):
    """
    Decision-based black-box adversarial examples (HopSkipJump): queries only the
    classifier's predicted labels, never gradients -- tests whether robustness/defence
    findings generalise beyond gradient-based attacks (FGSM/PGD).

    ART's defaults (max_iter=50, max_eval=10000, init_eval=100) cost ~500k queries per
    sample -- intractable at any real sample count. The reduced budget here trades
    attack strength (a fully-converged HopSkipJump finds a smaller, closer-to-minimal
    perturbation) for tractability on more than a handful of samples. Treat the
    resulting F1 as a LOWER BOUND on what a well-resourced black-box attacker could
    achieve, not the attack's ceiling -- state this budget explicitly wherever it's cited.
    """
    attack = HopSkipJump(classifier=classifier, max_iter=max_iter, max_eval=max_eval,
                         init_eval=init_eval, init_size=init_size, batch_size=batch_size,
                         verbose=False)
    X = X.astype(np.float32)
    return attack.generate(x=X, mask=mask) if mask is not None else attack.generate(x=X)
