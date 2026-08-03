"""
defense.py
==========
Adversarial training. Two variants live here, both crafting adversarial examples
from TRAIN signatures only (test-time adversarial examples are crafted separately
and never mix in):

- `adversarial_train_cnn` (static): PGD/FGSM examples are crafted ONCE from a
  frozen, undefended baseline, then appended to the clean training set (augment,
  not replace, to preserve clean-traffic accuracy) and a fresh CNN is trained on
  that fixed set. Cheap, but the attack the model is trained against never
  adapts to the model actually being defended -- the failure mode Kurakin et al.
  warn gives a false sense of robustness.

- `madry_adversarial_train_cnn` (iterative / min-max): PGD examples are crafted
  fresh, every batch, against the model's CURRENT weights, and the model is
  updated on those -- the inner-maximisation / outer-minimisation loop from
  Madry et al. (2018), the form of adversarial training the literature claims
  should survive a white-box attacker.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

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


def madry_adversarial_train_cnn(
    X_train, y_train,
    epsilon=None, step_size=None, max_iter=None,
    n_epochs=50, batch_size=64, lr=1e-3,
    class_weights=None, device="cpu", random_seed=42,
):
    """
    True (Madry-style) adversarial training: every batch, PGD adversarial examples
    are crafted against the model's CURRENT weights (not a frozen baseline), and
    the model is updated on those adversarial examples -- solving the inner
    maximisation and outer minimisation together, batch by batch, rather than
    training on a fixed set crafted once (see `adversarial_train_cnn`).

    Trains on adversarial batches only (no clean augmentation), matching the
    canonical Madry et al. formulation -- deliberately different from
    `adversarial_train_cnn`'s "augment, not replace" choice, since the property
    being tested here is specifically whether the min-max form of AT survives a
    white-box attacker, not the augmented-training recipe.

    epsilon/step_size/max_iter default to config.MADRY_TRAIN_EPSILON,
    config.PGD_STEP_SIZE, config.MADRY_TRAIN_MAX_ITER -- fewer inner-loop steps
    than the eval-time PGD attack (config.PGD_MAX_ITER), matching standard
    practice of a cheaper train-time attack verified by a stronger test-time one.
    """
    eps = config.MADRY_TRAIN_EPSILON if epsilon is None else epsilon
    step = config.PGD_STEP_SIZE if step_size is None else step_size
    iters = config.MADRY_TRAIN_MAX_ITER if max_iter is None else max_iter

    torch.manual_seed(random_seed)
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    model = CNN1D(n_features=n_features, n_classes=n_classes).to(device)
    # Wraps `model` by reference: PGD crafted through this classifier during
    # training reflects the model's weights AT THE TIME `generate_pgd` is called,
    # i.e. the current, still-training weights -- this is what makes it Madry-style
    # rather than the static variant's attack-a-frozen-snapshot approach.
    classifier = attacks.wrap_cnn_for_art(model, n_features=n_features, n_classes=n_classes, device=device)

    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_rows = X.shape[0]
    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n_rows, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_rows, batch_size):
            idx = perm[start:start + batch_size]
            xb_clean = X[idx].detach().cpu().numpy()
            # verbose=False: this runs every batch, every epoch (tens of thousands of
            # calls on ROAD) -- ART's progress bar is a live Jupyter widget, not text,
            # so leaving it on floods the notebook frontend and eventually crashes it.
            xb_adv = attacks.generate_pgd(classifier, xb_clean, epsilon=eps, step_size=step, max_iter=iters, verbose=False)
            xb = torch.tensor(xb_adv, dtype=torch.float32, device=device)
            yb = y[idx]

            model.train()   # ART's PGD generation may leave the model in eval() mode
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    [madry] epoch {epoch + 1:3d}/{n_epochs}     loss {epoch_loss / n_batches:.4f}")

    return model
