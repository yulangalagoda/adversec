"""
Experiment layer: attack generation, adversarial-training defence, cross-
validation, and physical-realism checks. All consume canonical arrays + meta and
are dataset-agnostic. Requires torch and the adversarial-robustness-toolbox.
"""

from .adversarial import run_adversarial
from .attack import generate_fgsm, generate_hopskipjump, generate_pgd, wrap_cnn_for_art
from .baseline import run_baseline
from .crossval import (
    crossval_defence_comparison,
    crossval_defended,
    crossval_perclass_robustness,
    crossval_rowlevel,
    crossval_signature_level,
)
from .defended import run_defended
from .defense import adversarial_train_cnn, build_adversarial_trainset
from .realism import clip_to_id_envelope, learn_observed_ranges, observed_range_mask, round_to_integer_frames

__all__ = [
    "run_baseline", "run_adversarial", "run_defended",
    "wrap_cnn_for_art", "generate_fgsm", "generate_pgd", "generate_hopskipjump",
    "build_adversarial_trainset", "adversarial_train_cnn",
    "round_to_integer_frames", "learn_observed_ranges", "observed_range_mask", "clip_to_id_envelope",
    "crossval_rowlevel", "crossval_signature_level", "crossval_defended",
    "crossval_perclass_robustness", "crossval_defence_comparison",
]
