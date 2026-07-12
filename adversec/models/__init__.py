"""Model definitions: a scikit-learn Random Forest and a PyTorch 1D-CNN."""

from .cnn import CNN1D, train_cnn
from .rf import build_random_forest

__all__ = ["CNN1D", "train_cnn", "build_random_forest"]
