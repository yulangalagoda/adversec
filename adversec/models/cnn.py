"""
cnn.py
======
A small 1D Convolutional Neural Network for 9-feature CAN frames, plus its
training loop.

Deliberately shallow: with only a few thousand training rows a large network
would overfit. Mirrors the kind of 1D-CNN used in prior CICIoV2024 work so the
comparison to the Random Forest is fair.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, n_features: int = 9, n_classes: int = 6):
        super().__init__()
        # Convolutional feature extractor: each frame is a single-channel length-9
        # sequence. padding=1 keeps the length at 9 through both conv layers.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Classifier head: after two padding=1 convs the length stays 9 with 32
        # channels, so the flattened size is 32 * n_features.
        self.fc1 = nn.Linear(32 * n_features, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x arrives as (batch, n_features); Conv1d needs (batch, channels, length).
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # raw logits; softmax is applied inside CrossEntropyLoss


def train_cnn(
    model,
    X_train,
    y_train,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    class_weights=None,
    device: str = "cpu",
    random_seed: int = 42,
):
    """Train a CNN1D with Adam + weighted cross-entropy. Returns the trained model."""
    torch.manual_seed(random_seed)
    model = model.to(device)

    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.long, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_rows = X.shape[0]
    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n_rows, device=device)   # reshuffle each epoch
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_rows, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx], y[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1:3d}/{n_epochs}     loss {epoch_loss / n_batches:.4f}")

    return model
