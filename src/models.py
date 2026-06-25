"""
models.py
=========
Stage 3 model definitions for the Adversec pipeline.

Two baselines:
    - build_random_forest(): scikit-learn RF
    - CNN1D + train_cnn(): PyTorch 1D-CNN

The RF is build and trained in a couple of calls.
The CNN is a torch.nn.Module with its own training loop, defined further down.
"""

# Import libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

# The RF builder
def build_random_forest(random_seed=42):
    """
    Create a Random Forest classifier configured for the imbalanced data.

    Returns an unfitted RF. The notebook calls .fit(X_train, y_train) on it.
    """

    rf = RandomForestClassifier(
        # Number of trees (considering 9 features)
        n_estimators=200,

        # Balanced class weight so the RF weight rare classes more heavily.
        class_weight="balanced",

        # Reproductability
        random_state=random_seed,

        # Use all CPU cores in parallel
        n_jobs=1,
    )

    return rf


# 1D-CNN class
class CNN1D(nn.Module):
    """
    A small 1D Convolutional Neural Network for 9-feature CAN frames.

    Deliberately shallow: with only ~3800 training rows, a large network would overfit.
    Mirrors the kind of 1D-CNN used in prior CICIoV2024 work so the comparison to RF is fair.
    """

    def __init__(self, n_features=9, n_classes=6):
        # Calling the parent constructor to set up nn.Module internals
        super().__init__()

        # --- Convolutional feature extractor ---
        # Each CAN frame is a single-channel sequence of 9 values -> in_channels=1
        # 16 filters -> out_channels=16
        # Keeps the length at 9 -> padding=1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

        # Second Conv layer: takes the 16 channels from conv1, produce 32
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # ReLU activation
        self.relu = nn.ReLU()

        # --- Classifier head ---
        # After two padding=1 convs the length stays 9, with 32 channels, so the flattened size is 32*9=288 per frame.
        self.fc1 = nn.Linear(32 * n_features, 64)
        self.fc2 = nn.Linear(64, n_classes)
    
    def forward(self, x):
        # x arrives as (batch, n_features).
        # Conv1d needs (batch, channels, length).
        # So insert a channel dimension of 1.
        x = x.unsqueeze(1)

        # Conv -> activation, twice.
        # Each conv slides its filters across the 9 features; relu adds the non-linearity.
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Flatten everything except the batch dimension
        x = x.flatten(start_dim=1)

        # Fully-connected layers: hidden layer + activation, then the output layer.
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # Return raw scores (logits), one per class. SOFTMAX to be applied later with the loss function internally.
        return x
    

# 1D-CNN Training function
def train_cnn(
        model,
        X_train,
        y_train,
        n_epochs=50,
        batch_size=64,
        lr=1e-3,
        class_weights=None,
        device="cpu",
        random_seed=42
):
    """
    Train a CNN1D with gradient descent and print the loss per epoch.

    Args:
        model: an instance of CNN1D
        X_train: scaled feature array, shape (n_rows, n_features)
        y_train: integer label array, shape (n_rows,)
        n_epochs: how many full passes over the data
        batch size: rows per gradient update
        lr: learning rate
        class_weights: optional per-class weights for the loss, to counter imbalance
        device: "cuda" or "cpu"
        random_seed: for reproducible batch shuffling and weight init.

    Returns:
        The trained model (also modified in place).
    """

    # Reproducibility
    torch.manual_seed(random_seed)

    # Move the model to chosen device
    model = model.to(device)

    # Numpy arrays to torch tensors
    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.long, device=device)

    # The loss function: CrossEntropyLoss (standard for multi-class classification)
    # Applies softmax internally
    # Weight=class_weights makes rare class mistakes cost more, countering imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # The optimizer: ADAM
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_rows = X.shape[0]

    # Put model in training mode.
    model.train()

    for epoch in range(n_epochs):
        # Shuffle the row order each epoch so batches differ run to run.
        perm = torch.randperm(n_rows, device=device)

        epoch_loss = 0.0
        n_batches = 0

        # Walk through the data in chunks of batch size
        for start in range(0, n_rows, batch_size):
            idx = perm[start:start + batch_size]
            xb = X[idx]
            yb = y[idx]

            # 1. Clear old gradients
            optimizer.zero_grad()

            # 2. Forward pass: model predicts logits for this batch
            logits = model(xb)

            # 3. Compute the loss
            loss = criterion(logits, yb)

            # 4. Backward pass
            loss.backward()

            # 5. Update step
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Print average loss this epoch: should fail over time.
        avg_loss = epoch_loss / n_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1:3d}/{n_epochs}     loss {avg_loss:.4f}")
    
    return model