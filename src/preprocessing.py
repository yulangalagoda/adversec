"""
preprocessing.py
================

Stage 2 of the Adversec pipeline: turn the cleaned CSVs into model-ready arrays.

Two jobs:
    1. Label encoding: map class strings to integers.
    2. Feature scaling: scale all 9 features to [0,1]. fit on TRAIN only.

The [0,1] scaled space is also the coordinate system for adversarial pertubation later.
So an epsilon is a fraction of each features range.
"""

# Library import
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler        # Encorder and Scaler
import config                                                       # Project's settings file


# Encoding the labels
def encode_labels(train_df, test_df, target_column="true_class"):
    """
    Map class-name strings to integer labels, fitting on TRAIN only.
    
    Returns the integer label arrays for train and test, plus the fitted encoder.
    So, we can translate integers back to class names when reporting.
    """

    # Create the encoder and fit it on the training labels
    encoder = LabelEncoder()
    encoder.fit(train_df[target_column])

    # Transform both splits using that same learned mapping.
    y_train = encoder.transform(train_df[target_column])
    y_test = encoder.transform(test_df[target_column])

    # Print ordered list of class names to check the mapping on record
    print("Label mapping:")
    for i, name in enumerate(encoder.classes_):
        print(f"    {i} -> {name}")

    return y_train, y_test, encoder


# Scaling the features
def scale_features(train_df, test_df, feature_columns):
    """
    Scale all features to [0,1], fitting the scalar on TRAIN only.

    Returns scaled train and test feature arrays plus the fitted scalar.
    Required to scale adversarial samples into the same space and to interpret perturbation sizes.
    """

    # Pull feature columns as plain arrays
    X_train_raw = train_df[feature_columns].values
    X_test_raw = test_df[feature_columns].values

    # Create the scaler and fit on training features only
    scaler = MinMaxScaler()
    scaler.fit(X_train_raw)

    # Transform both splits with the train-derived min/max
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, scaler