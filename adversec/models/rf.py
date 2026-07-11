"""
rf.py
=====
Random Forest baseline, configured for the imbalanced CAN data.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(random_seed: int = 42) -> RandomForestClassifier:
    """Create an unfitted Random Forest. The caller fits it on (X_train, y_train)."""
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",       # weight rare attack classes more heavily
        random_state=random_seed,
        # Single-threaded on purpose: the deployment-profile latency benchmark
        # times single-frame predict(), which must be measured without thread
        # dispatch overhead. Macro-F1 is identical regardless of n_jobs.
        n_jobs=1,
    )
