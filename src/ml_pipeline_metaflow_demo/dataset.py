"""Tiny deterministic classification dataset (sklearn-friendly, no network)."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_tiny_split(
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X_train, X_test, y_train, y_test from Iris (binary: setosa vs rest)."""
    iris = load_iris()
    # Binary: class 0 vs others — keeps a tiny logistic model accurate and fast
    y = (iris.target == 0).astype(int)
    return train_test_split(
        iris.data,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
