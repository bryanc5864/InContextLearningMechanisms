"""Linear probing for task identity classification."""

import warnings
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import torch

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def train_probe(
    activations: np.ndarray,
    labels: list[str],
    n_splits: int = 5,
    test_size: float = 0.2,
    **kwargs,
) -> dict:
    """Train a nearest-centroid probe (very fast, no hyperparameters).

    Args:
        activations: Array of shape (n_samples, d_model).
        labels: Task labels for each sample.
        n_splits: Number of random train/test splits.
        test_size: Fraction held out for testing.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)

    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    scores = []

    for train_idx, test_idx in splitter.split(activations, y):
        clf = NearestCentroid()
        clf.fit(activations[train_idx], y[train_idx])
        scores.append(float(clf.score(activations[test_idx], y[test_idx])))

    # Full fit
    clf_full = NearestCentroid()
    clf_full.fit(activations, y)

    return {
        "accuracy_mean": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "model": clf_full,
        "label_encoder": le,
        "cv_scores": scores,
    }


def train_linear_probe(
    activations: np.ndarray,
    labels: list[str],
    n_splits: int = 3,
    test_size: float = 0.2,
) -> dict:
    """Train an SGD linear classifier (fast for high-dim data).

    Uses SGDClassifier with log loss (equivalent to logistic regression)
    but much faster for high-dimensional data.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)

    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    scores = []

    for train_idx, test_idx in splitter.split(activations, y):
        clf = SGDClassifier(loss="log_loss", max_iter=100, random_state=42, tol=1e-3)
        clf.fit(activations[train_idx], y[train_idx])
        scores.append(float(clf.score(activations[test_idx], y[test_idx])))

    clf_full = SGDClassifier(loss="log_loss", max_iter=100, random_state=42, tol=1e-3)
    clf_full.fit(activations, y)

    return {
        "accuracy_mean": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "model": clf_full,
        "label_encoder": le,
        "cv_scores": scores,
    }


def probe_all_layers(
    layer_activations: dict[int, list[torch.Tensor]],
    labels: list[str],
    n_splits: int = 5,
    **kwargs,
) -> dict[int, dict]:
    """Train probes at every layer."""
    results = {}
    for layer in sorted(layer_activations.keys()):
        acts = torch.stack(layer_activations[layer]).numpy()
        results[layer] = train_probe(acts, labels, n_splits=n_splits)
        print(f"  Layer {layer:2d}: accuracy = {results[layer]['accuracy_mean']:.3f} "
              f"± {results[layer]['accuracy_std']:.3f}")
    return results


def find_optimal_layer(probe_results: dict[int, dict]) -> int:
    """Find the layer with highest probe accuracy."""
    return max(probe_results, key=lambda l: probe_results[l]["accuracy_mean"])
