"""Layer-wise trajectory analysis for representation evolution."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def compute_probe_trajectory(
    layer_activations: dict[int, list[torch.Tensor]],
    labels: list[str],
    C: float = 1.0,
) -> dict[int, float]:
    """Compute probe confidence (accuracy) at each layer.

    Returns:
        Dict mapping layer -> probe accuracy on full dataset.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)

    trajectory = {}
    for layer in sorted(layer_activations.keys()):
        acts = torch.stack(layer_activations[layer]).float().numpy()
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                 multi_class="multinomial", random_state=42)
        clf.fit(acts, y)
        trajectory[layer] = float(clf.score(acts, y))

    return trajectory


def compute_representational_change(
    layer_activations: dict[int, list[torch.Tensor]],
) -> dict[tuple[int, int], float]:
    """Compute layer-to-layer representational change using cosine distance.

    Returns:
        Dict mapping (layer_i, layer_j) -> mean cosine distance across samples.
    """
    layers = sorted(layer_activations.keys())
    changes = {}

    for i in range(len(layers) - 1):
        l_i, l_j = layers[i], layers[i + 1]
        acts_i = torch.stack(layer_activations[l_i]).float()
        acts_j = torch.stack(layer_activations[l_j]).float()

        # Mean cosine distance across samples
        cos_sim = torch.nn.functional.cosine_similarity(acts_i, acts_j, dim=1)
        cos_dist = 1.0 - cos_sim.mean().item()
        changes[(l_i, l_j)] = cos_dist

    return changes


def find_crystallization_layer(
    trajectory: dict[int, float],
    threshold: float = 0.9,
) -> int | None:
    """Find the first layer where probe accuracy exceeds threshold.

    Returns:
        Layer index or None if threshold never reached.
    """
    for layer in sorted(trajectory.keys()):
        if trajectory[layer] >= threshold:
            return layer
    return None
