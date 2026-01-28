"""Vector interpolation and arithmetic for compositionality analysis."""

import torch
import numpy as np


def interpolate_vectors(
    v_a: torch.Tensor,
    v_b: torch.Tensor,
    alphas: list[float],
) -> list[torch.Tensor]:
    """Linear interpolation between two vectors.

    v_alpha = alpha * v_a + (1 - alpha) * v_b

    When alpha=1, result is v_a. When alpha=0, result is v_b.
    """
    return [alpha * v_a + (1 - alpha) * v_b for alpha in alphas]


def compute_task_difference(v_a: torch.Tensor, v_b: torch.Tensor) -> torch.Tensor:
    """Compute task difference vector: delta = v_a - v_b."""
    return v_a - v_b


def apply_task_shift(
    v_base: torch.Tensor,
    delta: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply a task difference vector to a base vector."""
    return v_base + scale * delta


def measure_transition_sharpness(
    task_a_probs: list[float],
    task_b_probs: list[float],
    alphas: list[float],
) -> dict:
    """Measure how sharply task probabilities transition.

    Returns metrics characterizing the transition:
    - midpoint_slope: slope of task_a_prob at alpha=0.5
    - transition_width: alpha range over which dominant task changes
    - is_smooth: whether transition appears linear vs. step-like
    """
    alphas = np.array(alphas)
    probs_a = np.array(task_a_probs)
    probs_b = np.array(task_b_probs)

    # Midpoint slope (finite difference)
    mid_idx = len(alphas) // 2
    if mid_idx > 0 and mid_idx < len(alphas) - 1:
        da = alphas[mid_idx + 1] - alphas[mid_idx - 1]
        dp = probs_a[mid_idx + 1] - probs_a[mid_idx - 1]
        midpoint_slope = dp / da if da > 0 else 0.0
    else:
        midpoint_slope = 0.0

    # Transition width: range where neither task dominates (>0.3 and <0.7)
    mixed = (probs_a > 0.2) & (probs_a < 0.8)
    if mixed.any():
        transition_width = float(alphas[mixed].max() - alphas[mixed].min())
    else:
        transition_width = 0.0

    # Smoothness: correlate with linear prediction
    linear_pred = alphas  # Perfect linear would be probs_a = alpha
    correlation = float(np.corrcoef(probs_a, linear_pred)[0, 1]) if probs_a.std() > 0 else 0.0

    return {
        "midpoint_slope": float(midpoint_slope),
        "transition_width": transition_width,
        "linear_correlation": correlation,
        "is_smooth": transition_width > 0.3,
    }
