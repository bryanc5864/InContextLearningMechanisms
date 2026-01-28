"""Activation transplantation and intervention."""

import torch
from src.model import HookedModel


def transplant_and_generate(
    model: HookedModel,
    target_prompt: str,
    source_vector: torch.Tensor,
    layer: int,
    position: int,
    max_new_tokens: int = 30,
) -> str:
    """Replace activation at (layer, position) with source_vector and generate."""
    return model.generate_with_intervention(
        target_prompt, layer, position, source_vector, max_new_tokens
    )


def baseline_generate(
    model: HookedModel,
    prompt: str,
    max_new_tokens: int = 30,
) -> str:
    """Generate without any intervention (baseline)."""
    return model.generate(prompt, max_new_tokens=max_new_tokens)


def zero_ablation_generate(
    model: HookedModel,
    prompt: str,
    layer: int,
    position: int,
    max_new_tokens: int = 30,
) -> str:
    """Generate with zero vector at (layer, position)."""
    zero_vec = torch.zeros(model.d_model)
    return transplant_and_generate(
        model, prompt, zero_vec, layer, position, max_new_tokens
    )


def random_ablation_generate(
    model: HookedModel,
    prompt: str,
    layer: int,
    position: int,
    norm: float,
    max_new_tokens: int = 30,
    seed: int = 42,
) -> str:
    """Generate with random Gaussian vector (matched norm) at (layer, position)."""
    rng = torch.Generator().manual_seed(seed)
    rand_vec = torch.randn(model.d_model, generator=rng)
    rand_vec = rand_vec * (norm / rand_vec.norm())
    return transplant_and_generate(
        model, prompt, rand_vec, layer, position, max_new_tokens
    )
