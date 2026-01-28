"""Activation extraction and caching."""

import torch
from src.model import HookedModel


def get_position_index(
    model: HookedModel,
    prompt: str,
    position_type: str,
) -> int:
    """Find the token position index for a given position type.

    Args:
        model: The HookedModel.
        prompt: The full prompt string.
        position_type: One of "last_demo_token", "separator_after_demo", "first_query_token".

    Returns:
        Token position index (0-based).
    """
    input_ids = model.tokenize(prompt)[0]  # (seq_len,)

    # Decode each token to find structural markers
    # The prompt ends with: "...Output: <last_demo_answer>\n\nInput: <test>\nOutput:"
    # We need to find the boundaries

    # Strategy: find the last "Input:" in the token string
    # Build cumulative decoded string to map char positions to token positions
    token_strs = []
    for i in range(len(input_ids)):
        token_strs.append(model.tokenizer.decode(input_ids[i:i+1]))

    full_text = model.tokenizer.decode(input_ids)

    if position_type == "last_demo_token":
        # Last token before the final "Input:" (which starts the test query)
        last_input = full_text.rfind("\nInput:")
        if last_input == -1:
            last_input = full_text.rfind("Input:")
        # Token position just before this
        target_char = last_input - 1
        return _char_to_token_pos(full_text, token_strs, target_char)

    elif position_type == "separator_after_demo":
        # The newline separating last demo from test query
        last_input = full_text.rfind("\nInput:")
        return _char_to_token_pos(full_text, token_strs, last_input)

    elif position_type == "first_query_token":
        # First token of the test input value (after last "Input: ")
        last_input = full_text.rfind("Input: ")
        target_char = last_input + len("Input: ")
        return _char_to_token_pos(full_text, token_strs, target_char)

    else:
        raise ValueError(f"Unknown position type: {position_type}")


def _char_to_token_pos(full_text: str, token_strs: list[str], char_pos: int) -> int:
    """Convert character position to token index."""
    cumulative = 0
    for i, ts in enumerate(token_strs):
        cumulative += len(ts)
        if cumulative > char_pos:
            return i
    return len(token_strs) - 1


def extract_activations(
    model: HookedModel,
    prompt: str,
    layers: list[int] | None = None,
    position: int | None = None,
) -> dict[int, torch.Tensor]:
    """Extract residual stream activations at specified layers and position.

    Args:
        model: The HookedModel.
        prompt: The input prompt.
        layers: List of layer indices. If None, extract all.
        position: Token position. If None, return all positions.

    Returns:
        Dict mapping layer -> activation tensor.
        If position specified: shape (d_model,)
        If position is None: shape (seq_len, d_model)
    """
    input_ids = model.tokenize(prompt)
    cache = model.forward_with_cache(input_ids, layers=layers)

    if position is not None:
        return {l: cache[l][position] for l in cache}
    return cache


def extract_all_layer_activations(
    model: HookedModel,
    prompt: str,
    position: int,
) -> torch.Tensor:
    """Extract activations at a specific position across all layers.

    Returns:
        Tensor of shape (n_layers, d_model).
    """
    acts = extract_activations(model, prompt, position=position)
    return torch.stack([acts[l] for l in sorted(acts.keys())])
