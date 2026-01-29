#!/usr/bin/env python3
"""Experiment 8: Multi-Position Transplantation.

Test whether replacing activations at ALL demo token positions can transfer
task behavior. This tests the distributed encoding hypothesis.
"""

import json
import sys
import os
import logging
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from src.model import load_model
from src.tasks import TaskRegistry

INCLUDED_TASKS = [
    "uppercase", "first_letter", "repeat_word", "length",
    "linear_2x", "sentiment", "antonym", "pattern_completion",
]

# Use tasks with similar output structure for cleaner position alignment
TEST_PAIRS = [
    ("uppercase", "first_letter"),   # both produce single token/word
    ("uppercase", "repeat_word"),    # procedural tasks
    ("first_letter", "repeat_word"),
    ("uppercase", "sentiment"),      # procedural vs semantic
    ("linear_2x", "length"),         # numeric tasks
    ("sentiment", "antonym"),        # semantic tasks
]


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "exp8.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def find_demo_token_positions(tokenizer, prompt: str, n_demos: int = 5):
    """Find token positions for each demo (input and output portions).

    Returns:
        dict with keys:
            'all_demo': all demo token positions
            'input_only': just "Input: X" portions
            'output_only': just "Output: Y" portions
            'last_demo': just the last demo pair
    """
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_text = tokenizer.decode(tokens)

    # Split into lines to find demo structure
    lines = prompt.strip().split("\n")

    positions = {
        'all_demo': [],
        'input_only': [],
        'output_only': [],
        'last_demo': [],
    }

    current_pos = 0
    demo_count = 0
    last_demo_start = None

    for i, line in enumerate(lines):
        line_tokens = tokenizer.encode(line, add_special_tokens=False)
        line_len = len(line_tokens)

        # Check if this is the final "Input: X\nOutput:" (query)
        if i >= len(lines) - 2:  # Last two lines are query
            break

        if line.startswith("Input:"):
            for p in range(current_pos, current_pos + line_len):
                positions['all_demo'].append(p)
                positions['input_only'].append(p)
            if demo_count == n_demos - 1:  # Last demo
                last_demo_start = current_pos
                for p in range(current_pos, current_pos + line_len):
                    positions['last_demo'].append(p)

        elif line.startswith("Output:"):
            for p in range(current_pos, current_pos + line_len):
                positions['all_demo'].append(p)
                positions['output_only'].append(p)
            if last_demo_start is not None:
                for p in range(current_pos, current_pos + line_len):
                    positions['last_demo'].append(p)
            demo_count += 1

        # Account for newline token
        current_pos += line_len
        if i < len(lines) - 1:
            newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
            current_pos += len(newline_tokens)

    return positions


def extract_multi_position_activations(model, prompt, layer, positions):
    """Extract activations at multiple positions."""
    tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    activations = {}

    def hook_fn(module, input, output):
        # output is (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # hidden shape: [batch, seq, d_model]
        for pos in positions:
            if pos < hidden.shape[1]:
                activations[pos] = hidden[0, pos, :].detach().clone()

    layer_module = model.get_layer_module(layer)
    handle = layer_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        model.model(tokens)

    handle.remove()
    return activations


def generate_with_multi_intervention(model, prompt, layer, position_vectors, max_new_tokens=30):
    """Generate with intervention at multiple positions."""
    tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    intervention_done = [False]

    def hook_fn(module, input, output):
        if intervention_done[0]:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        # Replace at all specified positions
        for pos, vec in position_vectors.items():
            if pos < hidden.shape[1]:
                hidden[0, pos, :] = vec.to(hidden.device).to(hidden.dtype)

        intervention_done[0] = True

        if rest is not None:
            return (hidden,) + rest
        return hidden

    layer_module = model.get_layer_module(layer)
    handle = layer_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        output_ids = model.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=model.tokenizer.eos_token_id,
        )

    handle.remove()

    new_tokens = output_ids[0, tokens.shape[1]:]
    text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()


def run_multi_position(device="cuda:3", n_demos=5, n_test=10, output_dir="results/exp8"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 8: Multi-Position Transplantation")
    logger.info(f"Start: {datetime.now().isoformat()}")

    model = load_model(device=device)
    tasks = {name: TaskRegistry.get(name) for name in INCLUDED_TASKS}

    # Use layer 14 as in Phase 3 (or test multiple)
    test_layers = [8, 12, 14, 16]

    # Conditions to test
    conditions = ['all_demo', 'input_only', 'output_only', 'last_demo']

    all_results = []

    for source_name, target_name in TEST_PAIRS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pair: {source_name} → {target_name}")
        logger.info(f"{'='*60}")

        source_task = tasks[source_name]
        target_task = tasks[target_name]

        source_demos = source_task.generate_demos(n_demos)
        target_demos = target_task.generate_demos(n_demos)

        source_test_inputs = source_task.generate_test_inputs(n_test)
        target_test_inputs = target_task.generate_test_inputs(n_test)

        pair_results = {
            "source": source_name,
            "target": target_name,
            "conditions": {},
        }

        for layer in test_layers:
            logger.info(f"\n  Layer {layer}:")

            for condition in conditions:
                # Extract source activations at relevant positions
                source_position_vectors = {}

                for s_input in source_test_inputs:
                    prompt = source_task.format_prompt(source_demos, s_input)
                    positions = find_demo_token_positions(model.tokenizer, prompt, n_demos)
                    pos_list = positions[condition]

                    acts = extract_multi_position_activations(model, prompt, layer, pos_list)

                    for pos, vec in acts.items():
                        if pos not in source_position_vectors:
                            source_position_vectors[pos] = []
                        source_position_vectors[pos].append(vec)

                # Average across source examples
                mean_position_vectors = {
                    pos: torch.stack(vecs).mean(dim=0)
                    for pos, vecs in source_position_vectors.items()
                }

                # Transplant into target contexts
                transfer_count = 0
                preserve_count = 0
                neither_count = 0

                for t_input in target_test_inputs:
                    prompt = target_task.format_prompt(target_demos, t_input)

                    output = generate_with_multi_intervention(
                        model, prompt, layer, mean_position_vectors, max_new_tokens=30
                    )

                    try:
                        source_correct = source_task.score_output(t_input, output) == "correct"
                    except:
                        source_correct = False
                    try:
                        target_correct = target_task.score_output(t_input, output) == "correct"
                    except:
                        target_correct = False

                    if source_correct:
                        transfer_count += 1
                    elif target_correct:
                        preserve_count += 1
                    else:
                        neither_count += 1

                key = f"layer{layer}_{condition}"
                pair_results["conditions"][key] = {
                    "layer": layer,
                    "condition": condition,
                    "n_positions": len(mean_position_vectors),
                    "transfer_rate": transfer_count / n_test,
                    "preserve_rate": preserve_count / n_test,
                    "neither_rate": neither_count / n_test,
                }

                marker = "***" if transfer_count / n_test > 0.3 else ""
                logger.info(
                    f"    {condition:12s} ({len(mean_position_vectors):3d} pos): "
                    f"transfer={transfer_count/n_test:.2f} preserve={preserve_count/n_test:.2f} {marker}"
                )

        all_results.append(pair_results)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY BY CONDITION")
    logger.info("=" * 60)

    for condition in conditions:
        for layer in test_layers:
            key = f"layer{layer}_{condition}"
            transfers = [
                pr["conditions"][key]["transfer_rate"]
                for pr in all_results
                if key in pr["conditions"]
            ]
            if transfers:
                mean_t = np.mean(transfers)
                logger.info(f"  Layer {layer:2d}, {condition:12s}: mean transfer = {mean_t:.3f}")

    # Save
    results = {
        "metadata": {
            "test_layers": test_layers,
            "conditions": conditions,
            "pairs": TEST_PAIRS,
            "n_test": n_test,
            "timestamp": datetime.now().isoformat(),
        },
        "pair_results": all_results,
    }

    with open(Path(output_dir) / "multi_position_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(Path(output_dir) / "multi_position.csv", "w") as f:
        f.write("pair,layer,condition,n_positions,transfer_rate,preserve_rate,neither_rate\n")
        for pr in all_results:
            pair = f"{pr['source']}_{pr['target']}"
            for key, data in pr["conditions"].items():
                f.write(f"{pair},{data['layer']},{data['condition']},{data['n_positions']},"
                       f"{data['transfer_rate']},{data['preserve_rate']},{data['neither_rate']}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 8 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--output-dir", default="results/exp8")
    args = parser.parse_args()
    run_multi_position(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
