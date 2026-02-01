#!/usr/bin/env python3
"""Experiment 23: Proper Statistics (N=50, confidence intervals).

Re-run exp8 multi-position transplantation with N=50 on Llama-3B.
Add bootstrap 95% CIs and Wilson intervals to all transfer rates.
Report mean + CI rather than cherry-picked max.
"""

import json
import sys
import os
import logging
import math
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

TEST_PAIRS = [
    ("uppercase", "first_letter"),
    ("uppercase", "repeat_word"),
    ("first_letter", "repeat_word"),
    ("uppercase", "sentiment"),
    ("linear_2x", "length"),
    ("sentiment", "antonym"),
]


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "exp23.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def wilson_interval(successes, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return p_hat, lo, hi


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    means = sorted(means)
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_boot)]
    hi = means[int((1 - alpha) * n_boot)]
    return float(np.mean(values)), lo, hi


# Inline the core functions from exp8 to avoid import issues
def find_demo_token_positions(tokenizer, prompt, n_demos=5):
    """Find token positions for each demo (input and output portions)."""
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
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
        if i >= len(lines) - 2:
            break
        if line.startswith("Input:"):
            for p in range(current_pos, current_pos + line_len):
                positions['all_demo'].append(p)
                positions['input_only'].append(p)
            if demo_count == n_demos - 1:
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
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
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
            tokens, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=model.tokenizer.eos_token_id,
        )
    handle.remove()
    new_tokens = output_ids[0, tokens.shape[1]:]
    text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()


def run_proper_stats(model_name="meta-llama/Llama-3.2-3B-Instruct",
                     device="cuda:3", n_demos=5, n_test=50,
                     output_dir="results/exp23"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 23: Proper Statistics (N=50, CIs)")
    logger.info(f"Start: {datetime.now().isoformat()}")

    model = load_model(model_name, device=device)
    tasks = {name: TaskRegistry.get(name) for name in INCLUDED_TASKS}

    # Same fractional layers as exp8
    layer_fractions = [0.30, 0.44, 0.52, 0.59]
    test_layers = sorted(set(model.layer_at_fraction(f) for f in layer_fractions))
    conditions = ['all_demo', 'input_only', 'output_only', 'last_demo']

    logger.info(f"Model: {model_name}, N={n_test}, layers={test_layers}")

    all_results = []

    for source_name, target_name in TEST_PAIRS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pair: {source_name} -> {target_name}")
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
            for condition in conditions:
                # Extract source activations
                source_position_vectors = {}
                for s_input in source_test_inputs:
                    prompt = source_task.format_prompt(source_demos, s_input)
                    pos_info = find_demo_token_positions(model.tokenizer, prompt, n_demos)
                    pos_list = pos_info[condition]
                    acts = extract_multi_position_activations(model, prompt, layer, pos_list)
                    for pos, vec in acts.items():
                        if pos not in source_position_vectors:
                            source_position_vectors[pos] = []
                        source_position_vectors[pos].append(vec)

                mean_position_vectors = {
                    pos: torch.stack(vecs).mean(dim=0)
                    for pos, vecs in source_position_vectors.items()
                }

                # Per-instance binary outcomes for CIs
                transfer_outcomes = []
                preserve_outcomes = []

                for t_input in target_test_inputs:
                    prompt = target_task.format_prompt(target_demos, t_input)
                    output = generate_with_multi_intervention(
                        model, prompt, layer, mean_position_vectors
                    )
                    try:
                        source_correct = source_task.score_output(t_input, output) == "correct"
                    except Exception:
                        source_correct = False
                    try:
                        target_correct = target_task.score_output(t_input, output) == "correct"
                    except Exception:
                        target_correct = False

                    transfer_outcomes.append(1 if source_correct else 0)
                    preserve_outcomes.append(1 if target_correct else 0)

                transfer_count = sum(transfer_outcomes)
                preserve_count = sum(preserve_outcomes)
                neither_count = n_test - transfer_count - preserve_count

                # Wilson intervals
                tr_rate, tr_lo, tr_hi = wilson_interval(transfer_count, n_test)
                pr_rate, pr_lo, pr_hi = wilson_interval(preserve_count, n_test)

                # Bootstrap CI on transfer rate
                boot_mean, boot_lo, boot_hi = bootstrap_ci(transfer_outcomes)

                key = f"layer{layer}_{condition}"
                pair_results["conditions"][key] = {
                    "layer": layer,
                    "condition": condition,
                    "n_positions": len(mean_position_vectors),
                    "n_test": n_test,
                    "transfer_rate": tr_rate,
                    "transfer_wilson_ci": [round(tr_lo, 4), round(tr_hi, 4)],
                    "transfer_bootstrap_ci": [round(boot_lo, 4), round(boot_hi, 4)],
                    "preserve_rate": pr_rate,
                    "preserve_wilson_ci": [round(pr_lo, 4), round(pr_hi, 4)],
                    "neither_rate": neither_count / n_test,
                }

                logger.info(
                    f"  L{layer:2d} {condition:12s}: "
                    f"transfer={tr_rate:.2f} [{tr_lo:.2f},{tr_hi:.2f}] "
                    f"preserve={pr_rate:.2f} [{pr_lo:.2f},{pr_hi:.2f}]"
                )

        all_results.append(pair_results)

    # Summary: mean transfer rate per condition with CI
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Mean Transfer Rate by Condition (with bootstrap 95% CI)")
    logger.info("=" * 60)

    for condition in conditions:
        for layer in test_layers:
            key = f"layer{layer}_{condition}"
            rates = [
                pr["conditions"][key]["transfer_rate"]
                for pr in all_results
                if key in pr["conditions"]
            ]
            if rates:
                mean_r, ci_lo, ci_hi = bootstrap_ci(rates)
                logger.info(
                    f"  Layer {layer:2d}, {condition:12s}: "
                    f"mean={mean_r:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
                )

    # Save
    results = {
        "metadata": {
            "model": model_name,
            "test_layers": test_layers,
            "conditions": conditions,
            "pairs": TEST_PAIRS,
            "n_test": n_test,
            "timestamp": datetime.now().isoformat(),
        },
        "pair_results": all_results,
    }

    with open(Path(output_dir) / "proper_stats_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV with CIs
    with open(Path(output_dir) / "transfer_with_ci.csv", "w") as f:
        f.write("pair,layer,condition,n_test,transfer_rate,wilson_lo,wilson_hi,"
                "boot_lo,boot_hi,preserve_rate,preserve_wilson_lo,preserve_wilson_hi\n")
        for pr in all_results:
            pair = f"{pr['source']}_{pr['target']}"
            for key, data in pr["conditions"].items():
                wci = data["transfer_wilson_ci"]
                bci = data["transfer_bootstrap_ci"]
                pwci = data["preserve_wilson_ci"]
                f.write(f"{pair},{data['layer']},{data['condition']},{data['n_test']},"
                        f"{data['transfer_rate']},{wci[0]},{wci[1]},"
                        f"{bci[0]},{bci[1]},"
                        f"{data['preserve_rate']},{pwci[0]},{pwci[1]}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 23 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--output-dir", default="results/exp23")
    args = parser.parse_args()
    run_proper_stats(
        model_name=args.model,
        device=args.device,
        n_test=args.n_test,
        output_dir=args.output_dir,
    )
