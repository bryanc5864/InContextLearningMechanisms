#!/usr/bin/env python3
"""Experiment 12: Layer-Wise Ablation Study.

Skip individual layers (pass residual stream unchanged) and measure
which layers are necessary for task execution.
"""

import json
import sys
import os
import logging
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


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "exp12.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def generate_with_layer_skip(model, prompt, skip_layer, max_new_tokens=30):
    """Generate with a specific layer skipped (residual passthrough only)."""
    tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    skip_done = [False]

    def skip_hook(module, input, output):
        if skip_done[0]:
            return output

        # Return input unchanged (skip this layer's computation)
        # input is typically (hidden_states, attention_mask, position_ids, ...)
        if isinstance(input, tuple) and len(input) > 0:
            hidden = input[0]
        else:
            return output

        skip_done[0] = True

        # Return the input hidden states as if this layer did nothing
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    layer_module = model.get_layer_module(skip_layer)
    handle = layer_module.register_forward_hook(skip_hook)

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


def generate_with_phase_skip(model, prompt, skip_layers, max_new_tokens=30):
    """Generate with multiple layers skipped."""
    tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    skipped = set()

    def make_skip_hook(layer_idx):
        def skip_hook(module, input, output):
            if layer_idx in skipped:
                return output

            if isinstance(input, tuple) and len(input) > 0:
                hidden = input[0]
            else:
                return output

            skipped.add(layer_idx)

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return skip_hook

    handles = []
    for layer in skip_layers:
        layer_module = model.get_layer_module(layer)
        handle = layer_module.register_forward_hook(make_skip_hook(layer))
        handles.append(handle)

    with torch.no_grad():
        output_ids = model.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=model.tokenizer.eos_token_id,
        )

    for handle in handles:
        handle.remove()

    new_tokens = output_ids[0, tokens.shape[1]:]
    text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()


def run_layer_ablation(device="cuda:2", n_demos=5, n_test=15, output_dir="results/exp12"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 12: Layer-Wise Ablation Study")
    logger.info(f"Start: {datetime.now().isoformat()}")

    model = load_model(device=device)
    tasks = {name: TaskRegistry.get(name) for name in INCLUDED_TASKS}
    n_layers = model.n_layers

    # ═══════════════════════════════════════════════════════════════════
    # Part A: Single Layer Skip
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Part A: Single Layer Skip")
    logger.info("=" * 60)

    # Test every 2nd layer for speed
    test_layers = list(range(0, n_layers, 2))

    single_skip_results = []

    for task_name in INCLUDED_TASKS:
        logger.info(f"\nTask: {task_name}")
        task = tasks[task_name]
        demos = task.generate_demos(n_demos)
        test_inputs = task.generate_test_inputs(n_test)

        # Baseline
        baseline_correct = 0
        for ti in test_inputs:
            prompt = task.format_prompt(demos, ti)
            tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.model.generate(
                    tokens, max_new_tokens=30, do_sample=False,
                    pad_token_id=model.tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0, tokens.shape[1]:]
            output = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            output = output.strip().split("\n")[0].strip()
            if task.score_output(ti, output) == "correct":
                baseline_correct += 1

        baseline_acc = baseline_correct / n_test
        logger.info(f"  Baseline: {baseline_acc:.2f}")

        task_result = {"task": task_name, "baseline": baseline_acc, "layers": {}}

        for layer in test_layers:
            correct = 0
            for ti in test_inputs:
                prompt = task.format_prompt(demos, ti)
                output = generate_with_layer_skip(model, prompt, layer)
                if task.score_output(ti, output) == "correct":
                    correct += 1

            acc = correct / n_test
            drop = baseline_acc - acc
            task_result["layers"][str(layer)] = {"accuracy": acc, "drop": drop}

            marker = "***" if drop > 0.2 else ""
            logger.info(f"  Skip layer {layer:2d}: acc={acc:.2f} drop={drop:.2f} {marker}")

        single_skip_results.append(task_result)

    # ═══════════════════════════════════════════════════════════════════
    # Part B: Phase Skip (skip groups of layers)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Part B: Phase Skip")
    logger.info("=" * 60)

    phases = {
        "early": list(range(0, 8)),
        "mid": list(range(8, 16)),
        "late": list(range(16, 24)),
        "final": list(range(24, 28)),
    }

    phase_skip_results = []

    for task_name in INCLUDED_TASKS[:4]:  # Subset for speed
        logger.info(f"\nTask: {task_name}")
        task = tasks[task_name]
        demos = task.generate_demos(n_demos)
        test_inputs = task.generate_test_inputs(n_test)

        # Get baseline from single_skip_results
        baseline_acc = next(r["baseline"] for r in single_skip_results if r["task"] == task_name)

        task_result = {"task": task_name, "baseline": baseline_acc, "phases": {}}

        for phase_name, skip_layers in phases.items():
            correct = 0
            for ti in test_inputs:
                prompt = task.format_prompt(demos, ti)
                output = generate_with_phase_skip(model, prompt, skip_layers)
                if task.score_output(ti, output) == "correct":
                    correct += 1

            acc = correct / n_test
            drop = baseline_acc - acc
            task_result["phases"][phase_name] = {
                "layers": skip_layers,
                "accuracy": acc,
                "drop": drop,
            }

            marker = "***" if drop > 0.3 else ""
            logger.info(f"  Skip {phase_name:6s} ({skip_layers[0]}-{skip_layers[-1]}): "
                       f"acc={acc:.2f} drop={drop:.2f} {marker}")

        phase_skip_results.append(task_result)

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Mean accuracy drop by layer")
    logger.info("=" * 60)

    for layer in test_layers:
        drops = [r["layers"][str(layer)]["drop"] for r in single_skip_results]
        mean_drop = np.mean(drops)
        marker = "***" if mean_drop > 0.1 else ""
        logger.info(f"  Layer {layer:2d}: mean drop = {mean_drop:.3f} {marker}")

    logger.info("\nSUMMARY: Mean accuracy drop by phase")
    for phase_name in phases:
        drops = [r["phases"][phase_name]["drop"] for r in phase_skip_results]
        mean_drop = np.mean(drops)
        logger.info(f"  {phase_name:6s}: mean drop = {mean_drop:.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════════
    results = {
        "metadata": {
            "test_layers": test_layers,
            "phases": {k: list(v) for k, v in phases.items()},
            "n_layers": n_layers,
            "n_test": n_test,
            "timestamp": datetime.now().isoformat(),
        },
        "single_skip_results": single_skip_results,
        "phase_skip_results": phase_skip_results,
    }

    with open(Path(output_dir) / "layer_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(Path(output_dir) / "layer_ablation.csv", "w") as f:
        f.write("task,layer,accuracy,drop\n")
        for r in single_skip_results:
            for layer, data in r["layers"].items():
                f.write(f"{r['task']},{layer},{data['accuracy']},{data['drop']}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 12 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--n-test", type=int, default=15)
    parser.add_argument("--output-dir", default="results/exp12")
    args = parser.parse_args()
    run_layer_ablation(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
