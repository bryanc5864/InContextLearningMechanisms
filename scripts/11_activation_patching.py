#!/usr/bin/env python3
"""Experiment 11: Activation Patching (Causal Tracing).

Test which (layer, position) pairs are NECESSARY for task execution by
corrupting activations and measuring accuracy drop.
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
from src.extraction import get_position_index

INCLUDED_TASKS = [
    "uppercase", "first_letter", "repeat_word", "length",
    "linear_2x", "sentiment", "antonym", "pattern_completion",
]

POSITIONS = ["last_demo_token", "first_query_token"]


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "exp11.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def generate_with_noise(model, prompt, layer, position, noise_scale=1.0, max_new_tokens=30):
    """Generate with Gaussian noise added at (layer, position)."""
    tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    noise_added = [False]

    def hook_fn(module, input, output):
        if noise_added[0]:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        # Add Gaussian noise at the specified position
        if position < hidden.shape[1]:
            noise = torch.randn_like(hidden[0, position, :]) * noise_scale
            hidden[0, position, :] = hidden[0, position, :] + noise

        noise_added[0] = True

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


def run_patching(model_name="meta-llama/Llama-3.2-3B-Instruct", device="cuda:2", n_demos=5, n_test=20,
                  output_dir="results/exp11", noise_scales=None, positions=None):
    logger = setup_logging(output_dir)
    logger.info("Experiment 11: Activation Patching (Causal Tracing)")
    logger.info(f"Start: {datetime.now().isoformat()}")

    model = load_model(model_name, device=device)
    tasks = {name: TaskRegistry.get(name) for name in INCLUDED_TASKS}

    # Dynamic layer range: sample ~14 evenly-spaced layers
    step = max(1, model.n_layers // 14)
    test_layers = list(range(0, model.n_layers, step))
    logger.info(f"Model: {model_name} ({model.n_layers} layers), test_layers={test_layers}")

    # Allow overriding positions
    test_positions = positions if positions is not None else POSITIONS
    logger.info(f"Testing positions: {test_positions}")

    # Noise scales to test
    noise_scales = noise_scales if noise_scales is not None else [0.5, 1.0, 2.0, 5.0]

    all_results = []

    for task_name in INCLUDED_TASKS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Task: {task_name}")
        logger.info(f"{'='*50}")

        task = tasks[task_name]
        demos = task.generate_demos(n_demos)
        test_inputs = task.generate_test_inputs(n_test)

        # Baseline accuracy (no intervention)
        baseline_correct = 0
        for ti in test_inputs:
            prompt = task.format_prompt(demos, ti)
            tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.model.generate(
                    tokens,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=model.tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0, tokens.shape[1]:]
            output = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            output = output.strip().split("\n")[0].strip()
            if task.score_output(ti, output) == "correct":
                baseline_correct += 1

        baseline_acc = baseline_correct / n_test
        logger.info(f"  Baseline accuracy: {baseline_acc:.2f}")

        task_results = {
            "task": task_name,
            "baseline_accuracy": baseline_acc,
            "position_results": {},
        }

        for pos_type in test_positions:
            logger.info(f"\n  Position: {pos_type}")
            pos_results = {"layers": {}}

            for layer in test_layers:
                layer_data = {"noise_scales": {}}

                for noise_scale in noise_scales:
                    correct = 0

                    for ti in test_inputs:
                        prompt = task.format_prompt(demos, ti)
                        pos_idx = get_position_index(model, prompt, pos_type)

                        output = generate_with_noise(
                            model, prompt, layer, pos_idx, noise_scale
                        )

                        if task.score_output(ti, output) == "correct":
                            correct += 1

                    acc = correct / n_test
                    layer_data["noise_scales"][str(noise_scale)] = acc

                # Use the highest noise scale as primary disruption measure
                primary_scale = str(max(noise_scales))
                disruption = baseline_acc - layer_data["noise_scales"][primary_scale]
                layer_data["primary_noise_scale"] = primary_scale
                layer_data["disruption"] = disruption

                pos_results["layers"][str(layer)] = layer_data

                if disruption > 0.2:
                    marker = "***"
                elif disruption > 0.1:
                    marker = "**"
                else:
                    marker = ""

                logger.info(
                    f"    Layer {layer:2d}: acc@noise{primary_scale}={layer_data['noise_scales'][primary_scale]:.2f} "
                    f"disruption={disruption:.2f} {marker}"
                )

            task_results["position_results"][pos_type] = pos_results

        all_results.append(task_results)

    # Aggregate heatmap: which (layer, position) pairs are most critical?
    logger.info("\n" + "=" * 60)
    logger.info(f"CAUSAL IMPORTANCE HEATMAP (disruption at noise={max(noise_scales)})")
    logger.info("=" * 60)

    for pos_type in test_positions:
        logger.info(f"\nPosition: {pos_type}")
        logger.info("Layer:  " + "  ".join(f"{l:5d}" for l in test_layers))

        for task_result in all_results:
            task_name = task_result["task"]
            pos_data = task_result["position_results"].get(pos_type, {})
            layers_data = pos_data.get("layers", {})

            disruptions = []
            for l in test_layers:
                d = layers_data.get(str(l), {}).get("disruption", 0)
                disruptions.append(d)

            row = "  ".join(f"{d:5.2f}" for d in disruptions)
            logger.info(f"{task_name:12s}: {row}")

    # Find most critical layers
    logger.info("\n" + "=" * 60)
    logger.info("MOST CRITICAL LAYERS (mean disruption > 0.1)")
    logger.info("=" * 60)

    for pos_type in test_positions:
        layer_disruptions = {l: [] for l in test_layers}
        for task_result in all_results:
            pos_data = task_result["position_results"].get(pos_type, {})
            layers_data = pos_data.get("layers", {})
            for l in test_layers:
                d = layers_data.get(str(l), {}).get("disruption", 0)
                layer_disruptions[l].append(d)

        logger.info(f"\n{pos_type}:")
        for l in test_layers:
            mean_d = np.mean(layer_disruptions[l])
            if mean_d > 0.1:
                logger.info(f"  Layer {l:2d}: mean disruption = {mean_d:.3f}")

    # Save
    results = {
        "metadata": {
            "test_layers": test_layers,
            "positions": test_positions,
            "noise_scales": noise_scales,
            "n_test": n_test,
            "timestamp": datetime.now().isoformat(),
        },
        "task_results": all_results,
    }

    with open(Path(output_dir) / "patching_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV heatmap
    with open(Path(output_dir) / "disruption_heatmap.csv", "w") as f:
        f.write("task,position," + ",".join(f"layer_{l}" for l in test_layers) + "\n")
        for task_result in all_results:
            task_name = task_result["task"]
            for pos_type in test_positions:
                pos_data = task_result["position_results"].get(pos_type, {})
                layers_data = pos_data.get("layers", {})
                disruptions = [
                    str(layers_data.get(str(l), {}).get("disruption", 0))
                    for l in test_layers
                ]
                f.write(f"{task_name},{pos_type}," + ",".join(disruptions) + "\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 11 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--output-dir", default="results/exp11")
    parser.add_argument("--noise-scales", type=float, nargs="+", default=None,
                        help="Noise scales to test (default: 0.5 1.0 2.0 5.0)")
    parser.add_argument("--positions", nargs="+", default=None,
                        choices=["last_demo_token", "first_query_token"],
                        help="Position types to test (default: both)")
    args = parser.parse_args()
    run_patching(
        model_name=args.model,
        device=args.device,
        n_test=args.n_test,
        output_dir=args.output_dir,
        noise_scales=args.noise_scales,
        positions=args.positions,
    )
