#!/usr/bin/env python3
"""Experiment 27: Complete Baselines.

Three control conditions to validate that transplantation transfer is not
spurious:
  1. Random source — transplant activations from an unrelated third task.
  2. Shuffle positions — correct source task, but position mapping randomized.
  3. Magnitude-matched Gaussian noise — random vectors with matching L2 norms.

All run on Llama-3B at ~30% depth (layer 8) with the same task pairs and
metrics as exp8.
"""

import json
import sys
import os
import logging
import random
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
            logging.FileHandler(os.path.join(output_dir, "exp27.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def find_demo_token_positions(tokenizer, prompt, n_demos=5):
    """Find all demo token positions."""
    lines = prompt.strip().split("\n")
    positions = []
    current_pos = 0

    for i, line in enumerate(lines):
        line_tokens = tokenizer.encode(line, add_special_tokens=False)
        line_len = len(line_tokens)
        if i >= len(lines) - 2:
            break
        if line.startswith("Input:") or line.startswith("Output:"):
            for p in range(current_pos, current_pos + line_len):
                positions.append(p)
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

    handle = model.get_layer_module(layer).register_forward_hook(hook_fn)
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

    handle = model.get_layer_module(layer).register_forward_hook(hook_fn)
    with torch.no_grad():
        output_ids = model.model.generate(
            tokens, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=model.tokenizer.eos_token_id,
        )
    handle.remove()
    new_tokens = output_ids[0, tokens.shape[1]:]
    text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()


def evaluate_transfer(model, source_task, target_task, layer, position_vectors,
                      target_demos, target_test_inputs):
    """Evaluate transfer/preserve/neither rates."""
    n_test = len(target_test_inputs)
    transfer = 0
    preserve = 0

    for t_input in target_test_inputs:
        prompt = target_task.format_prompt(target_demos, t_input)
        output = generate_with_multi_intervention(model, prompt, layer, position_vectors)
        try:
            if source_task.score_output(t_input, output) == "correct":
                transfer += 1
                continue
        except Exception:
            pass
        try:
            if target_task.score_output(t_input, output) == "correct":
                preserve += 1
        except Exception:
            pass

    return {
        "transfer_rate": transfer / n_test,
        "preserve_rate": preserve / n_test,
        "neither_rate": (n_test - transfer - preserve) / n_test,
    }


def run_baselines(model_name="meta-llama/Llama-3.2-3B-Instruct",
                  device="cuda:3", n_demos=5, n_test=20,
                  output_dir="results/exp27", seed=42):
    logger = setup_logging(output_dir)
    logger.info("Experiment 27: Complete Baselines")
    logger.info(f"Start: {datetime.now().isoformat()}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_name, device=device)
    tasks = {name: TaskRegistry.get(name) for name in INCLUDED_TASKS}

    layer = model.layer_at_fraction(0.30)
    logger.info(f"Model: {model_name}, intervention layer: {layer}")

    # Pick a fixed third task for "random source" baseline
    third_task_name = "pattern_completion"
    third_task = tasks[third_task_name]

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

        pair_result = {
            "source": source_name,
            "target": target_name,
            "layer": layer,
            "baselines": {},
        }

        # ── Baseline 1: True source (normal transplant, for reference) ──
        source_vectors = {}
        for s_input in source_test_inputs:
            prompt = source_task.format_prompt(source_demos, s_input)
            positions = find_demo_token_positions(model.tokenizer, prompt, n_demos)
            acts = extract_multi_position_activations(model, prompt, layer, positions)
            for pos, vec in acts.items():
                if pos not in source_vectors:
                    source_vectors[pos] = []
                source_vectors[pos].append(vec)

        mean_source = {p: torch.stack(vs).mean(0) for p, vs in source_vectors.items()}
        true_result = evaluate_transfer(
            model, source_task, target_task, layer, mean_source,
            target_demos, target_test_inputs
        )
        pair_result["baselines"]["true_source"] = true_result
        logger.info(f"  True source:   transfer={true_result['transfer_rate']:.2f}")

        # ── Baseline 2: Random source (third unrelated task) ──
        # Use a third task that is neither source nor target
        rand_task = third_task
        if third_task_name in (source_name, target_name):
            # Fall back to a different task
            for t in INCLUDED_TASKS:
                if t not in (source_name, target_name):
                    rand_task = tasks[t]
                    break

        rand_demos = rand_task.generate_demos(n_demos)
        rand_test_inputs = rand_task.generate_test_inputs(n_test)

        rand_vectors = {}
        for r_input in rand_test_inputs:
            prompt = rand_task.format_prompt(rand_demos, r_input)
            positions = find_demo_token_positions(model.tokenizer, prompt, n_demos)
            acts = extract_multi_position_activations(model, prompt, layer, positions)
            for pos, vec in acts.items():
                if pos not in rand_vectors:
                    rand_vectors[pos] = []
                rand_vectors[pos].append(vec)

        mean_rand = {p: torch.stack(vs).mean(0) for p, vs in rand_vectors.items()}
        rand_result = evaluate_transfer(
            model, source_task, target_task, layer, mean_rand,
            target_demos, target_test_inputs
        )
        pair_result["baselines"]["random_source"] = rand_result
        logger.info(f"  Random source: transfer={rand_result['transfer_rate']:.2f}")

        # ── Baseline 3: Shuffled positions ──
        # Take the true source vectors but randomize which position gets which
        positions_list = sorted(mean_source.keys())
        shuffled_pos = positions_list.copy()
        random.shuffle(shuffled_pos)
        shuffled_vectors = {
            orig: mean_source[shuf]
            for orig, shuf in zip(positions_list, shuffled_pos)
        }
        shuffle_result = evaluate_transfer(
            model, source_task, target_task, layer, shuffled_vectors,
            target_demos, target_test_inputs
        )
        pair_result["baselines"]["shuffled_positions"] = shuffle_result
        logger.info(f"  Shuffled pos:  transfer={shuffle_result['transfer_rate']:.2f}")

        # ── Baseline 4: Magnitude-matched Gaussian noise ──
        noise_vectors = {}
        for pos, vec in mean_source.items():
            norm = vec.norm().item()
            noise = torch.randn_like(vec)
            noise = noise * (norm / noise.norm().item()) if noise.norm().item() > 0 else noise
            noise_vectors[pos] = noise

        noise_result = evaluate_transfer(
            model, source_task, target_task, layer, noise_vectors,
            target_demos, target_test_inputs
        )
        pair_result["baselines"]["magnitude_noise"] = noise_result
        logger.info(f"  Noise (mag):   transfer={noise_result['transfer_rate']:.2f}")

        all_results.append(pair_result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Pair':35s} {'True':>8s} {'Random':>8s} {'Shuffle':>8s} {'Noise':>8s}")
    logger.info("-" * 70)

    for pr in all_results:
        pair_name = f"{pr['source']} -> {pr['target']}"
        b = pr["baselines"]
        logger.info(
            f"{pair_name:35s} "
            f"{b['true_source']['transfer_rate']:8.2f} "
            f"{b['random_source']['transfer_rate']:8.2f} "
            f"{b['shuffled_positions']['transfer_rate']:8.2f} "
            f"{b['magnitude_noise']['transfer_rate']:8.2f}"
        )

    # Aggregate means
    for btype in ["true_source", "random_source", "shuffled_positions", "magnitude_noise"]:
        rates = [pr["baselines"][btype]["transfer_rate"] for pr in all_results]
        logger.info(f"  Mean {btype:25s}: {np.mean(rates):.3f}")

    # Save
    results = {
        "metadata": {
            "model": model_name,
            "layer": layer,
            "n_demos": n_demos,
            "n_test": n_test,
            "seed": seed,
            "third_task": third_task_name,
            "timestamp": datetime.now().isoformat(),
        },
        "pair_results": all_results,
    }

    with open(Path(output_dir) / "baseline_control_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(Path(output_dir) / "baseline_controls.csv", "w") as f:
        f.write("source,target,condition,transfer_rate,preserve_rate,neither_rate\n")
        for pr in all_results:
            for cond_name, cond_data in pr["baselines"].items():
                f.write(
                    f"{pr['source']},{pr['target']},{cond_name},"
                    f"{cond_data['transfer_rate']},{cond_data['preserve_rate']},"
                    f"{cond_data['neither_rate']}\n"
                )

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 27 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--output-dir", default="results/exp27")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_baselines(
        model_name=args.model,
        device=args.device,
        n_test=args.n_test,
        output_dir=args.output_dir,
        seed=args.seed,
    )
