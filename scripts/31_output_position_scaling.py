#!/usr/bin/env python3
"""Experiment 31: Output Token Position Scaling (Q3).

How many output positions are needed for transfer?
Shows scaling curve and redundancy in output position activations.

Design:
- Part A: Random subsets of output positions: sizes [1, 2, 3, 4, 5, 7, 10, ALL]
  - 10 random subsets per size, averaged
  - Primary pair: uppercase→repeat_word at layer 8
  - Secondary: uppercase→length, repeat_word→length
- Part B: Structured subsets:
  - first_output_token_per_demo (5 positions)
  - last_output_token_per_demo (5 positions)
  - every_other output position
  - first_demo_outputs only
  - last_demo_outputs only
- N=20
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

TEST_PAIRS = [
    ("uppercase", "repeat_word"),
    ("uppercase", "length"),
    ("repeat_word", "length"),
]

RANDOM_SUBSET_SIZES = [1, 2, 3, 4, 5, 7, 10]  # ALL handled separately
N_RANDOM_SUBSETS = 10
INTERVENTION_LAYER = 8
N_DEMOS = 5


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "exp31.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def find_demo_output_positions_detailed(tokenizer, prompt, n_demos=5):
    """Find output token positions with per-demo detail.

    Returns:
        dict with:
            'all_output': list of all output token positions
            'per_demo': list of n_demos lists, each with output positions for that demo
    """
    lines = prompt.strip().split("\n")
    all_output = []
    per_demo = []
    current_pos = 0
    demo_idx = 0

    for i, line in enumerate(lines):
        line_tokens = tokenizer.encode(line, add_special_tokens=False)
        line_len = len(line_tokens)

        # Last two lines are the query
        if i >= len(lines) - 2:
            break

        if line.startswith("Output:") and demo_idx < n_demos:
            # Output line positions
            demo_positions = list(range(current_pos, current_pos + line_len))
            all_output.extend(demo_positions)
            per_demo.append(demo_positions)
            demo_idx += 1

        current_pos += line_len
        if i < len(lines) - 1:
            newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
            current_pos += len(newline_tokens)

    return {
        'all_output': all_output,
        'per_demo': per_demo,
    }


def find_all_demo_positions(tokenizer, prompt, n_demos=5):
    """Find all demo token positions (input + output lines)."""
    lines = prompt.strip().split("\n")
    positions = []
    current_pos = 0

    for i, line in enumerate(lines):
        line_tokens = tokenizer.encode(line, add_special_tokens=False)
        line_len = len(line_tokens)

        if i >= len(lines) - 2:
            break

        if line.startswith("Input:") or line.startswith("Output:"):
            positions.extend(range(current_pos, current_pos + line_len))

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

        hidden = hidden.clone()
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


def evaluate_transfer(model, source_task, target_task, target_demos, test_inputs,
                      position_vectors, layer):
    """Evaluate transfer rate for a given set of position vectors."""
    transfer_count = 0
    for t_input in test_inputs:
        target_prompt = target_task.format_prompt(target_demos, t_input)
        output = generate_with_multi_intervention(
            model, target_prompt, layer, position_vectors
        )
        try:
            if source_task.score_output(t_input, output) == "correct":
                transfer_count += 1
        except Exception:
            pass
    return transfer_count / len(test_inputs)


def run_output_position_scaling(device="cuda:6", n_test=20, output_dir="results/exp31"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 31: Output Token Position Scaling")
    logger.info(f"Start: {datetime.now().isoformat()}")

    rng = np.random.RandomState(42)
    model = load_model(device=device)

    tasks = {
        "uppercase": TaskRegistry.get("uppercase"),
        "repeat_word": TaskRegistry.get("repeat_word"),
        "length": TaskRegistry.get("length"),
    }

    # ═══════════════════════════════════════════════════════════════════
    # Part A: Random Subsets
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Part A: Random Output Position Subsets")
    logger.info("=" * 60)

    part_a_results = []

    for source_name, target_name in TEST_PAIRS:
        logger.info(f"\n  Pair: {source_name} → {target_name}")

        source_task = tasks[source_name]
        target_task = tasks[target_name]

        source_demos = source_task.generate_demos(N_DEMOS)
        target_demos = target_task.generate_demos(N_DEMOS)
        test_inputs = target_task.generate_test_inputs(n_test)

        # Get all source activations at ALL demo positions (input + output)
        # We need the full activation map, then will select output-only subsets
        all_source_vectors = {}
        output_positions_info = None

        for s_input in test_inputs:
            prompt = source_task.format_prompt(source_demos, s_input)
            if output_positions_info is None:
                output_positions_info = find_demo_output_positions_detailed(
                    model.tokenizer, prompt, N_DEMOS
                )

            all_positions = find_all_demo_positions(model.tokenizer, prompt, N_DEMOS)
            acts = extract_multi_position_activations(model, prompt, INTERVENTION_LAYER, all_positions)

            for pos, vec in acts.items():
                if pos not in all_source_vectors:
                    all_source_vectors[pos] = []
                all_source_vectors[pos].append(vec)

        # Average across source examples
        mean_all_vectors = {
            pos: torch.stack(vecs).mean(dim=0)
            for pos, vecs in all_source_vectors.items()
        }

        all_output_positions = output_positions_info['all_output']
        total_output_positions = len(all_output_positions)
        logger.info(f"    Total output positions: {total_output_positions}")

        # Input-only positions (all demo positions minus output positions)
        all_demo_positions = set(mean_all_vectors.keys())
        output_set = set(all_output_positions)
        input_positions = sorted(all_demo_positions - output_set)

        # Baseline: ALL output positions (+ all input positions)
        all_vectors = {pos: mean_all_vectors[pos] for pos in mean_all_vectors}
        baseline_rate = evaluate_transfer(
            model, source_task, target_task, target_demos, test_inputs,
            all_vectors, INTERVENTION_LAYER
        )
        logger.info(f"    Baseline (ALL positions): {baseline_rate:.2f}")

        # Output-only baseline
        output_only_vectors = {pos: mean_all_vectors[pos] for pos in all_output_positions if pos in mean_all_vectors}
        output_only_rate = evaluate_transfer(
            model, source_task, target_task, target_demos, test_inputs,
            output_only_vectors, INTERVENTION_LAYER
        )
        logger.info(f"    Output-only (all {len(output_only_vectors)} positions): {output_only_rate:.2f}")

        pair_scaling = {
            "source": source_name,
            "target": target_name,
            "total_output_positions": total_output_positions,
            "baseline_all_rate": baseline_rate,
            "output_only_all_rate": output_only_rate,
            "subset_sizes": {},
        }

        for size in RANDOM_SUBSET_SIZES:
            if size > total_output_positions:
                continue

            subset_rates = []
            for trial in range(N_RANDOM_SUBSETS):
                # Sample random subset of output positions
                subset_indices = rng.choice(
                    len(all_output_positions), size=size, replace=False
                )
                subset_positions = [all_output_positions[i] for i in subset_indices]

                # Use input positions + subset of output positions
                subset_vectors = {pos: mean_all_vectors[pos] for pos in input_positions if pos in mean_all_vectors}
                for pos in subset_positions:
                    if pos in mean_all_vectors:
                        subset_vectors[pos] = mean_all_vectors[pos]

                rate = evaluate_transfer(
                    model, source_task, target_task, target_demos, test_inputs,
                    subset_vectors, INTERVENTION_LAYER
                )
                subset_rates.append(rate)

            mean_rate = np.mean(subset_rates)
            std_rate = np.std(subset_rates)

            pair_scaling["subset_sizes"][str(size)] = {
                "size": size,
                "mean_transfer_rate": float(mean_rate),
                "std_transfer_rate": float(std_rate),
                "all_rates": [float(r) for r in subset_rates],
            }

            logger.info(f"    Size {size:2d}: {mean_rate:.3f} ± {std_rate:.3f}")

        part_a_results.append(pair_scaling)

    # ═══════════════════════════════════════════════════════════════════
    # Part B: Structured Subsets
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Part B: Structured Output Position Subsets")
    logger.info("=" * 60)

    part_b_results = []

    for source_name, target_name in TEST_PAIRS:
        logger.info(f"\n  Pair: {source_name} → {target_name}")

        source_task = tasks[source_name]
        target_task = tasks[target_name]

        source_demos = source_task.generate_demos(N_DEMOS)
        target_demos = target_task.generate_demos(N_DEMOS)
        test_inputs = target_task.generate_test_inputs(n_test)

        # Re-extract activations
        all_source_vectors = {}
        output_positions_info = None

        for s_input in test_inputs:
            prompt = source_task.format_prompt(source_demos, s_input)
            if output_positions_info is None:
                output_positions_info = find_demo_output_positions_detailed(
                    model.tokenizer, prompt, N_DEMOS
                )

            all_positions = find_all_demo_positions(model.tokenizer, prompt, N_DEMOS)
            acts = extract_multi_position_activations(model, prompt, INTERVENTION_LAYER, all_positions)

            for pos, vec in acts.items():
                if pos not in all_source_vectors:
                    all_source_vectors[pos] = []
                all_source_vectors[pos].append(vec)

        mean_all_vectors = {
            pos: torch.stack(vecs).mean(dim=0)
            for pos, vecs in all_source_vectors.items()
        }

        all_output_positions = output_positions_info['all_output']
        per_demo = output_positions_info['per_demo']
        output_set = set(all_output_positions)
        all_demo_positions = set(mean_all_vectors.keys())
        input_positions = sorted(all_demo_positions - output_set)

        # Define structured subsets
        structured_subsets = {}

        # 1. First output token per demo
        first_per_demo = [demo_pos[0] for demo_pos in per_demo if len(demo_pos) > 0]
        structured_subsets["first_output_per_demo"] = first_per_demo

        # 2. Last output token per demo
        last_per_demo = [demo_pos[-1] for demo_pos in per_demo if len(demo_pos) > 0]
        structured_subsets["last_output_per_demo"] = last_per_demo

        # 3. Every other output position
        every_other = [all_output_positions[i] for i in range(0, len(all_output_positions), 2)]
        structured_subsets["every_other"] = every_other

        # 4. First demo outputs only
        if len(per_demo) > 0:
            structured_subsets["first_demo_only"] = per_demo[0]

        # 5. Last demo outputs only
        if len(per_demo) > 0:
            structured_subsets["last_demo_only"] = per_demo[-1]

        pair_structured = {
            "source": source_name,
            "target": target_name,
            "subsets": {},
        }

        for subset_name, subset_positions in structured_subsets.items():
            # Input positions + selected output positions
            subset_vectors = {pos: mean_all_vectors[pos] for pos in input_positions if pos in mean_all_vectors}
            for pos in subset_positions:
                if pos in mean_all_vectors:
                    subset_vectors[pos] = mean_all_vectors[pos]

            rate = evaluate_transfer(
                model, source_task, target_task, target_demos, test_inputs,
                subset_vectors, INTERVENTION_LAYER
            )

            pair_structured["subsets"][subset_name] = {
                "n_output_positions": len(subset_positions),
                "transfer_rate": float(rate),
                "positions": subset_positions,
            }

            logger.info(f"    {subset_name:25s} ({len(subset_positions):2d} pos): {rate:.2f}")

        part_b_results.append(pair_structured)

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info("\nPart A - Random subset scaling (mean across pairs):")
    for size in RANDOM_SUBSET_SIZES:
        rates = []
        for pr in part_a_results:
            if str(size) in pr["subset_sizes"]:
                rates.append(pr["subset_sizes"][str(size)]["mean_transfer_rate"])
        if rates:
            logger.info(f"  Size {size:2d}: {np.mean(rates):.3f}")

    baselines = [pr["output_only_all_rate"] for pr in part_a_results]
    logger.info(f"  ALL output: {np.mean(baselines):.3f}")

    logger.info("\nPart B - Structured subsets (mean across pairs):")
    subset_names = set()
    for pr in part_b_results:
        subset_names.update(pr["subsets"].keys())
    for sn in sorted(subset_names):
        rates = [
            pr["subsets"][sn]["transfer_rate"]
            for pr in part_b_results
            if sn in pr["subsets"]
        ]
        if rates:
            logger.info(f"  {sn:25s}: {np.mean(rates):.3f}")

    # Save
    results = {
        "metadata": {
            "intervention_layer": INTERVENTION_LAYER,
            "n_demos": N_DEMOS,
            "n_test": n_test,
            "random_subset_sizes": RANDOM_SUBSET_SIZES,
            "n_random_subsets": N_RANDOM_SUBSETS,
            "pairs": TEST_PAIRS,
            "timestamp": datetime.now().isoformat(),
        },
        "part_a_random_subsets": part_a_results,
        "part_b_structured_subsets": part_b_results,
    }

    with open(Path(output_dir) / "output_position_scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV for Part A
    with open(Path(output_dir) / "output_position_scaling_a.csv", "w") as f:
        f.write("pair,subset_size,mean_transfer_rate,std_transfer_rate\n")
        for pr in part_a_results:
            pair = f"{pr['source']}_{pr['target']}"
            for size_str, data in sorted(pr["subset_sizes"].items(), key=lambda x: int(x[0])):
                f.write(f"{pair},{data['size']},{data['mean_transfer_rate']},{data['std_transfer_rate']}\n")
            f.write(f"{pair},ALL,{pr['output_only_all_rate']},0.0\n")

    # CSV for Part B
    with open(Path(output_dir) / "output_position_scaling_b.csv", "w") as f:
        f.write("pair,subset_name,n_output_positions,transfer_rate\n")
        for pr in part_b_results:
            pair = f"{pr['source']}_{pr['target']}"
            for sn, data in pr["subsets"].items():
                f.write(f"{pair},{sn},{data['n_output_positions']},{data['transfer_rate']}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 31 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--output-dir", default="results/exp31")
    args = parser.parse_args()
    run_output_position_scaling(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
