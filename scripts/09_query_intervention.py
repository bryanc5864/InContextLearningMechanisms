#!/usr/bin/env python3
"""Experiment 9: Query Position Intervention.

Test whether intervening at the query position (where task identity is aggregated
via attention) can transfer task behavior. Phase 2 showed query position peaks
at layer 12 with 83% probe accuracy.
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
from src.extraction import extract_activations, get_position_index

INCLUDED_TASKS = [
    "uppercase", "first_letter", "repeat_word", "length",
    "linear_2x", "sentiment", "antonym", "pattern_completion",
]

# Test pairs across different regimes
TEST_PAIRS = [
    ("uppercase", "sentiment"),
    ("linear_2x", "uppercase"),
    ("sentiment", "first_letter"),
    ("pattern_completion", "length"),
    ("antonym", "repeat_word"),
    ("first_letter", "linear_2x"),
]


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "exp9.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def run_query_intervention(device="cuda:3", n_demos=5, n_test=10, output_dir="results/exp9"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 9: Query Position Intervention")
    logger.info(f"Start: {datetime.now().isoformat()}")

    model = load_model(device=device)
    tasks = {name: TaskRegistry.get(name) for name in INCLUDED_TASKS}

    # Layers to test: focus on the trajectory from Phase 2
    # Peak at layer 12, but test surrounding layers
    test_layers = [0, 4, 8, 10, 11, 12, 13, 14, 16, 20, 24, 27]
    logger.info(f"Testing layers: {test_layers}")

    all_results = []

    for source_name, target_name in TEST_PAIRS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Pair: {source_name} → {target_name}")
        logger.info(f"{'='*50}")

        source_task = tasks[source_name]
        target_task = tasks[target_name]

        source_demos = source_task.generate_demos(n_demos)
        target_demos = target_task.generate_demos(n_demos)

        # Use source task's test inputs for source vectors
        source_test_inputs = source_task.generate_test_inputs(n_test)
        # Use target task's test inputs for target contexts
        target_test_inputs = target_task.generate_test_inputs(n_test)

        pair_results = {
            "source": source_name,
            "target": target_name,
            "layer_results": [],
        }

        for layer in test_layers:
            # Extract source vectors at query position for this layer
            source_vectors = []
            for ti in source_test_inputs:
                prompt = source_task.format_prompt(source_demos, ti)
                pos_idx = get_position_index(model, prompt, "first_query_token")
                acts = extract_activations(model, prompt, layers=[layer], position=pos_idx)
                source_vectors.append(acts[layer])

            mean_source_vec = torch.stack(source_vectors).mean(dim=0)

            # Transplant into target contexts and measure transfer
            transfer_count = 0
            target_preserved = 0
            neither_count = 0
            details = []

            for ti in target_test_inputs:
                prompt = target_task.format_prompt(target_demos, ti)
                pos_idx = get_position_index(model, prompt, "first_query_token")

                # Generate with intervention at query position
                output = model.generate_with_intervention(
                    prompt, layer, pos_idx, mean_source_vec, max_new_tokens=30
                )

                # Score output
                try:
                    source_correct = source_task.score_output(ti, output) == "correct"
                except (ValueError, TypeError, KeyError):
                    source_correct = False
                try:
                    target_correct = target_task.score_output(ti, output) == "correct"
                except (ValueError, TypeError, KeyError):
                    target_correct = False

                if source_correct:
                    transfer_count += 1
                    classification = "source"
                elif target_correct:
                    target_preserved += 1
                    classification = "target"
                else:
                    neither_count += 1
                    classification = "neither"

                details.append({
                    "input": ti,
                    "output": output,
                    "classification": classification,
                })

            transfer_rate = transfer_count / n_test
            preserve_rate = target_preserved / n_test
            neither_rate = neither_count / n_test

            pair_results["layer_results"].append({
                "layer": layer,
                "transfer_rate": transfer_rate,
                "preserve_rate": preserve_rate,
                "neither_rate": neither_rate,
                "details": details,
            })

            marker = "***" if transfer_rate > 0.3 else ""
            logger.info(
                f"  Layer {layer:2d}: transfer={transfer_rate:.2f} "
                f"preserve={preserve_rate:.2f} neither={neither_rate:.2f} {marker}"
            )

        # Find peak transfer layer
        rates = [lr["transfer_rate"] for lr in pair_results["layer_results"]]
        peak_idx = int(np.argmax(rates))
        pair_results["peak_layer"] = test_layers[peak_idx]
        pair_results["peak_transfer"] = float(max(rates))

        all_results.append(pair_results)
        logger.info(f"  Peak: layer={pair_results['peak_layer']}, transfer={pair_results['peak_transfer']:.2f}")

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    # Aggregate by layer
    layer_transfers = {l: [] for l in test_layers}
    for pr in all_results:
        for lr in pr["layer_results"]:
            layer_transfers[lr["layer"]].append(lr["transfer_rate"])

    for layer in test_layers:
        mean_t = np.mean(layer_transfers[layer])
        logger.info(f"  Layer {layer:2d}: mean transfer = {mean_t:.3f}")

    # Save results
    results = {
        "metadata": {
            "test_layers": test_layers,
            "pairs": TEST_PAIRS,
            "n_test": n_test,
            "position": "first_query_token",
            "timestamp": datetime.now().isoformat(),
        },
        "pair_results": all_results,
        "layer_summary": {
            str(l): float(np.mean(layer_transfers[l])) for l in test_layers
        },
    }

    with open(Path(output_dir) / "query_intervention_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    with open(Path(output_dir) / "query_intervention.csv", "w") as f:
        f.write("pair,layer,transfer_rate,preserve_rate,neither_rate\n")
        for pr in all_results:
            pair = f"{pr['source']}_{pr['target']}"
            for lr in pr["layer_results"]:
                f.write(f"{pair},{lr['layer']},{lr['transfer_rate']},{lr['preserve_rate']},{lr['neither_rate']}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 9 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--output-dir", default="results/exp9")
    args = parser.parse_args()
    run_query_intervention(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
