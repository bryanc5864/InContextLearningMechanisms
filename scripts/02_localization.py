#!/usr/bin/env python3
"""Phase 2: Representation Localization via Linear Probing.

Extract residual stream activations at all layers and 3 token positions.
Train 8-way linear probes to find where task identity is encoded.
"""

import json
import sys
import os
import logging
import time
import pickle
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from src.model import load_model, get_model_info
from src.tasks import TaskRegistry
from src.extraction import extract_activations, get_position_index
from src.probing import train_probe, find_optimal_layer


# Tasks that passed Phase 1
INCLUDED_TASKS = [
    "uppercase", "first_letter", "repeat_word", "length",
    "linear_2x", "sentiment", "antonym", "pattern_completion",
]

POSITION_TYPES = ["last_demo_token", "separator_after_demo", "first_query_token"]


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "phase2.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def run_localization(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    device: str = "cuda:3",
    n_demos: int = 5,
    n_test: int = 50,
    output_dir: str = "results/phase2",
):
    logger = setup_logging(output_dir)
    logger.info("Phase 2: Representation Localization")
    logger.info(f"Start: {datetime.now().isoformat()}")
    logger.info(f"Tasks: {INCLUDED_TASKS}")
    logger.info(f"Positions: {POSITION_TYPES}")

    model = load_model(model_name, device=device)
    info = get_model_info(model)
    n_layers = info["n_layers"]
    logger.info(f"Model: {info}")

    # ── Step 1: Extract activations (or load from cache) ────────────
    logger.info("=" * 60)
    logger.info("Step 1: Extracting activations")
    logger.info("=" * 60)

    cache_path = Path(output_dir) / "activations_cache.pkl"
    if cache_path.exists():
        logger.info(f"Loading cached activations from {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        activations = cached["activations"]
        labels = cached["labels"]
        extract_elapsed = 0.0
        logger.info(f"Loaded. {len(labels[POSITION_TYPES[0]])} samples per position.")
    else:
        # Structure: activations[position_type][layer] = list of tensors
        #            labels[position_type] = list of task names
        activations = {pos: {l: [] for l in range(n_layers)} for pos in POSITION_TYPES}
        labels = {pos: [] for pos in POSITION_TYPES}

        total_extractions = len(INCLUDED_TASKS) * n_test
        extraction_count = 0
        extract_start = time.time()

        for task_name in INCLUDED_TASKS:
            task = TaskRegistry.get(task_name)
            demos = task.generate_demos(n_demos)
            test_inputs = task.generate_test_inputs(n_test)
            logger.info(f"  Extracting: {task_name} ({len(test_inputs)} inputs)")

            for i, test_input in enumerate(test_inputs):
                prompt = task.format_prompt(demos, test_input)

                for pos_type in POSITION_TYPES:
                    pos_idx = get_position_index(model, prompt, pos_type)
                    acts = extract_activations(model, prompt, position=pos_idx)

                    for layer_idx, act_tensor in acts.items():
                        activations[pos_type][layer_idx].append(act_tensor)
                    labels[pos_type].append(task_name)

                extraction_count += 1
                if (extraction_count) % 25 == 0:
                    elapsed = time.time() - extract_start
                    rate = extraction_count / elapsed
                    remaining = (total_extractions - extraction_count) / rate
                    logger.info(f"    [{extraction_count}/{total_extractions}] "
                               f"{rate:.1f} samples/s, ~{remaining:.0f}s remaining")

        extract_elapsed = time.time() - extract_start
        logger.info(f"Extraction complete: {extraction_count} samples in {extract_elapsed:.1f}s")

        # Save raw activations
        with open(cache_path, "wb") as f:
            pickle.dump({"activations": activations, "labels": labels}, f)
        logger.info(f"Saved activation cache to {cache_path}")

    # ── Step 2: Train probes ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Training linear probes")
    logger.info("=" * 60)

    probe_results = {}  # probe_results[pos_type][layer] = {accuracy_mean, accuracy_std, ...}

    for pos_type in POSITION_TYPES:
        logger.info(f"\n--- Position: {pos_type} ---")
        probe_results[pos_type] = {}

        for layer in range(n_layers):
            acts_list = activations[pos_type][layer]
            acts_array = torch.stack(acts_list).numpy()
            result = train_probe(acts_array, labels[pos_type], C=1.0, n_splits=3)
            probe_results[pos_type][layer] = {
                "accuracy_mean": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "cv_scores": result["cv_scores"],
            }
            if layer % 4 == 0 or layer == n_layers - 1:
                logger.info(f"  Layer {layer:2d}: {result['accuracy_mean']:.3f} "
                           f"+/- {result['accuracy_std']:.3f}")

    # ── Step 3: Find optimal (L*, p*) ────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 3: Finding optimal intervention coordinates")
    logger.info("=" * 60)

    best_acc = 0.0
    best_layer = 0
    best_pos = POSITION_TYPES[0]

    for pos_type in POSITION_TYPES:
        for layer in range(n_layers):
            acc = probe_results[pos_type][layer]["accuracy_mean"]
            if acc > best_acc:
                best_acc = acc
                best_layer = layer
                best_pos = pos_type

    logger.info(f"Optimal: layer={best_layer}, position={best_pos}, accuracy={best_acc:.4f}")

    # Also find best layer per position
    best_per_pos = {}
    for pos_type in POSITION_TYPES:
        bl = max(range(n_layers), key=lambda l: probe_results[pos_type][l]["accuracy_mean"])
        ba = probe_results[pos_type][bl]["accuracy_mean"]
        best_per_pos[pos_type] = {"layer": bl, "accuracy": ba}
        logger.info(f"  Best for {pos_type}: layer={bl}, accuracy={ba:.4f}")

    # ── Step 4: Save results ─────────────────────────────────────────
    results = {
        "metadata": {
            "model": model_name,
            "device": device,
            "n_demos": n_demos,
            "n_test": n_test,
            "tasks": INCLUDED_TASKS,
            "positions": POSITION_TYPES,
            "n_layers": n_layers,
            "timestamp": datetime.now().isoformat(),
            "extraction_time_s": round(extract_elapsed, 1),
        },
        "optimal": {
            "layer": best_layer,
            "position": best_pos,
            "accuracy": best_acc,
        },
        "best_per_position": best_per_pos,
        "probe_results": {},
    }

    # Convert probe results to serializable format
    for pos_type in POSITION_TYPES:
        results["probe_results"][pos_type] = {}
        for layer in range(n_layers):
            pr = probe_results[pos_type][layer]
            results["probe_results"][pos_type][str(layer)] = {
                "accuracy_mean": pr["accuracy_mean"],
                "accuracy_std": pr["accuracy_std"],
                "cv_scores": pr["cv_scores"],
            }

    # Save JSON
    json_path = Path(output_dir) / "localization_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # Save CSV: layer, position, accuracy_mean, accuracy_std
    csv_path = Path(output_dir) / "probe_accuracy.csv"
    with open(csv_path, "w") as f:
        f.write("layer,position,accuracy_mean,accuracy_std\n")
        for pos_type in POSITION_TYPES:
            for layer in range(n_layers):
                pr = probe_results[pos_type][layer]
                f.write(f"{layer},{pos_type},{pr['accuracy_mean']:.6f},{pr['accuracy_std']:.6f}\n")
    logger.info(f"Probe CSV saved to {csv_path}")

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Optimal intervention point: layer={best_layer}, position={best_pos}")
    logger.info(f"Probe accuracy at optimal: {best_acc:.4f}")
    for pos_type in POSITION_TYPES:
        bp = best_per_pos[pos_type]
        logger.info(f"  {pos_type}: best layer={bp['layer']}, accuracy={bp['accuracy']:.4f}")

    # Print full heatmap
    logger.info("\nProbe accuracy heatmap (layer × position):")
    header = f"{'Layer':>6s}"
    for pos_type in POSITION_TYPES:
        header += f"  {pos_type[:12]:>12s}"
    logger.info(header)
    for layer in range(n_layers):
        row = f"{layer:6d}"
        for pos_type in POSITION_TYPES:
            acc = probe_results[pos_type][layer]["accuracy_mean"]
            row += f"  {acc:12.4f}"
        logger.info(row)

    logger.info(f"\nPhase 2 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--n-demos", type=int, default=5)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--output-dir", default="results/phase2")
    args = parser.parse_args()

    run_localization(
        model_name=args.model,
        device=args.device,
        n_demos=args.n_demos,
        n_test=args.n_test,
        output_dir=args.output_dir,
    )
