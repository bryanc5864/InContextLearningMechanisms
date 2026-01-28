#!/usr/bin/env python3
"""Phase 7: Trajectory Analysis — how representations evolve across layers.

Track probe confidence, representational change, and crystallization.
"""

import json
import sys
import os
import logging
import pickle
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from src.model import load_model
from src.tasks import TaskRegistry
from src.extraction import extract_activations, get_position_index
from src.trajectory import (
    compute_probe_trajectory, compute_representational_change,
    find_crystallization_layer,
)

INCLUDED_TASKS = [
    "uppercase", "first_letter", "repeat_word", "length",
    "linear_2x", "sentiment", "antonym", "pattern_completion",
]

TASK_REGIMES = {
    "uppercase": "procedural", "first_letter": "procedural", "repeat_word": "procedural",
    "length": "counting", "linear_2x": "gd_like",
    "sentiment": "bayesian", "antonym": "retrieval",
    "pattern_completion": "induction",
}


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(output_dir, "phase7.log")), logging.StreamHandler()])
    return logging.getLogger(__name__)


def run_trajectory(device="cuda:3", n_demos=5, n_test=20, output_dir="results/phase7"):
    logger = setup_logging(output_dir)
    logger.info("Phase 7: Trajectory Analysis")
    logger.info(f"Start: {datetime.now().isoformat()}")

    with open("results/phase2/localization_results.json") as f:
        p2 = json.load(f)
    opt_pos_type = p2["optimal"]["position"]
    n_layers = p2["metadata"]["n_layers"]
    logger.info(f"Position type: {opt_pos_type}, n_layers={n_layers}")

    model = load_model(device=device)

    # ── Step 1: Extract activations at all layers ─────────────────────
    logger.info("Step 1: Extracting activations at all layers")
    # layer_activations[layer] = list of tensors (one per sample)
    layer_activations = {l: [] for l in range(n_layers)}
    labels = []

    for task_name in INCLUDED_TASKS:
        task = TaskRegistry.get(task_name)
        demos = task.generate_demos(n_demos)
        test_inputs = task.generate_test_inputs(n_test)
        logger.info(f"  {task_name}: {len(test_inputs)} inputs")

        for ti in test_inputs:
            prompt = task.format_prompt(demos, ti)
            pos_idx = get_position_index(model, prompt, opt_pos_type)
            acts = extract_activations(model, prompt, position=pos_idx)
            for l in range(n_layers):
                layer_activations[l].append(acts[l])
            labels.append(task_name)

    logger.info(f"Extracted: {len(labels)} samples × {n_layers} layers")

    # ── Step 2: Probe trajectory ──────────────────────────────────────
    logger.info("\nStep 2: Probe confidence trajectory")
    trajectory = compute_probe_trajectory(layer_activations, labels, C=1.0)

    for layer in sorted(trajectory.keys()):
        logger.info(f"  Layer {layer:2d}: probe accuracy = {trajectory[layer]:.4f}")

    # ── Step 3: Representational change ───────────────────────────────
    logger.info("\nStep 3: Layer-to-layer representational change")
    rep_change = compute_representational_change(layer_activations)

    for (l_i, l_j), dist in rep_change.items():
        logger.info(f"  Layer {l_i:2d} → {l_j:2d}: cosine dist = {dist:.6f}")

    # ── Step 4: Crystallization layer ─────────────────────────────────
    logger.info("\nStep 4: Crystallization layers")
    for threshold in [0.8, 0.9, 0.95]:
        crystal = find_crystallization_layer(trajectory, threshold)
        logger.info(f"  Threshold {threshold:.0%}: crystallization at layer {crystal}")

    # ── Step 5: Per-regime trajectories ───────────────────────────────
    logger.info("\nStep 5: Per-regime probe trajectories")
    regime_trajectories = {}

    unique_regimes = sorted(set(TASK_REGIMES[t] for t in INCLUDED_TASKS))
    for regime in unique_regimes:
        regime_tasks = [t for t in INCLUDED_TASKS if TASK_REGIMES[t] == regime]
        # Filter to samples from this regime
        regime_indices = [i for i, l in enumerate(labels) if TASK_REGIMES.get(l) == regime]

        if len(set(labels[i] for i in regime_indices)) < 2:
            logger.info(f"  {regime}: only 1 task, skipping per-regime probe")
            regime_trajectories[regime] = None
            continue

        regime_layer_acts = {l: [layer_activations[l][i] for i in regime_indices] for l in range(n_layers)}
        regime_labels = [labels[i] for i in regime_indices]

        regime_traj = compute_probe_trajectory(regime_layer_acts, regime_labels, C=1.0)
        regime_trajectories[regime] = regime_traj
        crystal = find_crystallization_layer(regime_traj, 0.9)
        logger.info(f"  {regime:12s}: crystallization@90% = layer {crystal}, "
                   f"final accuracy = {regime_traj[n_layers-1]:.4f}")

    # ── Save ─────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "tasks": INCLUDED_TASKS, "regimes": TASK_REGIMES,
            "position_type": opt_pos_type, "n_layers": n_layers,
            "n_test_per_task": n_test, "timestamp": datetime.now().isoformat(),
        },
        "probe_trajectory": {str(k): v for k, v in trajectory.items()},
        "representational_change": {f"{k[0]}_{k[1]}": v for k, v in rep_change.items()},
        "crystallization": {
            "80": find_crystallization_layer(trajectory, 0.8),
            "90": find_crystallization_layer(trajectory, 0.9),
            "95": find_crystallization_layer(trajectory, 0.95),
        },
        "regime_trajectories": {
            regime: ({str(k): v for k, v in traj.items()} if traj else None)
            for regime, traj in regime_trajectories.items()
        },
    }

    with open(Path(output_dir) / "trajectory_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV: trajectory
    with open(Path(output_dir) / "probe_trajectory.csv", "w") as f:
        f.write("layer,overall_accuracy")
        for regime in unique_regimes:
            f.write(f",{regime}_accuracy")
        f.write("\n")
        for l in range(n_layers):
            f.write(f"{l},{trajectory[l]:.6f}")
            for regime in unique_regimes:
                rt = regime_trajectories.get(regime)
                val = rt[l] if rt else ""
                f.write(f",{val:.6f}" if isinstance(val, float) else f",")
            f.write("\n")

    # CSV: representational change
    with open(Path(output_dir) / "representational_change.csv", "w") as f:
        f.write("layer_from,layer_to,cosine_distance\n")
        for (l_i, l_j), dist in rep_change.items():
            f.write(f"{l_i},{l_j},{dist:.6f}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Phase 7 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--output-dir", default="results/phase7")
    args = parser.parse_args()
    run_trajectory(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
