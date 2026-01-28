#!/usr/bin/env python3
"""Phase 3: Transplantation Experiments (Modularity Test).

For each task pair (A→B), transplant cached activation from A into B's context.
Measure transfer rate, disruption rate, preservation rate.
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

import torch
import numpy as np
from src.model import load_model, get_model_info
from src.tasks import TaskRegistry
from src.extraction import extract_activations, get_position_index
from src.intervention import (
    transplant_and_generate, baseline_generate,
    zero_ablation_generate, random_ablation_generate,
)

INCLUDED_TASKS = [
    "uppercase", "first_letter", "repeat_word", "length",
    "linear_2x", "sentiment", "antonym", "pattern_completion",
]


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(output_dir, "phase3.log")), logging.StreamHandler()])
    return logging.getLogger(__name__)


def classify_output(output: str, source_task, target_task, test_input: str) -> str:
    """Classify transplant output as: source_task / target_task / neither / malformed."""
    if not output or not output.strip():
        return "malformed"
    # Source task may not be able to score target task's inputs (e.g., linear_2x on words)
    try:
        source_score = source_task.score_output(test_input, output)
    except (ValueError, TypeError, KeyError):
        source_score = "incompatible"
    try:
        target_score = target_task.score_output(test_input, output)
    except (ValueError, TypeError, KeyError):
        target_score = "incompatible"
    if source_score == "correct":
        return "source_task"
    if target_score == "correct":
        return "target_task"
    if source_score in ("malformed", "incompatible") and target_score in ("malformed", "incompatible"):
        return "malformed"
    return "neither"


def run_transplantation(
    device: str = "cuda:3",
    n_demos: int = 5,
    n_test: int = 10,  # fewer per pair since O(n^2) pairs
    output_dir: str = "results/phase3",
):
    logger = setup_logging(output_dir)
    logger.info("Phase 3: Transplantation Experiments")
    logger.info(f"Start: {datetime.now().isoformat()}")

    # Load Phase 2 results to get optimal (L*, p*)
    p2_path = Path("results/phase2/localization_results.json")
    with open(p2_path) as f:
        p2 = json.load(f)
    opt_layer = p2["optimal"]["layer"]
    opt_pos_type = p2["optimal"]["position"]
    logger.info(f"Using optimal: layer={opt_layer}, position={opt_pos_type} "
               f"(probe accuracy={p2['optimal']['accuracy']:.4f})")

    model = load_model(device=device)

    tasks = {name: TaskRegistry.get(name) for name in INCLUDED_TASKS}
    n_tasks = len(INCLUDED_TASKS)

    # ── Step 1: Cache source activations ──────────────────────────────
    logger.info("Step 1: Caching source activations for all tasks")
    # For each task, cache mean activation vector across test inputs at (L*, p*)
    task_vectors = {}
    task_vector_norms = {}

    for task_name in INCLUDED_TASKS:
        task = tasks[task_name]
        demos = task.generate_demos(n_demos)
        test_inputs = task.generate_test_inputs(n_test)

        vecs = []
        for ti in test_inputs:
            prompt = task.format_prompt(demos, ti)
            pos_idx = get_position_index(model, prompt, opt_pos_type)
            acts = extract_activations(model, prompt, layers=[opt_layer], position=pos_idx)
            vecs.append(acts[opt_layer])

        mean_vec = torch.stack(vecs).mean(dim=0)
        task_vectors[task_name] = mean_vec
        task_vector_norms[task_name] = float(mean_vec.norm())
        logger.info(f"  {task_name}: mean vector norm = {mean_vec.norm():.2f}")

    # Save task vectors
    vec_path = Path(output_dir) / "task_vectors.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump(task_vectors, f)
    logger.info(f"Task vectors saved to {vec_path}")

    # ── Step 2: Cross-task transplantation ────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Cross-task transplantation")
    logger.info("=" * 60)

    transfer_matrix = np.zeros((n_tasks, n_tasks))  # source x target
    disruption_matrix = np.zeros((n_tasks, n_tasks))
    preservation_matrix = np.zeros((n_tasks, n_tasks))

    all_transplant_results = []

    for si, source_name in enumerate(INCLUDED_TASKS):
        for ti, target_name in enumerate(INCLUDED_TASKS):
            target_task = tasks[target_name]
            source_task = tasks[source_name]
            demos = target_task.generate_demos(n_demos)
            test_inputs = target_task.generate_test_inputs(n_test)

            source_vec = task_vectors[source_name]
            counts = {"source_task": 0, "target_task": 0, "neither": 0, "malformed": 0}
            details = []

            for test_input in test_inputs:
                prompt = target_task.format_prompt(demos, test_input)
                pos_idx = get_position_index(model, prompt, opt_pos_type)

                output = transplant_and_generate(
                    model, prompt, source_vec, opt_layer, pos_idx, max_new_tokens=30
                )
                classification = classify_output(output, source_task, target_task, test_input)
                counts[classification] += 1
                details.append({
                    "input": test_input, "output": output,
                    "classification": classification,
                })

            transfer_rate = counts["source_task"] / n_test
            disruption_rate = (counts["neither"] + counts["malformed"]) / n_test
            preservation_rate = counts["target_task"] / n_test

            transfer_matrix[si, ti] = transfer_rate
            disruption_matrix[si, ti] = disruption_rate
            preservation_matrix[si, ti] = preservation_rate

            pair_result = {
                "source": source_name, "target": target_name,
                "transfer_rate": transfer_rate,
                "disruption_rate": disruption_rate,
                "preservation_rate": preservation_rate,
                "counts": counts,
                "details": details,
            }
            all_transplant_results.append(pair_result)

            marker = "***" if transfer_rate > 0.5 and source_name != target_name else ""
            logger.info(f"  {source_name:20s} → {target_name:20s}: "
                       f"transfer={transfer_rate:.2f} disrupt={disruption_rate:.2f} "
                       f"preserve={preservation_rate:.2f} {marker}")

    # ── Step 3: Control conditions ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 3: Control conditions")
    logger.info("=" * 60)

    controls = {"baseline": {}, "zero_ablation": {}, "random_ablation": {}}
    mean_norm = float(np.mean(list(task_vector_norms.values())))

    for task_name in INCLUDED_TASKS:
        task = tasks[task_name]
        demos = task.generate_demos(n_demos)
        test_inputs = task.generate_test_inputs(n_test)

        baseline_correct = 0
        zero_correct = 0
        random_correct = 0

        for test_input in test_inputs:
            prompt = task.format_prompt(demos, test_input)
            pos_idx = get_position_index(model, prompt, opt_pos_type)

            # Baseline (no intervention)
            out_base = baseline_generate(model, prompt)
            if task.score_output(test_input, out_base) == "correct":
                baseline_correct += 1

            # Zero ablation
            out_zero = zero_ablation_generate(model, prompt, opt_layer, pos_idx)
            if task.score_output(test_input, out_zero) == "correct":
                zero_correct += 1

            # Random ablation
            out_rand = random_ablation_generate(
                model, prompt, opt_layer, pos_idx, norm=mean_norm
            )
            if task.score_output(test_input, out_rand) == "correct":
                random_correct += 1

        controls["baseline"][task_name] = baseline_correct / n_test
        controls["zero_ablation"][task_name] = zero_correct / n_test
        controls["random_ablation"][task_name] = random_correct / n_test
        logger.info(f"  {task_name:20s}: baseline={baseline_correct/n_test:.2f} "
                   f"zero={zero_correct/n_test:.2f} random={random_correct/n_test:.2f}")

    # ── Save everything ───────────────────────────────────────────────
    results = {
        "metadata": {
            "tasks": INCLUDED_TASKS,
            "optimal_layer": opt_layer,
            "optimal_position": opt_pos_type,
            "n_test_per_pair": n_test,
            "timestamp": datetime.now().isoformat(),
        },
        "transfer_matrix": transfer_matrix.tolist(),
        "disruption_matrix": disruption_matrix.tolist(),
        "preservation_matrix": preservation_matrix.tolist(),
        "controls": controls,
        "task_vector_norms": task_vector_norms,
        "transplant_details": all_transplant_results,
    }

    with open(Path(output_dir) / "transplant_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save transfer matrix as CSV
    with open(Path(output_dir) / "transfer_matrix.csv", "w") as f:
        f.write("source\\target," + ",".join(INCLUDED_TASKS) + "\n")
        for si, sn in enumerate(INCLUDED_TASKS):
            row = ",".join(f"{transfer_matrix[si,ti]:.3f}" for ti in range(n_tasks))
            f.write(f"{sn},{row}\n")

    logger.info(f"\nResults saved to {output_dir}")

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3 SUMMARY")
    logger.info("=" * 60)

    # Mean transfer rate (off-diagonal)
    mask = ~np.eye(n_tasks, dtype=bool)
    mean_transfer = transfer_matrix[mask].mean()
    mean_disruption = disruption_matrix[mask].mean()
    logger.info(f"Mean cross-task transfer rate: {mean_transfer:.3f}")
    logger.info(f"Mean disruption rate: {mean_disruption:.3f}")

    # Diagonal (same-task transplant = sanity check)
    diag_transfer = np.diag(transfer_matrix).mean()
    logger.info(f"Same-task transplant accuracy: {diag_transfer:.3f}")

    logger.info(f"\nPhase 3 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--output-dir", default="results/phase3")
    args = parser.parse_args()

    run_transplantation(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
