#!/usr/bin/env python3
"""Phase 4: Compositionality Analysis (Interpolation + Arithmetic).

Test linear vs. discrete structure of task representations.
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
from src.extraction import get_position_index
from src.intervention import transplant_and_generate
from src.interpolation import interpolate_vectors, compute_task_difference, apply_task_shift, measure_transition_sharpness

INCLUDED_TASKS = [
    "uppercase", "first_letter", "repeat_word", "length",
    "linear_2x", "sentiment", "antonym", "pattern_completion",
]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Test a subset of task pairs (representative from different regimes)
INTERPOLATION_PAIRS = [
    ("uppercase", "first_letter"),      # within procedural
    ("uppercase", "sentiment"),         # procedural vs bayesian
    ("linear_2x", "length"),            # gd_like vs counting
    ("sentiment", "antonym"),           # bayesian vs retrieval
    ("pattern_completion", "uppercase"),# induction vs procedural
    ("linear_2x", "sentiment"),        # gd_like vs bayesian
]


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(output_dir, "phase4.log")), logging.StreamHandler()])
    return logging.getLogger(__name__)


def run_interpolation(device="cuda:3", n_demos=5, n_test=10, output_dir="results/phase4"):
    logger = setup_logging(output_dir)
    logger.info("Phase 4: Compositionality Analysis")
    logger.info(f"Start: {datetime.now().isoformat()}")

    # Load Phase 2 optimal coordinates
    with open("results/phase2/localization_results.json") as f:
        p2 = json.load(f)
    opt_layer = p2["optimal"]["layer"]
    opt_pos_type = p2["optimal"]["position"]
    logger.info(f"Optimal: layer={opt_layer}, position={opt_pos_type}")

    # Load Phase 3 task vectors
    with open("results/phase3/task_vectors.pkl", "rb") as f:
        task_vectors = pickle.load(f)
    logger.info(f"Loaded task vectors for {list(task_vectors.keys())}")

    model = load_model(device=device)
    tasks = {name: TaskRegistry.get(name) for name in INCLUDED_TASKS}

    # ── Part A: Linear Interpolation ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("Part A: Linear Interpolation")
    logger.info("=" * 60)

    interpolation_results = []

    for task_a_name, task_b_name in INTERPOLATION_PAIRS:
        logger.info(f"\n--- {task_a_name} ↔ {task_b_name} ---")
        v_a = task_vectors[task_a_name]
        v_b = task_vectors[task_b_name]
        interpolated = interpolate_vectors(v_a, v_b, ALPHAS)

        task_a = tasks[task_a_name]
        task_b = tasks[task_b_name]

        # Use task_a's test inputs as a neutral test set
        demos_a = task_a.generate_demos(n_demos)
        test_inputs = task_a.generate_test_inputs(n_test)

        pair_results = {"task_a": task_a_name, "task_b": task_b_name, "alphas": ALPHAS, "alpha_results": []}

        task_a_probs = []
        task_b_probs = []

        for alpha, v_interp in zip(ALPHAS, interpolated):
            a_count = 0
            b_count = 0
            neither_count = 0
            details = []

            for ti in test_inputs:
                prompt = task_a.format_prompt(demos_a, ti)
                pos_idx = get_position_index(model, prompt, opt_pos_type)
                output = transplant_and_generate(model, prompt, v_interp, opt_layer, pos_idx)

                try:
                    a_correct = task_a.score_output(ti, output) == "correct"
                except (ValueError, TypeError, KeyError):
                    a_correct = False
                try:
                    b_correct = task_b.score_output(ti, output) == "correct"
                except (ValueError, TypeError, KeyError):
                    b_correct = False

                if a_correct:
                    a_count += 1
                elif b_correct:
                    b_count += 1
                else:
                    neither_count += 1

                details.append({"input": ti, "output": output,
                               "task_a_correct": a_correct, "task_b_correct": b_correct})

            a_prob = a_count / n_test
            b_prob = b_count / n_test
            task_a_probs.append(a_prob)
            task_b_probs.append(b_prob)

            pair_results["alpha_results"].append({
                "alpha": alpha, "task_a_prob": a_prob, "task_b_prob": b_prob,
                "neither_prob": neither_count / n_test, "details": details,
            })
            logger.info(f"  alpha={alpha:.1f}: {task_a_name}={a_prob:.2f} "
                       f"{task_b_name}={b_prob:.2f} neither={neither_count/n_test:.2f}")

        # Measure transition sharpness
        sharpness = measure_transition_sharpness(task_a_probs, task_b_probs, ALPHAS)
        pair_results["sharpness"] = sharpness
        logger.info(f"  Sharpness: slope={sharpness['midpoint_slope']:.3f} "
                   f"width={sharpness['transition_width']:.3f} "
                   f"smooth={sharpness['is_smooth']}")

        interpolation_results.append(pair_results)

    # ── Part B: Vector Arithmetic ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Part B: Vector Arithmetic")
    logger.info("=" * 60)

    arithmetic_results = []
    # Test: v_C + (v_A - v_B) should shift C toward A behavior
    arithmetic_triples = [
        ("uppercase", "first_letter", "repeat_word"),  # all procedural
        ("sentiment", "antonym", "linear_2x"),         # cross-regime
        ("length", "linear_2x", "sentiment"),          # numeric → semantic
    ]

    for a_name, b_name, c_name in arithmetic_triples:
        logger.info(f"\n--- Arithmetic: {c_name} + ({a_name} - {b_name}) ---")
        delta = compute_task_difference(task_vectors[a_name], task_vectors[b_name])
        v_shifted = apply_task_shift(task_vectors[c_name], delta)

        task_a = tasks[a_name]
        task_c = tasks[c_name]
        demos_c = task_c.generate_demos(n_demos)
        test_inputs = task_c.generate_test_inputs(n_test)

        a_count = c_count = neither = 0
        details = []

        for ti in test_inputs:
            prompt = task_c.format_prompt(demos_c, ti)
            pos_idx = get_position_index(model, prompt, opt_pos_type)
            output = transplant_and_generate(model, prompt, v_shifted, opt_layer, pos_idx)

            try:
                a_correct = task_a.score_output(ti, output) == "correct"
            except (ValueError, TypeError, KeyError):
                a_correct = False
            try:
                c_correct = task_c.score_output(ti, output) == "correct"
            except (ValueError, TypeError, KeyError):
                c_correct = False

            if a_correct:
                a_count += 1
            elif c_correct:
                c_count += 1
            else:
                neither += 1
            details.append({"input": ti, "output": output,
                           "task_a_correct": a_correct, "task_c_correct": c_correct})

        result = {
            "a": a_name, "b": b_name, "c": c_name,
            "operation": f"{c_name} + ({a_name} - {b_name})",
            "task_a_rate": a_count / n_test,
            "task_c_rate": c_count / n_test,
            "neither_rate": neither / n_test,
            "details": details,
        }
        arithmetic_results.append(result)
        logger.info(f"  Result: {a_name}={a_count/n_test:.2f} "
                   f"{c_name}={c_count/n_test:.2f} neither={neither/n_test:.2f}")

    # ── Save ─────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "optimal_layer": opt_layer, "optimal_position": opt_pos_type,
            "alphas": ALPHAS, "n_test": n_test,
            "timestamp": datetime.now().isoformat(),
        },
        "interpolation": interpolation_results,
        "arithmetic": arithmetic_results,
    }

    with open(Path(output_dir) / "interpolation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV: interpolation curves
    with open(Path(output_dir) / "interpolation_curves.csv", "w") as f:
        f.write("pair,alpha,task_a_prob,task_b_prob,neither_prob\n")
        for pr in interpolation_results:
            pair = f"{pr['task_a']}_{pr['task_b']}"
            for ar in pr["alpha_results"]:
                f.write(f"{pair},{ar['alpha']},{ar['task_a_prob']},{ar['task_b_prob']},{ar['neither_prob']}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Phase 4 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--output-dir", default="results/phase4")
    args = parser.parse_args()
    run_interpolation(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
