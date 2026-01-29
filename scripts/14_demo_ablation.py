#!/usr/bin/env python3
"""Experiment 14: Minimal Demonstration Ablation.

Test whether reducing demo count makes task representations more localized
(less redundancy), potentially enabling single-position intervention to work.
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
from src.probing import train_probe

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
            logging.FileHandler(os.path.join(output_dir, "exp14.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def run_demo_ablation(device="cuda:2", n_test=20, output_dir="results/exp14"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 14: Minimal Demonstration Ablation")
    logger.info(f"Start: {datetime.now().isoformat()}")

    model = load_model(device=device)
    tasks = {name: TaskRegistry.get(name) for name in INCLUDED_TASKS}

    demo_counts = [1, 2, 3, 4, 5]
    test_layer = 14  # Same as Phase 3
    position_type = "last_demo_token"

    all_results = []

    # Part A: Task accuracy vs demo count
    logger.info("=" * 60)
    logger.info("Part A: Task Accuracy vs Demo Count")
    logger.info("=" * 60)

    accuracy_by_demos = {k: {} for k in demo_counts}

    for task_name in INCLUDED_TASKS:
        task = tasks[task_name]
        test_inputs = task.generate_test_inputs(n_test)

        for n_demos in demo_counts:
            demos = task.generate_demos(n_demos)
            correct = 0

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
                    correct += 1

            acc = correct / n_test
            accuracy_by_demos[n_demos][task_name] = acc

        logger.info(f"  {task_name}: " + " ".join(
            f"{n}shot={accuracy_by_demos[n][task_name]:.2f}" for n in demo_counts
        ))

    # Part B: Transplant transfer rate vs demo count
    logger.info("\n" + "=" * 60)
    logger.info("Part B: Transplant Transfer Rate vs Demo Count")
    logger.info("=" * 60)

    # Use a few representative pairs
    test_pairs = [
        ("uppercase", "sentiment"),
        ("linear_2x", "first_letter"),
        ("pattern_completion", "repeat_word"),
    ]

    transfer_by_demos = {k: [] for k in demo_counts}

    for source_name, target_name in test_pairs:
        logger.info(f"\n  Pair: {source_name} → {target_name}")

        source_task = tasks[source_name]
        target_task = tasks[target_name]

        for n_demos in demo_counts:
            source_demos = source_task.generate_demos(n_demos)
            target_demos = target_task.generate_demos(n_demos)

            source_test = source_task.generate_test_inputs(n_test)
            target_test = target_task.generate_test_inputs(n_test)

            # Extract source vectors
            source_vecs = []
            for ti in source_test:
                prompt = source_task.format_prompt(source_demos, ti)
                pos_idx = get_position_index(model, prompt, position_type)
                acts = extract_activations(model, prompt, layers=[test_layer], position=pos_idx)
                source_vecs.append(acts[test_layer])

            mean_source = torch.stack(source_vecs).mean(dim=0)

            # Transplant and measure transfer
            transfer_count = 0
            for ti in target_test:
                prompt = target_task.format_prompt(target_demos, ti)
                pos_idx = get_position_index(model, prompt, position_type)

                output = model.generate_with_intervention(
                    prompt, test_layer, pos_idx, mean_source, max_new_tokens=30
                )

                try:
                    if source_task.score_output(ti, output) == "correct":
                        transfer_count += 1
                except:
                    pass

            transfer_rate = transfer_count / n_test
            transfer_by_demos[n_demos].append(transfer_rate)

        logger.info("    " + " ".join(
            f"{n}shot={np.mean([t for t in transfer_by_demos[n][-1:]]):.2f}"
            for n in demo_counts
        ))

    # Part C: Probe accuracy vs demo count
    logger.info("\n" + "=" * 60)
    logger.info("Part C: Probe Accuracy at Query Position vs Demo Count")
    logger.info("=" * 60)

    probe_by_demos = {}
    query_layer = 12  # Peak from Phase 2

    for n_demos in demo_counts:
        all_acts = []
        all_labels = []

        for task_name in INCLUDED_TASKS:
            task = tasks[task_name]
            demos = task.generate_demos(n_demos)
            test_inputs = task.generate_test_inputs(n_test)

            for ti in test_inputs:
                prompt = task.format_prompt(demos, ti)
                pos_idx = get_position_index(model, prompt, "first_query_token")
                acts = extract_activations(model, prompt, layers=[query_layer], position=pos_idx)
                all_acts.append(acts[query_layer])
                all_labels.append(task_name)

        # Train probe
        acts_array = torch.stack(all_acts).float().numpy()
        probe_result = train_probe(acts_array, all_labels)
        probe_acc = probe_result["accuracy_mean"]
        probe_by_demos[n_demos] = probe_acc

        logger.info(f"  {n_demos}-shot: probe accuracy = {probe_acc:.3f}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info("\nMean task accuracy by demo count:")
    for n in demo_counts:
        mean_acc = np.mean(list(accuracy_by_demos[n].values()))
        logger.info(f"  {n}-shot: {mean_acc:.3f}")

    logger.info("\nMean transfer rate by demo count:")
    for n in demo_counts:
        mean_transfer = np.mean(transfer_by_demos[n]) if transfer_by_demos[n] else 0
        logger.info(f"  {n}-shot: {mean_transfer:.3f}")

    logger.info("\nProbe accuracy (query position, layer 12) by demo count:")
    for n in demo_counts:
        logger.info(f"  {n}-shot: {probe_by_demos[n]:.3f}")

    # Save
    results = {
        "metadata": {
            "demo_counts": demo_counts,
            "test_layer": test_layer,
            "query_layer": query_layer,
            "position_type": position_type,
            "n_test": n_test,
            "test_pairs": test_pairs,
            "timestamp": datetime.now().isoformat(),
        },
        "accuracy_by_demos": accuracy_by_demos,
        "transfer_by_demos": {k: v for k, v in transfer_by_demos.items()},
        "probe_by_demos": probe_by_demos,
    }

    with open(Path(output_dir) / "demo_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(Path(output_dir) / "demo_ablation.csv", "w") as f:
        f.write("n_demos,mean_accuracy,mean_transfer,probe_accuracy\n")
        for n in demo_counts:
            mean_acc = np.mean(list(accuracy_by_demos[n].values()))
            mean_t = np.mean(transfer_by_demos[n]) if transfer_by_demos[n] else 0
            f.write(f"{n},{mean_acc:.4f},{mean_t:.4f},{probe_by_demos[n]:.4f}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 14 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--output-dir", default="results/exp14")
    args = parser.parse_args()
    run_demo_ablation(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
