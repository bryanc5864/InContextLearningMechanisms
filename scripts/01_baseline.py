#!/usr/bin/env python3
"""Phase 1: Baseline characterization.

Verify model competence on all 8 tasks and establish performance baselines.
Logs all inputs, outputs, scores, and summary statistics.
"""

import json
import sys
import os
import logging
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model import load_model, get_model_info
from src.tasks import TaskRegistry
from src.intervention import baseline_generate


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "phase1.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def run_baseline(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    device: str = "cuda:3",
    n_demos: int = 5,
    n_test: int = 50,
    output_dir: str = "results/phase1",
):
    logger = setup_logging(output_dir)
    logger.info(f"Phase 1: Baseline Characterization")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Model: {model_name}, Device: {device}, Demos: {n_demos}, Tests: {n_test}")

    # Load model
    model = load_model(model_name, device=device)
    model_info = get_model_info(model)
    logger.info(f"Model info: {json.dumps(model_info)}")

    # Run all tasks
    all_results = {"metadata": {
        "model": model_name,
        "device": device,
        "n_demos": n_demos,
        "n_test": n_test,
        "start_time": datetime.now().isoformat(),
        "model_info": model_info,
    }, "tasks": {}}

    tasks = TaskRegistry.all_tasks()
    task_times = {}

    for task_name, task in tasks.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task_name} (regime: {task.regime})")
        logger.info(f"{'='*60}")

        task_start = time.time()
        demos = task.generate_demos(n_demos)
        test_inputs = task.generate_test_inputs(n_test)

        logger.info(f"Demos: {demos}")
        logger.info(f"Test inputs ({len(test_inputs)}): {test_inputs[:5]}...")

        results = {
            "task": task_name,
            "regime": task.regime,
            "description": task.description,
            "n_demos": n_demos,
            "n_test": len(test_inputs),
            "demos": [{"input": d[0], "output": d[1]} for d in demos],
            "outputs": [],
        }

        correct = 0
        incorrect = 0
        malformed = 0

        for i, test_input in enumerate(test_inputs):
            prompt = task.format_prompt(demos, test_input)
            output = baseline_generate(model, prompt, max_new_tokens=30)
            expected = task.compute_answer(test_input)
            score = task.score_output(test_input, output)

            results["outputs"].append({
                "input": test_input,
                "expected": expected,
                "output": output,
                "score": score,
            })

            if score == "correct":
                correct += 1
            elif score == "incorrect":
                incorrect += 1
            else:
                malformed += 1

            if (i + 1) % 10 == 0 or score != "correct":
                logger.info(f"  [{i+1:3d}/{len(test_inputs)}] "
                           f"input={test_input!r:15s} expected={expected!r:15s} "
                           f"got={output!r:15s} -> {score}")

        task_elapsed = time.time() - task_start
        task_times[task_name] = task_elapsed
        accuracy = correct / len(test_inputs)

        results["accuracy"] = accuracy
        results["correct"] = correct
        results["incorrect"] = incorrect
        results["malformed"] = malformed
        results["elapsed_seconds"] = round(task_elapsed, 1)

        threshold = 0.85
        results["passes_threshold"] = accuracy >= threshold

        logger.info(f"\nResult: {accuracy:.1%} accuracy "
                   f"({correct}/{len(test_inputs)}) "
                   f"[{incorrect} incorrect, {malformed} malformed] "
                   f"in {task_elapsed:.1f}s")
        if accuracy < threshold:
            logger.warning(f"  BELOW {threshold:.0%} THRESHOLD — may exclude from later phases")

        all_results["tasks"][task_name] = results

    # Summary
    all_results["metadata"]["end_time"] = datetime.now().isoformat()
    all_results["metadata"]["total_elapsed"] = round(sum(task_times.values()), 1)

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    passing = []
    failing = []
    for name, res in all_results["tasks"].items():
        status = "PASS" if res["passes_threshold"] else "FAIL"
        logger.info(f"  {name:25s}: {res['accuracy']:5.1%}  [{status}]  "
                   f"(regime: {res['regime']}, {res['elapsed_seconds']:.1f}s)")
        if res["passes_threshold"]:
            passing.append(name)
        else:
            failing.append(name)

    logger.info(f"\nPassing ({len(passing)}): {passing}")
    logger.info(f"Failing ({len(failing)}): {failing}")
    logger.info(f"Total time: {sum(task_times.values()):.1f}s")

    # Save results
    output_path = Path(output_dir) / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Save summary CSV
    csv_path = Path(output_dir) / "baseline_summary.csv"
    with open(csv_path, "w") as f:
        f.write("task,regime,accuracy,correct,incorrect,malformed,passes_threshold,elapsed_s\n")
        for name, res in all_results["tasks"].items():
            f.write(f"{name},{res['regime']},{res['accuracy']:.4f},"
                   f"{res['correct']},{res['incorrect']},{res['malformed']},"
                   f"{res['passes_threshold']},{res['elapsed_seconds']}\n")
    logger.info(f"Summary CSV saved to {csv_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--n-demos", type=int, default=5)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--output-dir", default="results/phase1")
    args = parser.parse_args()

    run_baseline(
        model_name=args.model,
        device=args.device,
        n_demos=args.n_demos,
        n_test=args.n_test,
        output_dir=args.output_dir,
    )
