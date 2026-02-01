#!/usr/bin/env python3
"""Experiment 13: Instance-Level Analysis.

Analyze which specific test instances transfer successfully and why.
Look for patterns in transferable vs non-transferable instances.
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

INCLUDED_TASKS = [
    "uppercase", "first_letter", "repeat_word", "length",
    "linear_2x", "sentiment", "antonym", "pattern_completion",
]

TEST_PAIRS = [
    ("uppercase", "first_letter"),
    ("uppercase", "sentiment"),
    ("repeat_word", "first_letter"),
    ("pattern_completion", "repeat_word"),
]


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "exp13.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def find_demo_output_positions(tokenizer, prompt, n_demos=5):
    """Find positions of demo output tokens."""
    lines = prompt.strip().split("\n")
    positions = []
    current_text = ""

    for i, line in enumerate(lines):
        if line.startswith("Output:") and i < len(lines) - 1:  # Not the final Output:
            # Find where this output value ends
            output_value = line.replace("Output:", "").strip()
            current_text += line + "\n"
            tokens_so_far = tokenizer.encode(current_text, add_special_tokens=False)
            # The output tokens are at the end
            output_tokens = tokenizer.encode(" " + output_value, add_special_tokens=False)
            end_pos = len(tokens_so_far)
            start_pos = end_pos - len(output_tokens)
            positions.extend(range(max(0, start_pos), end_pos))
        else:
            current_text += line + "\n"

    return positions


def generate_with_multi_intervention(model, prompt, layer, position_vectors, max_new_tokens=30):
    """Generate with intervention at multiple positions."""
    tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    intervention_done = [False]

    def intervention_hook(module, input, output):
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
                hidden[0, pos, :] = vec.to(hidden.device)

        intervention_done[0] = True

        if rest is not None:
            return (hidden,) + rest
        return hidden

    layer_module = model.get_layer_module(layer)
    handle = layer_module.register_forward_hook(intervention_hook)

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


def extract_position_vectors(model, prompt, layer, positions):
    """Extract activation vectors at specified positions."""
    tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    vectors = {}

    def extract_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        for pos in positions:
            if pos < hidden.shape[1]:
                vectors[pos] = hidden[0, pos, :].clone()

    layer_module = model.get_layer_module(layer)
    handle = layer_module.register_forward_hook(extract_hook)

    with torch.no_grad():
        model.model(tokens)

    handle.remove()
    return vectors


def run_instance_analysis(model_name="meta-llama/Llama-3.2-3B-Instruct", device="cuda:2", n_demos=5, n_test=20, output_dir="results/exp13"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 13: Instance-Level Analysis")
    logger.info(f"Start: {datetime.now().isoformat()}")

    model = load_model(model_name, device=device)
    tasks = {name: TaskRegistry.get(name) for name in INCLUDED_TASKS}

    # Dynamic layer based on model depth (~30% depth, matching exp8 findings)
    intervention_layer = model.layer_at_fraction(0.30)
    logger.info(f"Model: {model_name} ({model.n_layers} layers), intervention_layer={intervention_layer}")

    # ═══════════════════════════════════════════════════════════════════
    # Part A: Instance-level transfer analysis
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Part A: Instance-Level Transfer Analysis")
    logger.info("=" * 60)

    instance_results = []

    for source_name, target_name in TEST_PAIRS:
        logger.info(f"\nTransfer: {source_name} → {target_name}")

        source_task = tasks[source_name]
        target_task = tasks[target_name]

        # Generate demos and test inputs
        source_demos = source_task.generate_demos(n_demos)
        target_demos = target_task.generate_demos(n_demos)
        test_inputs = target_task.generate_test_inputs(n_test)

        # Build source prompt for vector extraction
        source_prompt = source_task.format_prompt(source_demos, test_inputs[0])
        source_positions = find_demo_output_positions(model.tokenizer, source_prompt, n_demos)

        # Extract source vectors
        source_vectors = extract_position_vectors(model, source_prompt, intervention_layer, source_positions)

        pair_results = {
            "source": source_name,
            "target": target_name,
            "instances": [],
        }

        transferred = 0
        for ti in test_inputs:
            target_prompt = target_task.format_prompt(target_demos, ti)

            # Baseline output
            tokens = model.tokenizer.encode(target_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.model.generate(
                    tokens, max_new_tokens=30, do_sample=False,
                    pad_token_id=model.tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0, tokens.shape[1]:]
            baseline_output = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            baseline_output = baseline_output.strip().split("\n")[0].strip()

            # Intervened output
            intervened_output = generate_with_multi_intervention(
                model, target_prompt, intervention_layer, source_vectors
            )

            # Check if transfer happened
            target_correct = target_task.score_output(ti, baseline_output) == "correct"
            try:
                source_correct = source_task.score_output(ti, intervened_output) == "correct"
            except (ValueError, TypeError):
                # Input type mismatch between tasks
                source_correct = False

            instance = {
                "input": ti,
                "baseline_output": baseline_output,
                "intervened_output": intervened_output,
                "target_correct_baseline": target_correct,
                "source_correct_intervened": source_correct,
            }
            pair_results["instances"].append(instance)

            if source_correct:
                transferred += 1
                logger.info(f"  TRANSFER: '{ti}' → '{intervened_output}' (source task behavior)")

        transfer_rate = transferred / n_test
        pair_results["transfer_rate"] = transfer_rate
        instance_results.append(pair_results)

        logger.info(f"  Transfer rate: {transfer_rate:.2f} ({transferred}/{n_test})")

    # ═══════════════════════════════════════════════════════════════════
    # Part B: Input characteristics analysis
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Part B: Input Characteristics Analysis")
    logger.info("=" * 60)

    # Analyze what makes an instance transferable
    for pair_result in instance_results:
        source, target = pair_result["source"], pair_result["target"]
        logger.info(f"\n{source} → {target}:")

        transferred_inputs = []
        non_transferred_inputs = []

        for inst in pair_result["instances"]:
            if inst["source_correct_intervened"]:
                transferred_inputs.append(inst["input"])
            else:
                non_transferred_inputs.append(inst["input"])

        # Analyze input lengths
        if transferred_inputs:
            avg_len_trans = np.mean([len(str(x)) for x in transferred_inputs])
            logger.info(f"  Transferred ({len(transferred_inputs)}): avg input len = {avg_len_trans:.1f}")
            logger.info(f"    Examples: {transferred_inputs[:3]}")

        if non_transferred_inputs:
            avg_len_non = np.mean([len(str(x)) for x in non_transferred_inputs])
            logger.info(f"  Non-transferred ({len(non_transferred_inputs)}): avg input len = {avg_len_non:.1f}")
            logger.info(f"    Examples: {non_transferred_inputs[:3]}")

    # ═══════════════════════════════════════════════════════════════════
    # Part C: Output format overlap analysis
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Part C: Output Format Analysis")
    logger.info("=" * 60)

    for pair_result in instance_results:
        source, target = pair_result["source"], pair_result["target"]
        logger.info(f"\n{source} → {target}:")

        # Count output format types
        output_types = {"numeric": 0, "word": 0, "phrase": 0, "other": 0}

        for inst in pair_result["instances"]:
            out = inst["intervened_output"]
            if out.isdigit():
                output_types["numeric"] += 1
            elif out.isalpha() and len(out.split()) == 1:
                output_types["word"] += 1
            elif len(out.split()) > 1:
                output_types["phrase"] += 1
            else:
                output_types["other"] += 1

        logger.info(f"  Output types: {output_types}")

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for pair_result in instance_results:
        logger.info(f"  {pair_result['source']} → {pair_result['target']}: "
                   f"{pair_result['transfer_rate']:.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════════
    results = {
        "metadata": {
            "intervention_layer": intervention_layer,
            "n_demos": n_demos,
            "n_test": n_test,
            "timestamp": datetime.now().isoformat(),
        },
        "instance_results": instance_results,
    }

    with open(Path(output_dir) / "instance_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Write CSV of all instances
    with open(Path(output_dir) / "instances.csv", "w") as f:
        f.write("source,target,input,baseline_output,intervened_output,transferred\n")
        for pair in instance_results:
            for inst in pair["instances"]:
                transferred = "yes" if inst["source_correct_intervened"] else "no"
                f.write(f"{pair['source']},{pair['target']},{inst['input']},"
                       f"{inst['baseline_output']},{inst['intervened_output']},{transferred}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 13 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--output-dir", default="results/exp13")
    args = parser.parse_args()
    run_instance_analysis(model_name=args.model, device=args.device, n_test=args.n_test, output_dir=args.output_dir)
