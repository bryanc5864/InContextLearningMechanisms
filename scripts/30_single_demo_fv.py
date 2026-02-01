#!/usr/bin/env python3
"""Experiment 30: Single-Demo Function Vectors (Q1/W5).

Test whether multi-position transfer works with 1-shot source prompts.
Paper predicts it should fail/degrade because encoding is distributed across demos.

Design:
- Source demo counts: [1, 2, 3, 5]
- Target: always 5-shot
- Pairs: (uppercase→repeat_word), (uppercase→length), (repeat_word→length)
- Layer 8, all_demo condition, N=20
- Log number of positions transplanted at each demo count
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

SOURCE_DEMO_COUNTS = [1, 2, 3, 5]
TARGET_N_DEMOS = 5
INTERVENTION_LAYER = 8


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "exp30.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def find_demo_token_positions(tokenizer, prompt: str, n_demos: int = 5):
    """Find token positions for each demo (input and output portions).

    Returns dict with 'all_demo' positions covering all demo lines.
    """
    lines = prompt.strip().split("\n")
    positions = {
        'all_demo': [],
        'input_only': [],
        'output_only': [],
        'last_demo': [],
    }

    current_pos = 0
    demo_count = 0
    last_demo_start = None

    for i, line in enumerate(lines):
        line_tokens = tokenizer.encode(line, add_special_tokens=False)
        line_len = len(line_tokens)

        # Last two lines are the query (Input: X\nOutput:)
        if i >= len(lines) - 2:
            break

        if line.startswith("Input:"):
            for p in range(current_pos, current_pos + line_len):
                positions['all_demo'].append(p)
                positions['input_only'].append(p)
            if demo_count == n_demos - 1:
                last_demo_start = current_pos
                for p in range(current_pos, current_pos + line_len):
                    positions['last_demo'].append(p)

        elif line.startswith("Output:"):
            for p in range(current_pos, current_pos + line_len):
                positions['all_demo'].append(p)
                positions['output_only'].append(p)
            if last_demo_start is not None:
                for p in range(current_pos, current_pos + line_len):
                    positions['last_demo'].append(p)
            demo_count += 1

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


def run_single_demo_fv(device="cuda:1", n_test=20, output_dir="results/exp30"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 30: Single-Demo Function Vectors")
    logger.info(f"Start: {datetime.now().isoformat()}")
    logger.info(f"Source demo counts: {SOURCE_DEMO_COUNTS}")
    logger.info(f"Target always {TARGET_N_DEMOS}-shot")

    model = load_model(device=device)

    tasks = {
        "uppercase": TaskRegistry.get("uppercase"),
        "repeat_word": TaskRegistry.get("repeat_word"),
        "length": TaskRegistry.get("length"),
    }

    all_results = []

    for source_name, target_name in TEST_PAIRS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pair: {source_name} → {target_name}")
        logger.info(f"{'='*60}")

        source_task = tasks[source_name]
        target_task = tasks[target_name]

        # Fixed target demos (always 5-shot)
        target_demos = target_task.generate_demos(TARGET_N_DEMOS)
        test_inputs = target_task.generate_test_inputs(n_test)

        pair_results = {
            "source": source_name,
            "target": target_name,
            "demo_counts": {},
        }

        for n_source_demos in SOURCE_DEMO_COUNTS:
            logger.info(f"\n  Source demos: {n_source_demos}")

            source_demos = source_task.generate_demos(n_source_demos)

            # Extract source activations averaged over test inputs
            source_position_vectors = {}
            n_positions_list = []

            for s_input in test_inputs:
                prompt = source_task.format_prompt(source_demos, s_input)
                positions = find_demo_token_positions(model.tokenizer, prompt, n_source_demos)
                pos_list = positions['all_demo']
                n_positions_list.append(len(pos_list))

                acts = extract_multi_position_activations(model, prompt, INTERVENTION_LAYER, pos_list)

                for pos, vec in acts.items():
                    if pos not in source_position_vectors:
                        source_position_vectors[pos] = []
                    source_position_vectors[pos].append(vec)

            # Average across source examples
            mean_position_vectors = {
                pos: torch.stack(vecs).mean(dim=0)
                for pos, vecs in source_position_vectors.items()
            }

            avg_n_positions = np.mean(n_positions_list)
            logger.info(f"    Positions transplanted: {len(mean_position_vectors)} (avg tokens per prompt: {avg_n_positions:.1f})")

            # Transplant into target contexts
            transfer_count = 0
            preserve_count = 0
            neither_count = 0
            examples = []

            for t_input in test_inputs:
                target_prompt = target_task.format_prompt(target_demos, t_input)

                output = generate_with_multi_intervention(
                    model, target_prompt, INTERVENTION_LAYER, mean_position_vectors
                )

                try:
                    source_correct = source_task.score_output(t_input, output) == "correct"
                except Exception:
                    source_correct = False
                try:
                    target_correct = target_task.score_output(t_input, output) == "correct"
                except Exception:
                    target_correct = False

                if source_correct:
                    transfer_count += 1
                elif target_correct:
                    preserve_count += 1
                else:
                    neither_count += 1

                if len(examples) < 3:
                    examples.append({
                        "input": t_input,
                        "output": output,
                        "source_correct": source_correct,
                        "target_correct": target_correct,
                    })

            transfer_rate = transfer_count / n_test
            preserve_rate = preserve_count / n_test
            neither_rate = neither_count / n_test

            pair_results["demo_counts"][str(n_source_demos)] = {
                "n_source_demos": n_source_demos,
                "n_positions": len(mean_position_vectors),
                "transfer_rate": transfer_rate,
                "preserve_rate": preserve_rate,
                "neither_rate": neither_rate,
                "examples": examples,
            }

            marker = "***" if transfer_rate > 0.3 else ""
            logger.info(
                f"    transfer={transfer_rate:.2f}  preserve={preserve_rate:.2f}  "
                f"neither={neither_rate:.2f} {marker}"
            )

        all_results.append(pair_results)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Transfer Rate by Source Demo Count")
    logger.info("=" * 60)

    for pr in all_results:
        logger.info(f"\n  {pr['source']} → {pr['target']}:")
        for n_demos_str in sorted(pr["demo_counts"].keys(), key=int):
            data = pr["demo_counts"][n_demos_str]
            logger.info(
                f"    {data['n_source_demos']}-shot: transfer={data['transfer_rate']:.2f} "
                f"({data['n_positions']} positions)"
            )

    # Aggregate across pairs
    logger.info("\n  Mean transfer across pairs:")
    for n_demos in SOURCE_DEMO_COUNTS:
        rates = [
            pr["demo_counts"][str(n_demos)]["transfer_rate"]
            for pr in all_results
        ]
        logger.info(f"    {n_demos}-shot: {np.mean(rates):.3f} ± {np.std(rates):.3f}")

    # Save
    results = {
        "metadata": {
            "source_demo_counts": SOURCE_DEMO_COUNTS,
            "target_n_demos": TARGET_N_DEMOS,
            "intervention_layer": INTERVENTION_LAYER,
            "n_test": n_test,
            "pairs": TEST_PAIRS,
            "timestamp": datetime.now().isoformat(),
        },
        "pair_results": all_results,
    }

    with open(Path(output_dir) / "single_demo_fv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(Path(output_dir) / "single_demo_fv.csv", "w") as f:
        f.write("pair,n_source_demos,n_positions,transfer_rate,preserve_rate,neither_rate\n")
        for pr in all_results:
            pair = f"{pr['source']}_{pr['target']}"
            for n_demos_str in sorted(pr["demo_counts"].keys(), key=int):
                data = pr["demo_counts"][n_demos_str]
                f.write(
                    f"{pair},{data['n_source_demos']},{data['n_positions']},"
                    f"{data['transfer_rate']},{data['preserve_rate']},{data['neither_rate']}\n"
                )

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 30 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--output-dir", default="results/exp30")
    args = parser.parse_args()
    run_single_demo_fv(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
