#!/usr/bin/env python3
"""Experiment 29: Expanded Transfer Matrix.

Runs multi-position transfer (all_demo condition, ~30% depth) for ALL 56
ordered task pairs. This provides enough data points for a meaningful
template similarity correlation (addressing reviewer W4).
"""

import json
import sys
import os
import logging
import itertools
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


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "exp29.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def find_demo_token_positions(tokenizer, prompt, n_demos=5):
    """Find all demo token positions in a prompt."""
    lines = prompt.strip().split("\n")
    positions = []
    current_pos = 0

    for i, line in enumerate(lines):
        line_tokens = tokenizer.encode(line, add_special_tokens=False)
        line_len = len(line_tokens)

        if i >= len(lines) - 2:
            break

        if line.startswith("Input:") or line.startswith("Output:"):
            for p in range(current_pos, current_pos + line_len):
                positions.append(p)

        current_pos += line_len
        if i < len(lines) - 1:
            newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
            current_pos += len(newline_tokens)

    return positions


def extract_activations(model, prompt, layer, positions):
    """Extract activations at specified positions."""
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


def generate_with_intervention(model, prompt, layer, position_vectors, max_new_tokens=30):
    """Generate with activation replacement at multiple positions."""
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


def run_expanded_transfer(model_name="meta-llama/Llama-3.2-3B-Instruct",
                          device="cuda:3", n_demos=5, n_test=10,
                          output_dir="results/exp29"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 29: Expanded Transfer Matrix (all 56 pairs)")
    logger.info(f"Start: {datetime.now().isoformat()}")

    model = load_model(model_name, device=device)
    tasks = {name: TaskRegistry.get(name) for name in INCLUDED_TASKS}

    # Use ~30% depth (same as main exp8 finding)
    layer = model.layer_at_fraction(0.30)
    logger.info(f"Model: {model_name} ({model.n_layers} layers), intervention layer={layer}")

    all_pairs = list(itertools.permutations(INCLUDED_TASKS, 2))
    logger.info(f"Total pairs to test: {len(all_pairs)}")

    all_results = []

    for pair_idx, (source_name, target_name) in enumerate(all_pairs):
        logger.info(f"\n[{pair_idx+1}/{len(all_pairs)}] {source_name} -> {target_name}")

        source_task = tasks[source_name]
        target_task = tasks[target_name]

        source_demos = source_task.generate_demos(n_demos)
        target_demos = target_task.generate_demos(n_demos)

        source_test_inputs = source_task.generate_test_inputs(n_test)
        target_test_inputs = target_task.generate_test_inputs(n_test)

        # Extract source activations at all demo positions, averaged over inputs
        source_position_vectors = {}
        for s_input in source_test_inputs:
            prompt = source_task.format_prompt(source_demos, s_input)
            pos_list = find_demo_token_positions(model.tokenizer, prompt, n_demos)
            acts = extract_activations(model, prompt, layer, pos_list)
            for pos, vec in acts.items():
                if pos not in source_position_vectors:
                    source_position_vectors[pos] = []
                source_position_vectors[pos].append(vec)

        mean_position_vectors = {
            pos: torch.stack(vecs).mean(dim=0)
            for pos, vecs in source_position_vectors.items()
        }

        # Transplant into target contexts
        transfer_count = 0
        preserve_count = 0
        neither_count = 0

        for t_input in target_test_inputs:
            prompt = target_task.format_prompt(target_demos, t_input)
            output = generate_with_intervention(
                model, prompt, layer, mean_position_vectors, max_new_tokens=30
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

        result = {
            "source": source_name,
            "target": target_name,
            "layer": layer,
            "n_positions": len(mean_position_vectors),
            "transfer_rate": transfer_count / n_test,
            "preserve_rate": preserve_count / n_test,
            "neither_rate": neither_count / n_test,
        }
        all_results.append(result)

        marker = "***" if result["transfer_rate"] > 0.3 else ""
        logger.info(
            f"  transfer={result['transfer_rate']:.2f} "
            f"preserve={result['preserve_rate']:.2f} "
            f"neither={result['neither_rate']:.2f} "
            f"({result['n_positions']} positions) {marker}"
        )

    # Summary: transfer matrix
    logger.info("\n" + "=" * 60)
    logger.info("TRANSFER MATRIX (rows=source, cols=target)")
    logger.info("=" * 60)

    header = "            " + "  ".join(f"{t[:6]:>6s}" for t in INCLUDED_TASKS)
    logger.info(header)
    for src in INCLUDED_TASKS:
        row = f"{src[:10]:>10s}  "
        for tgt in INCLUDED_TASKS:
            if src == tgt:
                row += "   -  "
            else:
                r = next((x for x in all_results
                         if x["source"] == src and x["target"] == tgt), None)
                if r:
                    val = r["transfer_rate"]
                    row += f"  {val:.2f}"
                else:
                    row += "   ?  "
        logger.info(row)

    # Save results
    results = {
        "metadata": {
            "model": model_name,
            "layer": layer,
            "n_demos": n_demos,
            "n_test": n_test,
            "n_pairs": len(all_pairs),
            "condition": "all_demo",
            "timestamp": datetime.now().isoformat(),
        },
        "pair_results": all_results,
    }

    with open(Path(output_dir) / "expanded_transfer_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(Path(output_dir) / "transfer_matrix.csv", "w") as f:
        f.write("source,target,layer,n_positions,transfer_rate,preserve_rate,neither_rate\n")
        for r in all_results:
            f.write(f"{r['source']},{r['target']},{r['layer']},"
                    f"{r['n_positions']},{r['transfer_rate']},"
                    f"{r['preserve_rate']},{r['neither_rate']}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 29 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--output-dir", default="results/exp29")
    args = parser.parse_args()
    run_expanded_transfer(
        model_name=args.model,
        device=args.device,
        n_test=args.n_test,
        output_dir=args.output_dir,
    )
