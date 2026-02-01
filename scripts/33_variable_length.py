#!/usr/bin/env python3
"""Experiment 33: Variable-Length Output Tasks (Q2/W2).

Test whether ~30% depth finding holds for variable-length outputs.

Design:
- 2 inline tasks:
  - RepeatNTask: input "cat 3" → "cat cat cat" (length varies with N)
  - SpellOutTask: input "7" → "seven", "21" → "twenty-one" (variable word count)
- Part A: Baseline accuracy (5-shot)
- Part B: Layer sweep at [4, 6, 8, 10, 12, 15] for transfer pairs
- Part C: Length-dependent analysis for repeat_n (does N=2 transfer better than N=5?)
- N=15
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
from src.tasks import TaskRegistry, Task

# ═══════════════════════════════════════════════════════════════════════════
# Variable-length output tasks
# ═══════════════════════════════════════════════════════════════════════════

class RepeatNTask(Task):
    """Repeat word N times: input "cat 3" → "cat cat cat"."""
    name = "repeat_n"
    regime = "procedural"
    description = "Repeat word N times"

    WORDS = ["apple", "brain", "cloud", "dance", "eagle", "flame", "ghost",
             "house", "juice", "knife", "lemon", "mouse", "noble", "ocean",
             "piano", "queen", "river", "storm", "tiger", "urban"]

    _DEMO_ITEMS = [
        ("dog 2", "dog dog"),
        ("sun 3", "sun sun sun"),
        ("cat 4", "cat cat cat cat"),
        ("box 2", "box box"),
        ("hat 3", "hat hat hat"),
    ]

    _TEST_ITEMS = [
        ("apple 2", 2), ("brain 3", 3), ("cloud 4", 4), ("dance 2", 2),
        ("eagle 5", 5), ("flame 3", 3), ("ghost 2", 2), ("house 4", 4),
        ("juice 3", 3), ("knife 2", 2), ("lemon 5", 5), ("mouse 3", 3),
        ("noble 2", 2), ("ocean 4", 4), ("piano 3", 3), ("queen 2", 2),
        ("river 5", 5), ("storm 3", 3), ("tiger 4", 4), ("urban 2", 2),
    ]

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        return self._DEMO_ITEMS[:n]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = [item[0] for item in self._TEST_ITEMS]
        self.rng.shuffle(pool)
        return pool[:n]

    def compute_answer(self, inp: str) -> str:
        parts = inp.strip().split()
        word = parts[0]
        count = int(parts[1])
        return " ".join([word] * count)

    def score_output(self, inp: str, output: str) -> str:
        expected = self.compute_answer(inp)
        cleaned = output.strip().split("\n")[0].strip()
        if cleaned == expected:
            return "correct"
        if not cleaned:
            return "malformed"
        return "incorrect"

    def get_repeat_count(self, inp: str) -> int:
        """Get the N value for length-dependent analysis."""
        parts = inp.strip().split()
        return int(parts[1])


class SpellOutTask(Task):
    """Spell out a number: input "7" → "seven", "21" → "twenty-one"."""
    name = "spell_out"
    regime = "procedural"
    description = "Spell out a number as words"

    _NUMBER_WORDS = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
        15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
        19: "nineteen", 20: "twenty", 21: "twenty-one", 22: "twenty-two",
        23: "twenty-three", 24: "twenty-four", 25: "twenty-five",
        30: "thirty", 40: "forty", 50: "fifty",
    }

    _DEMO_ITEMS = [
        ("3", "three"),
        ("12", "twelve"),
        ("20", "twenty"),
        ("7", "seven"),
        ("15", "fifteen"),
    ]

    _TEST_ITEMS = [
        "1", "2", "4", "5", "6", "8", "9", "10", "11", "13",
        "14", "16", "17", "18", "19", "21", "22", "23", "24", "25",
    ]

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        return self._DEMO_ITEMS[:n]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = list(self._TEST_ITEMS)
        self.rng.shuffle(pool)
        return pool[:n]

    def compute_answer(self, inp: str) -> str:
        num = int(inp.strip())
        return self._NUMBER_WORDS.get(num, str(num))

    def score_output(self, inp: str, output: str) -> str:
        expected = self.compute_answer(inp)
        cleaned = output.strip().lower().split("\n")[0].strip()
        # Accept with or without hyphens
        if cleaned == expected or cleaned.replace(" ", "-") == expected or cleaned.replace("-", " ") == expected:
            return "correct"
        if not cleaned:
            return "malformed"
        return "incorrect"


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

LAYER_SWEEP = [4, 6, 8, 10, 12, 15]
N_DEMOS = 5

TRANSFER_PAIRS = [
    ("repeat_n", "repeat_word"),
    ("repeat_word", "repeat_n"),
    ("uppercase", "repeat_n"),
    ("repeat_n", "uppercase"),
    ("spell_out", "length"),
    ("length", "spell_out"),
]


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "exp33.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def find_demo_token_positions(tokenizer, prompt, n_demos=5):
    """Find all demo token positions."""
    lines = prompt.strip().split("\n")
    positions = []
    current_pos = 0

    for i, line in enumerate(lines):
        line_tokens = tokenizer.encode(line, add_special_tokens=False)
        line_len = len(line_tokens)

        if i >= len(lines) - 2:
            break

        if line.startswith("Input:") or line.startswith("Output:"):
            positions.extend(range(current_pos, current_pos + line_len))

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


def run_variable_length(device="cuda:6", n_test=15, output_dir="results/exp33"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 33: Variable-Length Output Tasks")
    logger.info(f"Start: {datetime.now().isoformat()}")

    model = load_model(device=device)

    # Task instances
    all_tasks = {
        "repeat_n": RepeatNTask(),
        "spell_out": SpellOutTask(),
        "repeat_word": TaskRegistry.get("repeat_word"),
        "uppercase": TaskRegistry.get("uppercase"),
        "length": TaskRegistry.get("length"),
    }

    # ═══════════════════════════════════════════════════════════════════
    # Part A: Baseline accuracy
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Part A: Baseline Accuracy (5-shot)")
    logger.info("=" * 60)

    baseline_results = {}

    for task_name in ["repeat_n", "spell_out", "repeat_word", "uppercase", "length"]:
        task = all_tasks[task_name]
        demos = task.generate_demos(N_DEMOS)
        test_inputs = task.generate_test_inputs(n_test)

        correct = 0
        examples = []

        for t_input in test_inputs:
            prompt = task.format_prompt(demos, t_input)
            tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.model.generate(
                    tokens, max_new_tokens=30, do_sample=False,
                    pad_token_id=model.tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0, tokens.shape[1]:]
            output = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            output = output.strip().split("\n")[0].strip()

            score = task.score_output(t_input, output)
            if score == "correct":
                correct += 1

            if len(examples) < 3:
                examples.append({
                    "input": t_input,
                    "output": output,
                    "expected": task.compute_answer(t_input),
                    "score": score,
                })

        accuracy = correct / n_test
        baseline_results[task_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": n_test,
            "examples": examples,
        }
        logger.info(f"  {task_name:15s}: {accuracy:.2f} ({correct}/{n_test})")
        for ex in examples:
            logger.info(f"    '{ex['input']}' → '{ex['output']}' (expected: '{ex['expected']}') [{ex['score']}]")

    # ═══════════════════════════════════════════════════════════════════
    # Part B: Layer sweep for transfer pairs
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Part B: Layer Sweep Transfer")
    logger.info("=" * 60)

    layer_sweep_results = []

    for source_name, target_name in TRANSFER_PAIRS:
        logger.info(f"\n  Pair: {source_name} → {target_name}")

        source_task = all_tasks[source_name]
        target_task = all_tasks[target_name]

        source_demos = source_task.generate_demos(N_DEMOS)
        target_demos = target_task.generate_demos(N_DEMOS)
        test_inputs = target_task.generate_test_inputs(n_test)

        pair_results = {
            "source": source_name,
            "target": target_name,
            "layers": {},
        }

        for layer in LAYER_SWEEP:
            # Extract source activations
            source_position_vectors = {}

            for s_input in test_inputs:
                prompt = source_task.format_prompt(source_demos, s_input)
                pos_list = find_demo_token_positions(model.tokenizer, prompt, N_DEMOS)
                acts = extract_multi_position_activations(model, prompt, layer, pos_list)

                for pos, vec in acts.items():
                    if pos not in source_position_vectors:
                        source_position_vectors[pos] = []
                    source_position_vectors[pos].append(vec)

            mean_position_vectors = {
                pos: torch.stack(vecs).mean(dim=0)
                for pos, vecs in source_position_vectors.items()
            }

            # Transplant into target
            transfer_count = 0
            preserve_count = 0

            for t_input in test_inputs:
                target_prompt = target_task.format_prompt(target_demos, t_input)
                output = generate_with_multi_intervention(
                    model, target_prompt, layer, mean_position_vectors
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

            transfer_rate = transfer_count / n_test
            preserve_rate = preserve_count / n_test

            pair_results["layers"][str(layer)] = {
                "layer": layer,
                "transfer_rate": transfer_rate,
                "preserve_rate": preserve_rate,
                "n_positions": len(mean_position_vectors),
            }

            marker = "***" if transfer_rate > 0.3 else ""
            logger.info(
                f"    Layer {layer:2d}: transfer={transfer_rate:.2f}  "
                f"preserve={preserve_rate:.2f} {marker}"
            )

        layer_sweep_results.append(pair_results)

    # Find best layer per pair
    logger.info("\n  Best layers per pair:")
    for pr in layer_sweep_results:
        best_layer = max(pr["layers"].items(), key=lambda x: x[1]["transfer_rate"])
        logger.info(
            f"    {pr['source']} → {pr['target']}: layer {best_layer[1]['layer']} "
            f"(transfer={best_layer[1]['transfer_rate']:.2f})"
        )

    # ═══════════════════════════════════════════════════════════════════
    # Part C: Length-dependent analysis for repeat_n
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Part C: Length-Dependent Analysis (repeat_n)")
    logger.info("=" * 60)

    repeat_n_task = all_tasks["repeat_n"]
    repeat_word_task = all_tasks["repeat_word"]

    # Get all test inputs and group by N
    all_repeat_inputs = [item[0] for item in RepeatNTask._TEST_ITEMS]
    inputs_by_n = {}
    for inp in all_repeat_inputs:
        n_val = repeat_n_task.get_repeat_count(inp)
        if n_val not in inputs_by_n:
            inputs_by_n[n_val] = []
        inputs_by_n[n_val].append(inp)

    summary = {k: len(v) for k, v in sorted(inputs_by_n.items())}
    logger.info(f"  Inputs by N: {summary}")

    # Use layer 8 for this analysis
    analysis_layer = 8
    source_demos = repeat_word_task.generate_demos(N_DEMOS)
    target_demos = repeat_n_task.generate_demos(N_DEMOS)

    # Extract source activations from repeat_word (fixed-length output)
    source_position_vectors = {}
    rw_test_inputs = repeat_word_task.generate_test_inputs(n_test)

    for s_input in rw_test_inputs:
        prompt = repeat_word_task.format_prompt(source_demos, s_input)
        pos_list = find_demo_token_positions(model.tokenizer, prompt, N_DEMOS)
        acts = extract_multi_position_activations(model, prompt, analysis_layer, pos_list)

        for pos, vec in acts.items():
            if pos not in source_position_vectors:
                source_position_vectors[pos] = []
            source_position_vectors[pos].append(vec)

    mean_position_vectors = {
        pos: torch.stack(vecs).mean(dim=0)
        for pos, vecs in source_position_vectors.items()
    }

    length_dependent_results = {}

    for n_val in sorted(inputs_by_n.keys()):
        inputs = inputs_by_n[n_val]
        transfer_count = 0

        for t_input in inputs:
            target_prompt = repeat_n_task.format_prompt(target_demos, t_input)
            output = generate_with_multi_intervention(
                model, target_prompt, analysis_layer, mean_position_vectors
            )

            # Check if repeat_word pattern emerges (word word)
            word = t_input.split()[0]
            expected_rw = f"{word} {word}"
            cleaned = output.strip().split("\n")[0].strip()
            if cleaned == expected_rw:
                transfer_count += 1

        rate = transfer_count / len(inputs) if inputs else 0
        length_dependent_results[n_val] = {
            "n_value": n_val,
            "n_inputs": len(inputs),
            "transfer_count": transfer_count,
            "transfer_rate": rate,
        }
        logger.info(f"  N={n_val}: transfer={rate:.2f} ({transfer_count}/{len(inputs)})")

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info("\nBaseline accuracy:")
    for name, data in baseline_results.items():
        logger.info(f"  {name:15s}: {data['accuracy']:.2f}")

    logger.info("\nBest transfer layer per pair:")
    for pr in layer_sweep_results:
        best = max(pr["layers"].values(), key=lambda x: x["transfer_rate"])
        depth_frac = best["layer"] / 28  # Llama-3.2-3B has 28 layers
        logger.info(
            f"  {pr['source']:15s} → {pr['target']:15s}: "
            f"layer {best['layer']} (~{depth_frac:.0%} depth), transfer={best['transfer_rate']:.2f}"
        )

    logger.info("\nLength-dependent transfer (repeat_word → repeat_n at layer 8):")
    for n_val, data in sorted(length_dependent_results.items()):
        logger.info(f"  N={n_val}: {data['transfer_rate']:.2f}")

    # Interpretation
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info("=" * 60)

    # Check if ~30% depth holds
    best_layers = []
    for pr in layer_sweep_results:
        best = max(pr["layers"].values(), key=lambda x: x["transfer_rate"])
        if best["transfer_rate"] > 0.1:
            best_layers.append(best["layer"])

    if best_layers:
        avg_best = np.mean(best_layers)
        avg_frac = avg_best / 28
        logger.info(f"\nAverage best layer: {avg_best:.1f} (~{avg_frac:.0%} depth)")
        if 0.2 <= avg_frac <= 0.4:
            logger.info("→ ~30% depth finding HOLDS for variable-length tasks")
        else:
            logger.info(f"→ Optimal depth shifted to ~{avg_frac:.0%} for variable-length tasks")

    # Check length dependence
    if length_dependent_results:
        n2_rate = length_dependent_results.get(2, {}).get("transfer_rate", 0)
        n5_rate = length_dependent_results.get(5, {}).get("transfer_rate", 0)
        if n2_rate > n5_rate + 0.1:
            logger.info("\nN=2 transfers better than N=5 → transfer degrades with output length mismatch")
        elif abs(n2_rate - n5_rate) < 0.1:
            logger.info("\nSimilar transfer across N values → transfer is robust to output length")

    # Save
    results = {
        "metadata": {
            "layer_sweep": LAYER_SWEEP,
            "n_demos": N_DEMOS,
            "n_test": n_test,
            "transfer_pairs": TRANSFER_PAIRS,
            "timestamp": datetime.now().isoformat(),
        },
        "baselines": baseline_results,
        "layer_sweep": layer_sweep_results,
        "length_dependent": length_dependent_results,
    }

    with open(Path(output_dir) / "variable_length_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV for layer sweep
    with open(Path(output_dir) / "variable_length_layers.csv", "w") as f:
        f.write("pair,layer,transfer_rate,preserve_rate,n_positions\n")
        for pr in layer_sweep_results:
            pair = f"{pr['source']}_{pr['target']}"
            for layer_str, data in sorted(pr["layers"].items(), key=lambda x: int(x[0])):
                f.write(
                    f"{pair},{data['layer']},{data['transfer_rate']},"
                    f"{data['preserve_rate']},{data['n_positions']}\n"
                )

    # CSV for length-dependent
    with open(Path(output_dir) / "variable_length_by_n.csv", "w") as f:
        f.write("n_value,n_inputs,transfer_count,transfer_rate\n")
        for n_val, data in sorted(length_dependent_results.items()):
            f.write(f"{data['n_value']},{data['n_inputs']},{data['transfer_count']},{data['transfer_rate']}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 33 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--n-test", type=int, default=15)
    parser.add_argument("--output-dir", default="results/exp33")
    args = parser.parse_args()
    run_variable_length(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
