#!/usr/bin/env python3
"""Experiment 15: Cross-Format Control.

Test whether semantically identical operations with different output formats
show transfer. If the "output format" hypothesis is correct, even semantically
identical operations should show 0% transfer when formats differ.

Task pairs:
- uppercase vs uppercase_with_period: Same operation, different format
- length vs length_word: Same operation (count chars), different format
- repeat_word vs repeat_word_comma: Same operation, different format
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
# Custom tasks with different output formats
# ═══════════════════════════════════════════════════════════════════════════

class UppercaseWithPeriodTask(Task):
    """Same as uppercase but output ends with period."""
    name = "uppercase_period"
    regime = "procedural"
    description = "Convert to uppercase with period"

    WORDS = ["hello", "world", "apple", "brain", "cloud", "dance", "eagle",
             "flame", "ghost", "house", "juice", "knife", "lemon", "mouse",
             "ocean", "piano", "queen", "river", "storm", "tiger"]

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        words = self.rng.sample(self.WORDS, min(n, len(self.WORDS)))
        return [(w, self.compute_answer(w)) for w in words]

    def generate_test_inputs(self, n: int) -> list[str]:
        return self.rng.sample(self.WORDS, min(n, len(self.WORDS)))

    def compute_answer(self, inp):
        return inp.upper() + "."


class LengthWordTask(Task):
    """Same as length but outputs word instead of digit."""
    name = "length_word"
    regime = "procedural"
    description = "Count characters, output as word"

    WORD_MAP = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

    WORDS = ["hi", "cat", "door", "apple", "banana", "elephant", "computer",
             "a", "go", "run", "jump", "house", "tree", "bird", "sky", "moon"]

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        words = self.rng.sample(self.WORDS, min(n, len(self.WORDS)))
        return [(w, self.compute_answer(w)) for w in words]

    def generate_test_inputs(self, n: int) -> list[str]:
        return self.rng.sample(self.WORDS, min(n, len(self.WORDS)))

    def compute_answer(self, inp):
        length = len(inp)
        return self.WORD_MAP.get(length, str(length))


class RepeatWordCommaTask(Task):
    """Same as repeat_word but with comma separator."""
    name = "repeat_comma"
    regime = "procedural"
    description = "Repeat word with comma"

    WORDS = ["apple", "brain", "cloud", "dance", "eagle", "flame", "ghost",
             "house", "juice", "knife", "lemon", "mouse", "noble", "ocean",
             "piano", "queen", "river", "storm", "tiger", "urban"]

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        words = self.rng.sample(self.WORDS, min(n, len(self.WORDS)))
        return [(w, self.compute_answer(w)) for w in words]

    def generate_test_inputs(self, n: int) -> list[str]:
        return self.rng.sample(self.WORDS, min(n, len(self.WORDS)))

    def compute_answer(self, inp):
        return f"{inp}, {inp}"


class ReverseTask(Task):
    """Reverse the input word."""
    name = "reverse"
    regime = "procedural"
    description = "Reverse the word"

    WORDS = ["hello", "world", "apple", "brain", "cloud", "dance", "eagle",
             "flame", "ghost", "house", "juice", "knife", "lemon", "mouse",
             "noble", "ocean", "piano", "queen", "river", "storm"]

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        words = self.rng.sample(self.WORDS, min(n, len(self.WORDS)))
        return [(w, self.compute_answer(w)) for w in words]

    def generate_test_inputs(self, n: int) -> list[str]:
        return self.rng.sample(self.WORDS, min(n, len(self.WORDS)))

    def compute_answer(self, inp):
        return inp[::-1]


class ReverseSpacedTask(Task):
    """Reverse the input word with spaces between letters."""
    name = "reverse_spaced"
    regime = "procedural"
    description = "Reverse with spaces"

    WORDS = ["hello", "world", "apple", "brain", "cloud", "dance", "eagle",
             "flame", "ghost", "house", "juice", "knife", "lemon", "mouse",
             "noble", "ocean", "piano", "queen", "river", "storm"]

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        words = self.rng.sample(self.WORDS, min(n, len(self.WORDS)))
        return [(w, self.compute_answer(w)) for w in words]

    def generate_test_inputs(self, n: int) -> list[str]:
        return self.rng.sample(self.WORDS, min(n, len(self.WORDS)))

    def compute_answer(self, inp):
        return " ".join(inp[::-1])


# ═══════════════════════════════════════════════════════════════════════════
# Test pairs: Same operation, different format
# ═══════════════════════════════════════════════════════════════════════════

FORMAT_PAIRS = [
    # (source_task, target_task, same_operation, same_format)
    ("uppercase", "uppercase_period", True, False),  # HELLO vs HELLO.
    ("length", "length_word", True, False),          # 5 vs five
    ("repeat_word", "repeat_comma", True, False),    # word word vs word, word
    ("reverse", "reverse_spaced", True, False),      # olleh vs o l l e h
]

# Control pairs: Different operation, same format (should also be 0%)
CONTROL_PAIRS = [
    ("uppercase", "first_letter", False, False),     # Different op, different format
    ("repeat_word", "pattern_completion", False, True),  # Different op, SAME format
]


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "exp15.log")),
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
        if line.startswith("Output:") and i < len(lines) - 1:
            output_value = line.replace("Output:", "").strip()
            current_text += line + "\n"
            tokens_so_far = tokenizer.encode(current_text, add_special_tokens=False)
            output_tokens = tokenizer.encode(" " + output_value, add_special_tokens=False)
            end_pos = len(tokens_so_far)
            start_pos = end_pos - len(output_tokens)
            positions.extend(range(max(0, start_pos), end_pos))
        else:
            current_text += line + "\n"

    return positions


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


def run_cross_format_control(device="cuda:0", n_demos=5, n_test=20, output_dir="results/exp15"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 15: Cross-Format Control")
    logger.info(f"Start: {datetime.now().isoformat()}")

    model = load_model(device=device)

    # Register custom tasks
    custom_tasks = {
        "uppercase_period": UppercaseWithPeriodTask(),
        "length_word": LengthWordTask(),
        "repeat_comma": RepeatWordCommaTask(),
        "reverse": ReverseTask(),
        "reverse_spaced": ReverseSpacedTask(),
    }

    # Get standard tasks
    standard_tasks = {
        "uppercase": TaskRegistry.get("uppercase"),
        "length": TaskRegistry.get("length"),
        "repeat_word": TaskRegistry.get("repeat_word"),
        "first_letter": TaskRegistry.get("first_letter"),
        "pattern_completion": TaskRegistry.get("pattern_completion"),
    }

    all_tasks = {**standard_tasks, **custom_tasks}

    # Use layer 8 based on Exp 8 findings
    intervention_layer = 8

    # ═══════════════════════════════════════════════════════════════════
    # Part A: Same Operation, Different Format
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Part A: Same Operation, Different Format")
    logger.info("Prediction: 0% transfer (format mismatch overrides semantic similarity)")
    logger.info("=" * 60)

    format_results = []

    for source_name, target_name, same_op, same_fmt in FORMAT_PAIRS:
        logger.info(f"\n{source_name} → {target_name}")
        logger.info(f"  Same operation: {same_op}, Same format: {same_fmt}")

        source_task = all_tasks[source_name]
        target_task = all_tasks[target_name]

        # Generate demos
        source_demos = source_task.generate_demos(n_demos)
        target_demos = target_task.generate_demos(n_demos)

        # Use shared test inputs (same words for both tasks)
        test_inputs = source_task.generate_test_inputs(n_test)

        # Build source prompt for vector extraction
        source_prompt = source_task.format_prompt(source_demos, test_inputs[0])
        source_positions = find_demo_output_positions(model.tokenizer, source_prompt, n_demos)
        source_vectors = extract_position_vectors(model, source_prompt, intervention_layer, source_positions)

        transferred = 0
        examples = []

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

            # Check if transfer happened (output matches SOURCE task format)
            try:
                source_correct = source_task.score_output(ti, intervened_output) == "correct"
            except (ValueError, TypeError):
                source_correct = False

            if source_correct:
                transferred += 1
                logger.info(f"  TRANSFER: '{ti}' → '{intervened_output}'")

            if len(examples) < 3:
                examples.append({
                    "input": ti,
                    "baseline": baseline_output,
                    "intervened": intervened_output,
                    "source_expected": source_task.compute_answer(ti),
                    "target_expected": target_task.compute_answer(ti),
                    "transferred": source_correct,
                })

        transfer_rate = transferred / n_test
        format_results.append({
            "source": source_name,
            "target": target_name,
            "same_operation": same_op,
            "same_format": same_fmt,
            "transfer_rate": transfer_rate,
            "examples": examples,
        })

        logger.info(f"  Transfer rate: {transfer_rate:.2f} ({transferred}/{n_test})")
        logger.info(f"  Examples:")
        for ex in examples:
            logger.info(f"    '{ex['input']}': baseline='{ex['baseline']}', "
                       f"intervened='{ex['intervened']}', "
                       f"source_expected='{ex['source_expected']}'")

    # ═══════════════════════════════════════════════════════════════════
    # Part B: Control - Different Operation with Same/Different Format
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Part B: Control Pairs")
    logger.info("=" * 60)

    control_results = []

    for source_name, target_name, same_op, same_fmt in CONTROL_PAIRS:
        logger.info(f"\n{source_name} → {target_name}")
        logger.info(f"  Same operation: {same_op}, Same format: {same_fmt}")

        source_task = all_tasks[source_name]
        target_task = all_tasks[target_name]

        source_demos = source_task.generate_demos(n_demos)
        target_demos = target_task.generate_demos(n_demos)
        test_inputs = target_task.generate_test_inputs(n_test)

        source_prompt = source_task.format_prompt(source_demos, source_task.generate_test_inputs(1)[0])
        source_positions = find_demo_output_positions(model.tokenizer, source_prompt, n_demos)
        source_vectors = extract_position_vectors(model, source_prompt, intervention_layer, source_positions)

        transferred = 0
        examples = []

        for ti in test_inputs:
            target_prompt = target_task.format_prompt(target_demos, ti)

            tokens = model.tokenizer.encode(target_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.model.generate(
                    tokens, max_new_tokens=30, do_sample=False,
                    pad_token_id=model.tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0, tokens.shape[1]:]
            baseline_output = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            baseline_output = baseline_output.strip().split("\n")[0].strip()

            intervened_output = generate_with_multi_intervention(
                model, target_prompt, intervention_layer, source_vectors
            )

            try:
                source_correct = source_task.score_output(ti, intervened_output) == "correct"
            except (ValueError, TypeError):
                source_correct = False

            if source_correct:
                transferred += 1
                logger.info(f"  TRANSFER: '{ti}' → '{intervened_output}'")

            if len(examples) < 3:
                examples.append({
                    "input": ti,
                    "baseline": baseline_output,
                    "intervened": intervened_output,
                    "transferred": source_correct,
                })

        transfer_rate = transferred / n_test
        control_results.append({
            "source": source_name,
            "target": target_name,
            "same_operation": same_op,
            "same_format": same_fmt,
            "transfer_rate": transfer_rate,
            "examples": examples,
        })

        logger.info(f"  Transfer rate: {transfer_rate:.2f} ({transferred}/{n_test})")

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info("\nSame Operation, Different Format (Hypothesis: 0% transfer):")
    for r in format_results:
        status = "CONFIRMED" if r["transfer_rate"] < 0.1 else "UNEXPECTED"
        logger.info(f"  {r['source']} → {r['target']}: {r['transfer_rate']:.2f} [{status}]")

    logger.info("\nControl Pairs:")
    for r in control_results:
        logger.info(f"  {r['source']} → {r['target']}: {r['transfer_rate']:.2f} "
                   f"(same_fmt={r['same_format']})")

    # ═══════════════════════════════════════════════════════════════════
    # Interpretation
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info("=" * 60)

    same_op_diff_fmt = [r for r in format_results if r["same_operation"] and not r["same_format"]]
    avg_same_op_diff_fmt = np.mean([r["transfer_rate"] for r in same_op_diff_fmt])

    diff_op_same_fmt = [r for r in control_results if not r["same_operation"] and r["same_format"]]
    avg_diff_op_same_fmt = np.mean([r["transfer_rate"] for r in diff_op_same_fmt]) if diff_op_same_fmt else 0

    logger.info(f"\nSame operation, different format: {avg_same_op_diff_fmt:.2f} mean transfer")
    logger.info(f"Different operation, same format: {avg_diff_op_same_fmt:.2f} mean transfer")

    if avg_same_op_diff_fmt < 0.1:
        logger.info("\n✓ FORMAT HYPOTHESIS CONFIRMED:")
        logger.info("  Even semantically identical operations show 0% transfer")
        logger.info("  when output formats differ. Transfer is about FORMAT, not TASK.")
    else:
        logger.info("\n✗ FORMAT HYPOTHESIS PARTIALLY REFUTED:")
        logger.info("  Some semantic transfer occurs despite format mismatch.")

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
        "format_pairs": format_results,
        "control_pairs": control_results,
        "summary": {
            "same_op_diff_fmt_mean": float(avg_same_op_diff_fmt),
            "diff_op_same_fmt_mean": float(avg_diff_op_same_fmt),
            "hypothesis_confirmed": bool(avg_same_op_diff_fmt < 0.1),
        }
    }

    with open(Path(output_dir) / "cross_format_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 15 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--output-dir", default="results/exp15")
    args = parser.parse_args()
    run_cross_format_control(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
