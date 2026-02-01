#!/usr/bin/env python3
"""Experiment 32: Sentiment Variant Transfer (Q4).

Test whether same-task-different-labels transfers. Distinguishes
"abstract task template" from "specific label token template."

Design:
- 3 sentiment variants:
  - Standard: positive/negative labels
  - GoodBad: good/bad labels
  - Symbol: +/- labels
- Part A: Baseline accuracy for all 3 variants
- Part B: 6 pairwise transfers + 2 negative controls
- Layer 8, all_demo, N=20
- Uses SOURCE task scorer to detect transfer
"""

import json
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from src.model import load_model
from src.tasks import TaskRegistry, Task

# ═══════════════════════════════════════════════════════════════════════════
# Sentiment variant tasks (inline, following exp15 pattern)
# ═══════════════════════════════════════════════════════════════════════════

# Exact copy of SentimentTask's _TEST_ITEMS and _DEMOS
_SENTIMENT_DEMOS = [
    ("joyful", "positive"),
    ("angry", "negative"),
    ("wonderful", "positive"),
    ("terrible", "negative"),
    ("cheerful", "positive"),
]

_SENTIMENT_TEST_ITEMS = [
    ("happy", "positive"), ("sad", "negative"), ("excited", "positive"),
    ("fearful", "negative"), ("grateful", "positive"), ("hostile", "negative"),
    ("hopeful", "positive"), ("anxious", "negative"), ("proud", "positive"),
    ("jealous", "negative"), ("loving", "positive"), ("bitter", "negative"),
    ("peaceful", "positive"), ("furious", "negative"), ("delighted", "positive"),
    ("miserable", "negative"), ("confident", "positive"), ("worried", "negative"),
    ("content", "positive"), ("gloomy", "negative"), ("radiant", "positive"),
    ("devastated", "negative"), ("thrilled", "positive"), ("depressed", "negative"),
    ("amused", "positive"), ("disgusted", "negative"), ("inspired", "positive"),
    ("frustrated", "negative"), ("optimistic", "positive"), ("pessimistic", "negative"),
    ("blissful", "positive"), ("resentful", "negative"), ("enthusiastic", "positive"),
    ("melancholy", "negative"), ("ecstatic", "positive"), ("sorrowful", "negative"),
    ("compassionate", "positive"), ("vindictive", "negative"), ("serene", "positive"),
    ("agitated", "negative"), ("jubilant", "positive"), ("despondent", "negative"),
    ("affectionate", "positive"), ("spiteful", "negative"), ("thankful", "positive"),
    ("envious", "negative"), ("elated", "positive"), ("dejected", "negative"),
    ("generous", "positive"), ("cruel", "negative"),
]

# Build sentiment lookup
_SENTIMENT_LOOKUP = {item[0]: item[1] for item in _SENTIMENT_TEST_ITEMS + list(_SENTIMENT_DEMOS)}


class SentimentGoodBadTask(Task):
    """Sentiment classification with good/bad labels."""
    name = "sentiment_goodbad"
    regime = "bayesian"
    description = "Classify sentiment as good or bad"

    _LABEL_MAP = {"positive": "good", "negative": "bad"}
    _REVERSE_MAP = {"good": "positive", "negative": "bad", "bad": "negative", "positive": "positive"}

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        demos = _SENTIMENT_DEMOS[:n]
        return [(word, self._LABEL_MAP[label]) for word, label in demos]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = [item[0] for item in _SENTIMENT_TEST_ITEMS]
        self.rng.shuffle(pool)
        return pool[:n]

    def compute_answer(self, inp: str) -> str:
        sentiment = _SENTIMENT_LOOKUP.get(inp, "positive")
        return self._LABEL_MAP[sentiment]

    def score_output(self, inp: str, output: str) -> str:
        expected = self.compute_answer(inp)
        cleaned = output.strip().lower().split("\n")[0].strip()
        if expected in cleaned:
            return "correct"
        other = "bad" if expected == "good" else "good"
        if other in cleaned:
            return "incorrect"
        return "malformed"


class SentimentSymbolTask(Task):
    """Sentiment classification with +/- labels."""
    name = "sentiment_symbol"
    regime = "bayesian"
    description = "Classify sentiment as + or -"

    _LABEL_MAP = {"positive": "+", "negative": "-"}

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        demos = _SENTIMENT_DEMOS[:n]
        return [(word, self._LABEL_MAP[label]) for word, label in demos]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = [item[0] for item in _SENTIMENT_TEST_ITEMS]
        self.rng.shuffle(pool)
        return pool[:n]

    def compute_answer(self, inp: str) -> str:
        sentiment = _SENTIMENT_LOOKUP.get(inp, "positive")
        return self._LABEL_MAP[sentiment]

    def score_output(self, inp: str, output: str) -> str:
        expected = self.compute_answer(inp)
        cleaned = output.strip().split("\n")[0].strip()
        # For +/- we need exact match or containment
        if expected == "+":
            if cleaned == "+" or cleaned.startswith("+"):
                return "correct"
            if cleaned == "-" or cleaned.startswith("-"):
                return "incorrect"
        elif expected == "-":
            if cleaned == "-" or cleaned.startswith("-"):
                return "correct"
            if cleaned == "+" or cleaned.startswith("+"):
                return "incorrect"
        return "malformed"


# ═══════════════════════════════════════════════════════════════════════════
# Transfer pairs
# ═══════════════════════════════════════════════════════════════════════════

# 6 directed pairs between 3 sentiment variants
SENTIMENT_PAIRS = [
    ("sentiment", "sentiment_goodbad"),
    ("sentiment_goodbad", "sentiment"),
    ("sentiment", "sentiment_symbol"),
    ("sentiment_symbol", "sentiment"),
    ("sentiment_goodbad", "sentiment_symbol"),
    ("sentiment_symbol", "sentiment_goodbad"),
]

# Negative controls: cross-regime tasks
CONTROL_PAIRS = [
    ("sentiment", "antonym"),       # semantic → retrieval
    ("uppercase", "sentiment"),     # procedural → bayesian
]

INTERVENTION_LAYER = 8
N_DEMOS = 5


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "exp32.log")),
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


def run_sentiment_variants(device="cuda:1", n_test=20, output_dir="results/exp32"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 32: Sentiment Variant Transfer")
    logger.info(f"Start: {datetime.now().isoformat()}")

    model = load_model(device=device)

    # Task instances
    all_tasks = {
        "sentiment": TaskRegistry.get("sentiment"),
        "sentiment_goodbad": SentimentGoodBadTask(),
        "sentiment_symbol": SentimentSymbolTask(),
        "antonym": TaskRegistry.get("antonym"),
        "uppercase": TaskRegistry.get("uppercase"),
    }

    # ═══════════════════════════════════════════════════════════════════
    # Part A: Baseline accuracy for all 3 sentiment variants
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Part A: Baseline Accuracy")
    logger.info("=" * 60)

    baseline_results = {}

    for task_name in ["sentiment", "sentiment_goodbad", "sentiment_symbol"]:
        task = all_tasks[task_name]
        demos = task.generate_demos(N_DEMOS)
        test_inputs = task.generate_test_inputs(n_test)

        correct = 0
        for t_input in test_inputs:
            prompt = task.format_prompt(demos, t_input)
            tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.model.generate(
                    tokens, max_new_tokens=15, do_sample=False,
                    pad_token_id=model.tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0, tokens.shape[1]:]
            output = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            output = output.strip().split("\n")[0].strip()

            if task.score_output(t_input, output) == "correct":
                correct += 1

        accuracy = correct / n_test
        baseline_results[task_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": n_test,
        }
        logger.info(f"  {task_name:25s}: {accuracy:.2f} ({correct}/{n_test})")

    # ═══════════════════════════════════════════════════════════════════
    # Part B: Transfer experiments
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Part B: Pairwise Transfer (+ Controls)")
    logger.info("=" * 60)

    all_transfer_pairs = SENTIMENT_PAIRS + CONTROL_PAIRS
    transfer_results = []

    for source_name, target_name in all_transfer_pairs:
        is_control = (source_name, target_name) in CONTROL_PAIRS
        label = "CONTROL" if is_control else "VARIANT"
        logger.info(f"\n  [{label}] {source_name} → {target_name}")

        source_task = all_tasks[source_name]
        target_task = all_tasks[target_name]

        source_demos = source_task.generate_demos(N_DEMOS)
        target_demos = target_task.generate_demos(N_DEMOS)

        # Use sentiment test inputs for sentiment pairs, target inputs for controls
        if not is_control:
            test_inputs = source_task.generate_test_inputs(n_test)
        else:
            test_inputs = target_task.generate_test_inputs(n_test)

        # Extract source activations averaged over test inputs
        source_position_vectors = {}

        for s_input in test_inputs:
            prompt = source_task.format_prompt(source_demos, s_input)
            pos_list = find_demo_token_positions(model.tokenizer, prompt, N_DEMOS)

            acts = extract_multi_position_activations(model, prompt, INTERVENTION_LAYER, pos_list)

            for pos, vec in acts.items():
                if pos not in source_position_vectors:
                    source_position_vectors[pos] = []
                source_position_vectors[pos].append(vec)

        # Average
        mean_position_vectors = {
            pos: torch.stack(vecs).mean(dim=0)
            for pos, vecs in source_position_vectors.items()
        }

        # Transplant into target
        transfer_count = 0
        preserve_count = 0
        neither_count = 0
        examples = []

        for t_input in test_inputs:
            target_prompt = target_task.format_prompt(target_demos, t_input)

            output = generate_with_multi_intervention(
                model, target_prompt, INTERVENTION_LAYER, mean_position_vectors
            )

            # Score using SOURCE task scorer (to detect transfer)
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

            if len(examples) < 5:
                examples.append({
                    "input": t_input,
                    "output": output,
                    "source_expected": source_task.compute_answer(t_input),
                    "target_expected": target_task.compute_answer(t_input) if not is_control else "N/A",
                    "source_correct": source_correct,
                    "target_correct": target_correct,
                })

        transfer_rate = transfer_count / n_test
        preserve_rate = preserve_count / n_test
        neither_rate = neither_count / n_test

        result = {
            "source": source_name,
            "target": target_name,
            "is_control": is_control,
            "n_positions": len(mean_position_vectors),
            "transfer_rate": transfer_rate,
            "preserve_rate": preserve_rate,
            "neither_rate": neither_rate,
            "examples": examples,
        }
        transfer_results.append(result)

        marker = "***" if transfer_rate > 0.3 else ""
        logger.info(
            f"    transfer={transfer_rate:.2f}  preserve={preserve_rate:.2f}  "
            f"neither={neither_rate:.2f} {marker}"
        )

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info("\nBaseline accuracy:")
    for name, data in baseline_results.items():
        logger.info(f"  {name:25s}: {data['accuracy']:.2f}")

    logger.info("\nSentiment variant transfers:")
    variant_rates = []
    for r in transfer_results:
        if not r["is_control"]:
            logger.info(f"  {r['source']:25s} → {r['target']:25s}: {r['transfer_rate']:.2f}")
            variant_rates.append(r["transfer_rate"])

    if variant_rates:
        logger.info(f"\n  Mean variant transfer: {np.mean(variant_rates):.3f}")

    logger.info("\nNegative controls:")
    for r in transfer_results:
        if r["is_control"]:
            logger.info(f"  {r['source']:25s} → {r['target']:25s}: {r['transfer_rate']:.2f}")

    # Interpretation
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info("=" * 60)

    mean_variant = np.mean(variant_rates) if variant_rates else 0
    control_rates = [r["transfer_rate"] for r in transfer_results if r["is_control"]]
    mean_control = np.mean(control_rates) if control_rates else 0

    if mean_variant > 0.3:
        logger.info("High variant transfer → model encodes ABSTRACT task template")
        logger.info("(sentiment classification rule transfers across label sets)")
    elif mean_variant > 0.1:
        logger.info("Moderate variant transfer → partial abstraction")
        logger.info("(some task structure transfers but label-specific encoding also matters)")
    else:
        logger.info("Low variant transfer → model encodes SPECIFIC label tokens")
        logger.info("(different labels = different task as far as the model is concerned)")

    if mean_variant > mean_control + 0.1:
        logger.info(f"Variant transfer ({mean_variant:.2f}) > control ({mean_control:.2f})")
        logger.info("→ Semantic similarity between variants contributes to transfer")

    # Save
    results = {
        "metadata": {
            "intervention_layer": INTERVENTION_LAYER,
            "n_demos": N_DEMOS,
            "n_test": n_test,
            "sentiment_pairs": SENTIMENT_PAIRS,
            "control_pairs": CONTROL_PAIRS,
            "timestamp": datetime.now().isoformat(),
        },
        "baselines": baseline_results,
        "transfer_results": transfer_results,
        "summary": {
            "mean_variant_transfer": float(mean_variant),
            "mean_control_transfer": float(mean_control),
        },
    }

    with open(Path(output_dir) / "sentiment_variant_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(Path(output_dir) / "sentiment_variants.csv", "w") as f:
        f.write("source,target,is_control,n_positions,transfer_rate,preserve_rate,neither_rate\n")
        for r in transfer_results:
            f.write(
                f"{r['source']},{r['target']},{r['is_control']},{r['n_positions']},"
                f"{r['transfer_rate']},{r['preserve_rate']},{r['neither_rate']}\n"
            )

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 32 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--output-dir", default="results/exp32")
    args = parser.parse_args()
    run_sentiment_variants(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
