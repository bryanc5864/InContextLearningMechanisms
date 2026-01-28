#!/usr/bin/env python3
"""Phase 5: Locality Analysis — transplantation across layers.

Test whether causal efficacy is localized or distributed across depth.
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
from src.extraction import extract_activations, get_position_index
from src.intervention import transplant_and_generate

INCLUDED_TASKS = [
    "uppercase", "first_letter", "repeat_word", "length",
    "linear_2x", "sentiment", "antonym", "pattern_completion",
]

# Representative cross-task pairs for locality sweep
LOCALITY_PAIRS = [
    ("uppercase", "sentiment"),          # procedural → bayesian
    ("linear_2x", "pattern_completion"), # gd_like → induction
    ("sentiment", "linear_2x"),          # bayesian → gd_like
    ("first_letter", "antonym"),         # procedural → retrieval
]


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(output_dir, "phase5.log")), logging.StreamHandler()])
    return logging.getLogger(__name__)


def run_locality(device="cuda:3", n_demos=5, n_test=10, output_dir="results/phase5"):
    logger = setup_logging(output_dir)
    logger.info("Phase 5: Locality Analysis")
    logger.info(f"Start: {datetime.now().isoformat()}")

    with open("results/phase2/localization_results.json") as f:
        p2 = json.load(f)
    opt_layer = p2["optimal"]["layer"]
    opt_pos_type = p2["optimal"]["position"]
    n_layers = p2["metadata"]["n_layers"]
    logger.info(f"Optimal layer={opt_layer}, n_layers={n_layers}")

    model = load_model(device=device)
    tasks = {name: TaskRegistry.get(name) for name in INCLUDED_TASKS}

    # Layers to test: every layer from 0 to n_layers-1
    test_layers = list(range(n_layers))
    logger.info(f"Testing layers: {test_layers}")

    all_results = []

    for source_name, target_name in LOCALITY_PAIRS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Pair: {source_name} → {target_name}")
        logger.info(f"{'='*50}")

        source_task = tasks[source_name]
        target_task = tasks[target_name]
        source_demos = source_task.generate_demos(n_demos)
        target_demos = target_task.generate_demos(n_demos)
        test_inputs = target_task.generate_test_inputs(n_test)

        # Extract source vectors at ALL layers
        logger.info(f"  Extracting source vectors at all {n_layers} layers...")
        source_vectors_per_layer = {}
        source_test = source_task.generate_test_inputs(n_test)
        for layer in test_layers:
            vecs = []
            for ti in source_test:
                prompt = source_task.format_prompt(source_demos, ti)
                pos_idx = get_position_index(model, prompt, opt_pos_type)
                acts = extract_activations(model, prompt, layers=[layer], position=pos_idx)
                vecs.append(acts[layer])
            source_vectors_per_layer[layer] = torch.stack(vecs).mean(dim=0)

        # Transplant at each layer
        pair_results = {"source": source_name, "target": target_name, "layer_results": []}

        for layer in test_layers:
            source_vec = source_vectors_per_layer[layer]
            transfer_count = 0

            for ti in test_inputs:
                prompt = target_task.format_prompt(target_demos, ti)
                pos_idx = get_position_index(model, prompt, opt_pos_type)
                output = transplant_and_generate(model, prompt, source_vec, layer, pos_idx)
                try:
                    if source_task.score_output(ti, output) == "correct":
                        transfer_count += 1
                except (ValueError, TypeError, KeyError):
                    pass

            transfer_rate = transfer_count / n_test
            pair_results["layer_results"].append({
                "layer": layer, "transfer_rate": transfer_rate,
            })
            logger.info(f"  Layer {layer:2d}: transfer_rate={transfer_rate:.2f}")

        # Compute localization index: variance of transfer rates
        rates = [lr["transfer_rate"] for lr in pair_results["layer_results"]]
        pair_results["localization_index"] = float(np.var(rates))
        pair_results["peak_layer"] = int(np.argmax(rates))
        pair_results["peak_rate"] = float(max(rates))
        logger.info(f"  Peak: layer={pair_results['peak_layer']}, "
                   f"rate={pair_results['peak_rate']:.2f}, "
                   f"localization_idx={pair_results['localization_index']:.4f}")

        all_results.append(pair_results)

    # ── Save ─────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "optimal_layer": opt_layer, "optimal_position": opt_pos_type,
            "n_layers": n_layers, "test_layers": test_layers,
            "pairs": LOCALITY_PAIRS, "n_test": n_test,
            "timestamp": datetime.now().isoformat(),
        },
        "pair_results": all_results,
    }

    with open(Path(output_dir) / "locality_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV: layer-wise transfer rates
    with open(Path(output_dir) / "locality_curves.csv", "w") as f:
        f.write("pair,source_regime,target_regime,layer,transfer_rate\n")
        for pr in all_results:
            pair = f"{pr['source']}_{pr['target']}"
            src_regime = tasks[pr['source']].regime
            tgt_regime = tasks[pr['target']].regime
            for lr in pr["layer_results"]:
                f.write(f"{pair},{src_regime},{tgt_regime},{lr['layer']},{lr['transfer_rate']}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Phase 5 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--output-dir", default="results/phase5")
    args = parser.parse_args()
    run_locality(device=args.device, n_test=args.n_test, output_dir=args.output_dir)
