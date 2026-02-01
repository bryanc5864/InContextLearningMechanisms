#!/usr/bin/env python3
"""Cross-Model Comparison.

Read results from all model subdirectories, normalize layers to fractional
depth, and produce comparison tables and CSVs.
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Models we expect to find results for
MODEL_DIRS = [
    "llama-3.2-3b-instruct",
    "llama-3.2-1b-instruct",
    "qwen2.5-1.5b-instruct",
    "gemma-2-2b-it",
]

MODEL_LAYERS = {
    "llama-3.2-3b-instruct": 28,
    "llama-3.2-1b-instruct": 16,
    "qwen2.5-1.5b-instruct": 28,
    "gemma-2-2b-it": 26,
}


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compare_baseline(results_root):
    """Compare exp1 baseline accuracies across models."""
    print("=" * 70)
    print("BASELINE ACCURACY COMPARISON (exp1)")
    print("=" * 70)

    rows = []
    all_tasks = set()

    for model_dir in MODEL_DIRS:
        data = load_json(results_root / model_dir / "exp1" / "baseline_results.json")
        if data is None:
            continue
        tasks = data.get("tasks", {})
        all_tasks.update(tasks.keys())
        row = {"model": model_dir}
        for task_name, task_data in tasks.items():
            row[task_name] = task_data.get("accuracy", 0)
        rows.append(row)

    if not rows:
        print("  No exp1 results found.\n")
        return

    all_tasks = sorted(all_tasks)
    # Header
    header = f"{'Model':30s}" + "".join(f"{t:>12s}" for t in all_tasks)
    print(header)
    print("-" * len(header))

    for row in rows:
        line = f"{row['model']:30s}"
        for t in all_tasks:
            acc = row.get(t, 0)
            line += f"{acc:12.2f}"
        print(line)

    print()
    return rows


def compare_patching(results_root):
    """Compare exp11 peak disruption layers across models (normalized depth)."""
    print("=" * 70)
    print("ACTIVATION PATCHING COMPARISON (exp11) — Peak Disruption")
    print("=" * 70)

    csv_rows = []

    for model_dir in MODEL_DIRS:
        n_layers = MODEL_LAYERS.get(model_dir, 28)
        data = load_json(results_root / model_dir / "exp11" / "patching_results.json")
        if data is None:
            continue

        print(f"\n  {model_dir} ({n_layers} layers):")

        for task_result in data.get("task_results", []):
            task_name = task_result["task"]
            for pos_type, pos_data in task_result.get("position_results", {}).items():
                layers_data = pos_data.get("layers", {})
                best_layer = None
                best_disruption = -1
                for layer_str, ldata in layers_data.items():
                    d = ldata.get("disruption", 0)
                    if d > best_disruption:
                        best_disruption = d
                        best_layer = int(layer_str)

                if best_layer is not None:
                    frac = best_layer / n_layers
                    print(f"    {task_name:20s} {pos_type:20s}: "
                          f"peak layer {best_layer:2d} (depth={frac:.2f}), "
                          f"disruption={best_disruption:.3f}")
                    csv_rows.append({
                        "model": model_dir,
                        "n_layers": n_layers,
                        "task": task_name,
                        "position": pos_type,
                        "peak_layer": best_layer,
                        "peak_depth_fraction": round(frac, 3),
                        "peak_disruption": round(best_disruption, 3),
                    })

    print()
    return csv_rows


def compare_transfer(results_root):
    """Compare exp8 transfer rates across models."""
    print("=" * 70)
    print("MULTI-POSITION TRANSFER COMPARISON (exp8)")
    print("=" * 70)

    csv_rows = []

    for model_dir in MODEL_DIRS:
        n_layers = MODEL_LAYERS.get(model_dir, 28)
        data = load_json(results_root / model_dir / "exp8" / "multi_position_results.json")
        if data is None:
            continue

        print(f"\n  {model_dir} ({n_layers} layers):")

        for pair_result in data.get("pair_results", []):
            source = pair_result["source"]
            target = pair_result["target"]
            # Find best condition for this pair
            best_key = None
            best_transfer = -1
            for key, cond_data in pair_result.get("conditions", {}).items():
                tr = cond_data.get("transfer_rate", 0)
                if tr > best_transfer:
                    best_transfer = tr
                    best_key = key

            if best_key:
                layer = pair_result["conditions"][best_key].get("layer", "?")
                condition = pair_result["conditions"][best_key].get("condition", "?")
                frac = layer / n_layers if isinstance(layer, int) else 0
                print(f"    {source:15s} -> {target:15s}: "
                      f"best={best_transfer:.2f} at {best_key} (depth={frac:.2f})")
                csv_rows.append({
                    "model": model_dir,
                    "source": source,
                    "target": target,
                    "best_transfer_rate": round(best_transfer, 3),
                    "best_condition": best_key,
                    "best_layer": layer,
                    "best_depth_fraction": round(frac, 3),
                })

    print()
    return csv_rows


def compare_instance(results_root):
    """Compare exp13 instance-level transfer across models."""
    print("=" * 70)
    print("INSTANCE-LEVEL TRANSFER COMPARISON (exp13)")
    print("=" * 70)

    csv_rows = []

    for model_dir in MODEL_DIRS:
        data = load_json(results_root / model_dir / "exp13" / "instance_analysis_results.json")
        if data is None:
            continue

        print(f"\n  {model_dir}:")
        for pair in data.get("instance_results", []):
            src = pair["source"]
            tgt = pair["target"]
            tr = pair.get("transfer_rate", 0)
            print(f"    {src:15s} -> {tgt:15s}: transfer_rate={tr:.2f}")
            csv_rows.append({
                "model": model_dir,
                "source": src,
                "target": tgt,
                "transfer_rate": round(tr, 3),
            })

    print()
    return csv_rows


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in fieldnames) + "\n")
    print(f"  Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-model result comparison")
    parser.add_argument("--results-dir", default="results",
                        help="Root results directory")
    parser.add_argument("--output-dir", default="results/cross_model",
                        help="Output directory for comparison CSVs")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = compare_baseline(results_root)
    patching_rows = compare_patching(results_root)
    transfer_rows = compare_transfer(results_root)
    instance_rows = compare_instance(results_root)

    print("=" * 70)
    print("SAVING CSVs")
    print("=" * 70)

    if baseline_rows:
        # Flatten baseline rows into per-task columns
        all_tasks = sorted(set(k for r in baseline_rows for k in r if k != "model"))
        write_csv(output_dir / "baseline_comparison.csv", baseline_rows,
                  fieldnames=["model"] + all_tasks)

    write_csv(output_dir / "patching_comparison.csv", patching_rows)
    write_csv(output_dir / "transfer_comparison.csv", transfer_rows)
    write_csv(output_dir / "instance_comparison.csv", instance_rows)

    # Summary: do peak disruption layers align across models?
    if patching_rows:
        print("\n" + "=" * 70)
        print("CROSS-MODEL PEAK DEPTH SUMMARY")
        print("=" * 70)
        fracs = [r["peak_depth_fraction"] for r in patching_rows if r["peak_disruption"] > 0.05]
        if fracs:
            print(f"  Mean peak depth fraction: {np.mean(fracs):.3f}")
            print(f"  Std:                      {np.std(fracs):.3f}")
            print(f"  Range:                    [{min(fracs):.3f}, {max(fracs):.3f}]")
        print()


if __name__ == "__main__":
    main()
