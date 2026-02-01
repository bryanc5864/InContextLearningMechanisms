#!/usr/bin/env python3
"""Experiment 19: Formal Template Similarity Metric.

Define structural features of task templates, compute pairwise similarity
for all task pairs, and correlate with transfer rates from exp8/exp18.
Report R^2, p-value, and bootstrap CI.
"""

import json
import sys
import os
import logging
import math
import re
import itertools
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
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
            logging.FileHandler(os.path.join(output_dir, "exp19.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_output_features(task, n_samples=50):
    """Extract structural features from a task's outputs.

    Returns a dict of feature name -> value (numeric).
    """
    demos = task.generate_demos(n_samples)
    outputs = [d[1] for d in demos]
    inputs = [d[0] for d in demos]

    # Feature 1: Average output token count (whitespace-split)
    avg_out_words = np.mean([len(o.split()) for o in outputs])

    # Feature 2: Average output character count
    avg_out_chars = np.mean([len(o) for o in outputs])

    # Feature 3: Average input token count
    avg_in_words = np.mean([len(i.split()) for i in inputs])

    # Feature 4: Output is numeric (fraction of outputs that are purely digits)
    numeric_frac = np.mean([1 if o.strip().lstrip("-").isdigit() else 0 for o in outputs])

    # Feature 5: Output is single word (no spaces)
    single_word_frac = np.mean([1 if len(o.split()) == 1 else 0 for o in outputs])

    # Feature 6: Output contains punctuation
    has_punct_frac = np.mean([
        1 if re.search(r"[^\w\s]", o) else 0 for o in outputs
    ])

    # Feature 7: Case pattern — fraction of outputs that are all uppercase
    all_upper_frac = np.mean([1 if o.isupper() else 0 for o in outputs])

    # Feature 8: Case pattern — fraction of outputs that are all lowercase
    all_lower_frac = np.mean([1 if o.islower() else 0 for o in outputs])

    # Feature 9: Repetition structure — output == input
    identity_frac = np.mean([1 if o.strip() == i.strip() else 0
                             for i, o in zip(inputs, outputs)])

    # Feature 10: Output length relative to input
    len_ratio = np.mean([
        len(o) / max(len(i), 1) for i, o in zip(inputs, outputs)
    ])

    return {
        "avg_out_words": float(avg_out_words),
        "avg_out_chars": float(avg_out_chars),
        "avg_in_words": float(avg_in_words),
        "numeric_frac": float(numeric_frac),
        "single_word_frac": float(single_word_frac),
        "has_punct_frac": float(has_punct_frac),
        "all_upper_frac": float(all_upper_frac),
        "all_lower_frac": float(all_lower_frac),
        "identity_frac": float(identity_frac),
        "len_ratio": float(len_ratio),
    }


def feature_vector(features):
    """Convert feature dict to numpy array (sorted keys for consistency)."""
    return np.array([features[k] for k in sorted(features.keys())])


def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))


# ─────────────────────────────────────────────────────────────────────────────
# Transfer rate loading
# ─────────────────────────────────────────────────────────────────────────────

def load_transfer_rates(results_dir):
    """Load transfer rates from exp8 results.

    Returns dict: (source, target) -> best transfer rate.
    """
    rates = {}

    # Try exp8
    exp8_path = Path(results_dir) / "exp8" / "multi_position_results.json"
    if exp8_path.exists():
        with open(exp8_path) as f:
            data = json.load(f)
        for pair in data.get("pair_results", []):
            src, tgt = pair["source"], pair["target"]
            best = max(
                (c.get("transfer_rate", 0) for c in pair.get("conditions", {}).values()),
                default=0,
            )
            rates[(src, tgt)] = best

    # Try exp29 (expanded transfer matrix — all 56 pairs)
    exp29_path = Path(results_dir) / "exp29" / "expanded_transfer_results.json"
    if exp29_path.exists():
        with open(exp29_path) as f:
            data = json.load(f)
        for pair in data.get("pair_results", []):
            src, tgt = pair["source"], pair["target"]
            key = (src, tgt)
            rate = pair.get("transfer_rate", 0)
            # Prefer exp29 data (all_demo at optimal layer) over exp8
            rates[key] = rate

    # Also try model-specific directories
    for model_dir in Path(results_dir).iterdir():
        if not model_dir.is_dir():
            continue
        p = model_dir / "exp8" / "multi_position_results.json"
        if p.exists() and p != exp8_path:
            with open(p) as f:
                data = json.load(f)
            for pair in data.get("pair_results", []):
                src, tgt = pair["source"], pair["target"]
                best = max(
                    (c.get("transfer_rate", 0) for c in pair.get("conditions", {}).values()),
                    default=0,
                )
                key = (src, tgt)
                if key not in rates:
                    rates[key] = best

    return rates


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def pearson_r(x, y):
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    if n < 3:
        return 0, 1.0
    xm = x - x.mean()
    ym = y - y.mean()
    r = np.sum(xm * ym) / (np.sqrt(np.sum(xm**2)) * np.sqrt(np.sum(ym**2)) + 1e-12)
    # t-test for significance
    t_stat = r * math.sqrt((n - 2) / (1 - r**2 + 1e-12))
    # Approximate p-value from t-distribution (two-tailed)
    # Using normal approximation for large n
    from scipy import stats as scipy_stats
    try:
        p_val = 2 * scipy_stats.t.sf(abs(t_stat), df=n - 2)
    except ImportError:
        p_val = float("nan")
    return float(r), float(p_val)


def bootstrap_r_ci(x, y, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap CI for Pearson r."""
    rng = np.random.RandomState(seed)
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = pearson_r(x[idx], y[idx])
        rs.append(r)
    rs = sorted(rs)
    alpha = (1 - ci) / 2
    lo = rs[int(alpha * n_boot)]
    hi = rs[int((1 - alpha) * n_boot)]
    return lo, hi


def run_template_similarity(results_dir="results", output_dir="results/exp19"):
    logger = setup_logging(output_dir)
    logger.info("Experiment 19: Template Similarity Metric")
    logger.info(f"Start: {datetime.now().isoformat()}")

    # Extract features for all tasks
    task_features = {}
    for name in INCLUDED_TASKS:
        task = TaskRegistry.get(name)
        feats = extract_output_features(task)
        task_features[name] = feats
        logger.info(f"  {name}: {feats}")

    # Compute pairwise similarity for all ordered pairs
    all_pairs = list(itertools.permutations(INCLUDED_TASKS, 2))
    logger.info(f"\nTotal pairs: {len(all_pairs)}")

    similarity_data = []
    for src, tgt in all_pairs:
        fv_src = feature_vector(task_features[src])
        fv_tgt = feature_vector(task_features[tgt])
        cos_sim = cosine_similarity(fv_src, fv_tgt)
        euc_dist = euclidean_distance(fv_src, fv_tgt)
        similarity_data.append({
            "source": src,
            "target": tgt,
            "cosine_similarity": round(cos_sim, 4),
            "euclidean_distance": round(euc_dist, 4),
        })

    # Load transfer rates
    transfer_rates = load_transfer_rates(results_dir)
    logger.info(f"\nLoaded transfer rates for {len(transfer_rates)} pairs")

    # Match similarities with transfer rates
    matched_sim = []
    matched_transfer = []
    matched_pairs = []

    for entry in similarity_data:
        key = (entry["source"], entry["target"])
        if key in transfer_rates:
            matched_sim.append(entry["cosine_similarity"])
            matched_transfer.append(transfer_rates[key])
            matched_pairs.append(key)
            entry["transfer_rate"] = transfer_rates[key]

    logger.info(f"Matched pairs: {len(matched_pairs)}")

    # Correlation analysis
    if len(matched_pairs) >= 3:
        r, p_val = pearson_r(matched_sim, matched_transfer)
        r_sq = r ** 2
        ci_lo, ci_hi = bootstrap_r_ci(matched_sim, matched_transfer)

        logger.info(f"\n{'='*60}")
        logger.info("CORRELATION: Template Similarity vs Transfer Rate")
        logger.info(f"{'='*60}")
        logger.info(f"  Pearson r:   {r:.4f}")
        logger.info(f"  R-squared:   {r_sq:.4f}")
        logger.info(f"  p-value:     {p_val:.6f}")
        logger.info(f"  95% CI (r):  [{ci_lo:.4f}, {ci_hi:.4f}]")

        for sim_val, tr_val, (s, t) in zip(matched_sim, matched_transfer, matched_pairs):
            logger.info(f"    {s:20s} -> {t:20s}: sim={sim_val:.3f}, transfer={tr_val:.3f}")
    else:
        r, r_sq, p_val = 0, 0, 1.0
        ci_lo, ci_hi = 0, 0
        logger.warning("Not enough matched pairs for correlation analysis.")

    # Save
    results = {
        "metadata": {
            "n_tasks": len(INCLUDED_TASKS),
            "n_pairs_total": len(all_pairs),
            "n_pairs_matched": len(matched_pairs),
            "feature_names": sorted(task_features[INCLUDED_TASKS[0]].keys()),
            "timestamp": datetime.now().isoformat(),
        },
        "task_features": task_features,
        "pairwise_similarity": similarity_data,
        "correlation": {
            "pearson_r": round(r, 4),
            "r_squared": round(r_sq, 4),
            "p_value": round(p_val, 6),
            "bootstrap_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        },
    }

    with open(Path(output_dir) / "template_similarity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV: pairwise similarity
    with open(Path(output_dir) / "pairwise_similarity.csv", "w") as f:
        f.write("source,target,cosine_similarity,euclidean_distance,transfer_rate\n")
        for entry in similarity_data:
            tr = entry.get("transfer_rate", "")
            f.write(f"{entry['source']},{entry['target']},"
                    f"{entry['cosine_similarity']},{entry['euclidean_distance']},{tr}\n")

    # CSV: task features
    feat_names = sorted(task_features[INCLUDED_TASKS[0]].keys())
    with open(Path(output_dir) / "task_features.csv", "w") as f:
        f.write("task," + ",".join(feat_names) + "\n")
        for name in INCLUDED_TASKS:
            vals = [str(task_features[name][k]) for k in feat_names]
            f.write(f"{name}," + ",".join(vals) + "\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Experiment 19 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results",
                        help="Root results directory (for loading transfer rates)")
    parser.add_argument("--output-dir", default="results/exp19")
    args = parser.parse_args()
    run_template_similarity(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )
