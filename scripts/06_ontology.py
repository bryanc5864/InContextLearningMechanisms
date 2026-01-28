#!/usr/bin/env python3
"""Phase 6: Task Ontology — clustering and similarity analysis.

Analyze task vector geometry: similarity matrix, hierarchical clustering,
PCA embedding, regime clustering quality.
"""

import json
import sys
import os
import logging
import pickle
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from src.tasks import TaskRegistry
from src.clustering import (
    compute_similarity_matrix, hierarchical_clustering,
    compute_regime_clustering_score, pca_embedding,
)

INCLUDED_TASKS = [
    "uppercase", "first_letter", "repeat_word", "length",
    "linear_2x", "sentiment", "antonym", "pattern_completion",
]

TASK_REGIMES = {
    "uppercase": "procedural", "first_letter": "procedural", "repeat_word": "procedural",
    "length": "counting", "linear_2x": "gd_like",
    "sentiment": "bayesian", "antonym": "retrieval",
    "pattern_completion": "induction",
}


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(output_dir, "phase6.log")), logging.StreamHandler()])
    return logging.getLogger(__name__)


def run_ontology(output_dir="results/phase6"):
    logger = setup_logging(output_dir)
    logger.info("Phase 6: Task Ontology Analysis")
    logger.info(f"Start: {datetime.now().isoformat()}")

    # Load task vectors from Phase 3
    with open("results/phase3/task_vectors.pkl", "rb") as f:
        task_vectors = pickle.load(f)
    logger.info(f"Loaded task vectors: {list(task_vectors.keys())}")

    # Filter to included tasks
    tv = {k: task_vectors[k] for k in INCLUDED_TASKS if k in task_vectors}
    logger.info(f"Using {len(tv)} task vectors")

    # ── 1. Similarity matrix ─────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("1. Pairwise cosine similarity")
    sim_matrix, names = compute_similarity_matrix(tv)

    logger.info(f"{'':>20s}" + "".join(f"{n[:8]:>10s}" for n in names))
    for i, n in enumerate(names):
        row = f"{n:>20s}" + "".join(f"{sim_matrix[i,j]:10.3f}" for j in range(len(names)))
        logger.info(row)

    # ── 2. Hierarchical clustering ───────────────────────────────────
    logger.info("\n2. Hierarchical clustering")
    Z, cluster_names = hierarchical_clustering(tv, method="average")
    logger.info(f"Linkage matrix:\n{Z}")

    # ── 3. PCA embedding ─────────────────────────────────────────────
    logger.info("\n3. PCA embedding (2D)")
    embedding, embed_names = pca_embedding(tv, n_components=2)
    for name, coords in zip(embed_names, embedding):
        regime = TASK_REGIMES[name]
        logger.info(f"  {name:20s} ({regime:12s}): PC1={coords[0]:+.4f} PC2={coords[1]:+.4f}")

    # Higher-dimensional embedding for more detail
    embedding_3d, _ = pca_embedding(tv, n_components=3)

    # ── 4. Regime clustering quality ─────────────────────────────────
    logger.info("\n4. Regime clustering quality")
    cluster_score = compute_regime_clustering_score(tv, TASK_REGIMES)
    logger.info(f"  Silhouette score: {cluster_score['silhouette_score']:.4f}")
    logger.info(f"  Permutation p-value: {cluster_score['p_value']:.4f}")
    logger.info(f"  N permutations: {cluster_score['n_permutations']}")

    # ── 5. Within-regime vs between-regime similarity ────────────────
    logger.info("\n5. Within-regime vs between-regime similarity")
    within = []
    between = []
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i >= j:
                continue
            sim = sim_matrix[i, j]
            if TASK_REGIMES[ni] == TASK_REGIMES[nj]:
                within.append(sim)
            else:
                between.append(sim)

    within_mean = float(np.mean(within)) if within else 0.0
    between_mean = float(np.mean(between)) if between else 0.0
    logger.info(f"  Within-regime mean similarity:  {within_mean:.4f} (n={len(within)})")
    logger.info(f"  Between-regime mean similarity: {between_mean:.4f} (n={len(between)})")
    logger.info(f"  Difference: {within_mean - between_mean:.4f}")

    # ── Save ─────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "tasks": INCLUDED_TASKS,
            "regimes": TASK_REGIMES,
            "timestamp": datetime.now().isoformat(),
        },
        "similarity_matrix": {
            "names": names,
            "matrix": sim_matrix.tolist(),
        },
        "clustering": {
            "linkage_matrix": Z.tolist(),
            "names": cluster_names,
        },
        "pca_embedding": {
            "2d": {name: coords.tolist() for name, coords in zip(embed_names, embedding)},
            "3d": {name: coords.tolist() for name, coords in zip(embed_names, embedding_3d)},
        },
        "regime_clustering": cluster_score,
        "regime_similarity": {
            "within_mean": within_mean,
            "between_mean": between_mean,
            "within_values": within,
            "between_values": between,
        },
    }

    with open(Path(output_dir) / "ontology_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV: similarity matrix
    with open(Path(output_dir) / "similarity_matrix.csv", "w") as f:
        f.write("task," + ",".join(names) + "\n")
        for i, n in enumerate(names):
            f.write(n + "," + ",".join(f"{sim_matrix[i,j]:.6f}" for j in range(len(names))) + "\n")

    # CSV: PCA coords
    with open(Path(output_dir) / "pca_embedding.csv", "w") as f:
        f.write("task,regime,pc1,pc2,pc3\n")
        for name, c2, c3 in zip(embed_names, embedding, embedding_3d):
            f.write(f"{name},{TASK_REGIMES[name]},{c2[0]:.6f},{c2[1]:.6f},{c3[2]:.6f}\n")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Phase 6 complete: {datetime.now().isoformat()}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/phase6")
    args = parser.parse_args()
    run_ontology(output_dir=args.output_dir)
