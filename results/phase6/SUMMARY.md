# Phase 6: Task Ontology — Results

**Date:** 2026-01-28
**Task vectors:** From Phase 3 (layer 14, last_demo_token, 3072-dim mean vectors)

## Key Findings

### 1. Cosine Similarity Matrix

|  | antonym | first_l | length | lin2x | pattern | repeat | sentim | upper |
|--|---------|---------|--------|-------|---------|--------|--------|-------|
| antonym | 1.000 | 0.736 | 0.710 | 0.689 | 0.724 | 0.746 | 0.660 | 0.766 |
| first_letter | 0.736 | 1.000 | 0.831 | 0.625 | 0.664 | 0.861 | 0.653 | 0.886 |
| length | 0.710 | 0.831 | 1.000 | 0.634 | 0.666 | 0.861 | 0.694 | 0.880 |
| linear_2x | 0.689 | 0.625 | 0.634 | 1.000 | 0.752 | 0.646 | 0.546 | 0.673 |
| pattern | 0.724 | 0.664 | 0.666 | 0.752 | 1.000 | 0.707 | 0.568 | 0.710 |
| repeat_word | 0.746 | 0.861 | 0.861 | 0.646 | 0.707 | 1.000 | 0.648 | 0.935 |
| sentiment | 0.660 | 0.653 | 0.694 | 0.546 | 0.568 | 0.648 | 1.000 | 0.670 |
| uppercase | 0.766 | 0.886 | 0.880 | 0.673 | 0.710 | 0.935 | 0.670 | 1.000 |

### 2. Clustering Structure

**Hierarchical clustering** (average linkage on cosine distance) reveals:

1. **Tightest cluster:** repeat_word ↔ uppercase (cos=0.935) — both involve simple string operations
2. **Second cluster:** first_letter joins (cos≈0.87-0.89) — all three are procedural tasks
3. **Third:** length joins the procedural cluster (cos≈0.83-0.88) — counting is close to string operations
4. **Separate cluster:** linear_2x ↔ pattern_completion (cos=0.752) — numeric/induction tasks
5. **Outliers:** sentiment (most isolated, cos=0.55-0.69 to most tasks) and antonym

### 3. PCA Embedding

| Task | Regime | PC1 | PC2 |
|------|--------|-----|-----|
| linear_2x | gd_like | +5.28 | -0.87 |
| pattern_completion | induction | +3.81 | -0.93 |
| antonym | retrieval | +1.31 | +0.96 |
| sentiment | bayesian | -0.89 | +6.09 |
| uppercase | procedural | -2.14 | -1.57 |
| first_letter | procedural | -2.23 | -1.00 |
| repeat_word | procedural | -2.43 | -2.03 |
| length | counting | -2.72 | -0.66 |

PC1 separates numeric/abstract tasks (positive) from string/procedural tasks (negative).
PC2 separates sentiment (high positive) from everything else.

### 4. Regime Clustering Quality

| Metric | Value |
|--------|-------|
| Within-regime mean similarity | 0.894 |
| Between-regime mean similarity | 0.699 |
| Difference | 0.195 |
| Silhouette score | 0.096 |
| Permutation p-value | **0.005** |

The regime structure is **statistically significant** (p=0.005): tasks from the same hypothesized regime (procedural) have higher cosine similarity than cross-regime pairs, even though the silhouette score is low (reflecting high baseline similarity across all tasks).

### 5. Interpretation

Even though these task vectors are NOT causally effective (Phases 3-5 showed no behavioral effect from intervention), they still **encode meaningful task structure**:

- Procedural tasks cluster tightly (cos > 0.86)
- Semantic tasks (sentiment, antonym) are more isolated
- Numeric/induction tasks form their own cluster
- The regime taxonomy has statistical support (p=0.005)

This suggests that task representations at the last demo token position ARE informative (the model encodes task identity there), even though overriding a single position is insufficient to change behavior.

## Files

- `ontology_results.json` — Full results
- `similarity_matrix.csv` — Pairwise cosine similarities
- `pca_embedding.csv` — PCA coordinates with regimes
- `phase6.log` — Full execution log
