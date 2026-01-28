# Phase 2: Representation Localization — Results

**Date:** 2026-01-28
**Model:** Llama-3.2-3B-Instruct (28 layers, d_model=3072)
**Method:** NearestCentroid probe, 5-fold stratified shuffle, 400 samples (50 per task × 8 tasks)

## Key Findings

### 1. Probe Accuracy by Position

| Position              | Best Layer | Best Accuracy | Pattern              |
|-----------------------|-----------|---------------|----------------------|
| last_demo_token       | ALL (0-27) | 100.0%       | Perfect at every layer |
| separator_after_demo  | ALL (0-27) | 100.0%       | Perfect at every layer |
| first_query_token     | 12         | 82.9%        | Peaks mid-network     |

### 2. First Query Token Trajectory (most informative)

Task identity at the query token position builds up gradually:

| Layers | Accuracy Range |
|--------|---------------|
| 0-6    | 45-53%        |
| 7-9    | 63-73%        |
| 10-12  | 79-83% (PEAK) |
| 13     | 76%           |
| 14-27  | 49-60% (drops)|

### 3. Interpretation

- **Demo positions (p1, p2) are trivially separable**: Different tasks have different demo text, so the residual stream at these positions contains literal task-specific tokens. 100% accuracy from layer 0 is expected and not a deep finding about task encoding.
- **Query position (p3) is the informative probe target**: Here, all tasks share the same structural format ("Input: {word}\nOutput:") and the model must have internally computed task identity. The peak at **layer 12** (mid-network) suggests task information is propagated from demos to the query position through attention in layers 7-12.
- **Late-layer drop at query position**: Accuracy drops after layer 13 at the query token, suggesting the representation shifts from "task identity" to "output preparation" in later layers.

### 4. Optimal Intervention Coordinates

For transplantation experiments, we use:
- **Position:** `last_demo_token` (strongest task signal, causally upstream of generation)
- **Layer:** 14 (mid-to-late network, after task identity crystallization)

Rationale: While all layers probe perfectly at demo positions, mid-network activations are more likely to contain abstract task representations rather than raw token embeddings.

## Files

- `localization_results.json` — Full results with probe accuracies
- `probe_accuracy.csv` — Layer × position accuracy table
- `activations_cache.pkl` — Cached activations (3.2GB)
- `phase2.log` — Full execution log
