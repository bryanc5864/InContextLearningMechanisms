# Phase 7: Trajectory Analysis — Results

**Date:** 2026-01-28
**Position:** last_demo_token
**N test inputs per task:** 20 (160 total)
**Probe classifier:** LogisticRegression (C=1.0)

## Key Findings

### 1. Probe Accuracy Trajectory

100% accuracy at ALL 28 layers (0-27). This is trivially expected at demo positions — different tasks have different demo text, making classification easy from the raw token embeddings at layer 0.

Crystallization layer: 0 (all thresholds: 80%, 90%, 95%)

### 2. Layer-to-Layer Representational Change

This is the most informative measure — how much the residual stream changes between consecutive layers:

| Layers | Cosine Distance | Interpretation |
|--------|----------------|----------------|
| 0→1 | 0.212 | High — early processing |
| 1→2 | 0.162 | Moderate |
| 2→3 | 0.220 | High |
| 3→4 | 0.223 | High |
| 4→5 | 0.217 | High |
| 5→6 | **0.257** | **Peak early change** |
| 6→7 | 0.254 | High |
| 7→8 | 0.233 | High |
| 8→9 | 0.198 | Transitioning |
| 9→10 | 0.186 | Transitioning |
| 10→11 | 0.236 | Bump |
| 11→12 | 0.180 | Moderate |
| 12→13 | 0.172 | Moderate |
| 13→14 | 0.197 | Moderate |
| 14→15 | 0.160 | Declining |
| 15→16 | 0.108 | Low |
| 16→17 | 0.106 | Low |
| 17→18 | 0.089 | Low |
| 18→19 | 0.081 | Low |
| 19→20 | 0.066 | Very low |
| 20→21 | 0.086 | Low |
| 21→22 | **0.053** | **Minimum** |
| 22→23 | 0.060 | Very low |
| 23→24 | 0.083 | Low |
| 24→25 | 0.066 | Very low |
| 25→26 | 0.084 | Low |
| 26→27 | **0.359** | **Spike — final layer** |

### 3. Interpretation

**Three-phase processing:**

1. **Early layers (0-8):** High representational change (0.16-0.26). The model is actively transforming input representations — likely building contextual embeddings and initial task-relevant features.

2. **Middle layers (9-15):** Moderate change (0.16-0.20). Task identity is being consolidated; the representational trajectory is stabilizing.

3. **Late layers (16-26):** Low change (0.05-0.11). Representations are largely stable — the model has computed its internal representation and is mainly refining output preparation.

4. **Final layer (26→27):** Dramatic spike (0.36) — the unembedding layer radically reshapes the representation to produce token logits.

### 4. Per-Regime Analysis

Only the procedural regime (uppercase, first_letter, repeat_word) had multiple tasks for within-regime probing. Within-regime classification was 100% at all layers, indicating that even within the same regime, tasks have distinct representations at demo positions.

## Files

- `trajectory_results.json` — Full results
- `probe_trajectory.csv` — Layer × accuracy table
- `representational_change.csv` — Layer-to-layer cosine distances
- `phase7.log` — Full execution log
