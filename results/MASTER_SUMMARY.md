# Master Summary: Characterizing the Computational Interface of In-Context Learning

**Date:** 2026-01-28
**Model:** Llama-3.2-3B-Instruct (28 layers, d_model=3072)
**Tasks:** 8 (uppercase, first_letter, repeat_word, length, linear_2x, sentiment, antonym, pattern_completion)

---

## Phase 1: Baseline Characterization

All 8 tasks achieve high ICL accuracy (96-100%) with 5-shot prompting. Tasks span 5 hypothesized regimes: procedural, counting, gd_like, bayesian, retrieval, and induction.

## Phase 2: Representation Localization

Probed activations at 3 positions across all 28 layers:

- **Demo positions (last_demo_token, separator_after_demo):** 100% probe accuracy at ALL layers — trivially separable because different tasks have different demo text.
- **First query token:** Peak probe accuracy 83% at layer 12, building gradually from 45% at layer 0. This is the informationally interesting position because all tasks share the same structural format at the query.

**Chosen intervention site:** Layer 14, last_demo_token (highest probe confidence at a causally upstream position).

## Phase 3: Transplantation (Modularity Test)

**Central negative result:** Cross-task transplantation at (layer 14, last_demo_token) produces **0% task transfer** across all 56 off-diagonal task pairs. The target task is preserved at ~97% despite the intervention.

**Critical control finding:** Zero ablation and random ablation at the same site also produce **100% task accuracy** — meaning the activation at this single position is not causally necessary for task execution.

Same-task transplant: 98.8% (sanity check passes).

## Phase 4: Compositionality Analysis

Interpolation between task vectors shows **no behavioral transition** — task_A persists at 100% across all alpha values from 0.0 to 1.0 for all 6 pairs tested. Vector arithmetic similarly has no effect. This is consistent with Phase 3's finding that the intervention site lacks causal power.

## Phase 5: Locality Sweep

Extended transplantation to **ALL 28 layers** (still at last_demo_token position): 0% transfer at every layer for all 4 pairs tested. This definitively shows that single-position intervention is insufficient at any depth.

## Phase 6: Task Ontology

Analysis of task vector geometry reveals meaningful structure despite the vectors' causal irrelevance:

- **Procedural tasks cluster tightly:** uppercase ↔ repeat_word cos=0.935, all procedural pairs > 0.86
- **Regime structure is statistically significant:** Within-regime similarity 0.894 vs between-regime 0.699 (permutation p=0.005)
- **PCA:** PC1 separates numeric/abstract tasks from string/procedural tasks; PC2 isolates sentiment
- **Sentiment is most isolated** (lowest cosine similarity to all other tasks)

## Phase 7: Trajectory Analysis

Representational change at last_demo_token reveals three processing phases:

1. **Early layers (0-8):** High change (cos dist 0.16-0.26) — active representation building
2. **Late layers (16-26):** Low change (0.05-0.11) — representations stabilize
3. **Final layer (26→27):** Spike (0.36) — unembedding transformation

---

## Key Conclusions

### 1. Task identity in ICL is distributed, not bottlenecked

The model distributes task information across ALL demo token positions. Overriding a single token position (even across all layers) cannot redirect task behavior. This falsifies the hypothesis of a single (layer, position) "task identity bottleneck."

### 2. Correlational ≠ Causal

Linear probes at demo positions achieve 100% accuracy (correlational), but the same activations are not causally necessary (zero ablation has no effect). This dissociation between probing success and causal importance is an important methodological warning for interpretability research.

### 3. Representational structure exists but is redundant

Task vectors encode meaningful regime structure (procedural tasks cluster, sentiment is isolated, p=0.005 for regime clustering). But this structure is one of many redundant copies across the token sequence — no single copy is necessary or sufficient.

### 4. Recommended next steps

To achieve actual task transfer, future experiments should:
- **Multi-position intervention:** Transplant activations at ALL demo token positions simultaneously
- **Attention manipulation:** Modify attention weights/patterns rather than residual stream values
- **Prompt-level intervention:** Construct hybrid prompts combining demos from different tasks
- **Patch-based methods:** Use activation patching (clean-run vs corrupted-run difference) rather than mean vector transplantation

---

## File Structure

```
results/
├── MASTER_SUMMARY.md          ← This file
├── phase1/
│   ├── baseline_results.json
│   ├── baseline_summary.csv
│   ├── SUMMARY.md
│   └── phase1.log
├── phase2/
│   ├── localization_results.json
│   ├── probe_accuracy.csv
│   ├── activations_cache.pkl
│   ├── SUMMARY.md
│   └── phase2.log
├── phase3/
│   ├── transplant_results.json
│   ├── transfer_matrix.csv
│   ├── task_vectors.pkl
│   ├── SUMMARY.md
│   └── phase3.log
├── phase4/
│   ├── interpolation_results.json
│   ├── interpolation_curves.csv
│   ├── SUMMARY.md
│   └── phase4.log
├── phase5/
│   ├── locality_results.json
│   ├── locality_curves.csv
│   ├── SUMMARY.md
│   └── phase5.log
├── phase6/
│   ├── ontology_results.json
│   ├── similarity_matrix.csv
│   ├── pca_embedding.csv
│   ├── SUMMARY.md
│   └── phase6.log
└── phase7/
    ├── trajectory_results.json
    ├── probe_trajectory.csv
    ├── representational_change.csv
    ├── SUMMARY.md
    └── phase7.log
```
