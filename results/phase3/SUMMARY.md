# Phase 3: Transplantation Experiments — Results

**Date:** 2026-01-28
**Model:** Llama-3.2-3B-Instruct (28 layers, d_model=3072)
**Intervention site:** Layer 14, last_demo_token position
**N test inputs per pair:** 10

## Key Findings

### 1. Cross-Task Transfer Matrix

Transplanting the mean task vector from source → target at (layer 14, last_demo_token):

| Source ↓ \ Target → | upper | first | repeat | length | lin2x | sent | anton | pattern |
|---------------------|-------|-------|--------|--------|-------|------|-------|---------|
| uppercase           | **1.00** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| first_letter        | 0.00 | **1.00** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| repeat_word         | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| length              | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 | 0.00 | 0.00 |
| linear_2x           | 0.00 | 0.00 | 0.00 | 0.00 | **0.90** | 0.00 | 0.00 | 0.00 |
| sentiment           | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 |
| antonym             | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 |
| pattern_completion  | 0.00 | 0.00 | **1.00*** | 0.00 | 0.00 | 0.00 | 0.00 | **1.00** |

*** = unexpected cross-task transfer

### 2. Summary Statistics

| Metric | Value |
|--------|-------|
| Mean same-task transfer (diagonal) | 98.8% |
| Mean cross-task transfer (off-diagonal) | 1.8% |
| Mean disruption rate (off-diagonal) | 0.5% |
| Mean preservation rate (off-diagonal) | 96.6% |

### 3. Control Conditions

All controls show 100% accuracy — the intervention site has NO causal effect:

| Task | Baseline | Zero Ablation | Random Ablation |
|------|----------|---------------|-----------------|
| uppercase | 1.00 | 1.00 | 1.00 |
| first_letter | 1.00 | 1.00 | 1.00 |
| repeat_word | 1.00 | 1.00 | 1.00 |
| length | 1.00 | 1.00 | 1.00 |
| linear_2x | 1.00 | 1.00 | 1.00 |
| sentiment | 1.00 | 1.00 | 1.00 |
| antonym | 1.00 | 1.00 | 1.00 |
| pattern_completion | 1.00 | 1.00 | 1.00 |

### 4. Interpretation

**The intervention at layer 14, last_demo_token is NOT causally effective:**

- Zero ablation (zeroing the activation) and random ablation (replacing with random noise) both produce 100% task accuracy. This means the model does not rely on information at this specific (layer, position) to execute the task.
- Cross-task transplant shows 0% transfer (with one exception) — the target task is preserved despite replacing the activation.
- Same-task transplant "works" (98.8%) but this is trivially expected since the model was already doing the right task.

**Why this happens:**
- Task identity information is likely **redundantly encoded** across many token positions. Overriding a single position (last_demo_token) at one layer is insufficient to change behavior.
- The probing results from Phase 2 showed 100% accuracy at demo positions because different tasks have different demo text — this is a **correlational signal, not a causal one**.
- The model likely propagates task identity through attention across ALL demo tokens, not just the last one.

**Anomaly: pattern_completion → repeat_word = 100% transfer:**
- This likely reflects scoring overlap rather than genuine transfer. Pattern completion repeats alternating elements (A B A B A → B), and repeat_word also repeats the input word. The generated output may happen to match repeat_word's scoring criteria.

### 5. Implications for Phase 5

Phase 5 (locality sweep) is now critical: we need to test interventions at **all layers** and potentially **all positions** to find which (layer, position) pairs are actually causally relevant for task behavior.

## Files

- `transplant_results.json` — Full results with per-example details
- `transfer_matrix.csv` — Source × target transfer rates
- `task_vectors.pkl` — Cached mean task vectors
- `phase3.log` — Full execution log
