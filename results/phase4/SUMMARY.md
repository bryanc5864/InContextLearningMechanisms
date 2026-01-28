# Phase 4: Compositionality Analysis — Results

**Date:** 2026-01-28
**Intervention site:** Layer 14, last_demo_token
**N test inputs:** 10 per condition

## Key Findings

### 1. Linear Interpolation: No Transition Observed

For all 6 task pairs tested, interpolating between task vectors v_A and v_B at (layer 14, last_demo_token) produces **no behavioral change**. The original task (task_A, whose demos are in the prompt) persists at 100% accuracy across all alpha values from 0.0 to 1.0:

| Pair | Task A at α=0.0 | Task A at α=1.0 | Transition? |
|------|-----------------|-----------------|-------------|
| uppercase ↔ first_letter | 1.00 | 1.00 | No |
| uppercase ↔ sentiment | 1.00 | 1.00 | No |
| linear_2x ↔ length | 1.00 | 1.00 | No |
| sentiment ↔ antonym | 1.00 | 1.00 | No |
| pattern_completion ↔ uppercase | 1.00 | 1.00 | No |
| linear_2x ↔ sentiment | 1.00 | 1.00 | No |

Transition sharpness = 0.000 for all pairs.

### 2. Vector Arithmetic: No Effect

Testing v_C + (v_A - v_B) — the shifted vector has no effect on behavior:

| Operation | Target task rate | Shift task rate |
|-----------|-----------------|-----------------|
| repeat_word + (uppercase - first_letter) | 1.00 | 0.00 |
| linear_2x + (sentiment - antonym) | 1.00 | 0.00 |
| sentiment + (length - linear_2x) | 1.00 | 0.00 |

### 3. Interpretation

These results are consistent with Phase 3's finding: **the intervention at (layer 14, last_demo_token) is not causally relevant for task behavior**. The model's task execution is robust to arbitrary perturbation at this site because:

1. Task information flows through multiple token positions simultaneously
2. A single-position intervention cannot override the distributed task representation
3. The remaining demo tokens provide sufficient redundant task signal

These compositionality experiments will need to be revisited once Phase 5 identifies a causally effective intervention site (if one exists at this single-position level).

## Files

- `interpolation_results.json` — Full results with per-example details
- `interpolation_curves.csv` — Alpha × task probability curves
- `phase4.log` — Full execution log
