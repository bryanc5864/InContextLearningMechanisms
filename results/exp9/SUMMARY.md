# Experiment 9: Query Position Intervention — Results

**Date:** 2026-01-28
**Position:** first_query_token (where task identity is aggregated via attention)
**Layers tested:** 0, 4, 8, 10, 11, 12, 13, 14, 16, 20, 24, 27
**N test inputs:** 10 per pair

## Hypothesis

Phase 2 showed the query position (first token of test input) has meaningful task encoding that builds up through layers, peaking at layer 12 with 83% probe accuracy. If this aggregated task representation is causally sufficient, intervening at the query position at layer 12 should transfer task behavior.

## Key Findings

### 1. Transfer Rate = 0% at All Layers

| Layer | Mean Transfer | Interpretation |
|-------|---------------|----------------|
| 0 | 0.000 | Pre-processing |
| 4 | 0.000 | Early layers |
| 8 | 0.000 | Before peak |
| 10 | 0.000 | Rising probe accuracy |
| 11 | 0.000 | Near peak |
| **12** | **0.000** | **Peak probe accuracy layer** |
| 13 | 0.000 | Post-peak |
| 14 | 0.000 | Declining |
| 16-27 | 0.000 | Late layers |

### 2. Disruption Patterns

While transfer is zero, some interventions disrupt the target task:

| Pair | Layer Range | Preservation | Neither | Notes |
|------|-------------|--------------|---------|-------|
| sentiment → first_letter | 10-13 | 0-10% | 90-100% | Strong disruption |
| antonym → repeat_word | 4-14 | 0-30% | 70-100% | Prolonged disruption |
| linear_2x → uppercase | 13-16 | 30-70% | 30-70% | Moderate disruption |

Disruption occurs but DOES NOT redirect to source task behavior — it produces malformed outputs.

### 3. Recovery at Late Layers

All pairs show recovery to target task behavior at late layers (20-27):
- Preservation returns to 100%
- Neither rate drops to 0%

This suggests the model's output-preparation layers are robust to mid-network perturbations.

## Interpretation

**Query position intervention also fails to transfer task behavior.** This extends the Phase 3-5 finding:

1. **Not just demo positions:** Even the aggregated task signal at the query position is not causally sufficient for task transfer.

2. **Probing ≠ Causation:** Layer 12 has the highest probe accuracy at query position (83%), but intervening there has zero causal effect on task behavior. This is a strong dissociation between correlational and causal importance.

3. **The model is robust:** The computation from (intervened query) → (correct output) is robust enough to recover from single-position perturbation at any layer.

## Implications

If neither demo positions (Phase 3-5) nor query position (Exp 9) can transfer task behavior when intervened individually, the remaining hypotheses are:

1. **Multi-position intervention is needed** (Exp 8) — all demo positions together
2. **Attention patterns encode task identity** (Exp 10) — not residual stream values
3. **Task identity is computed on-the-fly** — no static storage location exists

## Files

- `query_intervention_results.json` — Full results
- `query_intervention.csv` — Layer × transfer rates
- `exp9.log` — Full execution log
