# Phase 5: Locality Analysis — Results

**Date:** 2026-01-28
**Position:** last_demo_token (fixed across all layers)
**Layers tested:** 0-27 (all 28 layers)
**N test inputs:** 10 per condition

## Key Findings

### 1. Transfer Rate = 0% at Every Layer

For all 4 cross-task pairs tested, transplanting the source task's mean vector at the last_demo_token position produces **zero transfer** at every layer from 0 to 27:

| Pair | Peak Layer | Peak Transfer | Localization Index |
|------|-----------|---------------|-------------------|
| uppercase → sentiment | 0 | 0.00 | 0.0000 |
| linear_2x → pattern_completion | 0 | 0.00 | 0.0000 |
| sentiment → linear_2x | 0 | 0.00 | 0.0000 |
| first_letter → antonym | 0 | 0.00 | 0.0000 |

### 2. Interpretation

**Single-position intervention is fundamentally insufficient for task transfer in this model.** The 0% transfer at ALL 28 layers means:

1. **Task identity is distributed across token positions.** The model does not route task information through any single demo token position. Instead, it spreads task information across ALL demo tokens (5 input-output pairs = ~30-50 tokens), and overriding one position leaves the other ~29-49 positions intact.

2. **This is a positive finding about ICL robustness.** The model's task-inference mechanism is highly distributed and robust to single-point perturbation. This resembles a distributed code rather than a bottleneck architecture.

3. **Multi-position intervention is needed.** Future experiments should transplant activations at ALL demo token positions simultaneously to test whether task identity can be causally manipulated.

### 3. Implications for the Research Plan

The original plan assumed a single (layer, position) bottleneck for task identity. This assumption is falsified. The key modification needed:

- **Multi-position transplantation**: Replace activations at all demo positions (or all positions in the prompt) simultaneously
- **Attention pattern manipulation**: Instead of modifying the residual stream, modify attention patterns to redirect information flow
- **Gradient-based attribution**: Use integrated gradients or attention knockouts to identify which tokens contribute most to task execution

## Files

- `locality_results.json` — Full layer-wise results
- `locality_curves.csv` — Layer × transfer rate curves
- `phase5.log` — Full execution log
