# Experiment 10: Attention Pattern Intervention — Results

**Date:** 2026-01-29
**Method:** Zero out demo positions in residual stream at layer L
**N test inputs:** 10 per condition

## Key Finding: Demo Information is Fully Processed by Layer 16

### 1. Disruption by Layer

| Layer | Mean Disruption | Interpretation |
|-------|-----------------|----------------|
| 4 | **100%** | Demo info still critical |
| 8 | **100%** | Demo info still critical |
| 12 | **92.5%** | Demo info mostly critical |
| 16 | **2.5%** | Demo info fully processed |
| 20 | **0%** | Model has recovered |

### 2. Per-Task Results

| Task | Layer 4 | Layer 8 | Layer 12 | Layer 16 | Layer 20 |
|------|---------|---------|----------|----------|----------|
| uppercase | 100% | 100% | 100% | 0% | 0% |
| first_letter | 100% | 100% | 90% | 0% | 0% |
| repeat_word | 100% | 100% | 100% | 0% | 0% |
| length | 100% | 100% | 80% | 10% | 0% |

### 3. Interpretation

**This confirms the Layer 8-16 processing window:**

1. **Layers 0-12:** Demo information is actively being used
   - Zeroing demo positions causes complete task failure
   - Task identity is being read from demo tokens

2. **Layers 16+:** Demo information has been fully extracted
   - Zeroing demo positions has no effect
   - Task identity has been routed to query position

3. **Transition at Layer 12-16:**
   - Some tasks (uppercase, repeat_word) complete extraction earlier
   - Other tasks (first_letter, length) take longer

### 4. Relation to Other Findings

- **Exp 8 (Multi-position):** Layer 8 is optimal for intervention because task info is still in demo positions
- **Exp 11 (Causal tracing):** Query position becomes critical after layer 8
- **Exp 12 (Layer ablation):** Early layers (0-7) are absolutely critical (100% drop when skipped)

The attention knockout confirms the causal flow model:
```
Demos (layers 0-12) → Query Position (layers 8-16) → Output
```

## Files

- `attention_results.json` — Full results
- `exp10.log` — Full execution log
