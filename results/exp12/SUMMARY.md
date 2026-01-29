# Experiment 12: Layer-Wise Ablation Study — Results

**Date:** 2026-01-29
**Method:** Skip individual layers (pass residual unchanged)
**N test inputs:** 15 per condition

## Key Finding: Layer 0 is Critical for ALL Tasks

### 1. Single Layer Skip — Mean Drop by Layer

| Layer | Mean Drop | Interpretation |
|-------|-----------|----------------|
| **0** | **100%** | Critical for ALL tasks |
| 2 | 15.8% | Somewhat important |
| 4 | 2.5% | Not critical individually |
| 6 | 4.2% | Not critical individually |
| 8 | 3.3% | Not critical individually |
| 10 | 4.2% | Not critical individually |
| 12 | 1.7% | Not critical individually |
| 14 | 5.8% | Minor importance |
| 16+ | <2.5% | Not critical |

### 2. Phase Skip — Mean Drop by Phase

| Phase | Layers | Mean Drop | Interpretation |
|-------|--------|-----------|----------------|
| **Early** | 0-7 | **100%** | Absolutely critical |
| **Mid** | 8-15 | **75%** | Very important |
| Late | 16-23 | 35% | Less critical |
| Final | 24-27 | 33% | Some tasks sensitive |

### 3. Per-Task Analysis (Phase Skip)

| Task | Early (0-7) | Mid (8-15) | Late (16-23) | Final (24-27) |
|------|-------------|------------|--------------|---------------|
| uppercase | 100% | 100% | 53% | 100% |
| first_letter | 100% | 100% | 73% | 27% |
| repeat_word | 100% | 100% | 13% | 7% |
| length | 100% | 0% | 0% | 0% |

### 4. Interpretation

**Layer 0 is universally critical:**
- Skipping layer 0 causes 100% failure across ALL tasks
- This layer likely handles initial token embedding processing
- Cannot be bypassed regardless of task type

**Early layers (0-7) form essential processing:**
- Skipping the entire early phase causes complete failure
- This is where task recognition begins

**Mid layers (8-15) handle task execution:**
- 75% drop when skipped
- Some simple tasks (length) can survive without mid layers
- Most tasks require mid-layer processing

**Late layers (16+) are more redundant:**
- Individual layers can be skipped with minimal impact
- But skipping entire phases still causes significant drop
- The final layers (24-27) are important for output formatting

### 5. Implications for Intervention

- **Don't intervene at layer 0** — it's universally critical
- **Layers 8-14 are the optimal intervention window** — task identity is being processed but not finalized
- **Late layers are safe to skip individually** but not in groups

## Files

- `layer_ablation_results.json` — Full results
- `layer_ablation.csv` — Per-layer accuracy data
- `exp12.log` — Full execution log
