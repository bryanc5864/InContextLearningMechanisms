# Experiment 14: Demo Count Ablation — Results

**Date:** 2026-01-28
**Demo counts tested:** 1, 2, 3, 4, 5
**N test inputs:** 15 per condition

## Key Findings

### 1. Task Accuracy vs Demo Count

| Task | 1-shot | 2-shot | 3-shot | 4-shot | 5-shot |
|------|--------|--------|--------|--------|--------|
| uppercase | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| first_letter | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| repeat_word | 0.73 | 1.00 | 1.00 | 1.00 | 1.00 |
| length | 0.13 | 1.00 | 1.00 | 1.00 | 1.00 |
| linear_2x | 0.60 | 0.87 | 0.87 | 1.00 | 1.00 |
| sentiment | 0.67 | 0.80 | 1.00 | 1.00 | 1.00 |
| antonym | 0.87 | 0.93 | 1.00 | 1.00 | 1.00 |
| pattern | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **Mean** | **0.64** | **0.96** | **0.98** | **1.00** | **1.00** |

**Key insight:** 2 demos are sufficient for most tasks to achieve high accuracy. The model learns quickly.

### 2. Transfer Rate vs Demo Count

Single-position transplant transfer rate (layer 14, last_demo_token):

| Demo Count | Mean Transfer |
|------------|---------------|
| 1-shot | 0.333 |
| 2-shot | 0.333 |
| 3-shot | 0.333 |
| 4-shot | 0.333 |
| 5-shot | 0.333 |

**No improvement with fewer demos.** The 33.3% rate is due to pattern_completion → repeat_word showing 100% transfer (output format overlap), while other pairs show 0%.

### 3. Probe Accuracy at Query Position (Layer 12)

| Demo Count | Probe Accuracy |
|------------|----------------|
| 1-shot | 0.533 |
| 2-shot | 0.583 |
| 3-shot | 0.550 |
| 4-shot | 0.550 |
| 5-shot | 0.558 |

Probe accuracy at query position is relatively flat (~55%) regardless of demo count. This is lower than Phase 2's 83% — possibly because fewer samples or different test inputs.

### 4. Interpretation

**The hypothesis that fewer demos = more concentrated encoding is NOT supported:**

- Transfer rate is constant regardless of demo count
- Reducing to 1-shot doesn't enable single-position intervention
- The encoding is distributed even with minimal demonstrations

**What demo count DOES affect:**
- Task accuracy (1-shot = 64%, 5-shot = 100%)
- Some tasks need 2+ demos to work at all (first_letter = 0% at 1-shot)

**What demo count does NOT affect:**
- Transferability via single-position intervention
- The fundamental distributed nature of task encoding

## Files

- `demo_ablation_results.json` — Full results
- `demo_ablation.csv` — Summary table
- `exp14.log` — Full execution log
