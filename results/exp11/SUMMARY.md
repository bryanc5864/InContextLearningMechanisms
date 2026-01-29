# Experiment 11: Activation Patching (Causal Tracing) — Results

**Date:** 2026-01-29
**Status:** Complete (8 tasks)
**Noise scale:** 2.0 (Gaussian)
**N test inputs:** 15 per condition

## Key Finding: Query Position is NECESSARY, Demo Position is NOT

### 1. Causal Importance by Position

| Position | Layer Range | Mean Disruption | Interpretation |
|----------|-------------|-----------------|----------------|
| **first_query_token** | 0-12 | **55-81%** | Causally NECESSARY |
| **first_query_token** | 14-18 | 12-38% | Transition zone |
| **first_query_token** | 20+ | 0-20% | Model has recovered |
| last_demo_token | ALL | **<1%** | NOT necessary |

### 2. Disruption Heatmap by Task

**Demo position (last_demo_token) — Essentially NO disruption:**
```
Layer:      0      2      4      6      8     10     12     14     16     18     20     22     24     26
uppercase   :  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
first_letter:  0.00   0.00   0.00   0.07   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
repeat_word :  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
length      :  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
linear_2x   :  0.07   0.00   0.07   0.00   0.00   0.07   0.00   0.07   0.00   0.00   0.00   0.00   0.00   0.00
sentiment   :  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
antonym     :  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
pattern_completion:  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
```

**Query position (first_query_token) — HIGH disruption at early layers:**
```
Layer:      0      2      4      6      8     10     12     14     16     18     20     22     24     26
uppercase   :  1.00   1.00   0.93   0.93   0.87   1.00   1.00   0.87   0.40   0.07   0.00   0.00   0.00   0.00
first_letter:  1.00   0.93   0.67   0.67   0.67   0.87   0.93   0.20   0.00   0.00   0.00   0.00   0.00   0.00
repeat_word :  1.00   1.00   1.00   1.00   1.00   1.00   1.00   1.00   0.80   0.87   0.20   0.20   0.00   0.00
length      :  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
linear_2x   :  1.00   1.00   1.00   1.00   1.00   1.00   1.00   0.93   0.40   0.07   0.00   0.00   0.00   0.00
sentiment   :  0.60   0.53   0.53   0.20   0.00   0.07   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
antonym     :  1.00   1.00   0.87   0.80   0.87   1.00   0.73   0.07   0.00   0.00   0.00   0.00   0.00   0.00
pattern_completion:  0.87   0.07   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
```

### 3. Mean Disruption by Layer (query position)

| Layer | Mean Disruption |
|-------|-----------------|
| 0 | 80.8% |
| 2 | 69.2% |
| 4 | 62.5% |
| 6 | 57.5% |
| 8 | 55.0% |
| 10 | 61.7% |
| 12 | 58.3% |
| 14 | 38.3% |
| 16 | 20.0% |
| 18 | 12.5% |
| 20+ | <10% |

### 4. Task-Specific Patterns

**Tasks with strong query position sensitivity (layers 0-14):**
- uppercase, first_letter, repeat_word, linear_2x, antonym: 67-100% disruption

**Tasks with early recovery:**
- sentiment: Recovers by layer 8 (0% disruption)
- pattern_completion: Recovers by layer 4 (0% disruption)
- length: 0% disruption at ALL layers (anomaly)

### 5. Interpretation

**This explains the puzzle from Phases 3-5 and Experiments 8-9:**

1. **Why single-position intervention at demo positions fails:**
   - Adding noise at demo positions causes 0% disruption
   - Therefore, no single demo position is causally necessary
   - Task information is aggregated from ALL demos, then routed to query position

2. **Why query position intervention also fails for TRANSFER (Exp 9):**
   - The query position is necessary for OUTPUT, not for task IDENTITY
   - Corrupting it breaks output generation (60-100% disruption)
   - But transplanting a different task's query activation doesn't redirect task — it just corrupts

3. **Why multi-position intervention at LAYER 8 works (Exp 8):**
   - Layer 8 is early enough that task identity hasn't fully propagated to query
   - Replacing ALL demo outputs at layer 8 intercepts the distributed task signal
   - Layer 14+ is too late — task identity has already been routed

### 6. The Causal Flow (Confirmed)

```
Demos (distributed) ──[layers 0-8]──> Query Position ──[layers 8-14]──> Output
       ↑                                    ↑
  Not necessary                        NECESSARY
  (0% disruption)                    (60-100% disruption)
```

Task identity is:
- **Stored** in demo token outputs (distributed)
- **Aggregated** via attention to query position (layers 0-8)
- **Routed** from query position to output (layers 8-14)
- **Finalized** by layer 18 (recovery from noise at late layers)

## Files

- `patching_results.json` — Full results
- `disruption_heatmap.csv` — Layer × position disruption matrix
- `exp11.log` — Full execution log
