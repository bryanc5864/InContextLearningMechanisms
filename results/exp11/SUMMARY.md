# Experiment 11: Activation Patching (Causal Tracing) — Results

**Date:** 2026-01-28
**Status:** Preliminary (uppercase complete, 7 tasks remaining)
**Noise scale:** 2.0 (Gaussian)
**N test inputs:** 15 per condition

## Key Finding: Query Position is NECESSARY, Demo Position is NOT

### 1. Causal Importance by Position

| Position | Layer Range | Mean Disruption | Interpretation |
|----------|-------------|-----------------|----------------|
| **first_query_token** | 0-14 | **87-100%** | Causally NECESSARY |
| **first_query_token** | 16 | 40% | Transition zone |
| **first_query_token** | 18-26 | 0-7% | Model has recovered |
| last_demo_token | ALL | **0%** | NOT necessary |

### 2. Detailed Results for `uppercase` Task

**Demo position (last_demo_token):**
```
Layer  0-26: 0% disruption at all layers
```

**Query position (first_query_token):**
```
Layer  0: 100% disruption ***
Layer  2: 100% disruption ***
Layer  4:  93% disruption ***
Layer  6:  93% disruption ***
Layer  8:  87% disruption ***
Layer 10: 100% disruption ***
Layer 12: 100% disruption ***
Layer 14:  87% disruption ***
Layer 16:  40% disruption ***
Layer 18:   7% disruption
Layer 20:   0% disruption
Layer 22:   0% disruption
Layer 24:   0% disruption
Layer 26:   0% disruption
```

### 3. Interpretation

**This explains the puzzle from Phases 3-5 and Experiment 9:**

1. **Why single-position intervention at demo positions fails:**
   - Adding noise at demo positions causes 0% disruption
   - Therefore, no single demo position is causally necessary
   - Task information is aggregated from ALL demos, then routed to query position

2. **Why query position intervention also fails for TRANSFER (Exp 9):**
   - The query position is necessary for OUTPUT, not for task IDENTITY
   - Corrupting it breaks output generation (100% disruption)
   - But transplanting a different task's query activation doesn't redirect task — it just corrupts

3. **Why multi-position intervention at LAYER 8 works (Exp 8):**
   - Layer 8 is early enough that task identity hasn't fully propagated to query
   - Replacing ALL demo outputs at layer 8 intercepts the distributed task signal
   - Layer 14+ is too late — task identity has already been routed

### 4. The Causal Flow

```
Demos (distributed) ──[layers 0-8]──> Query Position ──[layers 8-14]──> Output
       ↑                                    ↑
  Not necessary                        NECESSARY
  (0% disruption)                    (100% disruption)
```

Task identity is:
- **Stored** in demo token outputs (distributed)
- **Aggregated** via attention to query position (layers 0-8)
- **Routed** from query position to output (layers 8-16)
- **Finalized** by layer 18 (recovery from noise at late layers)

## Files

- `patching_results.json` — Full results (when complete)
- `disruption_heatmap.csv` — Layer × position disruption matrix
- `exp11.log` — Full execution log
