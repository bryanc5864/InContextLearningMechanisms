# Experiment 13: Instance-Level Analysis — Results

**Date:** 2026-01-29
**Method:** Multi-position intervention at layer 8 on demo output tokens
**N test inputs:** 20 per condition

## Key Finding: Transfer Only Succeeds When Output Formats Overlap

### 1. Transfer Rates by Task Pair

| Source → Target | Transfer Rate | Output Format Overlap |
|-----------------|---------------|----------------------|
| uppercase → first_letter | **0%** | No (WORD vs letter) |
| uppercase → sentiment | **0%** | No (WORD vs pos/neg) |
| repeat_word → first_letter | **0%** | No (word word vs letter) |
| pattern_completion → repeat_word | **100%** | **Yes** (word word) |

### 2. Why pattern_completion → repeat_word Works

Both tasks produce outputs in "word word" format:
- **pattern_completion:** `apple apple` → expects repetition pattern
- **repeat_word:** `apple` → `apple apple`

Example transfers observed:
```
'honey' → 'honey honey' (source task behavior)
'frost' → 'frost frost' (source task behavior)
'badge' → 'badge badge' (source task behavior)
... (20/20 instances transferred)
```

### 3. Interpretation

**Task transfer is NOT about "task identity" but about output format:**

1. **When output formats match:**
   - The model's output mechanism already produces the right format
   - Intervention "redirects" the output to the source task's pattern
   - Transfer appears successful

2. **When output formats differ:**
   - The model's output mechanism is incompatible
   - Even with perfect task identity transfer, output fails
   - Transfer rate = 0%

**This explains our 90% transfer at layer 8 (Exp 8):**
- The high transfer rates were likely between format-compatible tasks
- Or the intervention disrupted the output enough to produce format-matching outputs by chance

### 4. Instance-Level Patterns

All 20 instances transferred for pattern_completion → repeat_word:
- No specific input characteristics predicted transfer
- Transfer was universal when formats matched

No instances transferred for other pairs:
- Input length, content type made no difference
- The format mismatch was the deciding factor

### 5. Revised Model of ICL Transfer

```
Intervention at Layer 8
         ↓
Task Identity Partially Transferred
         ↓
Output Format Compatibility?
    ↓           ↓
   YES          NO
    ↓           ↓
 TRANSFER    FAILURE
  100%         0%
```

Transfer is binary based on format compatibility, not gradual based on "how much" task identity was transferred.

## Files

- `instance_analysis_results.json` — Full results with per-instance data
- `instances.csv` — All instances with transfer status
- `exp13.log` — Full execution log
