# Phase 1: Baseline Characterization — Results

**Date:** 2026-01-27
**Model:** meta-llama/Llama-3.2-3B-Instruct (28 layers, d_model=3072, fp16)
**Device:** cuda:3 (RTX 2080 Ti, 11GB)
**Protocol:** 5-shot demonstrations, 50 test inputs per task, greedy decoding

## Summary Table

| Task               | Regime      | Accuracy | Status | Notes                        |
|--------------------|-------------|----------|--------|------------------------------|
| uppercase          | procedural  | 96.0%    | PASS   | 2 errors (glyph→GYLPH, trend→TRENDS) |
| first_letter       | procedural  | 100.0%   | PASS   | Replacement for reverse      |
| repeat_word        | procedural  | 100.0%   | PASS   | Replacement for pig_latin    |
| length             | counting    | 100.0%   | PASS   | All test words are 5 chars   |
| linear_2x          | gd_like     | 100.0%   | PASS   | Perfect on range 1-60        |
| sentiment          | bayesian    | 100.0%   | PASS   | Perfect positive/negative    |
| antonym            | retrieval   | 98.0%    | PASS   | Relaxed scoring (multi-valid)|
| pattern_completion | induction   | 100.0%   | PASS   | Perfect on alternating A B   |

## Excluded Tasks

| Task      | Regime     | Accuracy | Reason                                    |
|-----------|------------|----------|-------------------------------------------|
| reverse   | procedural | 56.0%    | Model struggles with char-level reversal   |
| pig_latin | procedural | 8.0%     | Model cannot learn pig latin from 5 demos  |

## Observations

1. **Semantic/numeric tasks are trivial**: sentiment, linear_2x, length, pattern_completion all hit 100%.
2. **Character-level manipulation is hard**: reverse (56%) and pig_latin (8%) both require character-level operations the model cannot reliably perform from few-shot examples.
3. **Antonym is mostly correct**: 82% with strict scoring, 98% with multi-valid antonyms (e.g., "stingy" accepted for "selfish").
4. **Uppercase nearly perfect**: Only 2/50 errors, both minor (extra letter or wrong letter order).

## Final Task Battery for Phases 2-7

8 tasks across 4 regimes:
- **Procedural** (string manipulation): uppercase, first_letter, repeat_word
- **Counting**: length
- **GD-like** (numeric regression): linear_2x
- **Bayesian** (semantic classification): sentiment
- **Retrieval** (lexical): antonym
- **Induction** (pattern): pattern_completion

## Files

- `baseline_results.json` — Full results for original 8 tasks
- `baseline_summary.csv` — CSV summary table
- `first_letter_results.json` — Replacement task results
- `repeat_word_results.json` — Replacement task results
- `antonym_results.json` — Re-scored antonym results
- `phase1.log` — Full execution log
- `phase1_replacements.log` — Replacement task log
