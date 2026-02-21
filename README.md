# Characterizing the Computational Interface of In-Context Learning

Code and experimental results for mechanistic interpretability research on how transformer language models encode and transfer task identity during in-context learning (ICL).

## Key Findings

1. **Task identity is distributed across demo output tokens.** Replacing activations at ALL demo output positions at ~30% network depth achieves up to 96% task transfer (N=50, 95% CI: [87%, 99%]). Single-position intervention yields 0%.

2. **The optimal intervention depth is ~30% across architectures.** This holds for LLaMA (1B, 3B), Qwen (1.5B), and Gemma (2B) — layer 8/28, 5/16, 8/28, and 8/26 respectively.

3. **Transfer is governed by output token-count templates.** Tasks with matching output token counts transfer at 50-100%; mismatched counts yield 0%. Variable-length outputs (e.g., "cat 3" -> "cat cat cat") transfer perfectly only when N matches the source output length.

4. **Abstract task templates are separable from specific labels.** Sentiment variants with different labels (positive/negative vs good/bad vs +/-) transfer at 58% mean, confirming the model encodes classification rules abstractly.

5. **Encoding requires a critical mass of demonstrations.** 1-3 source demos yield 0% transfer; 5 demos yield 93%. There is no gradual degradation — it is all-or-nothing.

## Experiments

| Exp | Description | Key Result |
|-----|-------------|------------|
| 1 | Baseline characterization | 96-100% ICL accuracy across 8 tasks |
| 2 | Representation localization | Demo positions: 100% probe accuracy; query: 83% peak |
| 3-5 | Single-position intervention | 0% transfer at all layers |
| 6 | Task ontology | Procedural tasks cluster (cos > 0.86) |
| 7 | Trajectory analysis | High change in early layers, stabilization in late |
| **8** | **Multi-position transplantation** | **90-96% transfer at ~30% depth** |
| 9 | Query position intervention | 0% transfer (necessary but not sufficient) |
| 10 | Attention pattern intervention | Demo info fully processed by layer 16 |
| 11 | Activation patching | Query position causally necessary; no single demo is |
| 12 | Layer-wise ablation | Layer 0 critical; early layers essential |
| 13 | Instance-level analysis | Transfer = output format matching |
| 14 | Demo count ablation | Distribution is fundamental, not artifact |
| 15 | Cross-format control | Structural format similarity drives transfer |
| 16 | Multi-model replication | Findings replicate across 4 models, 3 architectures |
| 19 | Template similarity metric | Surface similarity does NOT predict transfer (r=-0.05) |
| 23 | Proper statistics (N=50) | 96% [87%, 99%] with Wilson CIs |
| 27 | Baseline controls | Random/noise/shuffle all yield 0% |
| 28 | Tokenization confound analysis | Token alignment does not explain transfer |
| 29 | Full 56-pair transfer matrix | Clear cluster structure with asymmetric transfer |
| 30 | Single-demo function vectors | 0% with 1-3 demos, 93% with 5 |
| 31 | Output position scaling | Nearly ALL output positions required |
| 32 | Sentiment variant transfer | 58% mean across label variants |
| 33 | Variable-length output tasks | Token-count matching is the critical factor |

## Models Tested

- Llama-3.2-3B-Instruct (primary)
- Llama-3.2-1B-Instruct
- Qwen2.5-1.5B-Instruct
- Gemma-2-2B-IT

## Tasks

8 ICL tasks spanning procedural, semantic, and retrieval regimes:
- **Procedural:** uppercase, first_letter, repeat_word, length, linear_2x
- **Semantic/Bayesian:** sentiment
- **Retrieval:** antonym
- **Pattern:** pattern_completion
- **Variable-length (Exp 33):** repeat_n, spell_out

## Repository Structure

```
scripts/          Experiment scripts (01-33)
src/              Core library (model loading, task definitions, probing, intervention)
  tasks/          Task implementations
results/          All experimental outputs (JSON, CSV, logs)
  exp{N}/         Per-experiment results
  {model-name}/   Cross-model replication results
  cross_model/    Comparison CSVs
config/           Configuration files
tests/            Unit tests
```

## Causal Flow Model

```
Input Processing (Layer 0)           [CRITICAL]
         |
Demo Output Tokens (Layers 1-8)     Store task identity (distributed)
         |
Attention Aggregation (Layers 8-12)  Demo info -> Query position
         |
Query Position (Layers 12-16)       Task identity finalized
         |
Output Generation (Layers 16-28)    Output format applied
```

**Intervention window:** ~30% depth, ALL demo output positions.

## Usage

```bash
# Run a single experiment (e.g., baseline)
python scripts/01_baseline.py --device cuda:0

# Run multi-position transfer
python scripts/08_multi_position.py --device cuda:0 --n-test 20

# Run second-round experiments
python scripts/30_single_demo_fv.py --device cuda:1
python scripts/31_output_position_scaling.py --device cuda:6
python scripts/32_sentiment_variants.py --device cuda:1
python scripts/33_variable_length.py --device cuda:6
```

## Requirements

- Python 3.10+
- PyTorch (CUDA-compatible)
- Transformers (HuggingFace)
- NumPy
- scikit-learn (for probing experiments)

## Full Summary

See [results/MASTER_SUMMARY.md](results/MASTER_SUMMARY.md) for the complete experimental narrative, all results tables, and detailed interpretation.
