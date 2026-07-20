# Evaluation protocol

## Non-negotiable controls

1. Use the same base checkpoint, tokenizer, chat template, prompts, context length, and evaluator revision.
2. Run base, RTN, conventional quantization, and OpenBonsai under identical decoding. If a model-specific recommended temperature is also reported, show it as a separate column.
3. Score the hard exported checkpoint loaded through its deployment runtime.
4. Archive raw generations. Rule-only scores are primary; LLM-judge recovery is a labeled secondary analysis.
5. Report mean, bootstrap confidence interval, and seed/sample count. Two-point differences without intervals are not conclusions.
6. Separate ideal information bits, packed file bytes, resident weight bytes, peak runtime memory, and KV-cache memory.

## Evaluation ladder

### Level A — every checkpoint

- held-out token cross-entropy/perplexity;
- teacher forward KL on natural text, code, math, and tool prompts;
- hidden-state cosine/MSE at 5–9 depth taps;
- next-token top-1 and top-5 agreement;
- code alphabet, zero density, scale distribution, NaN/Inf, and file-size audit;
- 100-prompt generation smoke set.

### Level B — promising checkpoints

- MMLU-Redux and GPQA-Diamond (knowledge);
- MuSR (soft multi-step reasoning);
- GSM8K and MATH-500 (math);
- HumanEval+ and MBPP+ (execution-based code);
- IFEval and IFBench (constraint following);
- BFCL v3 single- and multi-turn reported separately;
- long-context retrieval with several lengths.

### Level C — finalists

- AIME with repeated samples and fixed sample budgets;
- LiveCodeBench with contamination-aware dates;
- Tau/Tau2-style multi-turn tool tasks;
- long-horizon run-test-repair coding trajectories;
- adversarial formatting, schema, refusal, and calibration tests;
- deployment throughput/energy on at least one Apple and one NVIDIA target.

## Core metrics

Let teacher distribution be `p`, student `q`, hard export `h`.

- Forward KL: `E[KL(p || h)]`; report by domain and token position.
- Retention: `score(h) / score(p)` only for ratio-scale accuracy metrics; also show absolute points.
- Reasoning survival: fraction of base-correct examples still correct after conversion.
- New-correct and flipped-wrong counts: averages hide asymmetric changes.
- Calibration: expected calibration error and Brier score on MCQ tasks.
- Trajectory survival: probability an entire multi-step tool episode remains valid.
- Density: `-log2(1-average_error) / actual_resident_GB`; publish the corresponding ideal-density value separately.

## PrismML-claim replication matrix

| Claim | Replication control |
|---|---|
| End-to-end 1/ternary-bit weights | Enumerate every tensor and its physical dtype/bytes |
| 95%/90% retained | Re-run identical decoding for all variants and publish per-task survival |
| Better than conventional 2-bit | Compare at measured average bits and measured resident bytes |
| 4-bit KV nearly lossless | Same prompt set, centering recipe, cache layout, and output forward-KL |
| Throughput gains | Same runtime commit, batch, prompt length, output length, clocks, and thermal state |
| Energy gains | State power domains, idle subtraction, wall time, and confidence/variance |

## Known PrismML comparability issues to correct

- The 27B whitepaper samples Qwen at temperature 1.0 and Bonsai at 0.7.
- Earlier and later papers use different IFEval/IFBench/BFCL conventions.
- Ternary density uses the ideal 1.71-bit size while fast released files use 2-bit code slots.
- The 8B suite disables thinking, while the 27B suite enables it; their averages are not on one scale.

These do not invalidate the models. They prevent the papers alone from serving
as the final independent comparison.

