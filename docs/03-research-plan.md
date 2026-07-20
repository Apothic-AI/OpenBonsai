# Research plan

## Principle

Spend compute only after a cheap experiment has identified which variable is
likely limiting. Every run needs a preregistered hypothesis, a matched control,
hard-export evaluation, and archived raw outputs.

## Phase 0 — artifact science and harness lock (completed/in progress)

Deliverables:

- exact paper/post evidence ledger;
- reproducible GGUF/safetensors forensics;
- independent model-size and true-bits accounting;
- common evaluator with identical decoding across base and quantized models;
- raw-generation archive and deterministic smoke suite.

Exit gate: reproduce the released model locally, validate its alphabet, and
obtain benchmark values close enough to distinguish a 2–3 point change. Do not
train while the evaluator is noisy.

## Phase 1 — matrix and block experiments

Use Qwen3-0.6B and a few layers from Qwen3-1.7B.

1. Capture 2K–16K representative token activations per layer.
2. Compare weight MSE RTN, activation MSE RTN, GPTQ/OBQ, discrepancy rounding, and curvature coordinate descent.
3. Measure layer-output MSE, cosine, teacher KL after reinsertion, and error propagation through 1, 2, 4, and all blocks.
4. Test exact group sizes 32/64/128/256 while holding true stored bits constant.
5. Test transform gauges: SmoothQuant-like channel scales, orthogonal rotations, permutations, and sign gauges.

Exit gate: an initializer that reduces whole-model teacher KL by at least 30%
relative to RTN without extra inference metadata.

## Phase 2 — 0.6B global QAT ablations

Run a factorial experiment, ternary first:

- initializer: RTN vs curvature-aware;
- schedule: immediate hard vs progressive;
- objective: CE only; CE+logit KL; CE+KL+hidden alignment;
- optimizer: AdamW vs Shampoo/K-FAC/PSGD-family preconditioning;
- robustness: none vs activation noise vs dual clean/noisy consistency;
- scale: analytic only vs analytic times learned foldable compensation.

Use the same data stream, token count, seeds, batch tokens, sequence curriculum,
and export format. Minimum three seeds for finalists.

Exit gates after hard export:

- ternary perplexity <=1.10x base;
- compact benchmark average >=90% of base;
- no invalid codes or high-precision matrix escape hatches;
- winner beats the next-simpler objective by more than seed variance.

## Phase 3 — 1.7B reproduction

1. Train the winning ternary recipe at 50M, 200M, 500M, and 1–2B token checkpoints.
2. Fit a tokens-to-retention scaling curve before committing to the longest run.
3. Continue the best ternary checkpoint to binary; compare against direct FP-to-binary.
4. Match the released forensic signature: sign agreement, zero fraction, scale ratios, layerwise norm movement, and output KL.
5. Independently run the six-benchmark PrismML suite plus reasoning-heavy tests.

Exit gate: within three points of released Ternary Bonsai 1.7B on the common
suite, or a clear scaling law showing the additional token budget required.

## Phase 4 — 8–9B reasoning conversion

Apply the validated method to a reasoning-capable target, with
Ornith-1.0-9B as the intended candidate from the prior model-selection work.
Keep a Qwen-family run as a controls-friendly bridge to PrismML.

Add reasoning-preservation data and metrics:

- teacher-generated long chains with final-answer verification;
- MATH-500, AIME, LiveCodeBench, and long tool-call trajectories;
- hidden-state taps weighted toward later reasoning tokens;
- response-format and tool-schema constraints;
- binary-vs-ternary transition checkpoints.

Exit gate: ternary retains >=93% and binary >=87% of a preregistered reasoning
aggregate at the actual shipped footprint.

## Phase 5 — systems and export

- pack exact Q1_0 g128 and a non-ambiguous ternary format;
- verify packed/unpacked logit parity on 1,000 prompts;
- upstream kernels rather than maintaining unnecessary forks;
- measure prompt processing, token generation, peak RSS/VRAM, and energy;
- evaluate 4-bit and sub-2-bit KV cache only after the weight model is stable.

## Compute budgeting

| Stage | Suggested hardware | Initial budget |
|---|---|---:|
| Matrix/block lab | 1x 24–80 GB GPU | hours |
| 0.6B QAT screen | 4–8x H100/A100 or TPU slice | 10–100M tokens/run |
| 1.7B finalists | 8x H100/A100 or TPU pod slice | 0.2–2B tokens |
| 8–9B final | multi-node FSDP/TPU | scale only after 1.7B curve |

Token budgets are experimental rungs, not claims about PrismML's cost.

## Run record requirements

Every run directory must contain:

- git commit, environment lock, model and data hashes;
- full resolved config and exact seed;
- token counts, sequence-length distribution, optimizer steps, and FLOPs estimate;
- per-layer scale/code statistics over time;
- shadow and hard-forward losses separately;
- packed hard-export checksum;
- raw evaluation generations and scorer versions;
- actual file/resident sizes and true average bits.

