# Training recipe v0

This is a deliberately staged reconstruction recipe. Do not move to the next
gate because a run “looks promising”; move when the stated hard-checkpoint
criterion is met.

## Phase 0 — matrix and 100M–600M validation

**Goal:** reject broken quantizers, losses, and schedules in days rather than
weeks.

1. Capture 128–512 million representative tokens and layer inputs from a small
   Qwen-family teacher. Preserve document boundaries and reasoning/code/math mix.
2. Compare group-128 RTN, activation-MSE, and curvature-coordinate initializers
   on at least attention Q, attention output, gate, up, and down projections.
3. Run four matched global jobs:
   - hard STE + CE;
   - progressive quantization + CE;
   - progressive + logit KL;
   - progressive + KL + selected hidden alignment.
4. For binary, add a fifth run initialized from the best converged ternary job.
5. Evaluate only fully hard weights with embeddings and LM head constrained.

**Gate:** progressive + distillation must beat hard STE on held-out KL and
perplexity at the same tokens, and a binary route must recover at least 90% of
the ternary route's normalized task-retention score. Otherwise revise before
scaling.

## Phase 1 — Qwen3-1.7B reproduction target

The 1.7B release is the best forensic target because both precision variants
and the claimed ancestor are publicly available.

### Data

Use a documented mixture with deterministic IDs:

| Slice | Initial share | Purpose |
|---|---:|---|
| General web/books | 45% | language coverage and perplexity |
| Code | 20% | syntax and long dependency preservation |
| Math/science | 15% | exact token and reasoning sensitivity |
| Instruction/chat | 15% | assistant behavior and formatting |
| Tool/structured output | 5% | schema and function-call robustness |

Deduplicate against evaluation prompts and keep licenses auditable. Start with
5–10B recovery tokens, but run a logarithmic budget sweep first (0.25B, 1B,
3B) to estimate the quality/token curve.

### Initialization

Initialize ternary from activation-aware groups. Initialize binary both from
the full-precision teacher and from the best ternary shadow checkpoint. Each
group gets a positive learnable scale initialized analytically. Fold any
positive/negative training-time compensation into the one exported scale before
the format gate; if it cannot be folded, it is a different model class.

### Schedule

- first 5% of steps: full-precision shadow forward, distillation warm-up;
- next 60%: increase hard mixture from 0 to 1 with a smooth cosine schedule;
- next 25%: fully hard forward with STE and trainable scales;
- final 10%: freeze codes for part of the ablation, optimize scales and
  higher-precision norms, then compare against leaving codes mobile.

Use a low peak learning rate (start at `2e-5` for 1.7B), 5% warm-up, cosine
decay, BF16 optimizer math, gradient clipping at 1.0, and effective sequences
large enough to stabilize teacher KL. These are experiment defaults, not a
claim about PrismML.

### Loss sweep

Test CE/KL ratios `(1,0)`, `(0.5,0.5)`, and `(0.25,0.75)`; temperatures 1, 2,
and 4; hidden weights 0, 0.05, and 0.1. A factorial sweep is wasteful—use
successive halving on short runs, then repeat the best two settings with three
seeds.

### Optimizer sweep

Compare AdamW with one feasible preconditioner at matched tokens and measured
FLOPs. Record sign-crossing rates by layer. The curvature hypothesis gains
support only if the preconditioner improves hard export at a cost-adjusted
frontier, not merely training loss.

### Phase 1 success criteria

- strict group-128 codes/scales for every linear matrix, embedding, and head;
- no undeclared residual or outlier tensors;
- physical bytes and runtime resident bytes reported;
- held-out perplexity within 15% relative of teacher for ternary and within 25%
  for binary;
- average normalized retention on the fixed task suite at least 0.85 ternary
  and 0.75 binary;
- three-seed uncertainty intervals and no benchmark-specific training leakage;
- hard-export scores reproduce from a clean environment.

These thresholds are engineering gates, not assertions about the public Bonsai
checkpoints.

## Phase 2 — 8B/9B reasoning target

Only after the 1.7B method is stable, apply the best ternary recipe to Qwen3-8B
or the previously selected Ornith-1.0-9B target. Preserve the exact tokenizer
and prompt template. Scale sequence length progressively (2K → 8K → target)
because quantization error can compound with context length.

Add reasoning-specific probes: GSM8K/MATH-style exact answer, code execution,
multi-turn tool calls, long-context retrieval, and refusal/over-refusal.
Reasoning traces may be distilled where licensing and policy permit, but the
evaluation must include answer-only and hidden test sets.

**Gate:** the method beats same-base 2-bit PTQ and a compute-matched QAT baseline
on a Pareto plot of hard quality versus actual bytes and conversion FLOPs.

## Phase 3 — KV and deployment robustness

Evaluate FP16, 8-, 6-, 5-, 4-, 3-, and 2-bit KV caches with a pinned runtime.
Then train a copy with stochastic KV fake quantization/noise. If robustness
improves without a clean-cache regression, the data support the explicit-noise
hypothesis. Measure time-to-first-token, decode rate, peak resident memory,
energy if available, and accuracy across context lengths.

## Stop conditions

Stop or redesign if:

- soft-forward gains disappear on hard export;
- improved weight MSE does not improve held-out KL/task retention;
- scale or metadata overhead violates the strict format;
- binary quality does not improve with an order-of-magnitude more tokens;
- benchmark gains fail prompt/seed replication;
- the best route is dominated by a conventional 2–3 bit model in bytes,
  latency, and quality.

