# Implementation design

OpenBonsai separates claims that can be tested cheaply from conversions that
consume serious compute. The first milestone is not “train an 8B model”; it is
to find a method whose scaling curves make that expenditure rational.

## System boundaries

The research stack has four layers:

1. **artifact forensics** — compare released GGUF tensors with their stated
   Qwen ancestors without loading either model into an accelerator;
2. **matrix laboratory** — test discrete solvers against captured layer inputs
   and Hessian surrogates;
3. **global recovery harness** — apply hard-forward, smooth-backward QAT and
   teacher alignment to a complete Hugging Face model;
4. **hard-export gate** — materialize only codes plus FP16 group scales and
   evaluate that representation independently.

The code in this repository implements the first three layers. A production
exporter should target a dedicated experimental GGUF type or a version-pinned
runtime; PrismML's public ternary file uses a legacy Q2_0 group-128 layout that
differs from recent llama.cpp Q2_0.

## Quantizer contract

For each contiguous 128-weight group, binary export stores

\[
\hat{w}_i = s z_i,\qquad z_i \in \{-1,+1\},\qquad s\in\mathrm{FP16}.
\]

Ternary export stores

\[
\hat{w}_i = s z_i,\qquad z_i \in \{-1,0,+1\},\qquad s\in\mathrm{FP16}.
\]

No residual matrix, outlier side channel, high-precision salient subset, or
per-weight correction is allowed in the strict Bonsai-compatible track. Such
methods remain useful baselines but must be labeled as a different format.

## Progressive QAT

Every quantized module owns a full-precision shadow weight during training.
The forward weight interpolates between shadow and hard quantized weights:

\[
w_f(t) = (1-\alpha(t))w + \alpha(t)Q_{\tau(t)}(w),
\]

where `alpha` increases from 0 to 1 and the ternary gate temperature `tau`
decreases. The hard-export evaluation always uses `alpha=1`; a good result at
an intermediate interpolation is not a result.

Binary runs should test two paths independently:

- direct FP → binary;
- FP → ternary convergence → binary continuation.

The second path is the leading hypothesis because the released binary and
ternary signs agree much more often with one another than either agrees with
the Qwen ancestor, and the ternary release is closer to Qwen.

## Loss design

The base objective is

\[
L = \lambda_{CE}L_{CE} + \lambda_{KL}T^2
\operatorname{KL}(p_T^{(T)}\,\|\,p_S^{(T)}) +
\lambda_h\sum_{l\in S} d(h_l^S,h_l^T) +
\lambda_n L_{noise}.
\]

Recommended defaults are CE/KL/hidden weights `0.25/0.75/0.10`, temperature
2, and cosine hidden-state distance at four evenly spaced layers. Treat those
as starting points, not settled facts. Log every component separately.

`L_noise` is initially zero. In a later ablation, perturb saved K/V tensors or
fake-quantize them to 4 bits during a fraction of batches. This tests whether
the unusually flat KV-cache degradation reported for 27B can be trained rather
than merely inherited.

## Curvature path

For a matrix `W` and captured inputs `X`, the local target is

\[
\min_{Z,s}\;\|(W-s\odot Z)X\|_F^2,
\]

or equivalently a quadratic form with `H = XXᵀ`. The included coordinate
solver greedily changes signs/codes against this loss. It is deliberately
small and readable. Scale it by operating per output row and per group, using
block-diagonal Hessians, and never materializing a full model Hessian.

Three initializers should enter the same global recovery run:

| Initializer | Purpose |
|---|---|
| Mean-absolute RTN | lower-bound control |
| Activation-MSE alternating solve | tests local function preservation |
| Curvature coordinate solve | tests second-order sensitivity |

If better initialization only improves the first few hundred steps but not the
final hard checkpoint, allocate compute to global objectives instead.

## Optimizer ablation

AdamW is the control. The high-value comparison is a curvature-aware
preconditioner such as Shampoo, PSGD, or a tractable Kronecker approximation,
with matched data order, token count, peak memory, and reported FLOPs. The
PrismML team's public optimization background makes this a plausible hidden
ingredient, but it is not established by released artifacts.

## Checkpoint and telemetry contract

Each run must save:

- resolved config and code revision;
- data manifest, tokenizer revision, and deterministic sample IDs;
- shadow checkpoint and hard code/scale checkpoint;
- optimizer/scheduler state;
- per-layer hard-forward KL and hidden-state cosine error;
- fraction of signs/codes changed since initialization;
- code occupancy, group-scale distribution, and gradient norms;
- validation perplexity under soft, hard, and exported execution.

Keep the best checkpoint by hard-export validation loss, never by shadow or
soft-forward loss.

## Scaling architecture

The supplied `train_hf.py` is a single-node discrimination harness. At 1.7B+
use FSDP or tensor parallelism for the student, put the teacher on separate
workers or cache top-k logits plus selected hidden states, and checkpoint
teacher targets by dataset hash. Full-vocabulary logits are expensive; top-k
teacher probabilities plus a residual mass bucket are a reasonable ablation.

## Explicit non-goals

- claiming the proprietary PrismML algorithm has been recovered;
- matching benchmark numbers before matching evaluation conditions;
- hiding precision exceptions in an average “bits per weight” number;
- optimizing only weight-space reconstruction;
- using released Bonsai outputs as proprietary training data.

