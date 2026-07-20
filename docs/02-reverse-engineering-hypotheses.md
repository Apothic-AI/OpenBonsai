# Reverse-engineering hypotheses

Each hypothesis below makes predictions that can be tested against released
artifacts or small conversion runs. A useful experiment must discriminate
between hypotheses rather than merely improve one score.

## H1 — progressive global QAT with distillation

Start from the full-precision checkpoint, introduce a smooth-to-hard quantizer,
and optimize language-model and teacher-alignment losses while increasing
discreteness. Learn foldable per-group scale compensation. For binary, either
continue directly from FP or pass through ternary first.

Why it fits:

- large sign changes and worse Euclidean weight error show that task/function loss dominates weight loss;
- TPU disclosure is consistent with a global recovery run;
- final runtime needs only discrete codes and one scale, which foldable QAT provides;
- the 27B checkpoint retains long-chain tasks that layerwise reconstruction usually loses first.

Predictions:

- direct hard-QAT has a large initial loss spike; progressive hardening avoids it;
- teacher-logit KL reduces late-layer drift more than CE-only training;
- hidden-state alignment improves reasoning/tool use disproportionately;
- ternary-to-binary continuation converges faster than FP-to-binary.

Closest public analogues: [Rethinking 1-bit Optimization](https://arxiv.org/abs/2508.06974),
[OneBit](https://arxiv.org/abs/2402.11295), and [EfficientQAT](https://arxiv.org/abs/2407.11062).

Discriminator: run identical 0.6B training tokens with hard STE, progressive
STE, and progressive STE + distillation. Track initial loss area-under-curve and
hard-export reasoning retention.

## H2 — curvature-preconditioned projected discrete training

Optimize a latent/full-precision shadow model, project every forward pass to
`{-s,+s}` or `{-s,0,+s}`, and use a curvature-aware preconditioner so steps in
sensitive directions are small while insensitive directions can cross sign
boundaries. The group scale is solved analytically or jointly learned and folded.

Why it fits:

- PrismML repeatedly emphasizes a mathematical Caltech result;
- the research team's public background includes curvature-informed optimization and PSGD;
- sign flips are too numerous for local rounding but structured enough to preserve the function;
- Euclidean distance is a poor objective when the relevant geometry is Fisher/Hessian weighted.

Predictions:

- released sign flips should concentrate in low-curvature directions;
- a Kronecker/Shampoo/PSGD optimizer should beat AdamW at matched tokens, especially for binary;
- quadratic-loss estimates should predict which groups acquire the largest scale changes;
- a Fisher-weighted distance to Qwen should be much smaller, relatively, than Euclidean distance.

Discriminator: capture activations and gradient/Fisher diagonals on a held-out
calibration set. Compare curvature of flipped vs unflipped weights and run the
optimizer ablation under fixed data order and FLOPs.

## H3 — Hessian/activation-aware layerwise initialization plus global recovery

First fit each matrix to preserve `WX` or block outputs using GPTQ/OBQ-style
curvature, alternating discrete codes and scales. Then perform a shorter global
distillation/QAT recovery run.

Why it fits:

- scale correlation with base magnitude remains substantial;
- known PTQ methods supply a much better starting point than sign RTN;
- local initialization reduces the catastrophic first QAT step;
- a global phase explains why final weights no longer minimize local error.

Relevant public baselines: [BiLLM](https://arxiv.org/abs/2402.04291),
[PB-LLM](https://arxiv.org/abs/2310.00034), [LeanQuant](https://arxiv.org/abs/2407.10032),
[DiscQuant](https://arxiv.org/abs/2501.06417), and [NanoQuant](https://arxiv.org/abs/2602.06694).
Most published binary PTQ methods use residuals, salient high-precision weights,
or extra metadata and therefore are not format-equivalent to Bonsai; their
initialization ideas remain useful.

Predictions:

- a layerwise curvature initializer should reduce the first hard-forward KL;
- after global recovery, layer output error on the original calibration set may increase while held-out model KL improves;
- code patterns should resemble an activation-aware solution more than an MSE solution before QAT, then drift.

Discriminator: use `openbonsai-matrix-lab` on captured Qwen layer inputs, then
initialize two otherwise identical global runs from RTN and curvature solutions.

## H4 — equivalence transformations plus discrepancy balancing

Exploit exact or approximate transformer symmetries—channel rescaling,
permutation, normalization compensation, and discrepancy-minimizing rounding—to
make weights binary-friendly before recovery.

Why it fits:

- equivalent transforms can increase sign/scale flexibility without changing the initial function;
- discrepancy theory can bound accumulated error better than independent rounding;
- selective norm movement may compensate channel transformations.

Predictions:

- paired adjacent matrices show reciprocal channel rescaling;
- permutations/sign gauges align the release with Qwen much better than direct comparison;
- error residuals are anti-correlated across adjacent layers rather than independent.

Discriminator: solve channel-wise permutation/sign/scale alignment between Qwen
and Bonsai. Measure whether aligned sign agreement and weight error improve
dramatically. This is a high-value next addition to the forensic pipeline.

## H5 — pure PTQ

Perform a sophisticated one-shot or iterative discrete solve with calibration
data and no meaningful token training.

Why it is unlikely:

- TPU-training disclosure;
- large norm-parameter movement in selected layers;
- exceptional reasoning retention at 27B;
- strong post-conversion KV-noise tolerance;
- worse Euclidean reconstruction than RTN.

It remains falsifiable: exact activation/Hessian optimization can move weights
far from Qwen. If a calibration-only method matches Bonsai-level held-out
reasoning and the released curvature fingerprints, this probability rises.

## Experiment matrix

| Experiment | H1 | H2 | H3 | H4 | H5 |
|---|---|---|---|---|---|
| Progressive schedule beats hard STE | Strong support | Neutral | Neutral | Neutral | Against |
| PSGD/Shampoo beats AdamW at matched FLOPs | Weak support | Strong support | Weak support | Neutral | Neutral |
| Curvature init reduces final tokens-to-quality | Support | Support | Strong support | Neutral | Support |
| Channel alignment explains most sign changes | Against | Neutral | Neutral | Strong support | Support |
| No-training solver reaches target | Against | Support | Support | Support | Strong support |
| Noise augmentation reproduces KV tolerance | Strong support | Support | Neutral | Neutral | Against |

