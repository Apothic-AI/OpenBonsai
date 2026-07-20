# Executive brief

## Bottom line

PrismML almost certainly performs substantial post-pretraining optimization.
The method is not ordinary PTQ, not a from-scratch BitNet recipe, and not a
runtime trick. PrismML says the 27B models start from an off-the-shelf pretrained
Qwen checkpoint and undergo a behavior-preserving “representation
transformation.” A launch release additionally says the 8B family was trained
on Google v4 TPUs. Our direct tensor comparison confirms large learned movement
away from the Qwen weights.

The best OpenBonsai bet is a global, progressive quantization-aware recovery
run initialized by activation/curvature-aware discrete fitting. The likely loss
combines language modeling, teacher-logit KL, intermediate hidden-state
alignment, and perturbation robustness. The likely optimizer is curvature-aware
or strongly preconditioned; that inference is based on the founders' research
background, not direct disclosure.

## What is known

- Language-model matrix weights use groups of 128 and one shared FP16 scale.
- Binary groups are exactly `{-s,+s}`: 1 sign bit plus 16/128 scale bits = 1.125 effective bits/weight.
- Ternary groups are exactly `{-s,0,+s}`: log2(3) plus 16/128 = about 1.71 information bits/weight.
- Embeddings, attention projections, MLP projections, and the LM head are all low-bit. Norm parameters remain higher precision.
- Architectures remain recognizably the base architecture; the 1.7B release does trim 267 embedding vocabulary rows, an undocumented small exception discovered by our artifact inspection.
- The binary 1.7B GGUF is natively packed at 1.125 bits/weight. Current ternary GGUF kernels use 2-bit code slots, so shipped storage is larger than the 1.71-bit information-theoretic size.
- The 27B paper claims extreme tolerance to later 4-bit KV-cache quantization, suggesting that the transformation learns broad discretization-noise robustness.

Primary PrismML sources: [1-bit announcement](https://prismml.com/news/bonsai-8b), [ternary announcement](https://prismml.com/news/ternary-bonsai), [27B announcement](https://prismml.com/news/bonsai-27b), and [demo/whitepaper repository](https://github.com/PrismML-Eng/Bonsai-demo).

## What our forensics establishes

The released 1.7B tensors are functionally optimized discrete checkpoints:

- across 1,409,286,144 comparable weights, binary/Qwen sign agreement is only 71.62%;
- ternary nonzero signs agree with Qwen 86.75% of the time and 39.89% of ternary codes are zero;
- binary and ternary releases agree on 89.76% of nonzero signs, much more than binary agrees with Qwen;
- released group scales are only moderately correlated with base mean-absolute scales and are about 2.36x/2.07x larger at the median for binary/ternary;
- binary relative Euclidean error is 1.690 versus 0.612 for groupwise RTN; ternary is 1.004 versus 0.443 for the alternating-MSE baseline;
- some non-quantized RMSNorm weights move substantially while others remain exactly unchanged.

These observations are incompatible with “take sign and store mean absolute
value.” They are compatible with a common trained trajectory, plausibly FP ->
ternary -> binary, although training order cannot be proven from final weights.

## Ranked method hypotheses

| Rank | Hypothesis | Current probability | Why |
|---:|---|---:|---|
| 1 | Progressive global QAT + teacher distillation + foldable group scales | 45% | Explains large sign movement, no runtime metadata, TPU training, and retained behavior |
| 2 | Curvature-preconditioned/projected discrete training + distillation | 25% | Explains the “mathematical” claim and founders' curvature/PSGD background |
| 3 | Hessian/activation-aware layerwise initialization followed by global QAT recovery | 20% | Fits known extreme-PTQ practice and scale/sign fingerprints |
| 4 | A more exotic equivalence transformation or discrepancy-balancing algorithm plus brief recovery | 8% | Possible Caltech IP angle, but no public artifact demands it |
| 5 | Pure PTQ without meaningful retraining | 2% | Strongly contradicted by tensor movement and TPU-training disclosure |

Probabilities are research priors, not confidence intervals. The experiments in
`02-reverse-engineering-hypotheses.md` are designed to move them.

## Recommended first milestone

Do not start at 8B. Convert Qwen3-0.6B and 1.7B to ternary g128 first, because
ternary is a much smoother optimization target and PrismML's quality loss is
smallest there. Require:

- hard-export perplexity within 10% of the base;
- average accuracy retention above 90% on a compact no-thinking suite;
- no more than two points lost on GSM8K and HumanEval+;
- exact ternary alphabet in every matrix tensor;
- a clean ablation showing that global distillation beats local reconstruction.

Then continue the winning ternary checkpoint to binary. Only after reproducing
the signature at 1.7B should OpenBonsai move to an 8–9B reasoning model such as
the previously selected Ornith-1.0-9B.
