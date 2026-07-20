# Evidence ledger

Labels: **observed** means directly disclosed or measured; **inference** means a
reasoned interpretation; **unknown** means the public record does not resolve it.

## Model representation and origin

| ID | Status | Evidence | Implication |
|---|---|---|---|
| E01 | Observed | The 8B paper describes Qwen3-8B with unchanged architecture and matrix-wide Q1_0 g128. | The result is a converted pretrained architecture, not a custom BitNet architecture. |
| E02 | Observed | The 27B paper explicitly says it starts from off-the-shelf Qwen3.6-27B and applies a “representation transformation.” | Rules out from-scratch-only training as the Bonsai recipe. |
| E03 | Observed | A [launch release](https://www.prnewswire.com/news-releases/prismml-launches-worlds-first-1-bit-ai-model-to-redefine-intelligence-at-the-edge-302730568.html) says the model was trained using Google v4 TPUs. | Strong evidence of a recovery/QAT training phase, not only offline packing. |
| E04 | Observed | Binary is `{-s,+s}`, ternary `{-s,0,+s}`, group size 128, FP16 scale. | Export format is fully reconstructable. |
| E05 | Observed | Embeddings, attention, MLP, and LM head are low-bit; normalization and scale metadata remain higher precision. | OpenBonsai must quantize every large matrix to make a fair claim. |
| E06 | Measured | Released 1.7B embedding rows: 151,669; base Qwen3-1.7B: 151,936. | “Architecture unchanged” has at least one small packaging/model exception. Token mapping must be audited. |

## Learned-transformation fingerprints

| ID | Status | Evidence | Implication |
|---|---|---|---|
| E07 | Measured | Across 1,409,286,144 comparable weights, binary/Qwen sign agreement is 71.62%. | Not sign-rounding PTQ. |
| E08 | Measured | Ternary/Qwen nonzero sign agreement is 86.75%; binary/ternary nonzero agreement is 89.76%; ternary zero occupancy is 39.89%. | A shared trajectory and ternary bridge are plausible, though not proven. |
| E09 | Measured | Binary relative weight error is 1.690 versus 0.612 for RTN; ternary is 1.004 versus 0.443 for an alternating-MSE baseline. | The optimizer accepts larger parameter error to reduce functional/task error. |
| E10 | Measured | Some non-quantized `attn_norm` weights move; some later norms remain bit-identical to the base. | Recovery is structured/selective, not a global blind rescale. |
| E11 | Observed | 27B Bonsai is much more tolerant of 4-bit KV-cache noise than the FP16 and Q4 references, per the PrismML paper. | A robustness/noise-shaping loss is plausible; independent replication is required. |

Reproduce E06–E10 with `scripts/run_forensics.sh`; the generated machine-readable
measurements live in `results/1.7b_forensics.json`.

## Evaluation and reporting caveats

| ID | Status | Evidence | Risk/control |
|---|---|---|---|
| C01 | Observed | The 27B evaluation uses temperature 1.0 for Qwen references and 0.7 for Bonsai. | “Matched decoding” is not literal. Re-evaluate every variant at both settings and a shared deterministic setting. |
| C02 | Observed | PrismML computes 27B ternary intelligence density using the ideal 5.9 GB, while the current resident GGUF language model is about 7.17 GB. | Report theoretical and physically deployed density separately. |
| C03 | Observed | The April 8B ternary paper headlines 1.75 GB ideal size but lists 2.16 GiB deployed MLX size. | Do not call information entropy a deployable file size. |
| C04 | Observed | The 8B paper excludes BFCL multi-turn; the 27B paper includes it and changes several scoring conventions. | Never compare paper averages across generations without re-running a common harness. |
| C05 | Observed | Several tasks use an LLM fallback judge. | Archive raw generations and report rule-only plus judge-recovered scores. |
| C06 | Unknown | Training-set overlap and benchmark contamination controls are not disclosed. | Use held-out post-training tasks and canary/decontamination checks. |
| C07 | Unknown | Total tokens, data mixture, optimizer, loss, schedule, and training FLOPs are proprietary. | These are the core variables OpenBonsai must identify experimentally. |

## Team and research provenance

The [PrismML team page](https://prismml.com/about) identifies Sahin Lale and
Omead Pooladzandi as co-heads of research. A public [speaker biography](https://aicouncil.com/speakers/omead-pooladzandi)
describes Pooladzandi's work as curvature-informed optimization and PSGD. Babak
Hassibi's classic Optimal Brain Surgeon work is foundational second-order model
compression. This makes curvature-aware training a rational hypothesis, but it
is circumstantial: no public source says Bonsai uses PSGD, OBS, K-FAC, Shampoo,
or any particular Hessian approximation.

## Public wording that should not be over-read

- “Mathematically grounded” is marketing-compatible and does not identify an algorithm.
- “Architecture unchanged” does not mean weights are rounded without training.
- “True 1-bit” refers to matrix-weight representation; activations and accumulators are higher precision.
- “1.58-bit deployed” is currently information-theoretic for ternary codes; released fast kernels store them in 2-bit slots.
- High benchmark retention alone cannot distinguish QAT, distillation, or a sufficiently strong discrete optimizer.
