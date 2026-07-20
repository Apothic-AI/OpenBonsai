# Released-weight forensic methodology

## Question

Can the public 1.7B Bonsai weights be explained by ordinary groupwise rounding
of Qwen3-1.7B, or did the conversion materially optimize the model after the
low-bit constraint was introduced?

## Inputs

The fetch script pins three public artifacts by SHA-256. The BF16 shard contains
the complete transformer matrices and embedding; Qwen's second shard repeats
the tied output head, so it is unnecessary for matrix comparison. Binary and
ternary files are compared only where tensor names and element counts map
unambiguously to the Qwen base.

The released embedding is excluded from numerical comparison because the
Bonsai vocabulary has 267 fewer rows and the removed-token mapping has not been
established. All 196 transformer matrices are included. The analysis also
compares 113 higher-precision normalization tensors.

## Decoder validation

`openbonsai.formats` parses the GGUF header and memory maps tensor payloads. It
implements the observed layouts:

- binary Q1_0 group-128: FP16 scale followed by 128 packed sign bits;
- PrismML ternary Q2_0 group-128: FP16 scale followed by 128 two-bit slots,
  decoded as `{-1,0,+1}`; the fourth state is counted as invalid.

Synthetic conformance tests verify bit order, code mapping, FP16 scales, scalar
tensors, and BF16 safetensors decoding. The full run observed zero invalid
ternary states.

## Measurements

For each 128-value group with Qwen weights `w`, the pipeline computes:

- binary/base sign agreement;
- ternary/base and binary/ternary sign agreement on nonzero ternary codes;
- ternary zero occupancy;
- released scale divided by `mean(abs(w))`;
- scale correlation;
- relative Euclidean error of released tensors;
- relative Euclidean error of same-base binary RTN;
- relative Euclidean error of an alternating code/scale ternary MSE solver.

Aggregates are weighted by individual weights, not by layers. The tool streams
memory-mapped chunks and accumulates sums of squares in float64. Per-family and
per-tensor metrics remain in the JSON.

## Result

| Metric | Released binary | Binary RTN | Released ternary | Ternary MSE baseline |
|---|---:|---:|---:|---:|
| Relative weight error | 1.690 | 0.612 | 1.004 | 0.443 |
| Base sign agreement | 71.62% | 100% | 86.75% nonzero | construction-dependent |
| Zero fraction | 0% | 0% | 39.89% | construction-dependent |

The released models are dramatically worse weight reconstructors than cheap
baselines. That is not evidence of a failed conversion: the public models
retain useful behavior that ordinary 1-bit rounding does not. It is evidence
that Euclidean closeness to Qwen was not the final optimization target.

The 89.76% binary/ternary agreement on nonzero signs is especially informative.
It supports a common optimization trajectory, possibly ternary then binary.
Final weights cannot determine temporal direction, so this is a ranked
hypothesis rather than a historical claim.

The agreement is depth-structured: it climbs from 85.4% in block 0 to a 92.4%
peak in block 17, then falls to 84.5% in block 27. Binary/base sign agreement
also declines from 72.5% to 68.5% from first to last block. A reconstruction
should therefore reproduce layerwise behavior, not only one global average.

## Threats to validity

- The declared ancestor could differ slightly from the public Qwen checkpoint.
- Equivalent permutations, sign gauges, or channel rescalings could exaggerate
  direct weight-space distance while preserving function.
- Final weights cannot reveal the corpus, loss, optimizer, token budget, or
  sequence of conversion stages uniquely.
- The analysis establishes transformation, not PrismML benchmark validity.
- Embedding rows require a tokenizer-diff audit before comparison.

The next discriminators are activation/Fisher statistics and adjacent-layer
alignment. If channel symmetries explain most movement, H4 rises; if sign flips
concentrate in low-curvature directions, H2/H3 rise; otherwise global QAT and
distillation remain the parsimonious explanation.
