# Roadmap and decision gates

## Milestone A — evidence package (complete in this snapshot)

- archive the public claim map and evaluation caveats;
- parse released binary/ternary GGUF and BF16 safetensors directly;
- compare signs, zeros, scales, Euclidean error, and norm movement;
- publish falsifiable, probability-ranked hypotheses;
- ship unit-tested RTN and curvature-aware reference solvers.

Exit artifact: `results/1.7b_forensics.md` plus machine-readable JSON.

## Milestone B — invariance and curvature forensics

1. Capture Qwen3-1.7B activations and gradient/Fisher diagonals on an auditable
   calibration sample.
2. Test whether released sign flips occur preferentially in low-sensitivity
   weights and groups.
3. Fit adjacent-layer channel permutations, sign gauges, and reciprocal scales.
4. Re-run forensic similarity after alignment.

Decision:

- large alignment gain → prioritize equivalence transforms (H4);
- strong curvature separation → prioritize preconditioned/projected training
  (H2/H3);
- neither → prioritize global distillation curriculum (H1).

## Milestone C — small-model ablation

Implement a reproducible 100M–600M benchmark with cached teacher targets and a
strict exporter. Run schedule, loss, initialization, and optimizer ablations.
Select by hard-checkpoint quality per conversion FLOP.

Minimum report: config, data hash, seeds, learning curves, code occupancy,
sign-crossing maps, full precision exceptions, actual bytes, and confidence
intervals.

## Milestone D — 1.7B reconstruction

Train ternary first, then binary continuation. Compare forensic fingerprints
with the public releases without optimizing directly against their weights:

- sign agreement to the Qwen ancestor;
- binary–ternary path agreement;
- zero fraction by tensor family;
- group-scale/base-scale correlation;
- norm movement by depth;
- functional KL and layerwise drift.

Matching fingerprints without matching function is failure; matching function
with different fingerprints is a valuable alternative method.

## Milestone E — reasoning-scale model

Scale the winning recipe to 8–9B. Require fixed evaluation conditions, hard
export, three-seed targeted probes where feasible, and an honest conventional
2–4 bit Pareto baseline. Publish conversion FLOPs and energy rather than
treating model size as the only cost.

## Milestone F — deployable open release

- versioned model card and training-data manifest;
- strict code/scale format specification;
- deterministic reference decoder and conformance vectors;
- conversion recipe and ablation report;
- model weights under compatible licenses;
- independent benchmark reproduction;
- CPU/Metal/CUDA kernels or a version-pinned runtime integration.

## Immediate issue queue

| Priority | Work item | Evidence gained |
|---:|---|---|
| P0 | Run full 1.7B forensic pipeline | establishes aggregate transformation fingerprint |
| P0 | Add hard exporter/loader round-trip | prevents fake-quant-only success |
| P0 | Build fixed 100M–600M eval harness | makes ablations comparable |
| P1 | Capture activation/Fisher statistics | discriminates H1 vs H2/H3 |
| P1 | Implement channel alignment solver | tests H4 |
| P1 | Cache sparse teacher targets | enables affordable 1.7B recovery |
| P2 | Add KV fake-quant/noise module | tests reported cache robustness |
| P2 | Benchmark runtime kernels | separates information rate from actual speed |

## Research integrity rule

The goal is an independently reproducible technique inspired by public facts,
not an assertion that private source code or trade secrets have been recovered.
Every result should preserve the labels **reported**, **measured**, **inferred**,
or **hypothesized**.

