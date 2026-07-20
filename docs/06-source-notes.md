# Source notes and claim map

This project distinguishes statements in PrismML material from independent
measurements and from OpenBonsai hypotheses. Access dates are 2026-07-19.

## PrismML primary material

- [The First 1-Bit LLM](https://prismml.com/news/bonsai-8b) describes the
  binary 8B launch, Qwen3-8B ancestry, end-to-end low-bit coverage, and public
  benchmark claims.
- [The World's First Ternary LLM](https://prismml.com/news/ternary-bonsai)
  describes the ternary 8B release and its `{-s,0,+s}` representation.
- [Extreme quantization at 27B](https://prismml.com/news/bonsai-27b) describes
  the 27B representation transformation, binary/ternary variants, vision
  component, KV-cache experiments, and benchmark protocol.
- [PrismML Bonsai demo repository](https://github.com/PrismML-Eng/Bonsai-demo/)
  contains runtime examples, GGUF/MLX links, and the three whitepapers supplied
  with the releases.
- [1-bit Bonsai 8B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/957a1a6eebc4faac724f3560add0505e3a95cc3e/1-bit-bonsai-8b-whitepaper.pdf)
  specifies 36 decoder blocks, group size 128, FP16 group scales, binary code
  coverage, and reported sizes.
- [Ternary Bonsai 8B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/957a1a6eebc4faac724f3560add0505e3a95cc3e/ternary-bonsai-8b-whitepaper.pdf)
  specifies the ternary representation and effective information rate.
- [Bonsai 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/957a1a6eebc4faac724f3560add0505e3a95cc3e/bonsai-27b-whitepaper.pdf)
  uses the phrase “representation transformation,” states the Qwen3.6-27B
  starting point, and documents physical versus ideal storage.
- [Launch announcement](https://www.prnewswire.com/news-releases/prismml-launches-worlds-first-1-bit-ai-model-to-redefine-intelligence-at-the-edge-302730568.html)
  states that the 8B model was trained on Google v4 TPUs. This is evidence
  against a purely local rounding pipeline, not proof of a particular loss.

## Publicly released checkpoints used in forensics

The reproducible manifest in `scripts/fetch_reference_artifacts.sh` pins these
artifacts by URL and SHA-256:

- `PrismML/Bonsai-1.7B-GGUF`, binary Q1_0;
- `PrismML/Ternary-Bonsai-1.7B-GGUF`, ternary Q2_0 group-128;
- `Qwen/Qwen3-1.7B`, BF16 transformer shard.

The Hugging Face files themselves are not redistributed in this project. The
forensic JSON records input paths and the script verifies hashes.

## Adjacent primary research

- [Rethinking 1-bit Optimization](https://arxiv.org/abs/2508.06974):
  progressive smooth-to-hard binary training and dual-scale compensation.
- [OneBit](https://arxiv.org/abs/2402.11295): distillation-aware one-bit LLM
  quantization.
- [EfficientQAT](https://arxiv.org/abs/2407.11062): efficient end-to-end QAT for
  pretrained LLMs.
- [BiLLM](https://arxiv.org/abs/2402.04291) and
  [PB-LLM](https://arxiv.org/abs/2310.00034): binary/post-training reference
  methods. Their side-information choices are not necessarily format-equivalent
  to Bonsai.
- [LeanQuant](https://arxiv.org/abs/2407.10032),
  [DiscQuant](https://arxiv.org/abs/2501.06417), and
  [NanoQuant](https://arxiv.org/abs/2602.06694): activation-, discrepancy-, and
  reconstruction-oriented low-bit alternatives.

These papers are sources of candidate mechanisms and baselines, not evidence
that PrismML used them.

## Team/provenance clues

PrismML publicly lists Sahin Lale and Omead Pooladzandi as research leads. A
public [AI Council speaker biography](https://aicouncil.com/speakers/omead-pooladzandi)
mentions curvature-informed optimization and preconditioned stochastic gradient
descent. This raises the prior for a curvature-aware optimizer but remains
circumstantial. OpenBonsai therefore assigns it an ablation, not a conclusion.

## Known comparability problems in the public reports

- The 27B benchmark table uses temperature 1.0 for Qwen references and 0.7 for
  Bonsai while describing the setup as matched.
- The 8B and 27B reports do not use one stable task/scoring suite; for example,
  BFCL coverage changes.
- “Ideal,” packed file, and resident runtime size are different quantities.
  The 27B ternary report explicitly distinguishes ideal information storage
  from two-bit-slot deployment storage.
- The 27B model family introduces a Qwen3.6 label and a separate vision module,
  so results should not be merged casually with earlier text-only Qwen3-8B
  releases.
- Public benchmark tables do not expose run-to-run variance, full prompts, all
  sampling seeds, or an independent score reproduction.

Accordingly, OpenBonsai treats the papers as specifications for hypotheses and
format targets, not as a validated evaluation oracle.

