# OpenBonsai

OpenBonsai is an open research program for reproducing the *conversion method*
behind PrismML's Bonsai models: transforming an ordinary pretrained language
model into an end-to-end groupwise binary or ternary model without destroying
its useful behavior.

This repository is not a reimplementation of PrismML's proprietary method and
does not claim access to it. It contains an evidence ledger, ranked and
falsifiable hypotheses, an independent evaluation design, released-weight
forensics, reference quantizers, a curvature-aware matrix laboratory, and an
experimental progressive-QAT/distillation scaffold.

## The most important finding

The released weights rule out ordinary round-to-nearest post-training
quantization. Our 1.7B comparison against Qwen3-1.7B finds that Bonsai changes a
large fraction of discrete signs and deliberately moves *farther* from the base
in Euclidean weight space than naïve binarization. Some higher-precision norm
parameters move too. Yet the model retains much more task capability than naïve
1-bit PTQ. The likely target of the optimization is therefore function space
(teacher logits, hidden states, and/or task loss), not weight reconstruction.

The current leading reconstruction hypothesis is:

1. activation/Hessian-aware discrete initialization;
2. a gradual FP -> ternary -> binary curriculum with hard-forward/smooth-backward QAT;
3. global teacher-logit plus intermediate-state distillation;
4. learned group-scale compensation folded into one FP16 scale at export;
5. explicit activation/KV-noise robustness training;
6. a curvature-aware optimizer or preconditioner to stabilize the extreme discrete transition.

Read the [executive brief](docs/00-executive-brief.md), [hypothesis matrix](docs/02-reverse-engineering-hypotheses.md), [forensic methodology](docs/07-forensic-methodology.md), and [concrete recipe](docs/08-training-recipe-v0.md) before spending compute.

## Repository map

- `docs/`: evidence, hypotheses, experiment plan, evaluation controls, and recipe
- `src/openbonsai/formats.py`: standalone GGUF/safetensors artifact readers
- `src/openbonsai/forensics.py`: released-weight comparison pipeline
- `src/openbonsai/quantize.py`: binary/ternary and curvature-aware reference methods
- `src/openbonsai/qat.py`: optional PyTorch progressive-QAT building blocks
- `scripts/train_hf.py`: small-scale Hugging Face distillation runner
- `tests/`: dependency-light unit tests
- `results/`: generated 1.7B forensic report and JSON

## Reproduce the artifact forensics

The comparison downloads about 4.2 GB. The second Qwen safetensors shard is a
duplicate tied LM head and is intentionally not needed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
./scripts/fetch_reference_artifacts.sh
./scripts/run_forensics.sh
```

Then inspect `results/1.7b_forensics.md`.

## Run the tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Run a pilot QAT experiment

Install optional dependencies and provide a JSONL calibration/training corpus
with a `text` field:

```bash
pip install -e '.[training]'
accelerate launch scripts/train_hf.py configs/phase0_ternary.json
```

The included runner is intentionally a hypothesis-discrimination harness, not
a claim of production-scale training. A real 1.7B+ run should shard the student,
cache teacher targets or isolate the teacher on separate workers, and checkpoint
the optimizer state.

## Project rules

- Compare against exact same-base RTN, GPTQ/AWQ, and full-precision checkpoints.
- Report both theoretical information rate and actual shipped/resident bytes.
- Quantize embeddings and the LM head; disclose every exception.
- Evaluate the hard exported checkpoint, never only fake-quant shadow weights.
- Use identical prompts, decoding, context, and scoring across precision variants.
- Treat every PrismML claim as a claim until independently reproduced.

Apache-2.0. Third-party model licenses and dataset terms still apply.
