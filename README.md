<p align="center">
  <img src="https://github.com/Apothic-AI/OpenBonsai/blob/master/openbonsai-logo-nobg-512.png?raw=true" alt="OpenBonsai" width="220" />
</p>

<h1 align="center">OpenBonsai</h1>

<p align="center">
  <strong>Open reverse-engineering research for extreme low-bit LLM conversion</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-%3E%3D3.10-3776AB.svg" alt="Python" /></a>
  <a href="docs/00-executive-brief.md"><img src="https://img.shields.io/badge/docs-executive%20brief-0B3D2E.svg" alt="Docs" /></a>
  <a href="results/1.7b_forensics.md"><img src="https://img.shields.io/badge/results-1.7B%20forensics-6B4EFF.svg" alt="Results" /></a>
</p>

---

OpenBonsai is an open research program for reproducing the *conversion method* behind PrismML's Bonsai models: transforming an ordinary pretrained language model into an end-to-end groupwise **binary** or **ternary** model without destroying its useful behavior.

This repository is **not** a reimplementation of PrismML's proprietary method and does not claim access to it. It contains:

| What you get | Where |
| --- | --- |
| Evidence ledger & ranked hypotheses | [`docs/`](docs/) |
| Released-weight forensics | [`src/openbonsai/forensics.py`](src/openbonsai/forensics.py) |
| Reference quantizers & matrix lab | [`src/openbonsai/quantize.py`](src/openbonsai/quantize.py), [`matrix_lab.py`](src/openbonsai/matrix_lab.py) |
| Progressive QAT / distillation scaffold | [`src/openbonsai/qat.py`](src/openbonsai/qat.py), [`scripts/train_hf.py`](scripts/train_hf.py) |
| 1.7B forensic report | [`results/1.7b_forensics.md`](results/1.7b_forensics.md) |

## The most important finding

The released weights rule out ordinary round-to-nearest post-training quantization.

Our 1.7B comparison against Qwen3-1.7B finds that Bonsai:

- changes a **large fraction of discrete signs**
- deliberately moves **farther** from the base in Euclidean weight space than naïve binarization
- also moves some higher-precision norm parameters
- yet retains **much more task capability** than naïve 1-bit PTQ

The likely optimization target is therefore **function space** (teacher logits, hidden states, and/or task loss) — not weight reconstruction.

### Leading reconstruction hypothesis

1. **Activation / Hessian-aware** discrete initialization
2. A gradual **FP → ternary → binary** curriculum with hard-forward / smooth-backward QAT
3. Global **teacher-logit + intermediate-state** distillation
4. Learned **group-scale compensation**, folded into one FP16 scale at export
5. Explicit **activation / KV-noise** robustness training
6. A **curvature-aware** optimizer or preconditioner to stabilize the extreme discrete transition

> **Before spending compute**, read the [executive brief](docs/00-executive-brief.md), [hypothesis matrix](docs/02-reverse-engineering-hypotheses.md), [forensic methodology](docs/07-forensic-methodology.md), and [concrete recipe](docs/08-training-recipe-v0.md).

## Repository map

```text
OpenBonsai/
├── docs/                  evidence, hypotheses, plan, eval, recipe
├── src/openbonsai/
│   ├── formats.py         standalone GGUF / safetensors readers
│   ├── forensics.py       released-weight comparison pipeline
│   ├── quantize.py        binary / ternary + curvature-aware methods
│   ├── matrix_lab.py      curvature-aware matrix laboratory
│   └── qat.py             progressive-QAT building blocks (optional torch)
├── scripts/train_hf.py    small-scale HF distillation runner
├── configs/               phase-0 experiment configs
├── tests/                 dependency-light unit tests
└── results/               generated 1.7B forensic report + JSON
```

## Quick start

### Reproduce the artifact forensics

The comparison downloads about **4.2 GB**. The second Qwen safetensors shard is a duplicate tied LM head and is intentionally not needed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
./scripts/fetch_reference_artifacts.sh
./scripts/run_forensics.sh
```

Then inspect [`results/1.7b_forensics.md`](results/1.7b_forensics.md).

### Run the tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

### Run a pilot QAT experiment

Install optional dependencies and provide a JSONL calibration/training corpus with a `text` field:

```bash
pip install -e '.[training]'
accelerate launch scripts/train_hf.py configs/phase0_ternary.json
```

The included runner is a **hypothesis-discrimination harness**, not a claim of production-scale training. A real 1.7B+ run should shard the student, cache teacher targets or isolate the teacher on separate workers, and checkpoint optimizer state.

## Project rules

- Compare against exact same-base RTN, GPTQ/AWQ, and full-precision checkpoints
- Report both theoretical information rate and actual shipped / resident bytes
- Quantize embeddings and the LM head; disclose every exception
- Evaluate the **hard exported checkpoint**, never only fake-quant shadow weights
- Use identical prompts, decoding, context, and scoring across precision variants
- Treat every PrismML claim as a claim until independently reproduced

## Documentation index

| Doc | Topic |
| --- | --- |
| [00 — Executive brief](docs/00-executive-brief.md) | High-level findings |
| [01 — Evidence ledger](docs/01-evidence-ledger.md) | Sourced claims & observations |
| [02 — Hypotheses](docs/02-reverse-engineering-hypotheses.md) | Ranked, falsifiable matrix |
| [03 — Research plan](docs/03-research-plan.md) | Experiment roadmap |
| [04 — Evaluation protocol](docs/04-evaluation-protocol.md) | Controls & metrics |
| [05 — Implementation design](docs/05-implementation-design.md) | System design |
| [06 — Source notes](docs/06-source-notes.md) | External references |
| [07 — Forensic methodology](docs/07-forensic-methodology.md) | Weight analysis method |
| [08 — Training recipe v0](docs/08-training-recipe-v0.md) | Concrete conversion recipe |
| [09 — Roadmap](docs/09-roadmap.md) | Near-term priorities |

---

<p align="center">
  <sub>Apache-2.0 · Third-party model licenses and dataset terms still apply</sub>
</p>
