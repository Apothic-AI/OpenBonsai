#!/usr/bin/env bash
set -euo pipefail

ARTIFACTS="${1:-artifacts}"
PYTHONPATH=src python -m openbonsai.forensics \
  --binary "$ARTIFACTS/Bonsai-1.7B-Q1_0.gguf" \
  --ternary "$ARTIFACTS/Ternary-Bonsai-1.7B-Q2_0.gguf" \
  --base "$ARTIFACTS/Qwen3-1.7B-base.safetensors" \
  --json results/1.7b_forensics.json \
  --markdown results/1.7b_forensics.md

