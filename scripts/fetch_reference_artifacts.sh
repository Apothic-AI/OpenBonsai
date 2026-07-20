#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${1:-artifacts}"
mkdir -p "$DESTINATION"

curl -L --fail --retry 3 -o "$DESTINATION/Bonsai-1.7B-Q1_0.gguf" \
  "https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B-Q1_0.gguf"
curl -L --fail --retry 3 -o "$DESTINATION/Ternary-Bonsai-1.7B-Q2_0.gguf" \
  "https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf/resolve/main/Ternary-Bonsai-1.7B-Q2_0.gguf"
curl -L --fail --retry 3 -o "$DESTINATION/Qwen3-1.7B-base.safetensors" \
  "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/model-00001-of-00002.safetensors"

cd "$DESTINATION"
sha256sum -c <<'CHECKSUMS'
3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3  Bonsai-1.7B-Q1_0.gguf
d97d94eb564590c9f0300e54d3f87bbbb25a78693d0ade9f6e177973dcb8228a  Ternary-Bonsai-1.7B-Q2_0.gguf
169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5  Qwen3-1.7B-base.safetensors
CHECKSUMS

