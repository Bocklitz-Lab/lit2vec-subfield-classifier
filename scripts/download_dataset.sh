#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ===
HF_DATASET="Bocklitz-Lab/lit2vec-subfield-classifier-dataset"
OUT_DIR="data"
SPLITS=("train" "validation" "test")

mkdir -p "$OUT_DIR"

echo "Downloading dataset from Hugging Face: $HF_DATASET"

for split in "${SPLITS[@]}"; do
    URL="https://huggingface.co/datasets/${HF_DATASET}/resolve/main/data/${split}.jsonl"
    OUT_FILE="${OUT_DIR}/${split}.jsonl"
    echo "  ↳ ${split} -> ${OUT_FILE}"
    curl -L -o "$OUT_FILE" "$URL"
    # quick JSONL sanity check
    head -n 1 "$OUT_FILE" | jq . >/dev/null 2>&1 \
      && echo "    ${split}.jsonl looks valid" \
      || echo "    Warning: ${split}.jsonl might be malformed"
done

echo "Dataset ready in ${OUT_DIR}/"
