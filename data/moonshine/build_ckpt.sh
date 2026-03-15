#!/usr/bin/env bash
# assumes Moonshine Streaming Medium (safetensors from HuggingFace) is available from data/moonshine/source
set -euo pipefail

mkdir -p ckpt/
if [[ ! -f "source/model.safetensors" ]]; then
    echo "missing model.safetensors in source/"
    exit 1
fi

TEMP_DIR=$(mktemp -d -t moonshine-ckpt-XXXX)
trap 'rm -rf "$TEMP_DIR"' EXIT

python3 "export.py" "source/" "$TEMP_DIR"
python3 "consolidate.py" "$TEMP_DIR" "ckpt/"

cp source/tokenizer.json ckpt/
