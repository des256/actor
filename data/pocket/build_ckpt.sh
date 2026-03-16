#!/usr/bin/env bash
# assumes Pocket TTS (safetensors from HuggingFace) is available from data/pocket/source
# requires: pip install pocket-tts onnx safetensors
set -euo pipefail

mkdir -p ckpt/
if [[ ! -f "source/tts_b6369a24.safetensors" ]]; then
    echo "missing tts_b6369a24.safetensors in source/"
    exit 1
fi

TEMP_DIR=$(mktemp -d -t pocket-ckpt-XXXX)
trap 'rm -rf "$TEMP_DIR"' EXIT

python3 "export.py" "source/" "$TEMP_DIR"
python3 "consolidate.py" "$TEMP_DIR" "ckpt/"

cp source/tokenizer.model ckpt/
cp source/tokenizer.json ckpt/
