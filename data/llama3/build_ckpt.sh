#!/usr/bin/env bash
# assumes Llama 3.2 3B Instruct (safetensors from HuggingFace) is available in data/llama3/source
set -euo pipefail

if [[ ! -d "source/" ]]; then
    echo "missing source/ directory with Llama 3.2 safetensors"
    exit 1
fi

mkdir -p onnx/

python3 "export.py" "source/" "onnx/"

cp source/tokenizer.json onnx/
