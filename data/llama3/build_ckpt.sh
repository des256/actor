#!/usr/bin/env bash
# assumes Llama3 Instruct 3b (safetensors from HuggingFace) is available from data/llama3/source
set -euo pipefail

mkdir -p ckpt/

docker run --rm --user $(id -u):$(id -g) --gpus all -v .:/local -w /local actor python3 "/local/export.py" --model_dir "source/" --output_dir "ckpt/" --dtype float16 --use_weight_only --weight_only_precision int4

cp source/tokenizer.json ckpt/
