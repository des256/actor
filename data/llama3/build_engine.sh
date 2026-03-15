#!/usr/bin/env bash
# run after build_ckpt.sh to build the engine locally
set -euo pipefail

mkdir -p engine

docker run --rm --user $(id -u):$(id -g) --gpus all -v .:/local -w /local actor trtllm-build \
    --checkpoint_dir "/local/ckpt/" \
    --output_dir "/local/engine/" \
    --max_batch_size 1024 \
    --max_seq_len 2048 \
    --gpt_attention_plugin auto \
    --gemm_plugin auto \
    --paged_kv_cache enable \
    --remove_input_padding enable \
    --context_fmha enable

cp ckpt/tokenizer.json engine/
