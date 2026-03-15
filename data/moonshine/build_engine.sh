#!/usr/bin/env bash
# run after build_ckpt.sh to build the engine locally
set -euo pipefail

mkdir -p engine
if [[ ! -f "ckpt/encoder_model.onnx" ]]; then
    echo "missing encoder_model.onnx in ckpt/, run build_ckpt.sh first"
    exit 1
fi
if [[ ! -f "ckpt/decoder_model.onnx" ]]; then
    echo "missing decoder_model.onnx in ckpt/, run build_ckpt.sh first"
    exit 1
fi

docker run --rm --user $(id -u):$(id -g) --gpus all -v .:/local actor trtexec \
    --onnx="/local/ckpt/encoder_model.onnx" \
    --saveEngine="/local/engine/encoder.engine" \
    --minShapes="input_values:1x32000,attention_mask:1x32000" \
    --optShapes="input_values:1x32000,attention_mask:1x32000" \
    --maxShapes="input_values:1x32000,attention_mask:1x32000" \
    --memPoolSize=workspace:4096 \
    --fp16

docker run --rm --user $(id -u):$(id -g) --gpus all -v .:/local actor trtexec \
    --onnx="/local/ckpt/decoder_model.onnx" \
    --saveEngine="/local/engine/decoder.engine" \
    --minShapes="input_ids:1x1,encoder_hidden_states:1x100x768,encoder_attention_mask:1x100,k_self:14x1x10x0x64,v_self:14x1x10x0x64" \
    --optShapes="input_ids:1x1,encoder_hidden_states:1x100x768,encoder_attention_mask:1x100,k_self:14x1x10x32x64,v_self:14x1x10x32x64" \
    --maxShapes="input_ids:1x1,encoder_hidden_states:1x100x768,encoder_attention_mask:1x100,k_self:14x1x10x64x64,v_self:14x1x10x64x64" \
    --memPoolSize=workspace:4096 \
    --fp16 --int8 --int4

cp ckpt/tokenizer.json engine/
