#!/usr/bin/env bash
# run after build_ckpt.sh to build TensorRT engines locally
set -euo pipefail

mkdir -p engine
for model in mimi_encoder_model backbone_model flow_net_model mimi_decoder_model; do
    if [[ ! -f "ckpt/${model}.onnx" ]]; then
        echo "missing ${model}.onnx in ckpt/, run build_ckpt.sh first"
        exit 1
    fi
done

# -- Mimi Encoder (voice cloning) --
docker run --rm --user $(id -u):$(id -g) --gpus all -v .:/local actor trtexec \
    --onnx="/local/ckpt/mimi_encoder_model.onnx" \
    --saveEngine="/local/engine/mimi_encoder.engine" \
    --minShapes="audio:1x1x1920" \
    --optShapes="audio:1x1x48000" \
    --maxShapes="audio:1x1x480000" \
    --memPoolSize=workspace:512 \
    --fp16 \
    --builderOptimizationLevel=5

# -- Backbone (autoregressive transformer) --
docker run --rm --user $(id -u):$(id -g) --gpus all -v .:/local actor trtexec \
    --onnx="/local/ckpt/backbone_model.onnx" \
    --saveEngine="/local/engine/backbone.engine" \
    --minShapes="input_emb:1x1x1024,k_self:6x1x16x0x64,v_self:6x1x16x0x64" \
    --optShapes="input_emb:1x1x1024,k_self:6x1x16x100x64,v_self:6x1x16x100x64" \
    --maxShapes="input_emb:1x200x1024,k_self:6x1x16x500x64,v_self:6x1x16x500x64" \
    --memPoolSize=workspace:512 \
    --fp16 \
    --builderOptimizationLevel=5

# -- Flow Net (flow-matching MLP) --
docker run --rm --user $(id -u):$(id -g) --gpus all -v .:/local actor trtexec \
    --onnx="/local/ckpt/flow_net_model.onnx" \
    --saveEngine="/local/engine/flow_net.engine" \
    --minShapes="conditioning:1x1024,s:1x1,t:1x1,x:1x32" \
    --optShapes="conditioning:1x1024,s:1x1,t:1x1,x:1x32" \
    --maxShapes="conditioning:1x1024,s:1x1,t:1x1,x:1x32" \
    --memPoolSize=workspace:512 \
    --fp16 \
    --builderOptimizationLevel=5

# -- Mimi Decoder (audio synthesis) --
docker run --rm --user $(id -u):$(id -g) --gpus all -v .:/local actor trtexec \
    --onnx="/local/ckpt/mimi_decoder_model.onnx" \
    --saveEngine="/local/engine/mimi_decoder.engine" \
    --minShapes="latent:1x32x1" \
    --optShapes="latent:1x32x1" \
    --maxShapes="latent:1x32x100" \
    --memPoolSize=workspace:512 \
    --fp16 \
    --builderOptimizationLevel=5

cp ckpt/tokenizer.model engine/
cp ckpt/tokenizer.json engine/
cp ckpt/metadata.safetensors engine/
