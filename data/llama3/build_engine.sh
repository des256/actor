#!/usr/bin/env bash
# Build TensorRT engine from ONNX model using trtexec (no Docker)
set -euo pipefail

# Check for ONNX model
if [[ ! -f "onnx/model.onnx" ]]; then
    echo "ERROR: onnx/model.onnx not found"
    echo "Run 'python3 export.py source/ onnx/' first"
    exit 1
fi

# Check for tokenizer
if [[ ! -f "source/tokenizer.json" ]]; then
    echo "ERROR: source/tokenizer.json not found"
    exit 1
fi

# Check for trtexec
if ! command -v trtexec &> /dev/null; then
    echo "ERROR: trtexec not found in PATH"
    echo "Install TensorRT and add trtexec to PATH, or run via Docker:"
    echo "  docker run --rm --gpus all -v \$(pwd):/work -w /work nvcr.io/nvidia/tensorrt:24.02-py3 bash build_engine.sh"
    exit 1
fi

mkdir -p engine

# Architecture constants (must match export.py)
NUM_LAYERS=28
NUM_KV_HEADS=8
HEAD_DIM=128

# Build shape specification strings programmatically
# Format: tensor1:shape1,tensor2:shape2,...
echo "Generating shape specifications for ${NUM_LAYERS} layers..."

min_shapes="input_ids:1x1,position_ids:1x1"
opt_shapes="input_ids:1x128,position_ids:1x128"
max_shapes="input_ids:1x2048,position_ids:1x2048"

for i in $(seq 0 $((NUM_LAYERS - 1))); do
    # KV cache shapes: [batch=1, num_kv_heads=8, seq_len, head_dim=128]
    min_shapes="${min_shapes},past_key_values.${i}.key:1x${NUM_KV_HEADS}x0x${HEAD_DIM}"
    min_shapes="${min_shapes},past_key_values.${i}.value:1x${NUM_KV_HEADS}x0x${HEAD_DIM}"

    opt_shapes="${opt_shapes},past_key_values.${i}.key:1x${NUM_KV_HEADS}x1024x${HEAD_DIM}"
    opt_shapes="${opt_shapes},past_key_values.${i}.value:1x${NUM_KV_HEADS}x1024x${HEAD_DIM}"

    max_shapes="${max_shapes},past_key_values.${i}.key:1x${NUM_KV_HEADS}x2048x${HEAD_DIM}"
    max_shapes="${max_shapes},past_key_values.${i}.value:1x${NUM_KV_HEADS}x2048x${HEAD_DIM}"
done

echo "✓ Generated shape specs for 58 inputs (input_ids + position_ids + ${NUM_LAYERS}×2 KV tensors)"

# Build the engine
echo "Building TensorRT engine (strongly typed — preserves FP32 RMSNorm)..."
echo "  This will take several minutes..."

trtexec \
    --onnx="onnx/model.onnx" \
    --saveEngine="engine/model.engine" \
    --minShapes="${min_shapes}" \
    --optShapes="${opt_shapes}" \
    --maxShapes="${max_shapes}" \
    --stronglyTyped \
    --builderOptimizationLevel=3 \
    --memPoolSize=workspace:4096 \
    --verbose

if [[ $? -ne 0 ]]; then
    echo "ERROR: trtexec failed"
    exit 1
fi

# Copy tokenizer to engine directory
cp source/tokenizer.json engine/

echo ""
echo "✓ Build complete:"
echo "  Engine: engine/model.engine ($(du -h engine/model.engine | cut -f1))"
echo "  Tokenizer: engine/tokenizer.json"
