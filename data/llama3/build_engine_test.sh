#!/usr/bin/env bash
set -euo pipefail

NUM_LAYERS=28
NUM_KV_HEADS=8
HEAD_DIM=128

min_shapes="input_ids:1x1,position_ids:1x1"
opt_shapes="input_ids:1x128,position_ids:1x128"
max_shapes="input_ids:1x2048,position_ids:1x2048"

for i in $(seq 0 $((NUM_LAYERS - 1))); do
    min_shapes="${min_shapes},past_key_values.${i}.key:1x${NUM_KV_HEADS}x0x${HEAD_DIM}"
    min_shapes="${min_shapes},past_key_values.${i}.value:1x${NUM_KV_HEADS}x0x${HEAD_DIM}"
    opt_shapes="${opt_shapes},past_key_values.${i}.key:1x${NUM_KV_HEADS}x1024x${HEAD_DIM}"
    opt_shapes="${opt_shapes},past_key_values.${i}.value:1x${NUM_KV_HEADS}x1024x${HEAD_DIM}"
    max_shapes="${max_shapes},past_key_values.${i}.key:1x${NUM_KV_HEADS}x2048x${HEAD_DIM}"
    max_shapes="${max_shapes},past_key_values.${i}.value:1x${NUM_KV_HEADS}x2048x${HEAD_DIM}"
done

echo "Building with --stronglyTyped --builderOptimizationLevel=1..."
trtexec \
    --onnx="onnx/model.onnx" \
    --saveEngine="engine/model_strongly_typed.engine" \
    --minShapes="${min_shapes}" \
    --optShapes="${opt_shapes}" \
    --maxShapes="${max_shapes}" \
    --stronglyTyped \
    --builderOptimizationLevel=1 \
    --memPoolSize=workspace:4096 \
    2>&1 | tail -30
