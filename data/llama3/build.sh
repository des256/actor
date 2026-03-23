#!/bin/bash
# build Llama 3.2 3B engines

REBUILD=false

# parse arguments
for arg in "$@"; do
    if [ "$arg" == "--rebuild" ]; then
        REBUILD=true
    fi
done

# make sure we're running inside the virtual environment
if [ ! -n "$VIRTUAL_ENV" ]; then
    echo "please run inside the virtual environment"
    exit 1
fi

# make sure the current directory is the model root
if [ ! [ "$PWD" == *"actor/data/llama3" ] ]; then
    echo "please run from the model root directory (../actor/data/<model>)"
    exit 1
fi

# download the model
if [ "$REBUILD" == true ] || [ ! -d "source" ]; then
    echo "model not found."
    rm -rf source
    mkdir -p source
    python3 download.py || exit 1
else
    echo "model found."
fi

# build the ONNX checkpoint
if [ "$REBUILD" == true ] || [ ! -d "ckpt" ]; then
    echo "checkpoint not found."
    rm -rf ckpt
    mkdir -p ckpt
    VENV_ROOT="$(python3 -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])')/lib"
    export LD_LIBRARY_PATH="${VENV_ROOT}:${LD_LIBRARY_PATH:-}"
    python3 export.py || exit 1
else
    echo "checkpoint found."
fi

# build the TensorRT INT8 engine
if [ "$REBUILD" == true ] || [ ! -d "engine" ]; then
    echo "engine not found."
    rm -rf engine
    mkdir -p engine

    # quantize ONNX model with Q/DQ nodes (INT8 via ORT static quantization)
    python3 calibrate.py || exit 1

    NUM_LAYERS=28
    NUM_KV_HEADS=8
    HEAD_DIM=128
    min_shapes="input_ids:1x1,position_ids:1x1"
    opt_shapes="input_ids:1x64,position_ids:1x64"
    max_shapes="input_ids:1x128,position_ids:1x128"
    for i in $(seq 0 $((NUM_LAYERS - 1))); do
        # KV cache shapes: [batch=1, num_kv_heads=8, seq_len, head_dim=128]
        min_shapes="${min_shapes},past_key_values.${i}.key:1x${NUM_KV_HEADS}x0x${HEAD_DIM}"
        min_shapes="${min_shapes},past_key_values.${i}.value:1x${NUM_KV_HEADS}x0x${HEAD_DIM}"

        opt_shapes="${opt_shapes},past_key_values.${i}.key:1x${NUM_KV_HEADS}x64x${HEAD_DIM}"
        opt_shapes="${opt_shapes},past_key_values.${i}.value:1x${NUM_KV_HEADS}x64x${HEAD_DIM}"

        max_shapes="${max_shapes},past_key_values.${i}.key:1x${NUM_KV_HEADS}x128x${HEAD_DIM}"
        max_shapes="${max_shapes},past_key_values.${i}.value:1x${NUM_KV_HEADS}x128x${HEAD_DIM}"
    done
    trtexec \
        --onnx="ckpt/model_int8.onnx" \
        --saveEngine="engine/model.engine" \
        --minShapes="${min_shapes}" \
        --optShapes="${opt_shapes}" \
        --maxShapes="${max_shapes}" \
        --stronglyTyped \
        --builderOptimizationLevel=3 \
        --memPoolSize=workspace:4096 \
        --skipInference \
        --verbose ||
        exit 1
    cp source/tokenizer.json engine/
else
    echo "engine found."
fi
