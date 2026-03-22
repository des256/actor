#!/bin/bash
# build Moonshine Streaming Medium engines

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
if [ ! [ "$PWD" == *"actor/data/moonshine" ] ]; then
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
    TEMP_DIR=$(mktemp -d -t moonshine-ckpt-XXXX)
    trap 'rm -rf "$TEMP_DIR"' EXIT
    python3 export.py "$TEMP_DIR" || exit 1
else
    echo "checkpoint found."
fi

# build the TensorRT engine
if [ "$REBUILD" == true ] || [ ! -d "engine" ]; then
    echo "engine not found."
    rm -rf engine
    mkdir -p engine
    trtexec \
        --onnx="ckpt/encoder.onnx" \
        --saveEngine="engine/encoder.engine" \
        --minShapes="input_values:1x32000,attention_mask:1x32000" \
        --optShapes="input_values:1x32000,attention_mask:1x32000" \
        --maxShapes="input_values:1x32000,attention_mask:1x32000" \
        --memPoolSize=workspace:512 \
        --fp16 \
        --builderOptimizationLevel=5 ||
        exit 1
    trtexec \
        --onnx="ckpt/decoder.onnx" \
        --saveEngine="engine/decoder.engine" \
        --minShapes="input_ids:1x1,encoder_hidden_states:1x100x768,encoder_attention_mask:1x100,k_self:14x1x10x0x64,v_self:14x1x10x0x64" \
        --optShapes="input_ids:1x1,encoder_hidden_states:1x100x768,encoder_attention_mask:1x100,k_self:14x1x10x16x64,v_self:14x1x10x16x64" \
        --maxShapes="input_ids:1x1,encoder_hidden_states:1x100x768,encoder_attention_mask:1x100,k_self:14x1x10x32x64,v_self:14x1x10x32x64" \
        --memPoolSize=workspace:512 \
        --fp16 --int4 \
        --builderOptimizationLevel=5 ||
        exit 1
    cp source/tokenizer.json engine/
else
    echo "engine found."
fi
