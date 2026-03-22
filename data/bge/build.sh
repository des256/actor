#!/bin/bash
# build BGE-small-en-v1.5 ONNX model

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
if [ ! [ "$PWD" == *"actor/data/bge" ] ]; then
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

# export to ONNX
if [ "$REBUILD" == true ] || [ ! -d "onnx" ]; then
    echo "ONNX not found."
    rm -rf onnx
    mkdir -p onnx
    python3 export.py || exit 1
else
    echo "ONNX found."
fi
