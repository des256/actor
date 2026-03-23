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
#if [ "$REBUILD" == true ] || [ ! -d "ckpt" ]; then
echo "checkpoint not found."
rm -rf ckpt
mkdir -p ckpt
VENV_ROOT="$(python3 -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])')/lib"
export LD_LIBRARY_PATH="${VENV_ROOT}:${LD_LIBRARY_PATH:-}"
python3 export.py || exit 1
#else
#    echo "checkpoint found."
#fi

# build the TensorRT INT8 engine
#if [ "$REBUILD" == true ] || [ ! -d "engine" ]; then
echo "building INT8 engine..."
rm -rf engine
mkdir -p engine
python3 build_engine.py || exit 1
cp source/tokenizer.json engine/
#else
#    echo "engine found."
#fi
