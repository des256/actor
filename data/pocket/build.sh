#!/bin/bash
# build Pocket TTS engines

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
if [[ ! "$PWD" == *"actor/data/pocket" ]]; then
    echo "please run from the model root directory (../actor/data/pocket)"
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
    python3 export.py source ckpt || exit 1
    python3 consolidate.py ckpt ckpt || exit 1
    python3 fix_scatter.py ckpt/mimi_decoder_model.onnx ckpt/mimi_decoder_model.onnx || exit 1
else
    echo "checkpoint found."
fi

# build the TensorRT engine
if [ "$REBUILD" == true ] || [ ! -d "engine" ]; then
    echo "engine not found."
    rm -rf engine
    mkdir -p engine
    for model in text_encoder_model flow_lm_main_model flow_lm_flow_model mimi_decoder_model; do
        if [[ ! -f "ckpt/${model}.onnx" ]]; then
            echo "missing ${model}.onnx in ckpt/, run build_ckpt.sh first"
            exit 1
        fi
    done
    trtexec \
        --onnx="ckpt/text_encoder_model.onnx" \
        --saveEngine="engine/text_encoder.engine" \
        --minShapes="token_ids:1x1" \
        --optShapes="token_ids:1x50" \
        --maxShapes="token_ids:1x500" \
        --timingCacheFile=engine/timing.cache \
        --memPoolSize=workspace:512 \
        --fp16 \
        --builderOptimizationLevel=5
    trtexec \
        --onnx="ckpt/flow_lm_main_model.onnx" \
        --saveEngine="engine/flow_lm_main.engine" \
        --minShapes="sequence:1x0x32,text_embeddings:1x0x1024,kv_cache:6x2x1x0x16x64" \
        --optShapes="sequence:1x1x32,text_embeddings:1x0x1024,kv_cache:6x2x1x100x16x64" \
        --maxShapes="sequence:1x200x32,text_embeddings:1x500x1024,kv_cache:6x2x1x800x16x64" \
        --timingCacheFile=engine/timing.cache \
        --memPoolSize=workspace:512 \
        --fp16 \
        --builderOptimizationLevel=5
    trtexec \
        --onnx="ckpt/flow_lm_flow_model.onnx" \
        --saveEngine="engine/flow_lm_flow.engine" \
        --timingCacheFile=engine/timing.cache \
        --memPoolSize=workspace:512 \
        --fp16 \
        --builderOptimizationLevel=5
    # mimi decoder runs WITHOUT fp16: the stateful feedback loop compounds
    # fp16 rounding errors across frames, degrading audio quality.
    trtexec \
        --onnx="ckpt/mimi_decoder_model.onnx" \
        --saveEngine="engine/mimi_decoder.engine" \
        --timingCacheFile=engine/timing.cache \
        --memPoolSize=workspace:512 \
        --builderOptimizationLevel=5
    python3 -c "
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDec
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='source/tokenizer.model')
vocab = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())]
tok = Tokenizer(Unigram(vocab))
tok.pre_tokenizer = Metaspace()
tok.decoder = MetaspaceDec()
tok.save('engine/tokenizer.json')
print('converted tokenizer.model -> tokenizer.json')
" || exit 1
else
    echo "engine found."
fi
