# assumes Moonshine Streaming Medium (safetensors from HuggingFace) is available from data/moonshine/source

set -euo pipefail

DATA_DIR="$(cd .. && pwd)"
MOONSHINE_DIR="${DATA_DIR}/moonshine"
SOURCE_DIR="${MOONSHINE_DIR}/source"
CKPT_DIR="${MOONSHINE_DIR}/ckpt"

mkdir -p "$CKPT_DIR"

if [[ ! -f "$SOURCE_DIR/model.safetensors" ]]; then
    echo "missing model.safetensors in $SOURCE_DIR"
    exit 1
fi
