#!/usr/bin/env bash
# Download BAAI/bge-small-en-v1.5 from HuggingFace and export to ONNX
set -euo pipefail

mkdir -p onnx/

echo "Exporting BAAI/bge-small-en-v1.5 to ONNX (will auto-download if needed)..."
python3 "export.py" "onnx/"

echo "Build complete. ONNX model and tokenizer available in onnx/"
