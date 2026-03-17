#!/usr/bin/env bash
# Test script for BGE ONNX export
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Testing BGE ONNX export..."

# Check that build script exists and is executable
if [[ ! -x "build_onnx.sh" ]]; then
    echo "FAIL: build_onnx.sh not found or not executable"
    exit 1
fi

# Check that required files exist after build
if [[ ! -f "onnx/model.onnx" ]]; then
    echo "FAIL: onnx/model.onnx not found"
    exit 1
fi

if [[ ! -f "onnx/tokenizer.json" ]]; then
    echo "FAIL: onnx/tokenizer.json not found"
    exit 1
fi

# Validate ONNX model structure
python3 -c "
import onnx
import sys

try:
    model = onnx.load('onnx/model.onnx')
    onnx.checker.check_model(model)

    # Check inputs
    inputs = [i.name for i in model.graph.input]
    expected_inputs = ['input_ids', 'attention_mask', 'token_type_ids']
    if inputs != expected_inputs:
        print(f'FAIL: Expected inputs {expected_inputs}, got {inputs}')
        sys.exit(1)

    # Check outputs (last_hidden_state is required, tanh is optional pooler)
    outputs = [o.name for o in model.graph.output]
    if 'last_hidden_state' not in outputs:
        print(f'FAIL: Expected last_hidden_state in outputs, got {outputs}')
        sys.exit(1)

    print('PASS: ONNX model structure is valid')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"

if [[ $? -ne 0 ]]; then
    exit 1
fi

echo "All tests passed!"
