#!/usr/bin/env python3
"""Quantize Parakeet ONNX models to int8 (dynamic quantization).

Weights: int8
Datapath: f32
Indices: kept as int32/int64
"""

import os

from onnxruntime.quantization import quantize_dynamic, QuantType

SRC = os.path.join(os.path.dirname(__file__), "..", "parakeet")
DEST = os.path.join(SRC, "int8")
os.makedirs(DEST, exist_ok=True)

MODELS = ["encoder.onnx", "decoder_joint.onnx"]

for name in MODELS:
    src_path = os.path.join(SRC, name)
    out_path = os.path.join(DEST, name)
    print(f"\n=== {name} ===")
    src_mb = os.path.getsize(src_path) / 1024 / 1024

    print("  Quantizing weights to int8...")
    quantize_dynamic(
        src_path,
        out_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        extra_options={"MatMulConstBOnly": True},
    )

    out_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  {src_mb:.1f} MB -> {out_mb:.1f} MB ({out_mb/src_mb*100:.0f}%)")

print("\nDone!")
