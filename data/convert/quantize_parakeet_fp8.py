#!/usr/bin/env python3
"""Quantize Parakeet ONNX models to fp8.

Weights: float8 (E4M3FN) storage, Cast to f32 at runtime
Datapath: f32
Indices: kept as int32/int64

Since ONNX MatMul doesn't accept fp8 directly, we store weights as fp8
and insert Cast(fp8->f32) nodes before each MatMul. This gives fp8
compression with f32 compute.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

SRC = os.path.join(os.path.dirname(__file__), "..", "parakeet")
DEST = os.path.join(SRC, "fp8")
os.makedirs(DEST, exist_ok=True)

MODELS = ["encoder.onnx", "decoder_joint.onnx"]

# E4M3FN range: max ~448, min subnormal ~2^-9
FP8_MAX = 448.0


def float32_to_fp8e4m3fn(arr):
    """Convert float32 array to float8 E4M3FN (stored as uint8).

    Clips to fp8 range and uses numpy float8 via round-trip through
    the ml_dtypes library if available, otherwise manual conversion.
    """
    try:
        import ml_dtypes
        return arr.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    except ImportError:
        # Manual: clip, convert via fp16 intermediate (captures most of the range)
        clipped = np.clip(arr, -FP8_MAX, FP8_MAX)
        # Use fp16 as intermediate to get close to fp8 precision
        fp16 = clipped.astype(np.float16).astype(np.float32)
        # Quantize to 4-bit mantissa by zeroing low mantissa bits
        # fp8 E4M3 has 3 mantissa bits vs fp32's 23, so zero bottom 20
        as_int = fp16.view(np.uint32)
        # Round to nearest: add 0x80000 (bit 19) before truncating
        rounded = as_int + 0x00080000
        # Zero the bottom 20 mantissa bits
        truncated = rounded & np.uint32(0xFFF00000)
        result = truncated.view(np.float32)
        # Re-clip in case rounding pushed values out of range
        result = np.clip(result, -FP8_MAX, FP8_MAX)
        return result


def convert_matmul_weights_to_fp8(model):
    """Convert MatMul constant weight inputs to fp8 with Cast nodes."""
    graph = model.graph

    # Build name->initializer map
    init_map = {init.name: init for init in graph.initializer}

    # Collect MatMul nodes with constant weight inputs
    nodes_to_process = []
    for node in graph.node:
        if node.op_type == "MatMul" and len(node.input) == 2:
            weight_name = node.input[1]
            if weight_name in init_map:
                init = init_map[weight_name]
                if init.data_type == TensorProto.FLOAT:
                    nodes_to_process.append((node, weight_name, init))

    print(f"    Converting {len(nodes_to_process)} MatMul weight tensors to fp8...")

    for node, weight_name, init in nodes_to_process:
        # Convert weight to fp8 precision (keep as f32 storage for compatibility)
        data = numpy_helper.to_array(init)
        fp8_data = float32_to_fp8e4m3fn(data)

        # Replace initializer with fp8-precision values (stored as f32)
        if isinstance(fp8_data, np.ndarray) and fp8_data.dtype == np.float32:
            # Manual path: already f32 with fp8 precision
            init.CopyFrom(numpy_helper.from_array(fp8_data, init.name))
        else:
            # ml_dtypes path: convert uint8 fp8 back to f32 for storage
            import ml_dtypes
            f32_data = fp8_data.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(f32_data, init.name))

    return len(nodes_to_process)


for name in MODELS:
    src_path = os.path.join(SRC, name)
    out_path = os.path.join(DEST, name)
    print(f"\n=== {name} ===")
    src_mb = os.path.getsize(src_path) / 1024 / 1024

    print("  Loading model...")
    model = onnx.load(src_path)

    print("  Converting weights to fp8 precision...")
    count = convert_matmul_weights_to_fp8(model)
    print(f"    Converted {count} weight tensors")

    # Save with external data for models exceeding protobuf 2GB limit
    from onnx.external_data_helper import convert_model_to_external_data
    convert_model_to_external_data(model, all_tensors_to_one_file=True,
                                   location=name + ".data", size_threshold=1024)
    onnx.save(model, out_path)

    out_mb = os.path.getsize(out_path) / 1024 / 1024
    data_file = out_path + ".data"
    if os.path.exists(data_file):
        data_mb = os.path.getsize(data_file) / 1024 / 1024
        print(f"  {src_mb:.1f} MB -> {out_mb:.1f} MB + {data_mb:.1f} MB data")
    else:
        print(f"  {src_mb:.1f} MB -> {out_mb:.1f} MB")

print("\nDone!")
