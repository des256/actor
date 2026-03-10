#!/usr/bin/env python3
"""Quantize Parakeet ONNX models to q4fp16.

Weights: int4 (packed into uint8) with fp16 scales (via MatMulNBits)
Datapath: fp16
Indices: kept as int32/int64
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

SRC = os.path.join(os.path.dirname(__file__), "..", "parakeet")
DEST = os.path.join(SRC, "q4fp16")
os.makedirs(DEST, exist_ok=True)

MODELS = ["encoder.onnx", "decoder_joint.onnx"]

# Integer types that should not be converted
INT_TYPES = {
    TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
    TensorProto.UINT8, TensorProto.UINT16, TensorProto.UINT32, TensorProto.UINT64,
    TensorProto.BOOL,
}


def convert_to_fp16(model):
    """Direct fp16 conversion: convert all float32 tensors/types to float16.

    Skips integer types and Cast node 'to' attributes. Much faster than
    onnxconverter_common because it skips shape inference.
    """
    # Convert initializers (weights, biases, scales, constants)
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            data = numpy_helper.to_array(init)
            init.CopyFrom(numpy_helper.from_array(data.astype(np.float16), init.name))

    # Convert graph inputs
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16

    # Convert graph outputs
    for out in model.graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT:
            out.type.tensor_type.elem_type = TensorProto.FLOAT16

    # Convert intermediate value_info
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16

    # Convert node attributes: Cast 'to' and Constant 'value' tensors
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
        elif node.op_type in ("Constant", "ConstantOfShape"):
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    data = numpy_helper.to_array(attr.t)
                    attr.t.CopyFrom(numpy_helper.from_array(data.astype(np.float16)))


for name in MODELS:
    src_path = os.path.join(SRC, name)
    out_path = os.path.join(DEST, name)
    print(f"\n=== {name} ===")
    src_mb = os.path.getsize(src_path) / 1024 / 1024

    # Step 1: Quantize MatMul weights to int4 (block_size=128)
    print("  Quantizing weights to int4...")
    model = onnx.load(src_path)
    quantizer = MatMulNBitsQuantizer(model, bits=4, block_size=128, is_symmetric=False)
    quantizer.process()
    q4_model = quantizer.model.model  # unwrap ONNXModel -> ModelProto

    # Step 2: Convert remaining f32 datapath to fp16
    print("  Converting datapath to fp16...")
    convert_to_fp16(q4_model)
    onnx.save(q4_model, out_path)

    out_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  {src_mb:.1f} MB -> {out_mb:.1f} MB ({out_mb/src_mb*100:.0f}%)")

print("\nDone!")
