"""Produce a quantized ONNX model with Q/DQ nodes for INT8 inference.

TRT's calibration mechanism OOMs on 16GB GPU regardless of approach (the
calibration execution context alone needs ~2GB on top of the ~7GB model).
Instead, we use ORT's static quantization to embed QuantizeLinear/DequantizeLinear
nodes with pre-computed scales directly in the ONNX model. trtexec then builds
an INT8 engine from this quantized model without any TRT calibration step.

The lm_head projection is excluded from quantization to preserve output quality.

Pipeline: ONNX model + C4 calibration data → ORT quantize_static → quantized ONNX
"""

from pathlib import Path

import numpy as np
from datasets import load_dataset
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.calibrate import CalibrationDataReader
from transformers import AutoTokenizer

SOURCE_DIR = Path("source")
ONNX_PATH = Path("ckpt/model.onnx")
QUANTIZED_PATH = Path("ckpt/model_int8.onnx")

# Architecture constants (must match export.py and Rust code)
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128

NUM_CALIB_SAMPLES = 64
CALIB_SEQ_LEN = 256


class C4DataReader(CalibrationDataReader):
    """Reads C4 dataset samples formatted as ONNX model inputs."""

    def __init__(self):
        self.samples = []
        self.idx = 0

        print("Loading calibration data from C4...")
        tokenizer = AutoTokenizer.from_pretrained(str(SOURCE_DIR))
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

        count = 0
        for sample in dataset:
            tokens = tokenizer.encode(sample["text"], add_special_tokens=False)
            if len(tokens) < CALIB_SEQ_LEN:
                continue

            feed = {
                "input_ids": np.array([tokens[:CALIB_SEQ_LEN]], dtype=np.int64),
                "position_ids": np.arange(CALIB_SEQ_LEN, dtype=np.int64).reshape(1, -1),
            }
            for i in range(NUM_LAYERS):
                feed[f"past_key_values.{i}.key"] = np.zeros(
                    (1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16
                )
                feed[f"past_key_values.{i}.value"] = np.zeros(
                    (1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16
                )

            self.samples.append(feed)
            count += 1
            if count % 10 == 0:
                print(f"  {count}/{NUM_CALIB_SAMPLES} samples")
            if count >= NUM_CALIB_SAMPLES:
                break

        print(f"Prepared {len(self.samples)} calibration samples")

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        sample = self.samples[self.idx]
        self.idx += 1
        return sample


def quantize():
    if QUANTIZED_PATH.exists():
        print(f"Quantized model already exists: {QUANTIZED_PATH}")
        return

    data_reader = C4DataReader()

    print("Running static quantization (QDQ format, INT8 symmetric, exclude lm_head)...")
    quantize_static(
        model_input=str(ONNX_PATH),
        model_output=str(QUANTIZED_PATH),
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["MatMul"],
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=True,
        calibration_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "CalibMovingAverage": True,
        },
        nodes_to_exclude=["/lm_head/MatMul"],
    )

    size_mb = sum(f.stat().st_size for f in QUANTIZED_PATH.parent.glob("model_int8*")) / (1024 * 1024)
    print(f"Quantized model saved: {QUANTIZED_PATH} ({size_mb:.0f} MB total)")


if __name__ == "__main__":
    quantize()
