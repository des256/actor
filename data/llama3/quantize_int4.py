"""INT4 AWQ quantization for Llama 3.2 3B ONNX model.

Quantizes the FP16 ONNX model in-place using NVIDIA modelopt's INT4 AWQ
algorithm. Uses C4 calibration data with 512 samples. The lm_head MatMul
is excluded from quantization to preserve output quality.

Usage:
    cd data/llama3
    /home/desmond/tensorrt/venv/bin/python3 quantize_int4.py
"""

import copy
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from datasets import load_dataset
from onnxruntime.quantization.calibrate import CalibrationDataReader
from transformers import AutoTokenizer


def _patched_save_onnx(model, onnx_path, save_as_external_data=False):
    """Patched save_onnx: handles EncodeError for >2GB models.

    modelopt's save_onnx only catches ValueError from SerializeToString(),
    but protobuf raises google.protobuf.message.EncodeError for >2GB models.
    """
    if not save_as_external_data:
        try:
            model_proto = model.SerializeToString()
            if len(model_proto) > 2 * (1024**3):
                save_as_external_data = True
        except Exception:
            save_as_external_data = True

    model.ir_version = 10
    if save_as_external_data:
        external_data_path = os.path.basename(onnx_path) + "_data"
        if os.path.exists(external_data_path):
            os.remove(external_data_path)
        model_copy = copy.deepcopy(model)
        onnx.save_model(
            model_copy,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path,
            size_threshold=1024,
        )
    else:
        onnx.save(model, onnx_path)


# Monkey-patch save_onnx in both modules before importing quantize
import modelopt.onnx.utils as _modelopt_utils  # noqa: E402
import modelopt.onnx.quantization.int4 as _int4_mod  # noqa: E402

_modelopt_utils.save_onnx = _patched_save_onnx
_int4_mod.save_onnx = _patched_save_onnx

from modelopt.onnx.quantization.int4 import quantize  # noqa: E402

# Paths
SCRIPT_DIR = Path(__file__).parent
ONNX_DIR = SCRIPT_DIR / "onnx"
ONNX_MODEL_PATH = ONNX_DIR / "model.onnx"
SOURCE_DIR = SCRIPT_DIR / "source"

# Architecture constants (must match export.py)
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128

# Calibration parameters
NUM_CALIB_SAMPLES = 128
SEQ_LEN = 256
BLOCK_SIZE = 128


class C4CalibrationDataReader(CalibrationDataReader):
    """Reads C4 calibration data formatted for the Llama3 ONNX model inputs."""

    def __init__(self, tokenizer_path: Path, num_samples: int, seq_len: int):
        self.seq_len = seq_len
        self.index = 0

        print(f"Loading tokenizer from {tokenizer_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        print(f"Loading C4 dataset (streaming, {num_samples} samples) ...")
        dataset = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
        )

        self.inputs = []
        count = 0
        for sample in dataset:
            tokens = tokenizer.encode(sample["text"], add_special_tokens=False)
            if len(tokens) < seq_len:
                continue

            input_ids = np.array([tokens[:seq_len]], dtype=np.int64)
            position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

            feed = {
                "input_ids": input_ids,
                "position_ids": position_ids,
            }
            # Empty KV cache for all 28 layers (prefill mode)
            for i in range(NUM_LAYERS):
                feed[f"past_key_values.{i}.key"] = np.zeros(
                    (1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16
                )
                feed[f"past_key_values.{i}.value"] = np.zeros(
                    (1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16
                )

            self.inputs.append(feed)
            count += 1
            if count % 100 == 0:
                print(f"  Prepared {count}/{num_samples} calibration samples")
            if count >= num_samples:
                break

        print(f"Prepared {len(self.inputs)} calibration samples")
        if len(self.inputs) < num_samples:
            print(
                f"WARNING: Only got {len(self.inputs)} samples "
                f"(requested {num_samples})"
            )

    def get_next(self):
        if self.index >= len(self.inputs):
            return None
        result = self.inputs[self.index]
        self.index += 1
        return result

    def __len__(self):
        return len(self.inputs)


def main():
    if not ONNX_MODEL_PATH.exists():
        print(f"ERROR: ONNX model not found at {ONNX_MODEL_PATH}")
        sys.exit(1)

    # Prepare calibration data
    calib_reader = C4CalibrationDataReader(SOURCE_DIR, NUM_CALIB_SAMPLES, SEQ_LEN)

    # Run INT4 AWQ quantization
    print(f"\nQuantizing {ONNX_MODEL_PATH} with INT4 AWQ (block_size={BLOCK_SIZE}) ...")
    print("  Excluding: /lm_head (default)")
    print("  Calibration EP: CUDAExecutionProvider")

    quantized_model = quantize(
        str(ONNX_MODEL_PATH),
        calibration_method="awq_clip",
        calibration_data_reader=calib_reader,
        calibration_eps=["cuda", "cpu"],
        use_external_data_format=True,
        block_size=BLOCK_SIZE,
        nodes_to_exclude=[r"/lm_head"],
    )

    # Upgrade opset to 21 if needed (INT4 DequantizeLinear requires opset >= 21)
    current_opset = max(o.version for o in quantized_model.opset_import if o.domain == "")
    if current_opset < 21:
        print(f"Upgrading opset from {current_opset} to 21 (required for INT4 DQ nodes)")
        for opset in quantized_model.opset_import:
            if opset.domain == "":
                opset.version = 21

    # Save quantized model (overwrites the original)
    print(f"\nSaving quantized model to {ONNX_MODEL_PATH} ...")
    _patched_save_onnx(quantized_model, str(ONNX_MODEL_PATH), save_as_external_data=True)

    print("\nQuantization complete!")
    print(f"  Output: {ONNX_MODEL_PATH}")


if __name__ == "__main__":
    main()
