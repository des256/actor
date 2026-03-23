"""Build TensorRT INT8 engine from FP16 ONNX model with entropy calibration.

Uses TRT's native INT8 support instead of ORT-based quantization,
which cannot handle >2GB models.

Pipeline: ckpt/model.onnx (FP16) → TRT INT8 calibration → engine/model.engine
"""

from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

SOURCE_DIR = Path("source")
ONNX_PATH = Path("ckpt/model.onnx")
ENGINE_DIR = Path("engine")
ENGINE_PATH = ENGINE_DIR / "model.engine"
CACHE_PATH = ENGINE_DIR / "calibration.cache"

# Architecture constants (must match export.py and Rust code)
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
MAX_SEQ_LEN = 2048

NUM_CALIB_SAMPLES = 32
CALIB_SEQ_LEN = 128


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 entropy calibrator using C4 dataset for activation range estimation."""

    def __init__(self):
        super().__init__()
        self.batch_idx = 0
        self.device_buffers = {}
        self.inputs = []

        if CACHE_PATH.exists():
            print(f"Using existing calibration cache: {CACHE_PATH}")
            return

        print("Loading calibration data...")
        tokenizer = AutoTokenizer.from_pretrained(str(SOURCE_DIR))
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

        count = 0
        for sample in dataset:
            tokens = tokenizer.encode(sample["text"], add_special_tokens=False)
            if len(tokens) < CALIB_SEQ_LEN:
                continue

            input_ids = np.array([tokens[:CALIB_SEQ_LEN]], dtype=np.int64)
            position_ids = np.arange(CALIB_SEQ_LEN, dtype=np.int64).reshape(1, -1)

            feed = {"input_ids": input_ids, "position_ids": position_ids}
            for i in range(NUM_LAYERS):
                feed[f"past_key_values.{i}.key"] = np.zeros(
                    (1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16
                )
                feed[f"past_key_values.{i}.value"] = np.zeros(
                    (1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16
                )

            self.inputs.append(feed)
            count += 1
            if count % 10 == 0:
                print(f"  {count}/{NUM_CALIB_SAMPLES} samples")
            if count >= NUM_CALIB_SAMPLES:
                break

        print(f"Prepared {len(self.inputs)} calibration samples")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.batch_idx >= len(self.inputs):
            return None

        batch = self.inputs[self.batch_idx]
        self.batch_idx += 1

        bindings = []
        for name in names:
            data = batch.get(name)
            if data is None or data.size == 0:
                # Zero-element tensor (empty KV cache for prefill)
                if name not in self.device_buffers:
                    self.device_buffers[name] = torch.empty(
                        0, dtype=torch.float16, device="cuda"
                    )
                bindings.append(self.device_buffers[name].data_ptr())
            else:
                tensor = torch.from_numpy(data.copy()).cuda()
                self.device_buffers[name] = tensor  # prevent GC
                bindings.append(tensor.data_ptr())

        return bindings

    def read_calibration_cache(self):
        if CACHE_PATH.exists():
            return CACHE_PATH.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        CACHE_PATH.write_bytes(cache)
        print(f"Calibration cache saved to {CACHE_PATH}")


def build_engine():
    """Build INT8 TensorRT engine from FP16 ONNX model."""
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)

    calibrator = Int8Calibrator()

    print("Parsing ONNX model...")
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(ONNX_PATH)):
        for i in range(parser.num_errors):
            print(f"  Parse error: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX model")

    print(
        f"Network: {network.num_inputs} inputs, "
        f"{network.num_outputs} outputs, {network.num_layers} layers"
    )

    # Force all outputs to FP16 (Rust code reads logits and KV cache as FP16)
    for i in range(network.num_outputs):
        network.get_output(i).dtype = trt.DataType.HALF

    # Configure builder for INT8 with FP16 fallback
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB
    config.builder_optimization_level = 3

    # Optimization profile (for inference)
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, 1), (1, 128), (1, MAX_SEQ_LEN))
    profile.set_shape("position_ids", (1, 1), (1, 128), (1, MAX_SEQ_LEN))
    for i in range(NUM_LAYERS):
        profile.set_shape(
            f"past_key_values.{i}.key",
            (1, NUM_KV_HEADS, 0, HEAD_DIM),
            (1, NUM_KV_HEADS, 1024, HEAD_DIM),
            (1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM),
        )
        profile.set_shape(
            f"past_key_values.{i}.value",
            (1, NUM_KV_HEADS, 0, HEAD_DIM),
            (1, NUM_KV_HEADS, 1024, HEAD_DIM),
            (1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM),
        )
    config.add_optimization_profile(profile)

    # Calibration profile (prefill mode: full seq_len, empty KV cache)
    calib_profile = builder.create_optimization_profile()
    calib_profile.set_shape(
        "input_ids",
        (1, CALIB_SEQ_LEN),
        (1, CALIB_SEQ_LEN),
        (1, CALIB_SEQ_LEN),
    )
    calib_profile.set_shape(
        "position_ids",
        (1, CALIB_SEQ_LEN),
        (1, CALIB_SEQ_LEN),
        (1, CALIB_SEQ_LEN),
    )
    for i in range(NUM_LAYERS):
        calib_profile.set_shape(
            f"past_key_values.{i}.key",
            (1, NUM_KV_HEADS, 0, HEAD_DIM),
            (1, NUM_KV_HEADS, 0, HEAD_DIM),
            (1, NUM_KV_HEADS, 0, HEAD_DIM),
        )
        calib_profile.set_shape(
            f"past_key_values.{i}.value",
            (1, NUM_KV_HEADS, 0, HEAD_DIM),
            (1, NUM_KV_HEADS, 0, HEAD_DIM),
            (1, NUM_KV_HEADS, 0, HEAD_DIM),
        )
    config.set_calibration_profile(calib_profile)

    print("Building INT8 engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    ENGINE_PATH.write_bytes(serialized_engine)
    print(f"Engine saved to {ENGINE_PATH}")


if __name__ == "__main__":
    build_engine()
