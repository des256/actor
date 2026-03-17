"""Compare step-0 (prefill) logits between ORT and TRT for the same prompt.

This isolates whether TRT diverges from ORT at the very first step.
"""
import ctypes
import numpy as np
import onnxruntime as ort
import tensorrt as trt
from tokenizers import Tokenizer

# Architecture constants
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 128256

# CUDA runtime
cuda = ctypes.CDLL("libcudart.so")

def cuda_check(rc, msg="CUDA call"):
    if rc != 0:
        raise RuntimeError(f"{msg} failed with rc={rc}")

def cuda_malloc(nbytes):
    ptr = ctypes.c_void_p()
    cuda_check(cuda.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc")
    return ptr

def cuda_memcpy_h2d(dst, src_np, nbytes):
    src_ptr = src_np.ctypes.data_as(ctypes.c_void_p)
    cuda_check(cuda.cudaMemcpy(dst, src_ptr, ctypes.c_size_t(nbytes), ctypes.c_int(1)), "cudaMemcpy H2D")

def cuda_memcpy_d2h(dst_np, src, nbytes):
    dst_ptr = dst_np.ctypes.data_as(ctypes.c_void_p)
    cuda_check(cuda.cudaMemcpy(dst_ptr, src, ctypes.c_size_t(nbytes), ctypes.c_int(2)), "cudaMemcpy D2H")

def cuda_free(ptr):
    cuda.cudaFree(ptr)


# Prompt
tokenizer = Tokenizer.from_file("engine/tokenizer.json")
prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You're a useful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Tell me something about cars<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
encoding = tokenizer.encode(prompt, add_special_tokens=False)
prompt_ids = np.array(encoding.ids, dtype=np.int64)
seq_len = len(prompt_ids)
print(f"Prompt tokens ({seq_len}): {prompt_ids[:10].tolist()}...")

# ========== ORT ==========
print("\n=== ORT (FP16 ONNX, disabled optimizations) ===")
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession("onnx/model.onnx", sess_options, providers=["CPUExecutionProvider"])

input_ids_ort = prompt_ids.reshape(1, -1)
position_ids_ort = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
past_kv = {
    f"past_key_values.{i}.{'key' if j == 0 else 'value'}": np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16)
    for i in range(NUM_LAYERS) for j in range(2)
}

feeds = {"input_ids": input_ids_ort, "position_ids": position_ids_ort, **past_kv}
outputs_ort = session.run(None, feeds)
logits_ort = outputs_ort[0]  # [1, seq_len, vocab]
last_logits_ort = logits_ort[0, -1, :].astype(np.float32)

print(f"  Logits shape: {logits_ort.shape}, dtype: {logits_ort.dtype}")
top5_ort = np.argsort(last_logits_ort)[-5:][::-1]
print(f"  Top-5: {[(int(i), float(last_logits_ort[i]), tokenizer.decode([int(i)])) for i in top5_ort]}")
print(f"  Logits stats: min={last_logits_ort.min():.4f} max={last_logits_ort.max():.4f} mean={last_logits_ort.mean():.4f}")

# ========== TRT ==========
print("\n=== TRT (from same ONNX, --fp16 engine) ===")
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open("engine/model.engine", "rb") as f:
    engine_data = f.read()
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

stream = ctypes.c_void_p()
cuda_check(cuda.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")

i64_size = np.dtype(np.int64).itemsize
f16_size = np.dtype(np.float16).itemsize
MAX_SEQ_LEN = 2048

input_ids_gpu = cuda_malloc(MAX_SEQ_LEN * i64_size)
position_ids_gpu = cuda_malloc(MAX_SEQ_LEN * i64_size)
logits_gpu = cuda_malloc(MAX_SEQ_LEN * VOCAB_SIZE * f16_size)
kv_bytes = NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM * f16_size
kv_in = [cuda_malloc(kv_bytes) for _ in range(NUM_LAYERS * 2)]
kv_out = [cuda_malloc(kv_bytes) for _ in range(NUM_LAYERS * 2)]

# Upload inputs
cuda_memcpy_h2d(input_ids_gpu, prompt_ids, seq_len * i64_size)
positions = np.arange(seq_len, dtype=np.int64)
cuda_memcpy_h2d(position_ids_gpu, positions, seq_len * i64_size)

# Set shapes and bind
context.set_input_shape("input_ids", (1, seq_len))
context.set_input_shape("position_ids", (1, seq_len))
context.set_tensor_address("input_ids", input_ids_gpu.value)
context.set_tensor_address("position_ids", position_ids_gpu.value)
context.set_tensor_address("logits", logits_gpu.value)

for layer in range(NUM_LAYERS):
    k_idx = layer * 2
    v_idx = layer * 2 + 1
    context.set_input_shape(f"past_key_values.{layer}.key", (1, NUM_KV_HEADS, 0, HEAD_DIM))
    context.set_input_shape(f"past_key_values.{layer}.value", (1, NUM_KV_HEADS, 0, HEAD_DIM))
    context.set_tensor_address(f"past_key_values.{layer}.key", kv_in[k_idx].value)
    context.set_tensor_address(f"past_key_values.{layer}.value", kv_in[v_idx].value)
    context.set_tensor_address(f"present.{layer}.key", kv_out[k_idx].value)
    context.set_tensor_address(f"present.{layer}.value", kv_out[v_idx].value)

ok = context.execute_async_v3(stream.value)
assert ok, "execute_async_v3 failed"
cuda_check(cuda.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

# Download last position logits (f16)
last_logits_f16 = np.zeros(VOCAB_SIZE, dtype=np.float16)
download_offset = (seq_len - 1) * VOCAB_SIZE * f16_size
src_ptr = ctypes.c_void_p(logits_gpu.value + download_offset)
cuda_memcpy_d2h(last_logits_f16, src_ptr, VOCAB_SIZE * f16_size)
last_logits_trt = last_logits_f16.astype(np.float32)

print("  Logits dtype from engine: HALF (downloaded as f16, converted to f32)")
top5_trt = np.argsort(last_logits_trt)[-5:][::-1]
print(f"  Top-5: {[(int(i), float(last_logits_trt[i]), tokenizer.decode([int(i)])) for i in top5_trt]}")
print(f"  Logits stats: min={last_logits_trt.min():.4f} max={last_logits_trt.max():.4f} mean={last_logits_trt.mean():.4f}")

# ========== Compare ==========
print("\n=== Comparison ===")
ort_top1 = int(np.argmax(last_logits_ort))
trt_top1 = int(np.argmax(last_logits_trt))
print(f"  ORT top-1: {ort_top1} ({tokenizer.decode([ort_top1])})")
print(f"  TRT top-1: {trt_top1} ({tokenizer.decode([trt_top1])})")
print(f"  Match: {ort_top1 == trt_top1}")

# Correlation
corr = np.corrcoef(last_logits_ort, last_logits_trt)[0, 1]
print(f"  Pearson correlation: {corr:.6f}")

# Absolute difference
diff = np.abs(last_logits_ort - last_logits_trt)
print(f"  Abs diff: mean={diff.mean():.4f} max={diff.max():.4f} median={np.median(diff):.4f}")

# Check if logits are all zeros or garbage
print(f"\n  ORT non-zero logits: {np.count_nonzero(last_logits_ort)} / {VOCAB_SIZE}")
print(f"  TRT non-zero logits: {np.count_nonzero(last_logits_trt)} / {VOCAB_SIZE}")

# Check ALL positions, not just last
print("\n=== TRT logits at different positions ===")
for pos in [0, seq_len // 2, seq_len - 1]:
    pos_f16 = np.zeros(VOCAB_SIZE, dtype=np.float16)
    off = pos * VOCAB_SIZE * f16_size
    cuda_memcpy_d2h(pos_f16, ctypes.c_void_p(logits_gpu.value + off), VOCAB_SIZE * f16_size)
    pos_f32 = pos_f16.astype(np.float32)
    top1 = int(np.argmax(pos_f32))
    print(f"  Position {pos}: top1={top1} ({tokenizer.decode([top1])}), "
          f"min={pos_f32.min():.4f}, max={pos_f32.max():.4f}, "
          f"nonzero={np.count_nonzero(pos_f32)}/{VOCAB_SIZE}")

# Cleanup
cuda_free(input_ids_gpu)
cuda_free(position_ids_gpu)
cuda_free(logits_gpu)
for buf in kv_in + kv_out:
    cuda_free(buf)
