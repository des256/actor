"""Check which positions in TRT logits output are zero vs valid."""
import ctypes
import numpy as np
import tensorrt as trt
from tokenizers import Tokenizer

VOCAB_SIZE = 128256
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
MAX_SEQ_LEN = 2048

cuda = ctypes.CDLL("libcudart.so")

def cuda_check(rc, msg=""):
    if rc != 0:
        raise RuntimeError(f"{msg} failed: {rc}")

def gpu_alloc(n):
    p = ctypes.c_void_p()
    cuda_check(cuda.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)), "malloc")
    return p

def h2d(dst, src, n):
    cuda_check(cuda.cudaMemcpy(dst, src.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(n), ctypes.c_int(1)), "h2d")

def d2h(dst, src, n):
    cuda_check(cuda.cudaMemcpy(dst.ctypes.data_as(ctypes.c_void_p), src, ctypes.c_size_t(n), ctypes.c_int(2)), "d2h")


tokenizer = Tokenizer.from_file("engine/tokenizer.json")
prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You're a useful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Tell me something about cars<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
encoding = tokenizer.encode(prompt, add_special_tokens=False)
prompt_ids = np.array(encoding.ids, dtype=np.int64)
seq_len = len(prompt_ids)
print(f"Prompt tokens: {seq_len}")

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open("engine/model.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
stream = ctypes.c_void_p()
cuda_check(cuda.cudaStreamCreate(ctypes.byref(stream)), "stream")

i64_sz = 8
f16_sz = 2
input_ids_gpu = gpu_alloc(MAX_SEQ_LEN * i64_sz)
position_ids_gpu = gpu_alloc(MAX_SEQ_LEN * i64_sz)
logits_gpu = gpu_alloc(MAX_SEQ_LEN * VOCAB_SIZE * f16_sz)
kv_bytes = NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM * f16_sz
kv_in = [gpu_alloc(kv_bytes) for _ in range(NUM_LAYERS * 2)]
kv_out = [gpu_alloc(kv_bytes) for _ in range(NUM_LAYERS * 2)]

h2d(input_ids_gpu, prompt_ids, seq_len * i64_sz)
positions = np.arange(seq_len, dtype=np.int64)
h2d(position_ids_gpu, positions, seq_len * i64_sz)

context.set_input_shape("input_ids", (1, seq_len))
context.set_input_shape("position_ids", (1, seq_len))
context.set_tensor_address("input_ids", input_ids_gpu.value)
context.set_tensor_address("position_ids", position_ids_gpu.value)
context.set_tensor_address("logits", logits_gpu.value)
for layer in range(NUM_LAYERS):
    k_idx, v_idx = layer * 2, layer * 2 + 1
    context.set_input_shape(f"past_key_values.{layer}.key", (1, NUM_KV_HEADS, 0, HEAD_DIM))
    context.set_input_shape(f"past_key_values.{layer}.value", (1, NUM_KV_HEADS, 0, HEAD_DIM))
    context.set_tensor_address(f"past_key_values.{layer}.key", kv_in[k_idx].value)
    context.set_tensor_address(f"past_key_values.{layer}.value", kv_in[v_idx].value)
    context.set_tensor_address(f"present.{layer}.key", kv_out[k_idx].value)
    context.set_tensor_address(f"present.{layer}.value", kv_out[v_idx].value)

assert context.execute_async_v3(stream.value), "execute_async_v3 failed"
cuda_check(cuda.cudaStreamSynchronize(stream), "sync")

print(f"\nChecking ALL {seq_len} positions:")
for pos in range(seq_len):
    buf = np.zeros(VOCAB_SIZE, dtype=np.float16)
    off = pos * VOCAB_SIZE * f16_sz
    d2h(buf, ctypes.c_void_p(logits_gpu.value + off), VOCAB_SIZE * f16_sz)
    f32 = buf.astype(np.float32)
    nz = np.count_nonzero(f32)
    top1 = int(np.argmax(f32))
    tok = tokenizer.decode([top1])
    print(f"  pos {pos:2d} (token {prompt_ids[pos]:6d}): nonzero={nz:6d}  top1={top1:6d} ({tok!r:15s})  max={f32.max():.4f}")
