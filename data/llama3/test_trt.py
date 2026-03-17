"""Test TensorRT engine inference directly via Python TRT API.

This mirrors the Rust llama3.rs inference logic exactly to isolate whether
garbage output is caused by the Rust FFI layer or TRT itself.
"""
import ctypes
import numpy as np
import tensorrt as trt
from tokenizers import Tokenizer

# Architecture constants (must match export.py and llama3.rs)
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 128256
MAX_SEQ_LEN = 2048

# Token IDs
BOS_TOKEN = 128000
EOS_TOKEN = 128001
EOT_TOKEN = 128009

import sys

ENGINE_PATH = sys.argv[1] if len(sys.argv) > 1 else "engine/model.engine"
TOKENIZER_PATH = "engine/tokenizer.json"

# CUDA runtime
cuda = ctypes.CDLL("libcudart.so")

def cuda_check(rc, msg="CUDA call"):
    if rc != 0:
        raise RuntimeError(f"{msg} failed with rc={rc}")

def cuda_stream_create():
    stream = ctypes.c_void_p()
    cuda_check(cuda.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    return stream

def cuda_stream_sync(stream):
    cuda_check(cuda.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

def cuda_malloc(nbytes):
    ptr = ctypes.c_void_p()
    cuda_check(cuda.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc")
    return ptr

def cuda_memcpy_h2d(dst, src_np, nbytes, stream=None):
    src_ptr = src_np.ctypes.data_as(ctypes.c_void_p)
    if stream:
        cuda_check(cuda.cudaMemcpyAsync(dst, src_ptr, ctypes.c_size_t(nbytes), ctypes.c_int(1), stream), "cudaMemcpyAsync H2D")
    else:
        cuda_check(cuda.cudaMemcpy(dst, src_ptr, ctypes.c_size_t(nbytes), ctypes.c_int(1)), "cudaMemcpy H2D")

def cuda_memcpy_d2h(dst_np, src, nbytes):
    dst_ptr = dst_np.ctypes.data_as(ctypes.c_void_p)
    cuda_check(cuda.cudaMemcpy(dst_ptr, src, ctypes.c_size_t(nbytes), ctypes.c_int(2)), "cudaMemcpy D2H")

def cuda_free(ptr):
    cuda.cudaFree(ptr)


class TrtInference:
    def __init__(self):
        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        print(f"Loading engine from {ENGINE_PATH}...")
        with open(ENGINE_PATH, "rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        assert self.engine is not None, "Failed to deserialize engine"

        self.context = self.engine.create_execution_context()
        assert self.context is not None, "Failed to create execution context"

        self.stream = cuda_stream_create()

        # Print engine I/O tensor info
        print("\nEngine I/O tensors:")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if i < 5 or "present" in name and ".0." in name:
                print(f"  [{i}] {name}: mode={mode}, dtype={dtype}, shape={shape}")
        print(f"  ... ({self.engine.num_io_tensors} total)")

        # Allocate GPU buffers — mirror Rust exactly
        i64_size = np.dtype(np.int64).itemsize
        f16_size = np.dtype(np.float16).itemsize

        self.input_ids_gpu = cuda_malloc(MAX_SEQ_LEN * i64_size)
        self.position_ids_gpu = cuda_malloc(MAX_SEQ_LEN * i64_size)
        self.logits_gpu = cuda_malloc(MAX_SEQ_LEN * VOCAB_SIZE * f16_size)

        # KV cache: double buffers (kv_a and kv_b), 56 each
        kv_bytes = 1 * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM * f16_size
        print(f"\nKV buffer size per tensor: {kv_bytes} bytes ({kv_bytes / 1024 / 1024:.1f} MB)")

        self.kv_a = [cuda_malloc(kv_bytes) for _ in range(NUM_LAYERS * 2)]
        self.kv_b = [cuda_malloc(kv_bytes) for _ in range(NUM_LAYERS * 2)]

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

        # State
        self.past_len = 0
        self.kv_idx = False  # False → A is input, B is output (matches Rust)

    def infer(self, input_token_ids: np.ndarray, seq_len: int):
        """Run one inference step — mirrors Direct::infer() in llama3.rs exactly."""
        # Upload input_ids
        cuda_memcpy_h2d(self.input_ids_gpu, input_token_ids, seq_len * np.dtype(np.int64).itemsize)

        # Upload position_ids
        positions = np.arange(self.past_len, self.past_len + seq_len, dtype=np.int64)
        cuda_memcpy_h2d(self.position_ids_gpu, positions, seq_len * np.dtype(np.int64).itemsize)

        # Set dynamic input shapes
        self.context.set_input_shape("input_ids", (1, seq_len))
        self.context.set_input_shape("position_ids", (1, seq_len))

        # Bind I/O tensors
        self.context.set_tensor_address("input_ids", self.input_ids_gpu.value)
        self.context.set_tensor_address("position_ids", self.position_ids_gpu.value)
        self.context.set_tensor_address("logits", self.logits_gpu.value)

        # Bind KV cache: past (input) and present (output)
        if not self.kv_idx:
            kv_in, kv_out = self.kv_a, self.kv_b
        else:
            kv_in, kv_out = self.kv_b, self.kv_a

        for layer in range(NUM_LAYERS):
            k_idx = layer * 2
            v_idx = layer * 2 + 1

            past_k_name = f"past_key_values.{layer}.key"
            past_v_name = f"past_key_values.{layer}.value"
            present_k_name = f"present.{layer}.key"
            present_v_name = f"present.{layer}.value"

            # Set KV cache shapes
            self.context.set_input_shape(past_k_name, (1, NUM_KV_HEADS, self.past_len, HEAD_DIM))
            self.context.set_input_shape(past_v_name, (1, NUM_KV_HEADS, self.past_len, HEAD_DIM))

            # Bind addresses
            self.context.set_tensor_address(past_k_name, kv_in[k_idx].value)
            self.context.set_tensor_address(past_v_name, kv_in[v_idx].value)
            self.context.set_tensor_address(present_k_name, kv_out[k_idx].value)
            self.context.set_tensor_address(present_v_name, kv_out[v_idx].value)

        # Execute
        ok = self.context.execute_async_v3(self.stream.value)
        assert ok, "execute_async_v3 failed"
        cuda_stream_sync(self.stream)

        # Update state
        self.past_len += seq_len
        self.kv_idx = not self.kv_idx

    def generate(self, prompt: str, max_new_tokens: int = 50):
        """Generate tokens — mirrors Direct::generate() in llama3.rs."""
        # Reset state
        self.past_len = 0
        self.kv_idx = False

        # Tokenize (no special tokens, same as Rust)
        encoding = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_tokens = np.array(encoding.ids, dtype=np.int64)
        print(f"\nPrompt tokens ({len(prompt_tokens)}): {prompt_tokens[:10].tolist()}...")

        # Prefill
        if len(prompt_tokens) > 0:
            self.infer(prompt_tokens, len(prompt_tokens))

        # Decode
        generated_tokens: list[int] = []
        prev_decoded_len = 0
        f16_size = np.dtype(np.float16).itemsize

        for step in range(max_new_tokens):
            # Download logits for last position (mirror Rust logic exactly)
            if not generated_tokens:
                # First iteration after prefill
                if len(prompt_tokens) == 0:
                    logits_offset = 0
                else:
                    logits_offset = (len(prompt_tokens) - 1) * VOCAB_SIZE
            else:
                # Decode iterations
                logits_offset = 0

            last_logits_f16 = np.zeros(VOCAB_SIZE, dtype=np.float16)
            download_offset = logits_offset * f16_size
            src_ptr = ctypes.c_void_p(self.logits_gpu.value + download_offset)
            cuda_memcpy_d2h(last_logits_f16, src_ptr, VOCAB_SIZE * f16_size)

            # Convert to f32 for argmax (same as Rust)
            last_logits = last_logits_f16.astype(np.float32)

            # Greedy argmax
            next_token = int(np.argmax(last_logits))

            # Print top-5 for comparison
            if step < 10:
                top5_idx = np.argsort(last_logits)[-5:][::-1]
                top5 = [(int(idx), float(last_logits[idx]),
                         self.tokenizer.decode([int(idx)], skip_special_tokens=False))
                        for idx in top5_idx]
                print(f"  Step {step}: token={next_token} top5={top5}")

            # Check EOS/EOT
            if next_token in (EOS_TOKEN, EOT_TOKEN):
                print(f"  Step {step}: EOS/EOT")
                break

            generated_tokens.append(next_token)

            # Decode and print delta (same as Rust)
            full_decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if len(full_decoded) > prev_decoded_len:
                delta = full_decoded[prev_decoded_len:]
                prev_decoded_len = len(full_decoded)
                # Print inline for streaming effect
                print(f"    delta: \"{delta}\"")

            # Infer next token
            next_token_i64 = np.array([next_token], dtype=np.int64)
            self.infer(next_token_i64, 1)

        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"\nGenerated ({len(generated_tokens)} tokens): {text}")
        return text

    def cleanup(self):
        cuda_free(self.input_ids_gpu)
        cuda_free(self.position_ids_gpu)
        cuda_free(self.logits_gpu)
        for buf in self.kv_a + self.kv_b:
            cuda_free(buf)


if __name__ == "__main__":
    trt_inf = TrtInference()

    # Same prompt as test_llama3.rs
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You're a useful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Tell me something about cars<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    print(f"Prompt: {prompt!r}")
    trt_inf.generate(prompt, max_new_tokens=50)
    trt_inf.cleanup()
