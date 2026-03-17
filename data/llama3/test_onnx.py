"""Test ONNX model generation with ONNX Runtime to verify export correctness."""
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 128256
EOS_TOKEN = 128001
EOT_TOKEN = 128009

# Load (disable graph optimizations for FP16 model compatibility)
print("Loading ONNX model...")
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession("onnx/model.onnx", sess_options, providers=["CPUExecutionProvider"])
tokenizer = Tokenizer.from_file("engine/tokenizer.json")

# Prompt (exact Llama 3 chat template)
prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You're a useful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Hi, tell me something about cars<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

# Tokenize
encoding = tokenizer.encode(prompt, add_special_tokens=False)
prompt_ids = encoding.ids
print(f"Prompt tokens ({len(prompt_ids)}): {prompt_ids[:10]}...")

# Prefill
input_ids = np.array([prompt_ids], dtype=np.int64)
position_ids = np.arange(len(prompt_ids), dtype=np.int64).reshape(1, -1)
past_kv = {
    f"past_key_values.{i}.{'key' if j == 0 else 'value'}": np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16)
    for i in range(NUM_LAYERS) for j in range(2)
}

feeds = {"input_ids": input_ids, "position_ids": position_ids, **past_kv}
outputs = session.run(None, feeds)

# Parse outputs: logits + 56 KV tensors
logits = outputs[0]  # [1, seq_len, vocab]
present_kv = outputs[1:]  # 56 tensors of [1, 8, seq_len, 128]

print(f"Prefill logits shape: {logits.shape}")
print(f"Present KV[0] shape: {present_kv[0].shape}")

# Generate tokens
generated = []
past_len = len(prompt_ids)

for step in range(50):
    # Get logits for last position
    if step == 0:
        last_logits = logits[0, -1, :]  # Last position from prefill
    else:
        last_logits = logits[0, 0, :]  # Only position from decode

    # Argmax
    next_token = int(np.argmax(last_logits))

    # Top-5 for first few steps
    if step < 5:
        top5_idx = np.argsort(last_logits)[-5:][::-1]
        top5 = [(int(idx), float(last_logits[idx]), tokenizer.decode([int(idx)], skip_special_tokens=False)) for idx in top5_idx]
        print(f"  Step {step}: token={next_token} top5={top5}")

    if next_token in (EOS_TOKEN, EOT_TOKEN):
        print(f"  Step {step}: EOS/EOT")
        break

    generated.append(next_token)

    # Decode step
    input_ids = np.array([[next_token]], dtype=np.int64)
    position_ids = np.array([[past_len]], dtype=np.int64)

    kv_feeds = {}
    for i in range(NUM_LAYERS):
        kv_feeds[f"past_key_values.{i}.key"] = present_kv[i * 2]
        kv_feeds[f"past_key_values.{i}.value"] = present_kv[i * 2 + 1]

    feeds = {"input_ids": input_ids, "position_ids": position_ids, **kv_feeds}
    outputs = session.run(None, feeds)
    logits = outputs[0]
    present_kv = outputs[1:]
    past_len += 1

# Decode result
text = tokenizer.decode(generated, skip_special_tokens=True)
print(f"\nGenerated ({len(generated)} tokens): {text}")
