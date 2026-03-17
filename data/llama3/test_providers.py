"""Test ONNX model with different ORT execution providers (CPU, CUDA, TRT).

Compares step-0 logits across providers to find where TRT diverges.
"""
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 128256

tokenizer = Tokenizer.from_file("engine/tokenizer.json")
prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You're a useful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Tell me something about cars<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
encoding = tokenizer.encode(prompt, add_special_tokens=False)
prompt_ids = np.array(encoding.ids, dtype=np.int64)
seq_len = len(prompt_ids)

input_ids = prompt_ids.reshape(1, -1)
position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
past_kv = {
    f"past_key_values.{i}.{'key' if j == 0 else 'value'}": np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16)
    for i in range(NUM_LAYERS) for j in range(2)
}
feeds = {"input_ids": input_ids, "position_ids": position_ids, **past_kv}

providers_to_test = [
    ("CPU", ["CPUExecutionProvider"]),
    ("CUDA", ["CUDAExecutionProvider"]),
    ("TensorRT", ["TensorrtExecutionProvider"]),
]

results = {}
for name, providers in providers_to_test:
    print(f"\n=== {name} ===")
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession("onnx/model.onnx", sess_options, providers=providers)
        actual_provider = session.get_providers()[0]
        print(f"  Active provider: {actual_provider}")

        outputs = session.run(None, feeds)
        logits = outputs[0]
        last_logits = logits[0, -1, :].astype(np.float32)

        top5_idx = np.argsort(last_logits)[-5:][::-1]
        top5 = [(int(i), float(last_logits[i]), tokenizer.decode([int(i)])) for i in top5_idx]
        print(f"  Logits shape: {logits.shape}, dtype: {logits.dtype}")
        print(f"  Top-5: {top5}")
        print(f"  Stats: min={last_logits.min():.4f} max={last_logits.max():.4f} mean={last_logits.mean():.4f}")

        # Check position 0
        pos0 = logits[0, 0, :].astype(np.float32)
        print(f"  Pos 0: nonzero={np.count_nonzero(pos0)}, max={pos0.max():.4f}")

        results[name] = last_logits
    except Exception as e:
        print(f"  FAILED: {e}")

# Compare
if "CPU" in results and "CUDA" in results:
    corr = np.corrcoef(results["CPU"], results["CUDA"])[0, 1]
    diff = np.abs(results["CPU"] - results["CUDA"]).max()
    print(f"\nCPU vs CUDA: correlation={corr:.6f}, max_diff={diff:.4f}")

if "CPU" in results and "TensorRT" in results:
    corr = np.corrcoef(results["CPU"], results["TensorRT"])[0, 1]
    diff = np.abs(results["CPU"] - results["TensorRT"]).max()
    print(f"CPU vs TRT:  correlation={corr:.6f}, max_diff={diff:.4f}")

if "CUDA" in results and "TensorRT" in results:
    corr = np.corrcoef(results["CUDA"], results["TensorRT"])[0, 1]
    diff = np.abs(results["CUDA"] - results["TensorRT"]).max()
    print(f"CUDA vs TRT: correlation={corr:.6f}, max_diff={diff:.4f}")
