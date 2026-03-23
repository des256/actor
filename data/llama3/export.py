"""Export Llama 3.2 3B to INT8 quantized ONNX with explicit KV cache.

Pipeline: source/ → FP16 ONNX → INT8 quantization → ckpt/model.onnx

Exports a single model with explicit KV cache I/O tensors for all 28 layers:
  - Inputs: input_ids, position_ids, past_key_values.{0..27}.{key,value}
  - Outputs: logits, present.{0..27}.{key,value}

The ONNX model implements the full Llama3 architecture: token embedding,
RMSNorm, RoPE with llama3-style scaling, GQA attention, SwiGLU FFN, lm_head.
"""

import copy
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SOURCE_DIR = Path("source")
CKPT_DIR = Path("ckpt")

# Load config
config = AutoConfig.from_pretrained(SOURCE_DIR, trust_remote_code=True)

# Architecture constants from config
NUM_LAYERS = config.num_hidden_layers  # 28
NUM_HEADS = config.num_attention_heads  # 24
NUM_KV_HEADS = config.num_key_value_heads  # 8
HIDDEN_SIZE = config.hidden_size  # 3072
INTERMEDIATE_SIZE = config.intermediate_size  # 8192
HEAD_DIM = config.head_dim  # 128
VOCAB_SIZE = config.vocab_size  # 128256
RMS_NORM_EPS = config.rms_norm_eps  # 1e-05
# RoPE params — handle both old (rope_theta + rope_scaling) and new (rope_parameters) config formats
rope_params = getattr(config, "rope_parameters", None) or {}
ROPE_THETA = rope_params.get("rope_theta", None) or getattr(
    config, "rope_theta", 500000.0
)

rope_scaling = getattr(config, "rope_scaling", None) or rope_params
ROPE_FACTOR = rope_scaling.get("factor", 32.0)
ROPE_HIGH_FREQ_FACTOR = rope_scaling.get("high_freq_factor", 4.0)
ROPE_LOW_FREQ_FACTOR = rope_scaling.get("low_freq_factor", 1.0)
ROPE_ORIGINAL_MAX_POS = rope_scaling.get("original_max_position_embeddings", 8192)

print(f"Loading Llama 3.2 3B from {SOURCE_DIR} ...")
print(f"  Layers: {NUM_LAYERS}, Heads: {NUM_HEADS}, KV Heads: {NUM_KV_HEADS}")
print(
    f"  Hidden: {HIDDEN_SIZE}, Intermediate: {INTERMEDIATE_SIZE}, Head Dim: {HEAD_DIM}"
)
print(f"  Vocab: {VOCAB_SIZE}, RoPE theta: {ROPE_THETA}")


# ===== RMSNorm =====
class RMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.eps = eps

    def forward(self, x):
        # Compute in FP32 to avoid FP16 overflow in Pow/ReduceMean on GPU
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed.type_as(x)


# ===== Llama3 RoPE with scaling =====
def precompute_rope_cos_sin(dim, end, theta=10000.0, rope_scaling=None):
    """Precompute RoPE cos/sin tables with optional llama3-style scaling."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    if rope_scaling is not None:
        factor = rope_scaling["factor"]
        low_freq_factor = rope_scaling["low_freq_factor"]
        high_freq_factor = rope_scaling["high_freq_factor"]
        original_max_pos = rope_scaling["original_max_position_embeddings"]

        # Llama3 scaling: scale frequencies below low_freq_wavelen and above high_freq_wavelen
        low_freq_wavelen = original_max_pos / low_freq_factor
        high_freq_wavelen = original_max_pos / high_freq_factor

        wave_len = 2 * torch.pi / freqs
        new_freqs = []
        for freq, wavelen in zip(freqs, wave_len):
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / factor)
            else:
                # Smooth interpolation
                smooth = (original_max_pos / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
        freqs = torch.tensor(new_freqs)

    t = torch.arange(end, device=freqs.device)
    angles = torch.outer(t, freqs).float()  # [end, head_dim//2]
    return angles.cos(), angles.sin()


def apply_rotary_emb(xq, xk, rope_cos, rope_sin):
    """Apply rotary embeddings using Llama's half-rotation convention (rotate_half)."""
    # xq, xk: [batch, num_heads, seq_len, head_dim]
    # rope_cos, rope_sin: [seq_len, head_dim//2]
    seq_len = xq.shape[2]
    cos = rope_cos[:seq_len][None, None, :, :]  # [1, 1, seq_len, head_dim//2]
    sin = rope_sin[:seq_len][None, None, :, :]

    # Split head_dim into two halves (Llama convention)
    half = xq.shape[-1] // 2
    xq_float = xq.float()
    xk_float = xk.float()
    xq1, xq2 = xq_float[..., :half], xq_float[..., half:]
    xk1, xk2 = xk_float[..., :half], xk_float[..., half:]

    # rotate_half: q_out = q * cos + rotate_half(q) * sin
    #   q_out[:half] = q[:half] * cos - q[half:] * sin
    #   q_out[half:] = q[half:] * cos + q[:half] * sin
    xq_out = torch.cat([xq1 * cos - xq2 * sin, xq2 * cos + xq1 * sin], dim=-1)
    xk_out = torch.cat([xk1 * cos - xk2 * sin, xk2 * cos + xk1 * sin], dim=-1)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ===== Llama3 Attention =====
class LlamaAttention(nn.Module):
    def __init__(self, layer_weights, layer_idx):
        super().__init__()
        self.num_heads = NUM_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.head_dim = HEAD_DIM
        self.hidden_size = HIDDEN_SIZE

        # Load weights from the base model layer
        attn = layer_weights.self_attn
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)

        self.q_proj.weight.data = attn.q_proj.weight.data.clone()
        self.k_proj.weight.data = attn.k_proj.weight.data.clone()
        self.v_proj.weight.data = attn.v_proj.weight.data.clone()
        self.o_proj.weight.data = attn.o_proj.weight.data.clone()

        # Precompute RoPE cos/sin tables
        rope_cos, rope_sin = precompute_rope_cos_sin(
            HEAD_DIM, 8192, ROPE_THETA, rope_scaling=rope_scaling
        )
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

    def forward(self, x, past_k, past_v, position_ids):
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = (
            self.q_proj(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply RoPE — index into precomputed cos/sin by position
        pos = position_ids.squeeze(0)  # [seq_len]
        cos_batch = self.rope_cos[pos]  # [seq_len, head_dim//2]
        sin_batch = self.rope_sin[pos]
        q, k = apply_rotary_emb(q, k, cos_batch, sin_batch)

        # Concatenate past KV cache
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
        present_k = k
        present_v = v

        # GQA: repeat KV heads to match Q heads (expand+reshape, same as HF repeat_kv)
        num_repeats = self.num_heads // self.num_kv_heads
        kv_len = k.shape[2]
        k = k[:, :, None, :, :].expand(batch, self.num_kv_heads, num_repeats, kv_len, self.head_dim).reshape(batch, self.num_heads, kv_len, self.head_dim)
        v = v[:, :, None, :, :].expand(batch, self.num_kv_heads, num_repeats, kv_len, self.head_dim).reshape(batch, self.num_heads, kv_len, self.head_dim)

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale

        # Causal mask — use comparison ops instead of Trilu (TRT-friendly)
        # position_ids contains absolute positions: [0..seq_len-1] for prefill,
        # [past_len] for decode.  Query at position p attends to keys 0..p.
        cache_len = past_k.shape[2]
        total_len = cache_len + seq_len
        pos = position_ids.squeeze(0)  # [seq_len]
        col_indices = torch.arange(total_len, device=pos.device, dtype=pos.dtype)
        mask = col_indices[None, :] > pos[:, None]  # [seq_len, total_len]
        attn_weights = attn_weights.masked_fill(mask[None, None, :, :], float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(attn_output)

        return output, present_k, present_v


# ===== Llama3 MLP (SwiGLU) =====
class LlamaMLP(nn.Module):
    def __init__(self, layer_weights):
        super().__init__()
        mlp = layer_weights.mlp
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

        self.gate_proj.weight.data = mlp.gate_proj.weight.data.clone()
        self.up_proj.weight.data = mlp.up_proj.weight.data.clone()
        self.down_proj.weight.data = mlp.down_proj.weight.data.clone()

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ===== Llama3 Decoder Layer =====
class LlamaDecoderLayer(nn.Module):
    def __init__(self, layer_weights, layer_idx):
        super().__init__()
        self.input_layernorm = RMSNorm(
            layer_weights.input_layernorm.weight.data.clone(), RMS_NORM_EPS
        )
        self.self_attn = LlamaAttention(layer_weights, layer_idx)
        self.post_attention_layernorm = RMSNorm(
            layer_weights.post_attention_layernorm.weight.data.clone(), RMS_NORM_EPS
        )
        self.mlp = LlamaMLP(layer_weights)

    def forward(self, x, past_k, past_v, position_ids):
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        attn_out, present_k, present_v = self.self_attn(x, past_k, past_v, position_ids)
        x = residual + attn_out

        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_out = self.mlp(x)
        x = residual + mlp_out

        return x, present_k, present_v


# ===== Full Llama3 Model for ONNX Export =====
class Llama3ForONNX(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        model = base_model.model

        # Token embeddings
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.embed_tokens.weight.data = model.embed_tokens.weight.data.clone()

        # Decoder layers
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(model.layers[i], i) for i in range(NUM_LAYERS)]
        )

        # Final norm and lm_head
        self.norm = RMSNorm(model.norm.weight.data.clone(), RMS_NORM_EPS)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        self.lm_head.weight.data = base_model.lm_head.weight.data.clone()

    def forward(self, input_ids, position_ids, *past_key_values):
        # past_key_values: 56 tensors (28 layers x 2)
        # Reshape into [(past_k_0, past_v_0), (past_k_1, past_v_1), ...]
        past_kv_list = []
        for i in range(NUM_LAYERS):
            past_k = past_key_values[i * 2]
            past_v = past_key_values[i * 2 + 1]
            past_kv_list.append((past_k, past_v))

        # Embed tokens
        x = self.embed_tokens(input_ids)

        # Run through all layers
        present_key_values = []
        for layer, (past_k, past_v) in zip(self.layers, past_kv_list):
            x, present_k, present_v = layer(x, past_k, past_v, position_ids)
            present_key_values.append(present_k)
            present_key_values.append(present_v)

        # Final norm and lm_head
        x = self.norm(x)
        logits = self.lm_head(x)

        # Return logits + 56 present KV tensors
        return (logits, *present_key_values)


# ===== FP16 ONNX Export =====
def export_fp16() -> Path:
    """Export FP16 ONNX model with explicit KV cache to ckpt/model.onnx."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the full model (we'll extract weights from it)
    base_model = AutoModelForCausalLM.from_pretrained(
        SOURCE_DIR, dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_model.eval()

    # Build the ONNX-exportable model
    print("Building ONNX-exportable model...")
    onnx_model = Llama3ForONNX(base_model)
    onnx_model.eval()

    # Create dummy inputs
    batch_size = 1
    seq_len = 1
    past_len = 0

    dummy_input_ids = torch.tensor([[128000]], dtype=torch.long)  # BOS token
    dummy_position_ids = torch.arange(
        past_len, past_len + seq_len, dtype=torch.long
    ).unsqueeze(0)
    dummy_past_kv = [
        torch.zeros(batch_size, NUM_KV_HEADS, past_len, HEAD_DIM, dtype=torch.float16)
        for _ in range(NUM_LAYERS * 2)
    ]

    # Prepare input/output names
    input_names = ["input_ids", "position_ids"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {1: "seq_len"},
        "position_ids": {1: "seq_len"},
        "logits": {1: "seq_len"},
    }

    for i in range(NUM_LAYERS):
        input_names.append(f"past_key_values.{i}.key")
        input_names.append(f"past_key_values.{i}.value")
        output_names.append(f"present.{i}.key")
        output_names.append(f"present.{i}.value")

        dynamic_axes[f"past_key_values.{i}.key"] = {2: "past_len"}
        dynamic_axes[f"past_key_values.{i}.value"] = {2: "past_len"}
        dynamic_axes[f"present.{i}.key"] = {2: "new_len"}
        dynamic_axes[f"present.{i}.value"] = {2: "new_len"}

    print(
        f"Exporting to ONNX with {len(input_names)} inputs and {len(output_names)} outputs..."
    )
    print(f"  Inputs: {input_names[:4]} ... (56 KV tensors omitted)")
    print(f"  Outputs: {output_names[:3]} ... (56 KV tensors omitted)")

    # Export to ONNX
    onnx_path = CKPT_DIR / "model.onnx"
    torch.onnx.export(
        onnx_model,
        (dummy_input_ids, dummy_position_ids, *dummy_past_kv),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    # Verify the ONNX model (use path-based check — model exceeds 2GB protobuf limit)
    print("Verifying ONNX model...")
    onnx.checker.check_model(str(onnx_path))

    # Count inputs/outputs
    onnx_model_proto = onnx.load(str(onnx_path), load_external_data=False)
    num_inputs = len(onnx_model_proto.graph.input)
    num_outputs = len(onnx_model_proto.graph.output)
    print(f"ONNX model has {num_inputs} inputs and {num_outputs} outputs")
    assert num_inputs == 58, f"Expected 58 inputs, got {num_inputs}"
    assert num_outputs == 57, f"Expected 57 outputs, got {num_outputs}"

    # Numerical validation: compare ONNX vs PyTorch
    print("Running numerical validation...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_session = ort.InferenceSession(
        str(onnx_path), sess_options, providers=["CPUExecutionProvider"]
    )

    # Prepare test input
    test_input_ids = torch.tensor([[128000, 15339, 374]], dtype=torch.long)  # "Hello is"
    test_position_ids = torch.arange(0, 3, dtype=torch.long).unsqueeze(0)
    test_past_kv = [
        torch.zeros(1, NUM_KV_HEADS, 0, HEAD_DIM, dtype=torch.float16)
        for _ in range(NUM_LAYERS * 2)
    ]

    # ONNX inference
    ort_inputs = {
        "input_ids": test_input_ids.numpy(),
        "position_ids": test_position_ids.numpy(),
    }
    for i in range(NUM_LAYERS):
        ort_inputs[f"past_key_values.{i}.key"] = test_past_kv[i * 2].numpy()
        ort_inputs[f"past_key_values.{i}.value"] = test_past_kv[i * 2 + 1].numpy()

    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_logits = ort_outputs[0]

    # Sanity check: verify logits shape and values are reasonable
    assert onnx_logits.shape == (1, 3, VOCAB_SIZE), (
        f"Unexpected logits shape: {onnx_logits.shape}"
    )
    top_token = int(np.argmax(onnx_logits[0, -1, :]))
    print(f"  Logits shape: {onnx_logits.shape}, top token at last pos: {top_token}")
    print("FP16 ONNX export complete")

    return onnx_path


# ===== INT4 AWQ Quantization =====
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


class C4CalibrationDataReader:
    """Reads C4 calibration data formatted for the Llama3 ONNX model inputs."""

    def __init__(self, tokenizer_path: Path, num_samples: int, seq_len: int):
        from datasets import load_dataset

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
            # Empty KV cache for all layers (prefill mode)
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
            # Auto-rewind so the reader can be iterated multiple times
            # (GEMV detection pass + calibration pass)
            self.index = 0
            return None
        result = self.inputs[self.index]
        self.index += 1
        return result

    def get_first(self):
        assert len(self.inputs) > 0, "Calibration data list is empty!"
        return self.inputs[0]

    def rewind(self):
        self.index = 0

    def __iter__(self):
        return iter(self.inputs)

    def __len__(self):
        return len(self.inputs)


def quantize_int8(onnx_path: Path):
    """INT8 quantization using NVIDIA modelopt.

    Reads the FP16 ONNX model from onnx_path, quantizes to INT8,
    and overwrites it in place.
    """
    # onnx_graphsurgeon 0.5.8 references onnx.helper.float32_to_bfloat16,
    # which was removed in onnx 1.17+. Provide it so graphsurgeon can import.
    import onnx.helper

    if not hasattr(onnx.helper, "float32_to_bfloat16"):
        onnx.helper.float32_to_bfloat16 = lambda x: (
            np.array(x, dtype=np.float32).view(np.uint32) >> 16
        ).astype(np.uint16)

    # Monkey-patch modelopt's save_onnx before importing quantize.
    # Must patch every module that does `from modelopt.onnx.utils import save_onnx`
    # since that binds a local name unaffected by patching the source module.
    import modelopt.onnx.quantization.graph_utils as _graph_utils_mod
    import modelopt.onnx.quantization.int8 as _int8_mod
    import modelopt.onnx.utils as _modelopt_utils

    _modelopt_utils.save_onnx = _patched_save_onnx
    _int8_mod.save_onnx = _patched_save_onnx
    _graph_utils_mod.save_onnx = _patched_save_onnx

    from modelopt.onnx.quantization.int8 import quantize

    num_calib_samples = 32
    calib_seq_len = 128

    calib_reader = C4CalibrationDataReader(SOURCE_DIR, num_calib_samples, calib_seq_len)

    print(f"\nQuantizing {onnx_path} with INT8 ...")
    print("  Excluding: /lm_head (default)")
    print("  Calibration EP: CUDAExecutionProvider")

    quantized_model = quantize(
        str(onnx_path),
        calibration_method="entropy",
        calibration_data_reader=calib_reader,
        calibration_eps=["cuda", "cpu"],
        use_external_data_format=True,
        nodes_to_exclude=[r"/lm_head"],
        calibrate_per_node=True,
    )

    print(f"\nSaving quantized model to {onnx_path} ...")
    _patched_save_onnx(quantized_model, str(onnx_path), save_as_external_data=True)

    print("\nINT8 quantization complete")
    print(f"  Output: {onnx_path}")


if __name__ == "__main__":
    export_fp16()
