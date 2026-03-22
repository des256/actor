"""Export Pocket TTS components to ONNX format.

Exports five models:
  - mimi_encoder_model.onnx   (audio → voice conditioning)
  - text_encoder_model.onnx   (token IDs → text embeddings)
  - flow_lm_main_model.onnx   (transformer backbone with KV cache)
  - flow_lm_flow_model.onnx   (flow-matching MLP)
  - mimi_decoder_model.onnx   (latent → audio waveform)
"""

import sys
from pathlib import Path

import onnx
import pocket_tts  # noqa: E402
import pocket_tts.modules.mimi_transformer as _mimi_xfmr
import torch
import torch.nn.functional as F
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import increment_steps, init_states
from pocket_tts.utils.config import load_config

source_dir = sys.argv[1]
export_dir = Path(sys.argv[2])

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
config = load_config(Path(pocket_tts.__file__).parent / "config/b6369a24.yaml")

# Point to local source files instead of HuggingFace
config.weights_path = str(Path(source_dir).resolve() / "tts_b6369a24.safetensors")
config.weights_path_without_voice_cloning = None
config.flow_lm.weights_path = None
config.mimi.weights_path = None
config.flow_lm.lookup_table.tokenizer_path = str(
    Path(source_dir).resolve() / "tokenizer.model"
)

print(f"Loading model from {source_dir} ...")
model = TTSModel._from_pydantic_config_with_weights(
    config, temp=0.7, lsd_decode_steps=3, noise_clamp=3.0, eos_threshold=-3.0
)
model.float()
model.eval()

# -- Architecture constants --
DEPTH = 6  # flow_lm transformer layers
NHEADS = 16  # attention heads
HEAD_DIM = 64  # d_model // num_heads
D_MODEL = 1024
LDIM = 32  # latent dimension
MIMI_CONTEXT = 250  # mimi decoder transformer context window


# ===== 1. Mimi Encoder (voice cloning: audio → conditioning) ==============
class MimiEncoderForExport(torch.nn.Module):
    """Encode a voice prompt into conditioning vectors.

    The caller must pre-pad audio to a multiple of frame_size (1920 samples).
    """

    def __init__(self, m: TTSModel):
        super().__init__()
        self.encoder = m.mimi.encoder
        self.encoder_transformer = m.mimi.encoder_transformer
        self.downsample = m.mimi.downsample
        self.speaker_proj_weight = m.flow_lm.speaker_proj_weight

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [1, 1, samples]
        emb = self.encoder(audio, model_state=None)
        (emb,) = self.encoder_transformer(emb, model_state=None)
        emb = self.downsample(emb, model_state=None)
        latents = emb.transpose(-1, -2).to(torch.float32)  # [1, T, 512]
        conditioning = F.linear(latents, self.speaker_proj_weight)  # [1, T, 1024]
        return conditioning


mimi_enc = MimiEncoderForExport(model)
mimi_enc.eval()

dummy_audio = torch.randn(1, 1, 48000)  # 2 s at 24 kHz, multiple of 1920

print("Exporting mimi_encoder ...")
torch.onnx.export(
    mimi_enc,
    (dummy_audio,),
    str(export_dir / "mimi_encoder_model.onnx"),
    input_names=["audio"],
    output_names=["conditioning"],
    dynamic_axes={
        "audio": {2: "audio_len"},
        "conditioning": {1: "cond_len"},
    },
    opset_version=18,
)


# ===== 2. Text Encoder (token IDs → text embeddings) =====================
class TextEncoderForExport(torch.nn.Module):
    """Look up text token embeddings from the conditioner vocabulary."""

    def __init__(self, m: TTSModel):
        super().__init__()
        self.embed = m.flow_lm.conditioner.embed

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [1, T] i64
        return self.embed(token_ids)  # [1, T, D_MODEL]


text_enc = TextEncoderForExport(model)
text_enc.eval()

dummy_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)

print("Exporting text_encoder ...")
torch.onnx.export(
    text_enc,
    (dummy_ids,),
    str(export_dir / "text_encoder_model.onnx"),
    input_names=["token_ids"],
    output_names=["text_embeddings"],
    dynamic_axes={
        "token_ids": {1: "text_len"},
        "text_embeddings": {1: "text_len"},
    },
    opset_version=18,
)


# ===== 3. Flow LM Main (transformer backbone with KV cache) ==============

# Monkey-patch beartype off modules that receive SymInt from the dynamo tracer.
# pocket_tts applies beartype_this_package() which rejects torch.SymInt
# (produced by PyTorch's dynamo-based ONNX exporter) for int-typed parameters.
import pocket_tts.modules.rope as _rope_mod  # noqa: E402
import pocket_tts.modules.transformer as _xfm_mod  # noqa: E402

for _fn_name in ("apply_rope",):
    _fn = getattr(_rope_mod, _fn_name)
    if hasattr(_fn, "__wrapped__"):
        setattr(_rope_mod, _fn_name, _fn.__wrapped__)
if hasattr(_rope_mod.RotaryEmbedding.forward, "__wrapped__"):
    _rope_mod.RotaryEmbedding.forward = _rope_mod.RotaryEmbedding.forward.__wrapped__

if hasattr(_xfm_mod._materialize_causal_mask, "__wrapped__"):
    _xfm_mod._materialize_causal_mask = _xfm_mod._materialize_causal_mask.__wrapped__
_materialize_causal_mask = _xfm_mod._materialize_causal_mask


class FlowLmMain(torch.nn.Module):
    """Flow LM transformer backbone with KV cache and latent projection.

    Handles both conditioning passes (text/voice via text_embeddings with
    empty sequence) and generation steps (latent via sequence with empty
    text_embeddings).  BOS is signaled by NaN in the sequence tensor.

    Bakes in input_linear and bos_emb weights.  emb_mean/emb_std are NOT
    used here — they de-normalize latents for the mimi decoder (see
    MimiDecoderForExport).
    """

    def __init__(self, m: TTSModel):
        super().__init__()
        self.layers = m.flow_lm.transformer.layers
        self.rope = m.flow_lm.transformer.rope
        self.out_norm = m.flow_lm.out_norm
        self.out_eos = m.flow_lm.out_eos
        self.depth = DEPTH

        # Latent processing weights (baked into the model)
        self.input_linear = m.flow_lm.input_linear  # Linear(LDIM → D_MODEL)
        self.register_buffer("bos_emb", m.flow_lm.bos_emb.data)  # [LDIM]

    def forward(
        self,
        sequence: torch.Tensor,  # [B, seq_len, LDIM]
        text_embeddings: torch.Tensor,  # [B, cond_len, D_MODEL]
        kv_cache: torch.Tensor,  # [DEPTH, 2, B, past_len, NHEADS, HEAD_DIM]
        cache_len: torch.Tensor,  # [1] i64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        past_len = kv_cache.shape[3]

        # Project latent sequence: NaN → BOS, linear project
        is_nan = torch.isnan(sequence[:, :, 0:1])  # [B, seq_len, 1]
        bos = self.bos_emb.view(1, 1, LDIM).expand_as(sequence)
        seq = torch.where(is_nan, bos, sequence)
        projected_seq = self.input_linear(seq)  # [B, seq_len, D_MODEL]

        # Combine: text_embeddings (conditioning) + projected sequence (generation)
        x = torch.cat([text_embeddings, projected_seq], dim=1)

        b, t, _ = x.shape
        new_ks = []
        new_vs = []

        for i in range(self.depth):
            layer = self.layers[i]
            attn = layer.self_attn

            # --- self-attention ---
            x_norm = layer.norm1(x)
            projected = attn.in_proj(x_norm)  # [B, T, 3·D]
            packed = projected.view(b, t, 3, NHEADS, HEAD_DIM)
            q, k, v = torch.unbind(packed, dim=2)  # each [B, T, H, D]

            q, k = self.rope(q, k, offset=past_len)

            # → [B, H, T, D] for attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Prepend cached KV (stored as [B, past_len, H, D] in combined cache)
            k_cached = kv_cache[i, 0].transpose(1, 2)  # → [B, H, past_len, D]
            v_cached = kv_cache[i, 1].transpose(1, 2)  # → [B, H, past_len, D]
            k = torch.cat([k_cached, k], dim=2)
            v = torch.cat([v_cached, v], dim=2)
            new_ks.append(k)
            new_vs.append(v)

            # causal mask (additive: 0 = attend, -inf = block)
            total_len = k.shape[2]
            mask = _materialize_causal_mask(
                (t, total_len), shift=past_len, device=q.device
            )

            # manual attention (more portable than SDPA for ONNX)
            scale = HEAD_DIM**-0.5
            attn_w = torch.matmul(q * scale, k.transpose(-2, -1))
            attn_w = attn_w + mask
            attn_w = F.softmax(attn_w, dim=-1)
            out = torch.matmul(attn_w, v)

            out = out.transpose(1, 2).reshape(b, t, -1)
            out = attn.out_proj(out)
            x = x.to(out) + layer.layer_scale_1(out)

            # --- feed-forward ---
            x = layer._ff_block(x)

        x = self.out_norm(x)
        last = x[:, -1]  # [B, D_MODEL]
        eos_logit = self.out_eos(last)  # [B, 1]

        # Build combined KV cache: [DEPTH, 2, B, new_len, H, D]
        # new_ks[i] is [B, H, new_len, D] → transpose to [B, new_len, H, D]
        out_k = torch.stack([k.transpose(1, 2) for k in new_ks])
        out_v = torch.stack([v.transpose(1, 2) for v in new_vs])
        new_kv_cache = torch.stack([out_k, out_v], dim=1)

        # Derive new_cache_len from output KV shape (traces to ONNX Shape op)
        kv_shape = torch._shape_as_tensor(new_kv_cache)
        new_cache_len = kv_shape[3:4]  # [1] i64

        return last, eos_logit, new_kv_cache, new_cache_len


flow_lm_main = FlowLmMain(model)
flow_lm_main.eval()

dummy_seq = torch.randn(1, 1, LDIM)
dummy_text_emb = torch.randn(1, 1, D_MODEL)
dummy_kv = torch.zeros(DEPTH, 2, 1, 1, NHEADS, HEAD_DIM)
dummy_cache_len = torch.tensor([1], dtype=torch.int64)

print("Exporting flow_lm_main ...")
torch.onnx.export(
    flow_lm_main,
    (dummy_seq, dummy_text_emb, dummy_kv, dummy_cache_len),
    str(export_dir / "flow_lm_main_model.onnx"),
    input_names=["sequence", "text_embeddings", "kv_cache", "cache_len"],
    output_names=["conditioning", "eos_logit", "new_kv_cache", "new_cache_len"],
    dynamic_axes={
        "sequence": {1: "seq_len"},
        "text_embeddings": {1: "cond_len"},
        "kv_cache": {3: "past_len"},
        "new_kv_cache": {3: "new_len"},
    },
    opset_version=18,
)


# ===== 4. Flow LM Flow (flow-matching MLP, fused) =========================
# forward(c, s, t, x) → latent  (= x + flow_dir, fused on GPU)
#   c      : [B, D_MODEL]  conditioning from flow_lm_main
#   s      : [B, 1]        start time
#   t      : [B, 1]        target time
#   x      : [B, LDIM]     noisy latent
#   latent : [B, LDIM]     noise + flow direction (final latent)


class FlowNetFused(torch.nn.Module):
    """Wraps flow_net to fuse the x + flow_dir addition on GPU."""

    def __init__(self, flow_net):
        super().__init__()
        self.flow_net = flow_net

    def forward(self, c, s, t, x):
        flow_dir = self.flow_net(c, s, t, x)
        return x + flow_dir


fused_flow = FlowNetFused(model.flow_lm.flow_net)
fused_flow.eval()

dummy_c = torch.randn(1, D_MODEL)
dummy_s = torch.zeros(1, 1)
dummy_t = torch.ones(1, 1)
dummy_x = torch.randn(1, LDIM)

print("Exporting flow_lm_flow ...")
torch.onnx.export(
    fused_flow,
    (dummy_c, dummy_s, dummy_t, dummy_x),
    str(export_dir / "flow_lm_flow_model.onnx"),
    input_names=["c", "s", "t", "x"],
    output_names=["latent"],
    opset_version=18,
)


# ===== 5. Mimi Decoder (latent → audio, stateful) ========================

# Monkey-patch the KV cache complete() function to use FUNCTIONAL operations
# instead of in-place scatter_ and end_offset[:] = ... which the ONNX tracer
# cannot correctly propagate (the attention reads pre-scatter/pre-update values).


def _functional_complete_kv(self, k, v, model_state):
    """Replacement for _complete_kv that avoids in-place ops for ONNX tracing."""
    if model_state is None:
        return _mimi_xfmr.KVCacheResult.from_kv(k, v)
    layer_state = self.get_state(model_state)
    cache = layer_state["cache"]
    end_offset = layer_state["end_offset"]

    capacity = cache.shape[3]
    B, H, T, D = k.shape

    indexes = torch.arange(T, device=end_offset.device, dtype=end_offset.dtype)
    indexes = indexes + end_offset.view(-1, 1)
    indexes = indexes % capacity

    this_indexes = indexes.view(B, 1, T, 1).expand(-1, H, T, D)

    # Functional scatter (returns new tensor instead of in-place modification)
    cache_0_new = cache[0].scatter(2, this_indexes, k)
    cache_1_new = cache[1].scatter(2, this_indexes, v)

    keys = cache_0_new
    values = cache_1_new

    # Write updated cache back to state
    layer_state["cache"] = torch.stack([cache_0_new, cache_1_new])

    indexes_all = torch.arange(capacity, device=end_offset.device, dtype=torch.long)
    last_offset = end_offset.view(-1, 1) + T - 1
    end_index = last_offset % capacity
    delta = indexes_all - end_index

    positions = torch.where(
        delta <= 0, last_offset + delta, last_offset + delta - capacity
    )

    # Functional end_offset update (no in-place [:] assignment)
    end_offset_new = end_offset + T
    layer_state["end_offset"] = end_offset_new

    invalid = indexes_all >= end_offset_new.view(-1, 1)
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)

    return _mimi_xfmr.KVCacheResult(keys, values, positions)


_mimi_xfmr.MimiStreamingMultiheadAttention._complete_kv = _functional_complete_kv


# Monkey-patch the attention forward to use manual SDPA with NaN-safe softmax.
# PyTorch's F.scaled_dot_product_attention returns zeros when all keys are masked
# (boolean all-False), but ONNX runtime's decomposed softmax produces NaN from
# exp(-inf).  This happens when query positions precede all cached key positions
# (offset < min_pos_k), which occurs once the KV ring buffer wraps.
_orig_mimi_attn_forward = _mimi_xfmr.MimiStreamingMultiheadAttention.forward


def _nan_safe_attn_forward(self, query, model_state=None):
    from einops import rearrange as _rearrange

    B, T = query.shape[:2]
    if model_state is None:
        offset = torch.zeros(B, device=query.device, dtype=torch.long)
    else:
        offset = self.get_state(model_state)["offset"]

    projected = self.in_proj(query)
    q, k, v = _rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    q, k = self.rope(q, k, offset)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)

    k, v, pos_k = self._complete_kv(k, v, model_state)
    pos_k = pos_k[:, None]
    pos_q = offset.view(-1, 1, 1) + torch.arange(
        T, device=q.device, dtype=torch.long
    ).view(-1, 1)
    delta = pos_q - pos_k
    attn_bias = (pos_k >= 0) & (delta >= 0)
    attn_bias = attn_bias & (delta < self.context)
    attn_bias = attn_bias[:, None]  # [B, 1, T_q, T_k]

    # Manual SDPA: convert boolean mask to additive float mask, then
    # replace NaN after softmax with 0 (matches PyTorch SDPA behaviour).
    dim_per_head = self.embed_dim // self.num_heads
    scale = dim_per_head ** -0.5
    attn_w = torch.matmul(q * scale, k.transpose(-2, -1))
    attn_w = attn_w + torch.where(
        attn_bias, torch.zeros_like(attn_w), torch.full_like(attn_w, float("-inf"))
    )
    attn_w = F.softmax(attn_w, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    x = torch.matmul(attn_w, v)

    x = _rearrange(x, "b h t d -> b t (h d)")
    x = self.out_proj(x)
    return x


_mimi_xfmr.MimiStreamingMultiheadAttention.forward = _nan_safe_attn_forward


class MimiDecoderStatefulForExport(torch.nn.Module):
    """Stateful mimi decoder for ONNX with explicit state I/O.

    Processes exactly one latent frame per call, maintaining internal conv
    history buffers, transformer KV caches, and upsample partials as explicit
    model inputs/outputs. Enables O(n) incremental decode instead of O(n²)
    batch re-decode.

    Bakes in emb_mean/emb_std to de-normalize flow_lm latents before
    feeding them to the mimi quantizer (see tts_model._decode_audio_worker).
    """

    def __init__(self, m: TTSModel):
        super().__init__()
        self.mimi = m.mimi
        self.register_buffer("emb_mean", m.flow_lm.emb_mean.data)  # [LDIM]
        self.register_buffer("emb_std", m.flow_lm.emb_std.data)  # [LDIM]

    def forward(
        self,
        latent: torch.Tensor,  # [1, LDIM, 1]
        state_0: torch.Tensor,   # decoder.model.0.previous [1, 512, 6]
        state_1: torch.Tensor,   # decoder.model.2.partial [1, 256, 6]
        state_2: torch.Tensor,   # decoder.model.3.block.1.previous [1, 256, 2]
        state_3: torch.Tensor,   # decoder.model.5.partial [1, 128, 5]
        state_4: torch.Tensor,   # decoder.model.6.block.1.previous [1, 128, 2]
        state_5: torch.Tensor,   # decoder.model.8.partial [1, 64, 4]
        state_6: torch.Tensor,   # decoder.model.9.block.1.previous [1, 64, 2]
        state_7: torch.Tensor,   # decoder.model.11.previous [1, 64, 2]
        state_8: torch.Tensor,   # decoder_transformer.layers.0.cache [2,1,8,250,64]
        state_9: torch.Tensor,   # decoder_transformer.layers.0.offset [1]
        state_10: torch.Tensor,  # decoder_transformer.layers.0.end_offset [1]
        state_11: torch.Tensor,  # decoder_transformer.layers.1.cache [2,1,8,250,64]
        state_12: torch.Tensor,  # decoder_transformer.layers.1.offset [1]
        state_13: torch.Tensor,  # decoder_transformer.layers.1.end_offset [1]
        state_14: torch.Tensor,  # upsample.convtr.partial [1, 512, 16]
    ) -> tuple[torch.Tensor, ...]:
        # latent: [1, LDIM, 1] single frame
        # De-normalize: emb_mean/emb_std are [LDIM], reshape to [1, LDIM, 1]
        latent = latent * self.emb_std.view(1, LDIM, 1) + self.emb_mean.view(1, LDIM, 1)

        # Build full state dict and override with input tensors
        state = init_states(self.mimi, batch_size=1, sequence_length=MIMI_CONTEXT)
        state["decoder.model.0"]["previous"] = state_0
        state["decoder.model.2"]["partial"] = state_1
        state["decoder.model.3.block.1"]["previous"] = state_2
        state["decoder.model.5"]["partial"] = state_3
        state["decoder.model.6.block.1"]["previous"] = state_4
        state["decoder.model.8"]["partial"] = state_5
        state["decoder.model.9.block.1"]["previous"] = state_6
        state["decoder.model.11"]["previous"] = state_7
        state["decoder_transformer.transformer.layers.0.self_attn"]["cache"] = state_8
        state["decoder_transformer.transformer.layers.0.self_attn"]["offset"] = state_9
        state["decoder_transformer.transformer.layers.0.self_attn"]["end_offset"] = state_10
        state["decoder_transformer.transformer.layers.1.self_attn"]["cache"] = state_11
        state["decoder_transformer.transformer.layers.1.self_attn"]["offset"] = state_12
        state["decoder_transformer.transformer.layers.1.self_attn"]["end_offset"] = state_13
        state["upsample.convtr"]["partial"] = state_14

        # Run decoder pipeline
        quantized = self.mimi.quantizer(latent)
        emb = self.mimi._to_encoder_framerate(quantized, state)
        (emb,) = self.mimi.decoder_transformer(emb, state)
        audio = self.mimi.decoder(emb, state)

        # Advance transformer offset positions (RoPE) for next frame.
        # Each latent frame produces 16 tokens via _to_encoder_framerate,
        # matching the standard library's increment_steps(self.mimi, state, 16).
        increment_steps(self.mimi, state, 16)

        # Extract updated states
        out_state_0 = state["decoder.model.0"]["previous"]
        out_state_1 = state["decoder.model.2"]["partial"]
        out_state_2 = state["decoder.model.3.block.1"]["previous"]
        out_state_3 = state["decoder.model.5"]["partial"]
        out_state_4 = state["decoder.model.6.block.1"]["previous"]
        out_state_5 = state["decoder.model.8"]["partial"]
        out_state_6 = state["decoder.model.9.block.1"]["previous"]
        out_state_7 = state["decoder.model.11"]["previous"]
        out_state_8 = state["decoder_transformer.transformer.layers.0.self_attn"]["cache"]
        out_state_9 = state["decoder_transformer.transformer.layers.0.self_attn"]["offset"]
        out_state_10 = state["decoder_transformer.transformer.layers.0.self_attn"]["end_offset"]
        out_state_11 = state["decoder_transformer.transformer.layers.1.self_attn"]["cache"]
        out_state_12 = state["decoder_transformer.transformer.layers.1.self_attn"]["offset"]
        out_state_13 = state["decoder_transformer.transformer.layers.1.self_attn"]["end_offset"]
        out_state_14 = state["upsample.convtr"]["partial"]

        return (
            audio,
            out_state_0, out_state_1, out_state_2, out_state_3, out_state_4,
            out_state_5, out_state_6, out_state_7, out_state_8, out_state_9,
            out_state_10, out_state_11, out_state_12, out_state_13, out_state_14,
        )


mimi_dec = MimiDecoderStatefulForExport(model)
mimi_dec.eval()

dummy_latent = torch.randn(1, LDIM, 1)  # one frame
# Initialize 15 dummy state tensors (all zeros, matching init_states)
dummy_state_0 = torch.zeros(1, 512, 6)
dummy_state_1 = torch.zeros(1, 256, 6)
dummy_state_2 = torch.zeros(1, 256, 2)
dummy_state_3 = torch.zeros(1, 128, 5)
dummy_state_4 = torch.zeros(1, 128, 2)
dummy_state_5 = torch.zeros(1, 64, 4)
dummy_state_6 = torch.zeros(1, 64, 2)
dummy_state_7 = torch.zeros(1, 64, 2)
dummy_state_8 = torch.zeros(2, 1, 8, 250, 64)
dummy_state_9 = torch.zeros(1, dtype=torch.int64)
dummy_state_10 = torch.zeros(1, dtype=torch.int64)
dummy_state_11 = torch.zeros(2, 1, 8, 250, 64)
dummy_state_12 = torch.zeros(1, dtype=torch.int64)
dummy_state_13 = torch.zeros(1, dtype=torch.int64)
dummy_state_14 = torch.zeros(1, 512, 16)

print("Exporting mimi_decoder ...")
torch.onnx.export(
    mimi_dec,
    (dummy_latent, dummy_state_0, dummy_state_1, dummy_state_2, dummy_state_3,
     dummy_state_4, dummy_state_5, dummy_state_6, dummy_state_7, dummy_state_8,
     dummy_state_9, dummy_state_10, dummy_state_11, dummy_state_12,
     dummy_state_13, dummy_state_14),
    str(export_dir / "mimi_decoder_model.onnx"),
    input_names=["latent", "state_0", "state_1", "state_2", "state_3", "state_4",
                 "state_5", "state_6", "state_7", "state_8", "state_9", "state_10",
                 "state_11", "state_12", "state_13", "state_14"],
    output_names=["audio", "out_state_0", "out_state_1", "out_state_2", "out_state_3",
                  "out_state_4", "out_state_5", "out_state_6", "out_state_7",
                  "out_state_8", "out_state_9", "out_state_10", "out_state_11",
                  "out_state_12", "out_state_13", "out_state_14"],
    opset_version=18,
)

# Strip the "reduction" attribute from ScatterND nodes.  PyTorch's ONNX
# exporter emits ScatterND with reduction="none" (from aten.slice_scatter),
# but TensorRT's parser rejects any ScatterND that carries a reduction
# attribute.  Since "none" is the default semantics (plain scatter, no
# reduction), removing the attribute is a no-op in terms of model behaviour.
print("Stripping unsupported ScatterND reduction attributes ...")
_decoder_path = str(export_dir / "mimi_decoder_model.onnx")
_onnx_model = onnx.load(_decoder_path)
for _node in _onnx_model.graph.node:
    if _node.op_type == "ScatterND":
        _to_remove = [a for a in _node.attribute if a.name == "reduction"]
        for a in _to_remove:
            _node.attribute.remove(a)
onnx.save(_onnx_model, _decoder_path)
del _onnx_model


del model, mimi_enc, text_enc, flow_lm_main, mimi_dec
print("ONNX export complete.")
