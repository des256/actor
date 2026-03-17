"""Export Pocket TTS components to ONNX format.

Exports four models:
  - mimi_encoder_model.onnx  (audio → voice conditioning)
  - backbone_model.onnx      (autoregressive transformer with KV cache)
  - flow_net_model.onnx      (flow-matching MLP)
  - mimi_decoder_model.onnx  (latent → audio waveform)

Plus metadata.safetensors with small tensors needed by the runtime
(embedding table, input projection, normalization stats, BOS token).
"""

import sys
from pathlib import Path

import onnx
import pocket_tts  # noqa: E402
import safetensors.torch
import torch
import torch.nn.functional as F
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states
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


# ===== 2. Backbone (autoregressive transformer with KV cache) =============
class BackboneWithKVCache(torch.nn.Module):
    """Flow LM transformer backbone with explicit KV-cache tensors for ONNX.

    Takes pre-embedded input (text embeddings and/or projected latents already
    concatenated in the D_MODEL space).  The caller handles the embedding
    lookup (conditioner.embed) and projection (input_linear) outside this
    model, using weights from metadata.safetensors.
    """

    def __init__(self, m: TTSModel):
        super().__init__()
        self.layers = m.flow_lm.transformer.layers
        self.rope = m.flow_lm.transformer.rope
        self.out_norm = m.flow_lm.out_norm
        self.out_eos = m.flow_lm.out_eos
        self.depth = DEPTH

    def forward(
        self, input_emb: torch.Tensor, k_self: torch.Tensor, v_self: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_emb : [B, T, D_MODEL]
        # k_self    : [DEPTH, B, NHEADS, past_len, HEAD_DIM]
        # v_self    : [DEPTH, B, NHEADS, past_len, HEAD_DIM]

        x = input_emb
        new_ks = []
        new_vs = []
        past_len = k_self.shape[3]

        for i in range(self.depth):
            layer = self.layers[i]
            attn = layer.self_attn

            # --- self-attention ---
            x_norm = layer.norm1(x)
            projected = attn.in_proj(x_norm)  # [B, T, 3·D]
            b, t, _ = projected.shape
            packed = projected.view(b, t, 3, NHEADS, HEAD_DIM)
            q, k, v = torch.unbind(packed, dim=2)  # each [B, T, H, D]

            q, k = self.rope(q, k, offset=past_len)

            # → [B, H, T, D] for attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # prepend cached KV
            k = torch.cat([k_self[i], k], dim=2)
            v = torch.cat([v_self[i], v], dim=2)
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
        last = x[:, -1]  # [B, D]
        is_eos = self.out_eos(last)  # [B, 1]

        out_k = torch.stack(new_ks)  # [DEPTH, B, H, new_len, D]
        out_v = torch.stack(new_vs)

        return last, is_eos, out_k, out_v


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

backbone = BackboneWithKVCache(model)
backbone.eval()

dummy_emb = torch.randn(1, 1, D_MODEL)
k_past = torch.zeros(DEPTH, 1, NHEADS, 1, HEAD_DIM)
v_past = torch.zeros(DEPTH, 1, NHEADS, 1, HEAD_DIM)

print("Exporting backbone ...")
torch.onnx.export(
    backbone,
    (dummy_emb, k_past, v_past),
    str(export_dir / "backbone_model.onnx"),
    input_names=["input_emb", "k_self", "v_self"],
    output_names=["hidden", "is_eos", "out_k_self", "out_v_self"],
    dynamic_axes={
        "input_emb": {1: "seq_len"},
        "k_self": {3: "past_len"},
        "v_self": {3: "past_len"},
        "out_k_self": {3: "new_len"},
        "out_v_self": {3: "new_len"},
    },
    opset_version=18,
)


# ===== 3. Flow Net (flow-matching MLP) ====================================
# forward(c, s, t, x) → v
#   c : [B, D_MODEL]  conditioning from backbone
#   s : [B, 1]         start time
#   t : [B, 1]         target time
#   x : [B, LDIM]      noisy latent
#   v : [B, LDIM]      velocity (flow direction)

dummy_c = torch.randn(1, D_MODEL)
dummy_s = torch.zeros(1, 1)
dummy_t = torch.ones(1, 1)
dummy_x = torch.randn(1, LDIM)

print("Exporting flow_net ...")
torch.onnx.export(
    model.flow_lm.flow_net,
    (dummy_c, dummy_s, dummy_t, dummy_x),
    str(export_dir / "flow_net_model.onnx"),
    input_names=["conditioning", "s", "t", "x"],
    output_names=["v"],
    opset_version=18,
)


# ===== 4. Mimi Decoder (latent → audio) ===================================
class MimiDecoderForExport(torch.nn.Module):
    """Non-streaming mimi decoder for ONNX.

    Processes all latent frames at once (suitable for batch decoding).
    Streaming decode with per-frame state is handled by the C++ runtime.
    """

    def __init__(self, m: TTSModel):
        super().__init__()
        self.mimi = m.mimi

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: [1, LDIM, num_frames]
        state = init_states(self.mimi, batch_size=1, sequence_length=MIMI_CONTEXT)
        quantized = self.mimi.quantizer(latent)
        emb = self.mimi._to_encoder_framerate(quantized, state)
        (emb,) = self.mimi.decoder_transformer(emb, state)
        out = self.mimi.decoder(emb, state)
        return out


mimi_dec = MimiDecoderForExport(model)
mimi_dec.eval()

dummy_latent = torch.randn(1, LDIM, 1)  # one frame

print("Exporting mimi_decoder ...")
torch.onnx.export(
    mimi_dec,
    (dummy_latent,),
    str(export_dir / "mimi_decoder_model.onnx"),
    input_names=["latent"],
    output_names=["audio"],
    dynamic_axes={
        "latent": {2: "num_frames"},
        "audio": {2: "audio_len"},
    },
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


# ===== 5. Metadata tensors ================================================
# Small tensors the runtime needs outside of the ONNX models:
#   - conditioner.embed.weight : text token → embedding lookup [4001, 1024]
#   - input_linear.weight      : latent projection [1024, 32]
#   - emb_mean / emb_std       : latent de-normalisation [32]
#   - bos_emb                  : beginning-of-sequence embedding [32]

print("Saving metadata tensors ...")
safetensors.torch.save_file(
    {
        "conditioner.embed.weight": model.flow_lm.conditioner.embed.weight.data,
        "input_linear.weight": model.flow_lm.input_linear.weight.data,
        "emb_mean": model.flow_lm.emb_mean.data,
        "emb_std": model.flow_lm.emb_std.data,
        "bos_emb": model.flow_lm.bos_emb.data,
    },
    str(export_dir / "metadata.safetensors"),
)


del model, mimi_enc, backbone, mimi_dec
print("ONNX export complete.")
