#!/usr/bin/env python3
"""Export Moonshine Streaming Medium to 5 ONNX models.

Models exported:
  frontend.onnx   - Stateful causal-conv frontend (audio → features)
  encoder.onnx    - Sliding-window encoder (features → encoded)
  adapter.onnx    - Positional embedding + projection (encoded → memory)
  cross_kv.onnx   - Cross-attention KV computation (memory → k_cross, v_cross)
  decoder_kv.onnx - Autoregressive decoder with self-attn KV cache

Audio input to the frontend must be a multiple of FRAME_LEN (80 samples = 5ms).
The Rust orchestration layer handles sample buffering to ensure this.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSpeechSeq2Seq

DEST = os.path.join(os.path.dirname(__file__), "..", "moonshine")
os.makedirs(DEST, exist_ok=True)

# ── Load HuggingFace model ──────────────────────────────────────────

print("Loading HuggingFace model...")
hf = AutoModelForSpeechSeq2Seq.from_pretrained(
    "UsefulSensors/moonshine-streaming-medium",
    dtype=torch.float32,
    attn_implementation="eager",
).eval()

ENCODER_DIM = hf.config.encoder_hidden_size          # 768
DECODER_DIM = hf.config.hidden_size                  # 640
DEPTH       = hf.config.num_hidden_layers             # 14
NHEADS      = hf.config.num_attention_heads           # 10
HEAD_DIM    = hf.config.head_dim                      # 64
VOCAB_SIZE  = hf.config.vocab_size                    # 32768
FRAME_LEN   = 80   # 5ms at 16kHz
C1          = ENCODER_DIM * 2                         # 1536
ROTARY_DIM  = int(HEAD_DIM * hf.config.rope_parameters["partial_rotary_factor"])  # 32

print(f"  encoder_dim={ENCODER_DIM} decoder_dim={DECODER_DIM} depth={DEPTH}")
print(f"  nheads={NHEADS} head_dim={HEAD_DIM} rotary_dim={ROTARY_DIM}")


# ── 1. Frontend ─────────────────────────────────────────────────────

class FrontendONNX(nn.Module):
    """Streaming frontend with causal convolution state.

    Audio length must be a multiple of FRAME_LEN (80).
    Inputs:  audio_chunk [1,N], conv1_buffer [1,768,4], conv2_buffer [1,1536,4]
    Outputs: features [1,F,768], conv1_buffer_out [1,768,4], conv2_buffer_out [1,1536,4]
    """
    def __init__(self, embedder):
        super().__init__()
        self.linear = embedder.linear
        self.log_k = embedder.comp.log_k
        self.conv1_weight = embedder.conv1.weight
        self.conv1_bias = embedder.conv1.bias
        self.conv2_weight = embedder.conv2.weight
        self.conv2_bias = embedder.conv2.bias
        self.eps = embedder.cmvn.eps

    def forward(self, audio_chunk, conv1_buffer, conv2_buffer):
        # Frame audio into FRAME_LEN windows (caller ensures N % FRAME_LEN == 0)
        num_frames = audio_chunk.shape[1] // FRAME_LEN
        frames = audio_chunk.reshape(1, num_frames, FRAME_LEN)

        # CMVN per frame
        mean = frames.mean(dim=-1, keepdim=True)
        centered = frames - mean
        rms = (centered.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        normed = centered / rms

        # Asinh compression
        k = torch.exp(self.log_k)
        kx = k * normed
        compressed = torch.log(kx + torch.sqrt(kx * kx + 1.0))

        # Linear + SiLU → [1, num_frames, 768]
        hidden = F.silu(self.linear(compressed))

        # Transpose to [1, 768, num_frames] for convolutions
        hidden = hidden.transpose(1, 2)

        # Causal Conv1: prepend buffer, conv, save new buffer
        conv1_input = torch.cat([conv1_buffer, hidden], dim=2)
        conv1_out = F.conv1d(conv1_input, self.conv1_weight, self.conv1_bias, stride=2)
        new_conv1_buffer = conv1_input[:, :, -4:]
        conv1_out = F.silu(conv1_out)

        # Causal Conv2: prepend buffer, conv, save new buffer
        conv2_input = torch.cat([conv2_buffer, conv1_out], dim=2)
        conv2_out = F.conv1d(conv2_input, self.conv2_weight, self.conv2_bias, stride=2)
        new_conv2_buffer = conv2_input[:, :, -4:]

        # Transpose back to [1, F, 768]
        features = conv2_out.transpose(1, 2)

        return features, new_conv1_buffer, new_conv2_buffer


# ── 2. Encoder ──────────────────────────────────────────────────────

ENC_NHEADS = hf.config.encoder_config.num_attention_heads     # 10
ENC_HEAD_DIM = hf.config.encoder_config.head_dim              # 64
ENC_PROJ_DIM = ENC_NHEADS * ENC_HEAD_DIM                      # 640
ENC_SCALE = 1.0 / (ENC_HEAD_DIM ** 0.5)                       # 0.125
SLIDING_WINDOWS = hf.config.encoder_config.sliding_windows    # [[16,4], ...]


class EncoderONNX(nn.Module):
    """Encoder with manual sliding-window attention for ONNX compatibility.

    features [1,W,768] → encoded [1,W,768]
    """
    def __init__(self, encoder):
        super().__init__()

        # Per-layer sliding window sizes [left, right]
        self.register_buffer(
            "windows", torch.tensor(SLIDING_WINDOWS, dtype=torch.int64),
        )

        # Collect all layer parameters
        self.input_layernorms = nn.ModuleList()
        self.input_gammas = nn.ParameterList()
        self.q_projs = nn.ModuleList()
        self.k_projs = nn.ModuleList()
        self.v_projs = nn.ModuleList()
        self.o_projs = nn.ModuleList()
        self.post_layernorms = nn.ModuleList()
        self.post_gammas = nn.ParameterList()
        self.fc1s = nn.ModuleList()
        self.fc2s = nn.ModuleList()

        for layer in encoder.layers:
            self.input_layernorms.append(layer.input_layernorm.ln)
            self.input_gammas.append(
                nn.Parameter(layer.input_layernorm.gamma + layer.input_layernorm.unit_offset),
            )
            self.q_projs.append(layer.self_attn.q_proj)
            self.k_projs.append(layer.self_attn.k_proj)
            self.v_projs.append(layer.self_attn.v_proj)
            self.o_projs.append(layer.self_attn.o_proj)
            self.post_layernorms.append(layer.post_attention_layernorm.ln)
            self.post_gammas.append(
                nn.Parameter(layer.post_attention_layernorm.gamma + layer.post_attention_layernorm.unit_offset),
            )
            self.fc1s.append(layer.mlp.fc1)
            self.fc2s.append(layer.mlp.fc2)

        self.final_ln = encoder.final_norm.ln
        self.final_gamma = nn.Parameter(
            encoder.final_norm.gamma + encoder.final_norm.unit_offset,
        )

    def forward(self, features):
        hidden = features
        B, S, _ = hidden.shape

        # Build distance matrix for sliding window masks: dist[i,j] = i - j
        pos = torch.arange(S, device=hidden.device)
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)           # [S, S]

        for i in range(len(self.q_projs)):
            residual = hidden

            # LayerNorm + gamma
            normed = self.input_layernorms[i](hidden) * self.input_gammas[i]

            # Q/K/V projections → [B, H, S, HD]
            q = self.q_projs[i](normed).reshape(B, S, ENC_NHEADS, ENC_HEAD_DIM).transpose(1, 2)
            k = self.k_projs[i](normed).reshape(B, S, ENC_NHEADS, ENC_HEAD_DIM).transpose(1, 2)
            v = self.v_projs[i](normed).reshape(B, S, ENC_NHEADS, ENC_HEAD_DIM).transpose(1, 2)

            # Sliding window mask matching HF formula:
            #   left_mask:  dist >= 0 AND dist < left_window
            #   right_mask: dist < 0 AND -dist < right_window
            left = self.windows[i, 0]
            right = self.windows[i, 1]
            left_ok = (dist >= 0) & (dist < left)
            right_ok = (dist < 0) & (-dist < right)
            in_window = left_ok | right_ok                        # [S, S]
            mask = torch.where(in_window, 0.0, -1e9)             # [S, S]
            mask = mask.unsqueeze(0).unsqueeze(0)                 # [1, 1, S, S]

            # Attention: softmax((Q @ K^T) * scale + mask) @ V
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * ENC_SCALE + mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_out = torch.matmul(attn_weights, v)

            attn_out = attn_out.transpose(1, 2).reshape(B, S, ENC_PROJ_DIM)
            hidden = residual + self.o_projs[i](attn_out)

            # MLP
            residual = hidden
            normed = self.post_layernorms[i](hidden) * self.post_gammas[i]
            hidden = residual + self.fc2s[i](F.gelu(self.fc1s[i](normed)))

        # Final norm
        hidden = self.final_ln(hidden) * self.final_gamma
        return hidden


# ── 3. Adapter ──────────────────────────────────────────────────────

class AdapterONNX(nn.Module):
    """Adapter: encoded [1,F,768] + pos_offset [1] → memory [1,F,640]"""
    def __init__(self, decoder):
        super().__init__()
        self.pos_emb = decoder.pos_emb
        self.proj = decoder.proj

    def forward(self, encoded, pos_offset):
        seq_len = encoded.shape[1]
        positions = torch.arange(seq_len, device=encoded.device) + pos_offset[0]
        pos = self.pos_emb(positions).unsqueeze(0)  # [1, F, 768]
        return self.proj(encoded + pos)  # [1, F, 640]


# ── 4. Cross KV ─────────────────────────────────────────────────────

class CrossKVONNX(nn.Module):
    """Cross KV: memory [1,M,640] → k_cross [D,1,H,M,HD], v_cross [D,1,H,M,HD]"""
    def __init__(self, decoder_layers):
        super().__init__()
        self.k_projs = nn.ModuleList([layer.encoder_attn.k_proj for layer in decoder_layers])
        self.v_projs = nn.ModuleList([layer.encoder_attn.v_proj for layer in decoder_layers])

    def forward(self, memory):
        B, M, _ = memory.shape
        k_layers = []
        v_layers = []
        for k_proj, v_proj in zip(self.k_projs, self.v_projs):
            k = k_proj(memory).reshape(B, M, NHEADS, HEAD_DIM).permute(0, 2, 1, 3)
            v = v_proj(memory).reshape(B, M, NHEADS, HEAD_DIM).permute(0, 2, 1, 3)
            k_layers.append(k)
            v_layers.append(v)
        k_cross = torch.stack(k_layers, dim=0)  # [D,1,H,M,HD]
        v_cross = torch.stack(v_layers, dim=0)
        return k_cross, v_cross


# ── 5. Decoder KV ───────────────────────────────────────────────────

class DecoderKVONNX(nn.Module):
    """Single-step autoregressive decoder with explicit KV cache.

    Inputs:  token [1,1], k_self [D,1,H,S,HD], v_self [D,1,H,S,HD],
             out_k_cross [D,1,H,M,HD], out_v_cross [D,1,H,M,HD]
    Outputs: logits [1,1,V], out_k_self [D,1,H,S+1,HD], out_v_self [D,1,H,S+1,HD]
    """
    def __init__(self, model):
        super().__init__()
        dec = model.model.decoder
        self.embed_tokens = dec.embed_tokens
        self.norm = dec.norm
        self.layers = dec.layers
        self.proj_out = model.proj_out
        self.register_buffer("inv_freq", dec.rotary_emb.inv_freq.clone())

    def _rope(self, x, offset):
        """Apply interleaved RoPE to the first ROTARY_DIM dims of x [B,H,1,HD].

        Matches HuggingFace's rotate_half + repeat_interleave pattern:
        pairs adjacent elements (x[0],x[1]), (x[2],x[3]), ... with each frequency.
        """
        pos = offset.float()
        freqs = self.inv_freq * pos                         # [ROTARY_DIM//2]
        # Interleave: [f0*p, f0*p, f1*p, f1*p, ...]
        interleaved = freqs.repeat_interleave(2).reshape(1, 1, 1, ROTARY_DIM)
        cos = interleaved.cos()
        sin = interleaved.sin()

        x_rot = x[..., :ROTARY_DIM]
        x_pass = x[..., ROTARY_DIM:]

        # rotate_half: pair even/odd indices → [-x1, x0, -x3, x2, ...]
        x1 = x_rot[..., 0::2]   # even indices
        x2 = x_rot[..., 1::2]   # odd indices
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)

        x_rot = x_rot * cos + rotated * sin
        return torch.cat([x_rot, x_pass], dim=-1)

    def forward(self, token, k_self, v_self, out_k_cross, out_v_cross):
        hidden = self.embed_tokens(token)
        seq_pos = k_self.shape[3]

        new_k_layers = []
        new_v_layers = []

        for i, layer in enumerate(self.layers):
            residual = hidden

            # Self-attention
            normed = layer.input_layernorm(hidden)
            sa = layer.self_attn

            q = sa.q_proj(normed).reshape(1, 1, NHEADS, HEAD_DIM).transpose(1, 2)
            k = sa.k_proj(normed).reshape(1, 1, NHEADS, HEAD_DIM).transpose(1, 2)
            v = sa.v_proj(normed).reshape(1, 1, NHEADS, HEAD_DIM).transpose(1, 2)

            # Apply RoPE
            pos_tensor = torch.tensor(seq_pos, dtype=torch.float32, device=token.device)
            q = self._rope(q, pos_tensor)
            k = self._rope(k, pos_tensor)

            # Append to KV cache
            k_cache = torch.cat([k_self[i : i + 1], k.unsqueeze(0)], dim=3)
            v_cache = torch.cat([v_self[i : i + 1], v.unsqueeze(0)], dim=3)
            new_k_layers.append(k_cache.squeeze(0))
            new_v_layers.append(v_cache.squeeze(0))

            # Scaled dot-product attention
            attn = F.scaled_dot_product_attention(
                q, k_cache.squeeze(0), v_cache.squeeze(0), is_causal=False
            )
            attn = attn.transpose(1, 2).reshape(1, 1, DECODER_DIM)
            hidden = residual + sa.o_proj(attn)

            # Cross-attention
            residual = hidden
            normed = layer.post_attention_layernorm(hidden)
            ca = layer.encoder_attn

            q_cross = ca.q_proj(normed).reshape(1, 1, NHEADS, HEAD_DIM).transpose(1, 2)
            k_cross_i = out_k_cross[i]
            v_cross_i = out_v_cross[i]

            attn = F.scaled_dot_product_attention(q_cross, k_cross_i, v_cross_i, is_causal=False)
            attn = attn.transpose(1, 2).reshape(1, 1, DECODER_DIM)
            hidden = residual + ca.o_proj(attn)

            # MLP
            residual = hidden
            normed = layer.final_layernorm(hidden)
            hidden = residual + layer.mlp(normed)

        hidden = self.norm(hidden)
        logits = self.proj_out(hidden)

        out_k_self = torch.stack(new_k_layers, dim=0)
        out_v_self = torch.stack(new_v_layers, dim=0)

        return logits, out_k_self, out_v_self


# ── Export helper ────────────────────────────────────────────────────

def export(name, module, inputs, input_names, output_names, dynamic_axes):
    path = os.path.join(DEST, f"{name}.onnx")
    print(f"Exporting {name}...")
    module.eval()
    torch.onnx.export(
        module,
        inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  -> {path} ({size_mb:.1f} MB)")


# ── Export all models ────────────────────────────────────────────────

with torch.no_grad():
    # 1. Frontend (simplified: no sample buffer, audio must be multiple of 80)
    frontend = FrontendONNX(hf.model.encoder.embedder)
    export(
        "frontend", frontend,
        (torch.randn(1, 1280), torch.zeros(1, ENCODER_DIM, 4), torch.zeros(1, C1, 4)),
        ["audio_chunk", "conv1_buffer", "conv2_buffer"],
        ["features", "conv1_buffer_out", "conv2_buffer_out"],
        {"audio_chunk": {1: "audio_len"}, "features": {1: "num_features"}},
    )

    # 2. Encoder
    encoder = EncoderONNX(hf.model.encoder)
    export(
        "encoder", encoder,
        (torch.randn(1, 16, ENCODER_DIM),),
        ["features"],
        ["encoded"],
        {"features": {1: "seq_len"}, "encoded": {1: "seq_len"}},
    )

    # 3. Adapter
    adapter = AdapterONNX(hf.model.decoder)
    export(
        "adapter", adapter,
        (torch.randn(1, 8, ENCODER_DIM), torch.zeros(1, dtype=torch.int64)),
        ["encoded", "pos_offset"],
        ["memory"],
        {"encoded": {1: "num_frames"}, "memory": {1: "num_frames"}},
    )

    # 4. Cross KV
    cross = CrossKVONNX(hf.model.decoder.layers)
    export(
        "cross_kv", cross,
        (torch.randn(1, 8, DECODER_DIM),),
        ["memory"],
        ["k_cross", "v_cross"],
        {"memory": {1: "mem_len"}, "k_cross": {3: "mem_len"}, "v_cross": {3: "mem_len"}},
    )

    # 5. Decoder KV (use self_len=1 to avoid zero-length cache export issues)
    decoder = DecoderKVONNX(hf)
    export(
        "decoder_kv", decoder,
        (
            torch.tensor([[1]], dtype=torch.int64),
            torch.randn(DEPTH, 1, NHEADS, 1, HEAD_DIM),
            torch.randn(DEPTH, 1, NHEADS, 1, HEAD_DIM),
            torch.randn(DEPTH, 1, NHEADS, 8, HEAD_DIM),
            torch.randn(DEPTH, 1, NHEADS, 8, HEAD_DIM),
        ),
        ["token", "k_self", "v_self", "out_k_cross", "out_v_cross"],
        ["logits", "out_k_self", "out_v_self"],
        {
            "k_self": {3: "self_len"}, "v_self": {3: "self_len"},
            "out_k_self": {3: "self_len_plus1"}, "out_v_self": {3: "self_len_plus1"},
            "out_k_cross": {3: "mem_len"}, "out_v_cross": {3: "mem_len"},
        },
    )

print("\nDone! Models exported to", DEST)
