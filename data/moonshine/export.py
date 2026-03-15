import sys
from pathlib import Path

import torch
from transformers import MoonshineStreamingForConditionalGeneration

source_dir = sys.argv[1]
export_dir = Path(sys.argv[2])

print(f"Loading model from {source_dir} ...")
# Eager attention avoids SymBool tracing issues with SDPA during export.
model = MoonshineStreamingForConditionalGeneration.from_pretrained(
    source_dir,
    attn_implementation="eager",
)
model.eval()


DEPTH = 14  # decoder_num_hidden_layers
NHEADS = 10  # decoder_num_attention_heads
HEAD_DIM = 64  # decoder_hidden_size // decoder_num_attention_heads


class DecoderWithKVCache(torch.nn.Module):
    """Wraps decoder + proj_out with explicit KV-cache tensors for ONNX."""

    def __init__(self, m):
        super().__init__()
        self.decoder = m.get_decoder()
        self.proj_out = m.proj_out
        self.depth = DEPTH

    def forward(
        self, input_ids, encoder_hidden_states, encoder_attention_mask, k_self, v_self
    ):
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        sa_cache = DynamicCache()
        for i in range(self.depth):
            sa_cache.update(k_self[i], v_self[i], i)
        ca_cache = DynamicCache()
        past_kv = EncoderDecoderCache(sa_cache, ca_cache)

        dec_out = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            past_key_values=past_kv,
        )
        logits = self.proj_out(dec_out.last_hidden_state)

        new_sa = dec_out.past_key_values.self_attention_cache
        new_k = torch.stack([layer.keys for layer in new_sa.layers])
        new_v = torch.stack([layer.values for layer in new_sa.layers])
        return logits, new_k, new_v


# -- Encoder --
encoder = model.get_encoder()
dummy_audio = torch.randn(1, 16000)
dummy_mask = torch.ones(1, 16000, dtype=torch.long)

print("Exporting encoder ...")
torch.onnx.export(
    encoder,
    (dummy_audio, dummy_mask),
    str(export_dir / "encoder_model.onnx"),
    input_names=["input_values", "attention_mask"],
    output_names=["last_hidden_state", "encoder_attention_mask"],
    dynamic_axes={
        "input_values": {0: "batch", 1: "audio_len"},
        "attention_mask": {0: "batch", 1: "audio_len"},
        "last_hidden_state": {0: "batch", 1: "enc_len"},
        "encoder_attention_mask": {0: "batch", 1: "enc_len"},
    },
    opset_version=18,
)

# -- Decoder (with KV cache) --
with torch.no_grad():
    enc_out = encoder(input_values=dummy_audio, attention_mask=dummy_mask)
enc_hidden = enc_out.last_hidden_state
enc_attn_mask = torch.ones(1, enc_hidden.shape[1], dtype=torch.long)
dec_ids = torch.tensor([[1]], dtype=torch.long)
k_past = torch.zeros(DEPTH, 1, NHEADS, 1, HEAD_DIM)
v_past = torch.zeros(DEPTH, 1, NHEADS, 1, HEAD_DIM)

dec_wrapper = DecoderWithKVCache(model)
dec_wrapper.eval()

print("Exporting decoder (with KV cache) ...")
torch.onnx.export(
    dec_wrapper,
    (dec_ids, enc_hidden, enc_attn_mask, k_past, v_past),
    str(export_dir / "decoder_model.onnx"),
    input_names=[
        "input_ids",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "k_self",
        "v_self",
    ],
    output_names=["logits", "out_k_self", "out_v_self"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "dec_len"},
        "encoder_hidden_states": {0: "batch", 1: "enc_len"},
        "encoder_attention_mask": {0: "batch", 1: "enc_len"},
        "k_self": {3: "past_len"},
        "v_self": {3: "past_len"},
        "logits": {0: "batch", 1: "dec_len"},
        "out_k_self": {3: "new_len"},
        "out_v_self": {3: "new_len"},
    },
    opset_version=18,
)

del model, encoder, dec_wrapper
print("ONNX export complete.")
