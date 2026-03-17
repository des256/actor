import sys
from pathlib import Path

import onnx
import torch
from transformers import AutoModel, AutoTokenizer

export_dir = Path(sys.argv[1])

print("Loading model (will auto-download from HuggingFace if needed) ...")
model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

# Save tokenizer to output directory for Rust usage
tokenizer.save_pretrained(export_dir)

# Create dummy inputs for ONNX export
dummy_input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]], dtype=torch.long)  # [batch, seq_len]
dummy_attention_mask = torch.ones_like(dummy_input_ids, dtype=torch.long)
dummy_token_type_ids = torch.zeros_like(dummy_input_ids, dtype=torch.long)

print("Exporting model to ONNX ...")
# Note: BertModel exports both last_hidden_state and pooler_output (tanh).
# We only use last_hidden_state to extract the CLS token for embeddings.
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
    str(export_dir / "model.onnx"),
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "token_type_ids": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
    },
    opset_version=18,
    do_constant_folding=True,
)

# Validate the exported model
onnx_model = onnx.load(str(export_dir / "model.onnx"))
onnx.checker.check_model(onnx_model)

# Print input/output info for verification
print("\nModel inputs:")
for inp in onnx_model.graph.input:
    print(f"  - {inp.name}: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]}")

print("\nModel outputs:")
for out in onnx_model.graph.output:
    print(f"  - {out.name}: {[d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]}")

del model
print("\nONNX export complete.")
