import sys
from pathlib import Path

import onnx

export_dir = Path(sys.argv[1])
ckpt_dir = Path(sys.argv[2])

PROTO_LIMIT = 2 * 1024**3  # 2 GB protobuf serialization ceiling

for onnx_file in sorted(export_dir.glob("*.onnx")):
    print(f"Consolidating {onnx_file.name} ...")
    proto = onnx.load(str(onnx_file), load_external_data=True)
    dst = ckpt_dir / onnx_file.name
    if proto.ByteSize() < PROTO_LIMIT:
        onnx.save(proto, str(dst))
    else:
        # Model exceeds 2 GB — must use external data so protobuf doesn't
        # silently corrupt the file.  trtexec handles this transparently.
        onnx.save_model(
            proto,
            str(dst),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=dst.stem + ".onnx_data",
        )
    del proto

# Also copy non-ONNX files (metadata.safetensors)
for extra in sorted(export_dir.glob("*.safetensors")):
    print(f"Copying {extra.name} ...")
    dst = ckpt_dir / extra.name
    dst.write_bytes(extra.read_bytes())

print(f"Checkpoints written to {ckpt_dir}")
