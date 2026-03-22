import os
from huggingface_hub import snapshot_download

repo_id = "BAAI/bge-small-en-v1.5"
local_dir = "source/"
print(f"downloading {repo_id}...")
try:
    model_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
    )
    print(f"download complete.")
except Exception as e:
    print(f"error: {e}")
