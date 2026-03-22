from huggingface_hub import snapshot_download

repo_id = "kyutai/pocket-tts"
local_dir = "source/"
print(f"downloading {repo_id}...")
try:
    model_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print("download complete.")
except Exception as e:
    print(f"error: {e}")
