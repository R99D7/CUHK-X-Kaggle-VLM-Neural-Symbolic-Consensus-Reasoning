import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYTHONUTF8"] = "1"

from huggingface_hub import snapshot_download

print("Downloading Qwen2-VL-2B-Instruct using huggingface_hub without progress bars...")

path = snapshot_download(
    repo_id="Qwen/Qwen2-VL-2B-Instruct",
    local_dir=r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\Qwen2-VL-2B-Instruct",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=1
)

print(f"Downloaded fully to {path}!")
