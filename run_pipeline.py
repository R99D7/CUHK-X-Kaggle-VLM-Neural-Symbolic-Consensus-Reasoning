import os
import time
import subprocess
import shutil

TARGET_SIZE = 3854538968
FILE_PATH = "model.safetensors"
SNAPSHOT_DIR = r"C:\Users\MUTHURAMANRAMANATHAN\.cache\huggingface\hub\models--vikhyatk--moondream2\snapshots\6b714b26eea5cbd9f31e4edb2541c170afa935ba"

print("Waiting for robust_download.py to finish...")
while True:
    if os.path.exists(FILE_PATH):
        size = os.path.getsize(FILE_PATH)
        if size >= TARGET_SIZE:
            print("Download complete!")
            break
        else:
            print(f"Current size: {size / 1024 / 1024:.2f} MB / {TARGET_SIZE / 1024 / 1024:.2f} MB")
    time.sleep(10)

print(f"Moving {FILE_PATH} to {SNAPSHOT_DIR}...")
try:
    shutil.copy(FILE_PATH, os.path.join(SNAPSHOT_DIR, "model.safetensors"))
    print("Moved successfully!")
except Exception as e:
    print("Error moving file:", e)

print("Starting Moondream inference on 25% of files (GPU)...")
subprocess.run("py run_local_moondream.py", shell=True)

print("Blending results with submission_ultimate_v3.csv...")
subprocess.run("py blend_moondream.py", shell=True)

print("ALL DONE! The final file is submission_ultimate_v5.csv (Wait, blend_moondream.py creates v5).")
