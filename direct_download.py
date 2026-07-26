import os
import requests

url = "https://huggingface.co/vikhyatk/moondream2/resolve/main/model.safetensors"
local_path = "model.safetensors"

print(f"Downloading {url} to {local_path}...")
try:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192*100):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"Downloaded {downloaded / 1e6:.2f} MB / {total / 1e6:.2f} MB", end='\r')
except Exception as e:
    print(f"Error: {e}")
print("\nDone.")
