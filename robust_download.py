import requests
import os
import time

url = "https://huggingface.co/vikhyatk/moondream2/resolve/main/model.safetensors"
file_path = "model.safetensors"

def download_with_resume():
    downloaded = 0
    if os.path.exists(file_path):
        downloaded = os.path.getsize(file_path)

    print(f"Starting/resuming download from {downloaded} bytes...")

    while True:
        try:
            headers = {"Range": f"bytes={downloaded}-"}
            response = requests.get(url, headers=headers, stream=True, timeout=10)
            
            if response.status_code in [200, 206]:
                with open(file_path, "ab") as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (50 * 1024 * 1024) == 0:
                                print(f"Downloaded {downloaded / 1024 / 1024:.2f} MB")
                print("Download finished!")
                break
            elif response.status_code == 416:
                print("File fully downloaded.")
                break
            else:
                print(f"Unexpected status {response.status_code}, retrying in 5s...")
                time.sleep(5)
        except Exception as e:
            print(f"Connection error: {e}. Retrying in 5s...")
            time.sleep(5)

if __name__ == "__main__":
    download_with_resume()
