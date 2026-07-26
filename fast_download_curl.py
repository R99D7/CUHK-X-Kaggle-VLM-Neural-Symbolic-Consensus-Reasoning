import os
import subprocess
import threading
import time
import json
import urllib.request

URL1 = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/main/model-00001-of-00002.safetensors"
URL2 = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/main/model-00002-of-00002.safetensors"

def get_real_url_and_size(url):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req) as response:
            real_url = response.geturl()
            size = int(response.headers.get('Content-Length', 0))
            if size == 0:
                # If HEAD fails, try GET Range 0-0
                req = urllib.request.Request(real_url, headers={"Range": "bytes=0-0"})
                with urllib.request.urlopen(req) as r2:
                    content_range = r2.headers.get('Content-Range')
                    if content_range:
                        size = int(content_range.split('/')[-1])
            return real_url, size
    except Exception as e:
        print(f"Failed to get size for {url}: {e}")
        return None, 0

def download_chunk_curl(url, start, end, filename, chunk_index):
    part_file = f"{filename}.part{chunk_index}"
    expected_size = end - start + 1
    
    if os.path.exists(part_file) and os.path.getsize(part_file) == expected_size:
        return True

    cmd = ["curl.exe", "-s", "-L", "-r", f"{start}-{end}", "-o", part_file, url]
    for attempt in range(10):
        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(part_file) and os.path.getsize(part_file) == expected_size:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def fast_download_curl(url, filename, num_threads=16):
    print(f"Resolving URL for {filename}...")
    real_url, total_size = get_real_url_and_size(url)
    
    if total_size < 1000000:
        print(f"Failed to get real size for {filename} (Got {total_size}).")
        return
        
    print(f"Total size: {total_size / (1024**2):.2f} MB. Starting {num_threads} curl threads...")
    chunk_size = total_size // num_threads
    threads = []
    
    for i in range(num_threads):
        start = i * chunk_size
        end = total_size - 1 if i == num_threads - 1 else (start + chunk_size - 1)
        t = threading.Thread(target=download_chunk_curl, args=(real_url, start, end, filename, i))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    print(f"Merging {filename}...")
    with open(filename, "wb") as outfile:
        for i in range(num_threads):
            part_file = f"{filename}.part{i}"
            if os.path.exists(part_file):
                with open(part_file, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(part_file)
            else:
                print(f"Missing part {i}")
    print(f"Finished {filename}!")

if __name__ == "__main__":
    os.makedirs(r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\Qwen2-VL-2B-Instruct", exist_ok=True)
    os.chdir(r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\Qwen2-VL-2B-Instruct")
    
    t1 = threading.Thread(target=fast_download_curl, args=(URL1, "model-00001-of-00002.safetensors", 16))
    t2 = threading.Thread(target=fast_download_curl, args=(URL2, "model-00002-of-00002.safetensors", 16))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    print("ALL DOWNLOADS COMPLETE!")
