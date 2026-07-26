import os
import requests
import threading
import time

URL1 = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/main/model-00001-of-00002.safetensors"
URL2 = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/main/model-00002-of-00002.safetensors"

def download_chunk(url, start, end, filename, chunk_index):
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(10):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            if r.status_code in (206, 200):
                with open(f"{filename}.part{chunk_index}", "wb") as f:
                    for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                return True
        except Exception as e:
            time.sleep(2)
    return False

def fast_download(url, filename, num_threads=16):
    print(f"Resolving URL for {filename}...")
    # Follow redirects to get the real CloudFront URL
    s = requests.Session()
    r = s.head(url, allow_redirects=True)
    real_url = r.url
    
    # Sometimes HEAD on S3/CloudFront fails or omits content-length if not explicitly requested
    r2 = requests.get(real_url, headers={"Range": "bytes=0-0"})
    content_range = r2.headers.get('content-range')
    if content_range:
        total_size = int(content_range.split('/')[-1])
    else:
        total_size = int(r.headers.get('content-length', 0))
        
    if total_size < 1000000:
        print(f"Failed to get real size for {filename} (Got {total_size}).")
        return
        
    print(f"Total size: {total_size / (1024**2):.2f} MB. Starting {num_threads} threads...")
    chunk_size = total_size // num_threads
    threads = []
    
    for i in range(num_threads):
        start = i * chunk_size
        end = total_size - 1 if i == num_threads - 1 else (start + chunk_size - 1)
        t = threading.Thread(target=download_chunk, args=(real_url, start, end, filename, i))
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
    
    t1 = threading.Thread(target=fast_download, args=(URL1, "model-00001-of-00002.safetensors", 16))
    t2 = threading.Thread(target=fast_download, args=(URL2, "model-00002-of-00002.safetensors", 16))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    print("ALL DOWNLOADS COMPLETE!")
