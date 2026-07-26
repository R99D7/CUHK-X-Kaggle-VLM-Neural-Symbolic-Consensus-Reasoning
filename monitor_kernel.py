import time
import subprocess
import os

kernel = "muthuramanraman7/cuhk-qwen2-vl-inference"

print("Monitoring Kaggle kernel: " + kernel)
while True:
    result = subprocess.run(['kaggle', 'kernels', 'status', '-k', kernel], capture_output=True, text=True)
    out = result.stdout
    print(out.strip())
    
    if "complete" in out.lower():
        print("Kernel completed! Downloading output...")
        subprocess.run(['kaggle', 'kernels', 'output', kernel, '-p', 'kaggle_output'])
        print("Done downloading output.")
        break
    elif "error" in out.lower():
        print("Kernel failed with error!")
        break
    
    time.sleep(60)
