import os
import time
import subprocess

kernels = [
    'muthuramanraman7/cuhk-deberta-v3-large',
    'muthuramanraman7/cuhk-timesformer-video'
]

def check_status(kernel):
    result = subprocess.run(['kaggle', 'kernels', 'status', kernel], capture_output=True, text=True)
    return result.stdout.strip()

def main():
    finished = set()
    
    while len(finished) < len(kernels):
        for k in kernels:
            if k in finished:
                continue
            status = check_status(k)
            print(f"{k}: {status}")
            
            if 'COMPLETE' in status:
                print(f"Downloading output for {k}...")
                subprocess.run(['kaggle', 'kernels', 'output', k])
                finished.add(k)
            elif 'ERROR' in status:
                print(f"WARNING: {k} failed with an error!")
                finished.add(k)
                
        if len(finished) < len(kernels):
            print("Waiting 60 seconds...")
            time.sleep(60)
            
    print("All kernels finished executing!")
    
if __name__ == "__main__":
    main()
