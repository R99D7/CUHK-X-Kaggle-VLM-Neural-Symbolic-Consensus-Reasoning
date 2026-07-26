import time
import subprocess
import json

kernel_id = "muthuramanraman7/cuhk-moondream-cpu"
print(f"Monitoring Kaggle kernel: {kernel_id}...")

while True:
    result = subprocess.run(['kaggle', 'kernels', 'status', kernel_id], capture_output=True, text=True)
    status = result.stdout.strip()
    print(status)
    
    if "complete" in status.lower() or "error" in status.lower() or "cancel" in status.lower() or "fail" in status.lower():
        break
    time.sleep(30)

print("Kernel finished. Fetching output...")
subprocess.run(['kaggle', 'kernels', 'output', kernel_id, '-p', '.'])
print("Done! Check for submission_ultimate_v6.csv")
