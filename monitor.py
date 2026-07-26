import time
import kaggle
import sys

kernel_ref = 'muthuramanraman7/cuhk-qwen2-vl-7b-inference'

print(f"Monitoring {kernel_ref}...")
kaggle.api.authenticate()

while True:
    try:
        status = kaggle.api.kernels_status(kernel_ref)
        status_str = str(status.status)
        if 'RUNNING' not in status_str and 'QUEUED' not in status_str:
            print(f"Kernel finished with status: {status_str}")
            break
        print(f"Still {status_str}... checking again in 120s")
    except Exception as e:
        print(f"Error checking status: {e}. Retrying in 120s...")
    sys.stdout.flush()
    time.sleep(120)
