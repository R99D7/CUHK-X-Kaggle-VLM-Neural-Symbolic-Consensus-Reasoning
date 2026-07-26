import time
import subprocess
import sys

print('Polling...', flush=True)
start = time.time()
while time.time() - start < 300:
    out = subprocess.check_output('kaggle kernels status muthuramanraman7/cuhk-moondream-gpu', shell=True).decode()
    print(out.strip(), flush=True)
    if 'complete' in out.lower() or 'error' in out.lower():
        sys.exit(0)
    time.sleep(15)
