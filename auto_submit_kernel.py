import os
import time
import subprocess
import datetime

KERNEL_SLUG = "muthuramanraman7/cuhk-qwen2-vl-7b-inference"
COMPETITION = "cuhk-x-competition-large-model-track"
POLL_INTERVAL = 60  # seconds

def log(msg):
    print(f"[{datetime.datetime.now().isoformat()}] {msg}")

def check_status():
    result = subprocess.run(["kaggle", "kernels", "status", KERNEL_SLUG], capture_output=True, text=True)
    out = result.stdout.lower() + result.stderr.lower()
    if "complete" in out:
        return "complete"
    elif "running" in out or "queued" in out:
        return "running"
    elif "error" in out or "failed" in out:
        return "error"
    return "unknown"

def main():
    log(f"Starting background monitor for {KERNEL_SLUG}")
    
    while True:
        status = check_status()
        log(f"Current status: {status}")
        
        if status == "complete":
            log("Kernel finished! Downloading output...")
            os.system(f"kaggle kernels output {KERNEL_SLUG} -p . --force")
            
            log("Submitting to leaderboard...")
            os.system(f"kaggle competitions submit -c {COMPETITION} -f submission.csv -m \"Auto-submitted Qwen2-VL-7B (4-bit) inference output\"")
            
            log("All done! Exiting monitor.")
            break
            
        elif status == "error":
            log("Kernel failed with an error! Exiting monitor.")
            break
            
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
