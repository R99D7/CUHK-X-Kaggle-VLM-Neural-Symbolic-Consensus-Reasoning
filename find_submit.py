"""
Search for submission scripts, kaggle CLI capability, and investigate what built v265/v267.
"""
import os
import glob
import subprocess

print("Checking for Kaggle CLI or library...")
try:
    import kaggle
    print("Kaggle package imported successfully.")
except ImportError:
    print("Kaggle python library not installed.")
    
# Check if kaggle command works
res = subprocess.run(["kaggle", "--version"], capture_output=True, text=True, shell=True)
if res.returncode == 0:
    print(f"Kaggle CLI available: {res.stdout.strip()}")
else:
    print("Kaggle CLI command not running directly.")
    
# Search python scripts for submission logic
print("\nSearching .py files for kaggle submit...")
for p in glob.glob("*.py") + glob.glob("*.bat") + glob.glob("*.sh"):
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
            if "kaggle competitions submit" in txt or "auto_submit" in txt:
                print(f"Found auto-submit reference in: {p}")
    except Exception as e:
        pass

# Search for what built v263, v265, v267
print("\nSearching for generator scripts of v263, v265, v267:")
for p in glob.glob("*.py"):
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
            if "v263" in txt or "v265" in txt or "v267" in txt:
                print(f"File {p} mentions high scoring versions!")
    except Exception as e:
        pass
