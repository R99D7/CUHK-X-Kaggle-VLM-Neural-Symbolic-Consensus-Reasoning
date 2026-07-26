import os
import json
import shutil

folder = 'kaggle_qwen2_surgical'
os.makedirs(folder, exist_ok=True)

with open('kaggle_cpu_submit/kaggle_qwen2_2b.ipynb', 'r') as f:
    nb = json.load(f)

code = nb['cells'][1]['source']
new_code = []

for line in code:
    if "device_map='cpu'" in line:
        line = line.replace("device_map='cpu'", "device_map='cuda'")
    if "to('cpu')" in line:
        line = line.replace("to('cpu')", "to('cuda')")
        
    if "test_df = pd.read_csv" in line:
        new_code.append(line)
        new_code.append("contentious = ['test_0112', 'test_0113', 'test_0114', 'test_0116', 'test_0117', 'test_0119', 'test_0120', 'test_0123', 'test_0124', 'test_0125']\n")
        new_code.append("test_df = test_df[test_df['qa_id'].isin(contentious)]\n")
        continue
        
    new_code.append(line)

nb['cells'][1]['source'] = new_code

with open(f'{folder}/kaggle_qwen2_surgical.ipynb', 'w') as f:
    json.dump(nb, f)

metadata = {
  "id": "muthuramanraman7/cuhk-qwen2-vl-surgical",
  "title": "cuhk qwen2 vl surgical",
  "code_file": "kaggle_qwen2_surgical.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": ["cuhk-x-competition-large-model-track"],
  "kernel_sources": []
}

with open(f'{folder}/kernel-metadata.json', 'w') as f:
    json.dump(metadata, f)
    
print("Surgical split created!")
