import os
import json
import shutil
import subprocess

def create_kaggle_split(part_num, start_idx, end_idx):
    folder = f'kaggle_qwen2_part{part_num}'
    os.makedirs(folder, exist_ok=True)
    
    # Read original CPU notebook
    with open('kaggle_cpu_submit/kaggle_qwen2_2b.ipynb', 'r') as f:
        nb = json.load(f)
        
    code = nb['cells'][1]['source']
    new_code = []
    
    for line in code:
        # Convert to GPU
        if "device_map='cpu'" in line:
            line = line.replace("device_map='cpu'", "device_map='cuda'")
        if "to('cpu')" in line:
            line = line.replace("to('cpu')", "to('cuda')")
            
        # Add slicing to test_df
        if "test_df = pd.read_csv" in line:
            new_code.append(line)
            # Add slice logic
            new_code.append(f"test_df = test_df.iloc[{start_idx}:{end_idx}]\n")
            continue
            
        new_code.append(line)
        
    nb['cells'][1]['source'] = new_code
    
    with open(f'{folder}/kaggle_qwen2_2b_part{part_num}.ipynb', 'w') as f:
        json.dump(nb, f)
        
    # Create metadata
    metadata = {
      "id": f"muthuramanraman7/cuhk-qwen2-vl-part{part_num}",
      "title": f"CUHK Qwen2-VL GPU Part {part_num}",
      "code_file": f"kaggle_qwen2_2b_part{part_num}.ipynb",
      "language": "python",
      "kernel_type": "notebook",
      "is_private": "true",
      "enable_gpu": "true",
      "enable_internet": "true",
      "dataset_sources": ["muthuramanraman7/cuhk-x-competition-large-model-track"],
      "competition_sources": [],
      "kernel_sources": []
    }
    
    with open(f'{folder}/kernel-metadata.json', 'w') as f:
        json.dump(metadata, f)
        
    print(f"Created Part {part_num} in {folder}")
    
create_kaggle_split(1, 0, 341)
create_kaggle_split(2, 341, 682)
