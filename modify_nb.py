import json

with open('kaggle_cpu_submit/kaggle_qwen2_2b.ipynb', 'r') as f:
    nb = json.load(f)

source = nb['cells'][1]['source']
new_source = []
for s in source:
    s = s.replace("device_map='cpu'", "device_map='cuda'")
    s = s.replace("to('cpu')", "to('cuda')")
    new_source.append(s)

nb['cells'][1]['source'] = new_source

with open('kaggle_cpu_submit/kaggle_qwen2_2b_gpu.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
print("Saved GPU notebook.")
