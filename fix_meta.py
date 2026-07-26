import json

for p in [1, 2]:
    f = f'kaggle_qwen2_part{p}/kernel-metadata.json'
    data = json.load(open(f))
    data['dataset_sources'] = []
    data['competition_sources'] = ['cuhk-x-competition-large-model-track']
    json.dump(data, open(f, 'w'))
