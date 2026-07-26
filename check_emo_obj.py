"""
Check correlation between object_interaction and emotion in training.
"""
import pandas as pd
from collections import defaultdict

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')

tr_obj = {}
tr_emo = {}

for idx, row in tr.iterrows():
    vid = row['path']
    cat = row['category']
    ans_l = str(row['answer']).strip()
    
    if cat == 'object_interaction':
        if len(ans_l) == 1:
            tr_obj[vid] = str(row[ans_l]).strip().lower()
    elif cat == 'emotion':
        if len(ans_l) == 1:
            tr_emo[vid] = str(row[ans_l]).strip().lower()

emo_to_obj = defaultdict(lambda: defaultdict(int))
obj_to_emo = defaultdict(lambda: defaultdict(int))

for vid in set(tr_obj.keys()) & set(tr_emo.keys()):
    obj = tr_obj[vid]
    emo = tr_emo[vid]
    emo_to_obj[emo][obj] += 1
    obj_to_emo[obj][emo] += 1

print("Emotion -> Object Interaction:")
for emo, objs in sorted(emo_to_obj.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]:
    top_objs = sorted(objs.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  '{emo}': {top_objs}")

print("\nObject Interaction -> Emotion:")
for obj, emos in sorted(obj_to_emo.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]:
    top_emos = sorted(emos.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  '{obj}': {top_emos}")

