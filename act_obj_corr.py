"""
Analyze correlations between actions (from single/combination/multi/sequence)
and the object_interaction / emotion answers on the same video.
"""
import pandas as pd
from collections import defaultdict

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Build known actions for each training video
tr_actions = defaultdict(set)
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
    else:
        # Extract actions
        if cat in ['single', 'sequence']:
            if len(ans_l) == 1 or cat == 'sequence':
                for l in ans_l:
                    if l in ['A', 'B', 'C', 'D']:
                        tr_actions[vid].add(str(row[l]).strip().lower())
        elif cat == 'combination':
            if len(ans_l) == 1:
                ans_text = str(row[ans_l]).strip().lower()
                for a in ans_text.split(','):
                    tr_actions[vid].add(a.strip())
        elif cat == 'multi':
            for l in ans_l:
                if l in ['A', 'B', 'C', 'D']:
                    tr_actions[vid].add(str(row[l]).strip().lower())

# For each object_interaction answer, what actions frequently co-occur?
obj_to_acts = defaultdict(lambda: defaultdict(int))
for vid, obj_ans in tr_obj.items():
    if vid in tr_actions:
        for act in tr_actions[vid]:
            obj_to_acts[obj_ans][act] += 1

print("Object interaction -> Action co-occurrence in training:")
for obj, acts in sorted(obj_to_acts.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]:
    top_acts = sorted(acts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  '{obj}': {top_acts}")

print("\nAction -> Object interaction in training:")
act_to_objs = defaultdict(lambda: defaultdict(int))
for vid, obj_ans in tr_obj.items():
    if vid in tr_actions:
        for act in tr_actions[vid]:
            act_to_objs[act][obj_ans] += 1

for act, objs in sorted(act_to_objs.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]:
    top_objs = sorted(objs.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  '{act}': {top_objs}")

# Check test set: how many object_interaction questions remain?
te_obj = te[te['category'] == 'object_interaction']
print(f"\nTest object_interaction questions: {len(te_obj)}")

# Let's do the same for emotion
emo_to_acts = defaultdict(lambda: defaultdict(int))
for vid, emo_ans in tr_emo.items():
    if vid in tr_actions:
        for act in tr_actions[vid]:
            emo_to_acts[emo_ans][act] += 1

print("\nEmotion -> Action co-occurrence in training:")
for emo, acts in sorted(emo_to_acts.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]:
    top_acts = sorted(acts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  '{emo}': {top_acts}")
