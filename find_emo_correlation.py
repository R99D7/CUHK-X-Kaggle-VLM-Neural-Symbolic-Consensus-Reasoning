"""
Find correlation between actions and emotions in training set.
"""
import pandas as pd
from collections import defaultdict

tr = pd.read_csv('training_qa.csv')
vid_to_emo = {}
vid_to_acts = defaultdict(set)

# First get all emotions
for idx, row in tr[tr['category'] == 'emotion'].iterrows():
    vid = row['path']
    ans = str(row['answer']).strip()
    emo_txt = str(row[ans]).strip().lower()
    vid_to_emo[vid] = emo_txt

# Get all actions for a video (from single, multi, combination, sequence)
for idx, row in tr[tr['category'].isin(['single', 'multi', 'combination', 'sequence'])].iterrows():
    vid = row['path']
    ans = str(row['answer']).strip()
    
    if row['category'] in ['single', 'combination']:
        opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        txt = opts.get(ans, "")
        acts = [a.strip() for a in txt.split(',')]
        vid_to_acts[vid].update(acts)
    elif row['category'] in ['multi', 'sequence']:
        opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        for l in ans:
            txt = opts.get(l, "")
            vid_to_acts[vid].add(txt)

# Now count co-occurrences
act_to_emo = defaultdict(lambda: defaultdict(int))
for vid, emo_txt in vid_to_emo.items():
    for act in vid_to_acts[vid]:
        if act:
            act_to_emo[act][emo_txt] += 1

# Print strongest correlations
print("Strongest Action -> Emotion correlations:")
for act, emo_counts in act_to_emo.items():
    total = sum(emo_counts.values())
    for emo, count in emo_counts.items():
        if count >= 15 and count / total > 0.6:
            print(f"  {act} -> {emo} (count: {count}/{total} = {count/total:.2f})")
