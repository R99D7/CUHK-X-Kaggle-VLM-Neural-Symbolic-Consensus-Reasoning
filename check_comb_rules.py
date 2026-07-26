"""
Check if correct COMB option is a maximal subset.
"""
import pandas as pd
from collections import defaultdict

tr = pd.read_csv('training_qa.csv')
vid_to_all_acts = defaultdict(set)

# Collect all true actions for each video
for idx, row in tr[tr['category'].isin(['single', 'multi', 'combination', 'sequence'])].iterrows():
    vid = row['path']
    ans = str(row['answer']).strip()
    
    if row['category'] == 'single':
        opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        txt = opts.get(ans, "")
        acts = [a.strip() for a in txt.split(',')]
        vid_to_all_acts[vid].update(acts)
    elif row['category'] in ['multi', 'sequence']:
        opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        for l in ans:
            txt = opts.get(l, "")
            if txt: vid_to_all_acts[vid].add(txt)

# Now check COMBINATION questions
violations = 0
total_comb = 0
for idx, row in tr[tr['category'] == 'combination'].iterrows():
    total_comb += 1
    vid = row['path']
    ans = str(row['answer']).strip()
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    # get acts in correct option
    correct_acts = set([a.strip() for a in opts.get(ans, "").split(',')])
    
    # Check if there are other options that contain MORE true actions
    # An option is "valid" if ALL its actions are true actions.
    valid_opts = []
    for l, txt in opts.items():
        acts = set([a.strip() for a in txt.split(',')])
        if acts.issubset(vid_to_all_acts[vid]):
            valid_opts.append(l)
            
    # if the correct option doesn't have the maximum number of true actions among valid options
    max_len = max([len(opts[l].split(',')) for l in valid_opts]) if valid_opts else 0
    if len(correct_acts) < max_len:
        print(f"COMB {idx}: Correct is {ans} (len {len(correct_acts)}). But there is a valid option with len {max_len}")
        violations += 1

print(f"\nTotal COMB violations: {violations} out of {total_comb}")
