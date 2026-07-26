"""
Check if combination predictions contain actions NOT in multi predictions.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Build multi acts
vid_to_multi_acts = {}
for idx, row in te[te['category'] == 'multi'].iterrows():
    vid = row['path']
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    pred_acts = set()
    for l in pred:
        if l in opts:
            pred_acts.add(opts[l])
    vid_to_multi_acts[vid] = pred_acts

# Check combination
changes = 0
for idx, row in te[te['category'] == 'combination'].iterrows():
    vid = row['path']
    if vid not in vid_to_multi_acts: continue
    multi_acts = vid_to_multi_acts[vid]
    
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    valid_opts = []
    for l, txt in opts.items():
        acts = set([a.strip() for a in txt.split(',')])
        # If the combination acts are a subset of the predicted multi acts
        if acts.issubset(multi_acts):
            valid_opts.append(l)
            
    # if the current prediction is NOT valid, and there is exactly 1 valid option, switch!
    if len(pred) == 1 and pred not in valid_opts and len(valid_opts) == 1:
        print(f"COMBINATION {row['qa_id']}: pred {pred} has acts {opts[pred]} NOT in multi {multi_acts}. Should be {valid_opts[0]} ({opts[valid_opts[0]]})")
        changes += 1

print(f"Total COMBINATION fixes from MULTI: {changes}")
