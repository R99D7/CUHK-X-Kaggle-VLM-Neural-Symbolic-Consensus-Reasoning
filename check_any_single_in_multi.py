"""
Check if ANY single option is in the multi prediction.
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

changes = 0
for idx, row in te[te['category'] == 'single'].iterrows():
    vid = row['path']
    if vid not in vid_to_multi_acts: continue
    multi_acts = vid_to_multi_acts[vid]
    
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D'] if str(row[l]) != 'nan'}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    # Options in single that are predicted by multi
    opts_in_multi = [l for l, txt in opts.items() if txt in multi_acts]
    
    if len(opts_in_multi) == 1 and pred != opts_in_multi[0]:
        print(f"SINGLE {row['qa_id']}: pred {pred} ({opts.get(pred, 'nan')}) -> change to {opts_in_multi[0]} ({opts[opts_in_multi[0]]}) (only option in MULTI)")
        changes += 1

print(f"\nTotal SINGLE fixes from MULTI: {changes}")
