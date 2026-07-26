"""
Check if any single option is predicted in combination.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Build comb acts
vid_to_comb_acts = {}
for idx, row in te[te['category'] == 'combination'].iterrows():
    vid = row['path']
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    pred_acts = set()
    if len(pred) == 1 and pred in opts:
        txt = opts[pred]
        pred_acts = set([a.strip() for a in txt.split(',')])
        
    vid_to_comb_acts[vid] = pred_acts

changes = 0
for idx, row in te[te['category'] == 'single'].iterrows():
    vid = row['path']
    if vid not in vid_to_comb_acts: continue
    comb_acts = vid_to_comb_acts[vid]
    
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D'] if str(row[l]) != 'nan'}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    opts_in_comb = [l for l, txt in opts.items() if txt in comb_acts]
    
    if len(opts_in_comb) == 1 and pred != opts_in_comb[0]:
        print(f"SINGLE {row['qa_id']}: pred {pred} ({opts.get(pred, 'nan')}) -> change to {opts_in_comb[0]} ({opts[opts_in_comb[0]]}) (predicted by COMBINATION)")
        changes += 1

print(f"\nTotal SINGLE fixes from COMB prediction: {changes}")
