"""
Check if single predictions contain actions NOT in multi predictions.
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

# Check single
changes = 0
for idx, row in te[te['category'] == 'single'].iterrows():
    vid = row['path']
    if vid not in vid_to_multi_acts: continue
    multi_acts = vid_to_multi_acts[vid]
    
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    if len(pred) == 1 and pred in opts:
        single_act = opts[pred]
        if single_act not in multi_acts and single_act != 'nan':
            print(f"SINGLE {row['qa_id']} pred {pred} ({single_act}) NOT in MULTI for {vid}")
            # Which letter in MULTI is this single_act?
            multi_q = te[(te['path'] == vid) & (te['category'] == 'multi')].iloc[0]
            multi_opts = {l: str(multi_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
            for ml, mtxt in multi_opts.items():
                if single_act == mtxt:
                    print(f"  -> Should add {ml} to MULTI {multi_q['qa_id']}!")
                    changes += 1

print(f"Total SINGLE fixes from MULTI: {changes}")
