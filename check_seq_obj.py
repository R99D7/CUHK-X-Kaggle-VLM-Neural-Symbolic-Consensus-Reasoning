"""
Pull sequence questions and object_interaction for the 21 HAU test videos.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

te_obj = te[te['category'] == 'object_interaction']
vids = list(te_obj['path'])

te_for_vids = te[te['path'].isin(vids)]

for vid in vids:
    print(f"\n--- {vid} ---")
    vid_rows = te_for_vids[te_for_vids['path'] == vid]
    
    seq_actions = set()
    for idx, row in vid_rows.iterrows():
        cat = row['category']
        if cat == 'sequence':
            # Option A contains the 4 actions
            seq_a = str(row['A']).strip().lower()
            seq_actions = set([a.strip() for a in seq_a.split(',')])
            break
            
    for idx, row in vid_rows.iterrows():
        cat = row['category']
        if cat == 'object_interaction':
            opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
            pred_l = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
            print(f"object_interaction: {row['qa_id']}")
            print(f"  Sequence actions: {seq_actions}")
            print(f"  Options: {opts}")
            print(f"  Pred: {pred_l} ({opts.get(pred_l, '')})")
