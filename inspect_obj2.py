"""
Inspect the 21 object_interaction questions in the test set.
Check what actions are predicted for these videos, INCLUDING sequence questions!
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

te_obj = te[te['category'] == 'object_interaction']

# Build predicted acts per video
pred_acts = {}
for idx, row in te.iterrows():
    vid = row['path']
    cat = row['category']
    if cat == 'object_interaction': continue
    
    if vid not in pred_acts: pred_acts[vid] = set()
    
    if cat == 'sequence':
        # Sequence answers give us the 4 actions in the video
        for l in ['A', 'B', 'C', 'D']:
            pred_acts[vid].add(str(row[l]).strip().lower())
    
    pred_l = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    if cat == 'single':
        if len(pred_l) == 1:
            pred_acts[vid].add(str(row[pred_l]).strip().lower())
    elif cat == 'combination':
        if len(pred_l) == 1:
            pred_text = str(row[pred_l]).strip().lower()
            for a in pred_text.split(','):
                pred_acts[vid].add(a.strip())
    elif cat == 'multi':
        for l in pred_l:
            if l in ['A', 'B', 'C', 'D']:
                pred_acts[vid].add(str(row[l]).strip().lower())

print("Test object_interaction questions and their predicted actions on the same video:")
for idx, row in te_obj.iterrows():
    vid = row['path']
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred_l = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    print(f"\n{row['qa_id']} (vid: {vid}):")
    print(f"  Options: {opts}")
    print(f"  Current Pred: {pred_l} ({opts.get(pred_l, '')})")
    print(f"  Actions in video: {pred_acts.get(vid, set())}")

