"""
Properly extract sequence actions for BOTH HAU and HARn.
And check SEQ->MULTI leaks.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

vid_to_seq_acts = {}
for idx, row in te[te['category'] == 'sequence'].iterrows():
    vid = row['path']
    if 'large_model_track' in vid:  # HAU
        acts = set([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    else:  # HARn
        acts = set([a.strip().lower() for a in str(row['A']).split(',')])
    vid_to_seq_acts[vid] = acts

print("Proper SEQ->MULTI leaks:")
changes = 0
for idx, row in te[te['category'] == 'multi'].iterrows():
    vid = row['path']
    if vid not in vid_to_seq_acts: continue
    seq_acts = vid_to_seq_acts[vid]
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    in_seq = [l for l, txt in opts.items() if txt in seq_acts]
    missing = [l for l in in_seq if l not in pred]
    
    if missing:
        new_pred_letters = sorted(set(list(pred.replace('nan', '')) + missing))
        new_pred = ''.join(new_pred_letters)
        print(f"SEQ->MULTI {row['qa_id']} (vid {vid}): {pred} -> {new_pred} (added {missing}). acts={seq_acts}")
        changes += 1

print(f"Total SEQ->MULTI additions: {changes}")
