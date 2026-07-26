"""
Check if any single option is in the sequence actions.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Build seq acts
vid_to_seq_acts = {}
for idx, row in te[te['category'] == 'sequence'].iterrows():
    vid = row['path']
    acts = set([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    vid_to_seq_acts[vid] = acts

changes = 0
for idx, row in te[te['category'] == 'single'].iterrows():
    vid = row['path']
    if vid not in vid_to_seq_acts: continue
    seq_acts = vid_to_seq_acts[vid]
    
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D'] if str(row[l]) != 'nan'}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    # Are any single options in sequence acts?
    opts_in_seq = [l for l, txt in opts.items() if txt in seq_acts]
    
    if len(opts_in_seq) == 1 and pred != opts_in_seq[0]:
        print(f"SINGLE {row['qa_id']}: pred {pred} ({opts.get(pred, 'nan')}) -> change to {opts_in_seq[0]} ({opts[opts_in_seq[0]]}) (predicted by SEQUENCE)")
        changes += 1
    elif len(opts_in_seq) > 1:
        print(f"SINGLE {row['qa_id']} has MULTIPLE options in sequence: {opts_in_seq}")

print(f"\nTotal SINGLE fixes from SEQUENCE: {changes}")
