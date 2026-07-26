"""
Check if there are any multi or combination or single options that ARE in the 
sequence actions but we failed to predict them!
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

# Build known actions from sequence
vid_to_seq_acts = {}
for idx, row in te[te['category'] == 'sequence'].iterrows():
    vid = row['path']
    acts = set([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    vid_to_seq_acts[vid] = acts

# Check SINGLE
print("Checking SINGLE against sequence:")
for idx, row in te[te['category'] == 'single'].iterrows():
    vid = row['path']
    if vid not in vid_to_seq_acts: continue
    seq_acts = vid_to_seq_acts[vid]
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    in_seq = [l for l, txt in opts.items() if txt in seq_acts]
    if len(in_seq) == 1 and pred != in_seq[0]:
        print(f"  SINGLE {row['qa_id']}: pred {pred}, should be {in_seq[0]} ({opts[in_seq[0]]})")

# Check MULTI
print("\nChecking MULTI against sequence:")
for idx, row in te[te['category'] == 'multi'].iterrows():
    vid = row['path']
    if vid not in vid_to_seq_acts: continue
    seq_acts = vid_to_seq_acts[vid]
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    in_seq = [l for l, txt in opts.items() if txt in seq_acts]
    
    missing = [l for l in in_seq if l not in pred]
    if missing:
        print(f"  MULTI {row['qa_id']}: pred {pred} is missing {missing} which are in seq acts: {[opts[l] for l in missing]}")

# Check COMBINATION
print("\nChecking COMBINATION against sequence:")
for idx, row in te[te['category'] == 'combination'].iterrows():
    vid = row['path']
    if vid not in vid_to_seq_acts: continue
    seq_acts = vid_to_seq_acts[vid]
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    valid_opts = []
    for l, txt in opts.items():
        acts = set([a.strip() for a in txt.split(',')])
        if acts.issubset(seq_acts):
            valid_opts.append(l)
    
    if len(valid_opts) == 1 and pred != valid_opts[0]:
        print(f"  COMBINATION {row['qa_id']}: pred {pred}, should be {valid_opts[0]} ({opts[valid_opts[0]]})")
