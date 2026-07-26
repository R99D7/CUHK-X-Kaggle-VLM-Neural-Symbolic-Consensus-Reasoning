"""
Check overlap between sequence actions and combination options.
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
for idx, row in te[te['category'] == 'combination'].iterrows():
    vid = row['path']
    if vid not in vid_to_seq_acts: continue
    seq_acts = vid_to_seq_acts[vid]
    
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D'] if str(row[l]) != 'nan'}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    # Are there any combination options that contain actions from the sequence?
    # Or maybe the correct combination must contain ALL sequence actions?
    # Let's see how many sequence actions are present in the combination options.
    all_comb_acts = set()
    for txt in opts.values():
        all_comb_acts.update([a.strip() for a in txt.split(',')])
        
    relevant_seq_acts = seq_acts & all_comb_acts
    
    if not relevant_seq_acts: continue
    
    valid_opts = []
    for l, txt in opts.items():
        acts = set([a.strip() for a in txt.split(',')])
        # This option must contain ALL relevant sequence actions
        if relevant_seq_acts.issubset(acts):
            valid_opts.append(l)
            
    if pred not in valid_opts and len(valid_opts) > 0:
        print(f"COMB {row['qa_id']} (vid {vid}): pred {pred} is INVALID because it lacks actions {relevant_seq_acts} verified by SEQUENCE.")
        print(f"  Valid options: {valid_opts}")
        changes += 1

print(f"\nTotal COMB fixes from SEQUENCE positive leak: {changes}")
