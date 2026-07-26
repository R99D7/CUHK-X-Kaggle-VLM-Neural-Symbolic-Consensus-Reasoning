"""
Check if any of the 23 single predictions missing from multi are in sequence actions.
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

# Build seq acts
vid_to_seq_acts = {}
for idx, row in te[te['category'] == 'sequence'].iterrows():
    vid = row['path']
    acts = set([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    vid_to_seq_acts[vid] = acts

for idx, row in te[te['category'] == 'single'].iterrows():
    vid = row['path']
    if vid not in vid_to_multi_acts: continue
    multi_acts = vid_to_multi_acts[vid]
    
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    if len(pred) == 1 and pred in opts:
        single_act = opts[pred]
        if single_act not in multi_acts and single_act != 'nan':
            multi_q = te[(te['path'] == vid) & (te['category'] == 'multi')].iloc[0]
            multi_opts = {l: str(multi_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
            
            # Check if this single_act is one of the multi options
            is_in_multi_opts = False
            for ml, mtxt in multi_opts.items():
                if single_act == mtxt:
                    is_in_multi_opts = True
                    break
                    
            if is_in_multi_opts:
                seq_acts = vid_to_seq_acts.get(vid, set())
                in_seq = "YES" if single_act in seq_acts else "NO"
                print(f"SINGLE {row['qa_id']} pred ({single_act}) -> in SEQ? {in_seq}")
