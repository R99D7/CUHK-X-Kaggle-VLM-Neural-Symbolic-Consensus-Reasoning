"""
Check if combination options are strictly exhaustive.
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

no_valid_comb = 0
for idx, row in te[te['category'] == 'combination'].iterrows():
    vid = row['path']
    if vid not in vid_to_seq_acts: continue
    seq_acts = vid_to_seq_acts[vid]
    
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D'] if str(row[l]) != 'nan'}
    
    valid_opts = []
    for l, txt in opts.items():
        acts = set([a.strip() for a in txt.split(',')])
        if seq_acts.issubset(acts):
            valid_opts.append(l)
            
    if len(valid_opts) == 0:
        print(f"vid {vid}: NO combination option contains sequence {seq_acts}")
        no_valid_comb += 1

print(f"\nTotal videos with NO valid comb options: {no_valid_comb}")
