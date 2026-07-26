"""
Properly extract actions from HAU test videos and check object_interaction.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

te_obj = te[te['category'] == 'object_interaction']

# Extract actions from sequence questions for the same videos
vid_to_acts = {}
for idx, row in te[te['category'] == 'sequence'].iterrows():
    vid = row['path']
    # All options in a sequence question have the same actions, just permuted
    # Just take option A and split it
    seq_a = str(row['A']).strip().lower()
    acts = set([a.strip() for a in seq_a.split(',')])
    vid_to_acts[vid] = acts

print("Test object_interaction questions with properly extracted sequence actions:")
for idx, row in te_obj.iterrows():
    vid = row['path']
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    acts = vid_to_acts.get(vid, set())
    print(f"\n{row['qa_id']} (vid: {vid}):")
    print(f"  Actions: {acts}")
    print(f"  Options: {opts}")
    print(f"  Pred: {pred} ({opts.get(pred, '')})")
