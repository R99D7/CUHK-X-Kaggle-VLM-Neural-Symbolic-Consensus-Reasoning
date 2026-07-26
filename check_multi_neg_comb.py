"""
Check negative leaks from multi to combination.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')
raw = pd.read_csv('transformer_fixed_raw_predictions.csv')

# Build multi negative acts
vid_to_multi_neg = {}
for idx, row in te[te['category'] == 'multi'].iterrows():
    vid = row['path']
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    neg_acts = set()
    for l, txt in opts.items():
        if l not in pred:
            neg_acts.add(txt)
    vid_to_multi_neg[vid] = neg_acts

changes = 0
for idx, row in te[te['category'] == 'combination'].iterrows():
    vid = row['path']
    if vid not in vid_to_multi_neg: continue
    neg_acts = vid_to_multi_neg[vid]
    
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D'] if str(row[l]) != 'nan'}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    valid_opts = []
    for l, txt in opts.items():
        acts = set([a.strip() for a in txt.split(',')])
        # If any act in this combination is in neg_acts, it's invalid
        if not (acts & neg_acts):
            valid_opts.append(l)
            
    if pred not in valid_opts and len(valid_opts) > 0:
        # The predicted option is INVALID!
        print(f"COMB {row['qa_id']}: pred {pred} is INVALID because it contains an action ruled out by MULTI.")
        print(f"  Valid options: {valid_opts}")
        
        # Pick the valid option with highest raw prob
        r = raw[raw['qa_id'] == row['qa_id']].iloc[0]
        best_opt = max(valid_opts, key=lambda x: r[f'raw_prob_{x}'])
        print(f"  -> Change to {best_opt}")
        changes += 1

print(f"\nTotal COMB fixes from MULTI negative leak: {changes}")
