"""
Check positive leaks from multi to combination.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')
raw = pd.read_csv('transformer_fixed_raw_predictions.csv')

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

changes = 0
for idx, row in te[te['category'] == 'combination'].iterrows():
    vid = row['path']
    if vid not in vid_to_multi_acts: continue
    multi_acts = vid_to_multi_acts[vid]
    if not multi_acts: continue
    
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D'] if str(row[l]) != 'nan'}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    # The correct combination option MUST contain all actions that were predicted by MULTI 
    # IF those actions are part of the union of all combination options.
    
    # Wait, some actions predicted by MULTI might not be in any combination option.
    # We only care about MULTI actions that appear in at least one COMB option.
    all_comb_acts = set()
    for txt in opts.values():
        all_comb_acts.update([a.strip() for a in txt.split(',')])
        
    relevant_multi_acts = multi_acts & all_comb_acts
    
    if not relevant_multi_acts: continue
    
    valid_opts = []
    for l, txt in opts.items():
        acts = set([a.strip() for a in txt.split(',')])
        # This option must contain ALL relevant multi actions
        if relevant_multi_acts.issubset(acts):
            valid_opts.append(l)
            
    if pred not in valid_opts and len(valid_opts) > 0:
        print(f"COMB {row['qa_id']} (vid {vid}): pred {pred} is INVALID because it lacks actions {relevant_multi_acts} predicted by MULTI.")
        print(f"  Valid options: {valid_opts}")
        
        # Pick the valid option with highest raw prob
        r = raw[raw['qa_id'] == row['qa_id']].iloc[0]
        best_opt = max(valid_opts, key=lambda x: r[f'raw_prob_{x}'])
        print(f"  -> Change to {best_opt}")
        changes += 1

print(f"\nTotal COMB fixes from MULTI positive leak: {changes}")
