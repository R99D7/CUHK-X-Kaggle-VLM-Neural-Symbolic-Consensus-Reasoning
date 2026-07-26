"""
Check if we can fix single predictions using multi predictions.
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
for idx, row in te[te['category'] == 'single'].iterrows():
    vid = row['path']
    if vid not in vid_to_multi_acts: continue
    multi_acts = vid_to_multi_acts[vid]
    
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    
    if len(pred) == 1 and pred in opts:
        single_act = opts[pred]
        if single_act not in multi_acts and single_act != 'nan':
            
            # This single prediction contradicts multi.
            # Are there any single options that ARE in multi?
            valid_opts = [l for l, txt in opts.items() if txt in multi_acts]
            
            if len(valid_opts) == 1:
                print(f"SINGLE {row['qa_id']}: pred {pred} ({single_act}) -> change to {valid_opts[0]} ({opts[valid_opts[0]]}) (only valid option in MULTI)")
                changes += 1
            elif len(valid_opts) > 1:
                # pick the one with highest raw probability
                raw_probs = raw[raw['qa_id'] == row['qa_id']].iloc[0]
                best_opt = max(valid_opts, key=lambda x: raw_probs[f'raw_prob_{x}'])
                print(f"SINGLE {row['qa_id']}: pred {pred} ({single_act}) -> change to {best_opt} ({opts[best_opt]}) (best among {valid_opts} in MULTI)")
                changes += 1
            else:
                print(f"SINGLE {row['qa_id']}: pred {pred} ({single_act}) -> NO options are in MULTI!")

print(f"\nTotal possible SINGLE fixes: {changes}")
