"""
Check confidence of the 23 single predictions.
"""
import pandas as pd

raw = pd.read_csv('transformer_fixed_raw_predictions.csv')
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

changes = []
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
            multi_opt_letter = None
            for ml, mtxt in multi_opts.items():
                if single_act == mtxt:
                    multi_opt_letter = ml
                    break
                    
            if multi_opt_letter:
                raw_probs = raw[raw['qa_id'] == row['qa_id']]
                if not raw_probs.empty:
                    p = raw_probs.iloc[0][f'prob_{pred}']
                    print(f"SINGLE {row['qa_id']} pred={pred} ({single_act}) prob={p:.4f} -> MULTI {multi_q['qa_id']} add {multi_opt_letter}")
                    if p > 0.8:
                        changes.append((multi_q['qa_id'], multi_opt_letter))

print(f"\nHighly confident (>0.8) to add: {len(changes)}")
