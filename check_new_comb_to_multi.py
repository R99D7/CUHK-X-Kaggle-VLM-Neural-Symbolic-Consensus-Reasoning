"""
Check if there are any new COMB->MULTI fixes after recent COMB updates.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

changes = 0
for idx, row in te[te['category'] == 'multi'].iterrows():
    vid = row['path']
    comb_q = te[(te['path'] == vid) & (te['category'] == 'combination')]
    if comb_q.empty: continue
    comb_q = comb_q.iloc[0]
    
    comb_pred = str(sub[sub['qa_id'] == comb_q['qa_id']]['prediction'].values[0])
    if len(comb_pred) != 1: continue
    
    comb_opts = {l: str(comb_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    comb_acts = set([a.strip() for a in comb_opts.get(comb_pred, '').split(',')])
    
    multi_pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0])
    multi_opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    missing = []
    for l, txt in multi_opts.items():
        if txt in comb_acts and l not in multi_pred:
            missing.append(l)
            
    if missing:
        print(f"MULTI {row['qa_id']}: {multi_pred} is missing {missing} from NEW COMB {comb_q['qa_id']} ({comb_pred})")
        changes += 1

print(f"\nTotal new MULTI fixes from COMB: {changes}")
