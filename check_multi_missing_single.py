"""
Check if SINGLE predicts an action that MULTI missed.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

changes = 0
for idx, single_q in te[te['category'] == 'single'].iterrows():
    vid = single_q['path']
    multi_q = te[(te['path'] == vid) & (te['category'] == 'multi')]
    if multi_q.empty: continue
    multi_q = multi_q.iloc[0]
    
    single_pred = str(sub[sub['qa_id'] == single_q['qa_id']]['prediction'].values[0]).strip()
    single_opts = {l: str(single_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    single_act = single_opts.get(single_pred)
    
    multi_opts = {l: str(multi_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    multi_pred = str(sub[sub['qa_id'] == multi_q['qa_id']]['prediction'].values[0])
    
    # Is single_act in MULTI options?
    for ml, mtxt in multi_opts.items():
        if mtxt == single_act and ml not in multi_pred:
            print(f"MULTI {multi_q['qa_id']}: missing {ml} ({mtxt}) predicted by SINGLE {single_q['qa_id']}")
            changes += 1

print(f"\nTotal MULTI missing SINGLE predictions: {changes}")
