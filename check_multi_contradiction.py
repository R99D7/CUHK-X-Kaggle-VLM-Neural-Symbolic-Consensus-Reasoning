"""
Check if MULTI contradicts the 3 remaining COMB->SINGLE fixes.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

fixes = {
    'test_0074': 'C',
    'test_0075': 'D',
    'test_0555': 'C'
}

for qa_id, new_pred in fixes.items():
    single_q = te[te['qa_id'] == qa_id].iloc[0]
    vid = single_q['path']
    multi_q = te[(te['path'] == vid) & (te['category'] == 'multi')]
    if multi_q.empty: 
        print(f"{qa_id}: No multi question.")
        continue
    multi_q = multi_q.iloc[0]
    multi_pred = sub[sub['qa_id'] == multi_q['qa_id']]['prediction'].values[0]
    multi_opts = {l: str(multi_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    single_opts = {l: str(single_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    new_act = single_opts[new_pred]
    
    print(f"\n{qa_id} (vid {vid}): COMB wants to set {new_pred} ({new_act})")
    print(f"  MULTI options: {multi_opts}")
    print(f"  MULTI prediction: {multi_pred}")
    
    # Is the new_act an option in MULTI?
    for ml, mtxt in multi_opts.items():
        if mtxt == new_act:
            if ml not in multi_pred:
                print(f"  WARNING: MULTI had '{new_act}' as option {ml} but did NOT predict it!")
            else:
                print(f"  SAFE: MULTI predicted '{new_act}'!")
