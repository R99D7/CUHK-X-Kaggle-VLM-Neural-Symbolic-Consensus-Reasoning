"""
Apply newly verified COMB actions to MULTI.
"""
import pandas as pd

sub = pd.read_csv('submission.csv')
te = pd.read_csv('test_qa.csv')

# The QA IDs of the COMBs that were deterministically fixed:
verified_combs = [
    'test_0227', 'test_0236', 'test_0245', 'test_0318', 'test_0328', 'test_0329', # from SEQUENCE
    'test_0230', 'test_0280', 'test_0292', 'test_0293', 'test_0299', 'test_0313', 'test_0621' # from MULTI pos
]

changes = 0
for comb_id in verified_combs:
    comb_q = te[te['qa_id'] == comb_id].iloc[0]
    vid = comb_q['path']
    multi_q = te[(te['path'] == vid) & (te['category'] == 'multi')]
    if multi_q.empty: continue
    multi_q = multi_q.iloc[0]
    
    comb_pred = str(sub[sub['qa_id'] == comb_id]['prediction'].values[0])
    comb_opts = {l: str(comb_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    comb_acts = set([a.strip() for a in comb_opts.get(comb_pred, '').split(',')])
    
    multi_pred = str(sub[sub['qa_id'] == multi_q['qa_id']]['prediction'].values[0])
    multi_opts = {l: str(multi_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    missing = []
    for l, txt in multi_opts.items():
        if txt in comb_acts and l not in multi_pred:
            missing.append(l)
            
    if missing:
        new_pred = "".join(sorted(set(multi_pred) | set(missing)))
        print(f"MULTI {multi_q['qa_id']}: {multi_pred} -> {new_pred} (added {missing} from verified COMB {comb_id})")
        sub.loc[sub['qa_id'] == multi_q['qa_id'], 'prediction'] = new_pred
        changes += 1

sub.to_csv('submission_v267_NEWCOMB2MULTI.csv', index=False)
sub.to_csv('submission.csv', index=False)
print(f"Applied {changes} changes. Saved to submission.csv")
