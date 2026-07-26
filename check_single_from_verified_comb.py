"""
Check SINGLE fixes from verified COMBINATION.
"""
import pandas as pd

sub = pd.read_csv('submission.csv')
te = pd.read_csv('test_qa.csv')

verified_combs = [
    'test_0227', 'test_0236', 'test_0245', 'test_0318', 'test_0328', 'test_0329', 
    'test_0230', 'test_0280', 'test_0292', 'test_0293', 'test_0299', 'test_0313', 'test_0621'
]

vid_to_verified_comb = {}
for comb_id in verified_combs:
    comb_q = te[te['qa_id'] == comb_id].iloc[0]
    vid_to_verified_comb[comb_q['path']] = comb_id

changes = 0
for idx, single_q in te[te['category'] == 'single'].iterrows():
    vid = single_q['path']
    if vid not in vid_to_verified_comb: continue
    comb_id = vid_to_verified_comb[vid]
    
    comb_q = te[te['qa_id'] == comb_id].iloc[0]
    comb_pred = str(sub[sub['qa_id'] == comb_id]['prediction'].values[0])
    comb_opts = {l: str(comb_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    comb_acts = set([a.strip() for a in comb_opts.get(comb_pred, '').split(',')])
    
    single_pred = str(sub[sub['qa_id'] == single_q['qa_id']]['prediction'].values[0]).strip()
    single_opts = {l: str(single_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D'] if str(single_q[l]) != 'nan'}
    
    opts_in_comb = [l for l, txt in single_opts.items() if txt in comb_acts]
    
    if len(opts_in_comb) == 1 and single_pred != opts_in_comb[0]:
        print(f"SINGLE {single_q['qa_id']}: {single_pred} -> {opts_in_comb[0]} (predicted by verified COMB {comb_id})")
        
        # Check MULTI contradiction
        multi_q = te[(te['path'] == vid) & (te['category'] == 'multi')]
        if not multi_q.empty:
            multi_q = multi_q.iloc[0]
            multi_pred = str(sub[sub['qa_id'] == multi_q['qa_id']]['prediction'].values[0])
            multi_opts = {l: str(multi_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
            
            new_act = single_opts[opts_in_comb[0]]
            for ml, mtxt in multi_opts.items():
                if mtxt == new_act and ml not in multi_pred:
                    print(f"  WARNING: MULTI has {new_act} as option {ml} but did NOT predict it! Conflict!")
        
        changes += 1

print(f"\nTotal SINGLE fixes from verified COMB: {changes}")
