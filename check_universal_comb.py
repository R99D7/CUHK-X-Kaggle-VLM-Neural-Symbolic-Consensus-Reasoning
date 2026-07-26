"""
Check if any actions are universal across all COMB options.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

changes = 0
for idx, row in te[te['category'] == 'combination'].iterrows():
    vid = row['path']
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D'] if str(row[l]) != 'nan'}
    
    # Get sets of actions for each option
    opt_acts = [set([a.strip() for a in txt.split(',')]) for txt in opts.values()]
    
    # Intersection of all options
    if not opt_acts: continue
    universal_acts = set.intersection(*opt_acts)
    
    if universal_acts:
        print(f"COMB {row['qa_id']} (vid {vid}): universal actions {universal_acts}!")
        # check if MULTI misses them
        multi_q = te[(te['path'] == vid) & (te['category'] == 'multi')]
        if not multi_q.empty:
            multi_q = multi_q.iloc[0]
            multi_opts = {l: str(multi_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
            multi_pred = str(sub[sub['qa_id'] == multi_q['qa_id']]['prediction'].values[0])
            
            missing = []
            for l, txt in multi_opts.items():
                if txt in universal_acts and l not in multi_pred:
                    missing.append(l)
            if missing:
                print(f"  -> MULTI {multi_q['qa_id']} is missing {missing}!")
                changes += 1

print(f"\nTotal universal MULTI fixes: {changes}")
