"""
Inspect all 43 cases where single action predicted is NOT in combination / multi actions.
We print out all available options for single, combination, and multi for each case,
so we can deterministically find every single valid fix!
"""
import pandas as pd

sub = pd.read_csv("submission_v270_TRUE_SUMMIT.csv")
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

grouped = te.groupby('path')

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    if 'single' in cats and ('combination' in cats or 'multi' in cats):
        s_row = cats['single']
        s_opts = {l: str(s_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        s_pred = str(s_row['pred']).strip()
        s_act = s_opts.get(s_pred, '')
        
        # Get verified actions from combination and multi
        verified = set()
        for c in ['combination', 'multi', 'sequence']:
            if c in cats:
                r = cats[c]
                p = str(r['pred']).strip()
                opts = {l: [x.strip().lower() for x in str(r[l]).split(',')] for l in ['A', 'B', 'C', 'D']}
                for char in p:
                    if char in opts:
                        for a in opts[char]:
                            verified.add(a)
                            
        # If the currently predicted single action is NOT in the verified action pool
        if s_act not in verified:
            # Check which options in single DO appear in the verified pool
            matching_opts = {l: act for l, act in s_opts.items() if act in verified}
            print(f"[SINGLE vs CONSENSUS] Vid: {vid_path} | QID: {s_row['qa_id']}")
            print(f"   Current Single: {s_pred} ({s_act}) -> Matching choices in consensus {verified}: {matching_opts}")
            # If there is exactly ONE choice in single that matches the consensus, or if combination needs to change
            if 'combination' in cats:
                c_row = cats['combination']
                c_opts = {l: str(c_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
                print(f"   Comb ({c_row['pred']}): {c_opts.get(c_row['pred'], '')} | All Comb: {c_opts}")
            print("-" * 60)
