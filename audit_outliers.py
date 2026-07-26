"""
Find clear outlier errors in submission_v270_TRUE_SUMMIT.csv where one category (e.g. single or comb)
contradicts the consensus of all other categories for the same video, and a perfectly matching option is available!
"""
import pandas as pd
from collections import Counter

sub = pd.read_csv("submission_v270_TRUE_SUMMIT.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv") # To check confidence or tiebreak
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

grouped = te.groupby('path')
fixes = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
    # Get all actions voted by multi, combination, and sequence
    consensus_actions = Counter()
    for cat in ['multi', 'combination', 'sequence']:
        if cat in cats:
            row = cats[cat]
            pred = str(row['pred']).strip()
            opts = {l: [x.strip().lower() for x in str(row[l]).split(',')] for l in ['A', 'B', 'C', 'D']}
            for l in pred:
                if l in opts:
                    for act in opts[l]:
                        consensus_actions[act] += 1
                        
    # Now evaluate SINGLE question against the consensus
    if 'single' in cats and consensus_actions:
        s_row = cats['single']
        s_opts = {l: str(s_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        s_pred = str(s_row['pred']).strip()
        curr_act = s_opts.get(s_pred, '')
        
        # If the currently predicted single action is NOT in the consensus (0 votes from other categories)
        if consensus_actions[curr_act] == 0:
            # Check if any OTHER option in single has strong support in consensus
            candidates = [(l, act, consensus_actions[act]) for l, act in s_opts.items() if consensus_actions[act] >= 2]
            if candidates:
                best_cand = max(candidates, key=lambda x: x[2])
                fixes.append({
                    'vid': vid_path,
                    'cat': 'single',
                    'qa_id': s_row['qa_id'],
                    'old_pred': s_pred,
                    'new_pred': best_cand[0],
                    'reason': f"Old single '{curr_act}' had 0 consensus votes; switching to '{best_cand[1]}' which has {best_cand[2]} consensus votes from multi/comb/seq."
                })

    # Evaluate COMBINATION question against consensus of (single + multi + sequence)
    if 'combination' in cats:
        c_row = cats['combination']
        c_opts = {l: set([x.strip().lower() for x in str(c_row[l]).split(',')]) for l in ['A', 'B', 'C', 'D']}
        c_pred = str(c_row['pred']).strip()
        
        # Build consensus from single, multi, seq
        non_comb_actions = set()
        for cat in ['single', 'multi', 'sequence']:
            if cat in cats:
                row = cats[cat]
                pred = str(row['pred']).strip()
                opts = {l: [x.strip().lower() for x in str(row[l]).split(',')] for l in ['A', 'B', 'C', 'D']}
                for l in pred:
                    if l in opts:
                        for act in opts[l]:
                            non_comb_actions.add(act)
                            
        # Check current overlap vs available options
        curr_overlap = len(c_opts.get(c_pred, set()) & non_comb_actions)
        best_opt = c_pred
        best_overlap = curr_overlap
        for l, acts in c_opts.items():
            if len(acts & non_comb_actions) > best_overlap:
                best_overlap = len(acts & non_comb_actions)
                best_opt = l
        if best_opt != c_pred and (best_overlap - curr_overlap >= 2): # At least 2 more verified actions
            fixes.append({
                'vid': vid_path,
                'cat': 'combination',
                'qa_id': c_row['qa_id'],
                'old_pred': c_pred,
                'new_pred': best_opt,
                'reason': f"Comb opt {best_opt} overlaps with {best_overlap} verified actions ({c_opts[best_opt] & non_comb_actions}), vs current {c_pred} ({c_opts.get(c_pred, set()) & non_comb_actions})"
            })

df_fixes = pd.DataFrame(fixes)
print(f"Found {len(df_fixes)} high-confidence outlier corrections!")
if len(df_fixes) > 0:
    for idx, r in df_fixes.head(30).iterrows():
        print(f"[{r['cat'].upper()}] {r['qa_id']}: {r['old_pred']} -> {r['new_pred']} | {r['reason']}")
