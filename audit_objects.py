"""
Audit object_interaction predictions against verified action consensus for the exact same video clip!
"""
import pandas as pd
from collections import defaultdict

sub = pd.read_csv("submission_v270_TRUE_SUMMIT.csv")
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

grouped = te.groupby('path')
object_cases = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    if 'object_interaction' not in cats:
        continue
        
    obj_row = cats['object_interaction']
    obj_opts = {l: str(obj_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    obj_pred = str(obj_row['pred']).strip()
    curr_obj = obj_opts.get(obj_pred, '')
    
    # Gather verified action words from single, multi, sequence, comb
    verified_acts = set()
    for c in ['single', 'multi', 'sequence', 'combination']:
        if c in cats:
            r = cats[c]
            p = str(r['pred']).strip()
            opts = {l: [x.strip().lower() for x in str(r[l]).split(',')] for l in ['A', 'B', 'C', 'D']}
            for letter in p:
                if letter in opts:
                    for a in opts[letter]:
                        verified_acts.add(a)
                        
    object_cases.append({
        'qid': obj_row['qa_id'],
        'vid': vid_path,
        'pred_letter': obj_pred,
        'pred_object': curr_obj,
        'all_opts': obj_opts,
        'verified_actions': list(verified_acts)
    })

df_objs = pd.DataFrame(object_cases)
print(f"Auditing all {len(df_objs)} object_interaction predictions against action context:")
for idx, r in df_objs.iterrows():
    print(f"[{r['qid']}] Pred Object: '{r['pred_object']}' | Actions: {r['verified_actions']} | Options: {r['all_opts']}")
