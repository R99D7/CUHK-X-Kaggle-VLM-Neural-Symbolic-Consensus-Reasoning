"""
Identify pure noise hallucinations in MULTI predictions where an included letter:
1. Has ZERO corroboration from single, sequence, or combination in the same video clip.
2. Has raw transformer probability < 0.35 (model itself considers it unlikely).
Also inspect if removing these noisy letters leaves at least one valid prediction, or if a verified letter should be added!
"""
import pandas as pd
from collections import Counter

sub = pd.read_csv("submission_v272_SUMMIT_PRO.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

grouped = te.groupby('path')
surgical_prunes = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    if 'multi' not in cats:
        continue
        
    m_row = cats['multi']
    qid = m_row['qa_id']
    m_pred = str(m_row['pred']).strip()
    m_opts = {l: str(m_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    corroborated_acts = set()
    for cat in ['single', 'sequence', 'combination']:
        if cat in cats:
            r = cats[cat]
            p = str(r['pred']).strip()
            opts = {l: [x.strip().lower() for x in str(r[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
            for char in p:
                if char in opts:
                    for act in opts[char]:
                        corroborated_acts.add(act)
                        
    noisy_letters = []
    retained_letters = []
    for l in m_pred:
        act_name = m_opts.get(l, '')
        prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
        if act_name not in corroborated_acts and prob < 0.35:
            noisy_letters.append((l, act_name, round(prob, 3)))
        else:
            retained_letters.append(l)
            
    # Also check if there are any corroborated letters NOT in m_pred
    missing_corroborated = []
    for l, act_name in m_opts.items():
        if act_name in corroborated_acts and l not in m_pred:
            missing_corroborated.append((l, act_name))
            
    if noisy_letters:
        new_pred = "".join(sorted(retained_letters + [x[0] for x in missing_corroborated]))
        # Ensure we never produce an empty answer
        if len(new_pred) == 0 and not missing_corroborated:
            # Let's see what options have highest prob or corroboration
            continue
        surgical_prunes.append({
            'qid': qid, 'old_pred': m_pred, 'new_pred': new_pred,
            'pruned': noisy_letters,
            'added': missing_corroborated,
            'corroborated': list(corroborated_acts)
        })

df_prune = pd.DataFrame(surgical_prunes)
print(f"Total multi questions with < 0.35 uncorroborated noise to prune: {len(df_prune)}")
for idx, r in df_prune.iterrows():
    print(f"[SURGICAL PRUNE] {r['qid']}: {r['old_pred']} -> {r['new_pred']} | Pruned: {r['pruned']} | Added: {r['added']} | Pool: {r['corroborated']}")
