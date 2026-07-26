"""
Audit MULTI questions in submission_v272_SUMMIT_PRO.csv for uncorroborated action letters!
If a letter in a MULTI prediction has ZERO votes from single, sequence, or combination for the same video,
and has low neural network confidence or conflicts with physical reality, flag it immediately!
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
multi_flags = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    if 'multi' not in cats:
        continue
        
    m_row = cats['multi']
    qid = m_row['qa_id']
    m_pred = str(m_row['pred']).strip()
    m_opts = {l: str(m_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    # Collect all actions confirmed by sequence, combination, or single in this exact video clip
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
                        
    # Now inspect every letter currently included in our MULTI answer
    unsupported_letters = []
    for l in m_pred:
        act_name = m_opts.get(l, '')
        if act_name and act_name not in corroborated_acts:
            prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
            unsupported_letters.append((l, act_name, round(prob, 3)))
            
    if unsupported_letters:
        multi_flags.append({
            'qid': qid, 'vid': vid_path, 'current_pred': m_pred,
            'unsupported': unsupported_letters,
            'corroborated_pool': list(corroborated_acts)
        })

df_flags = pd.DataFrame(multi_flags)
print(f"Total multi questions containing uncorroborated letters: {len(df_flags)}")
for idx, r in df_flags.head(30).iterrows():
    print(f"[MULTI AUDIT] {r['qid']}: Pred={r['current_pred']} | Unsupported Letters={r['unsupported']} | Corroborated={r['corroborated_pool']}")
