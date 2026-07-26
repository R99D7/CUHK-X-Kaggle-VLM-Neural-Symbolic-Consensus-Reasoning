"""
Inspect ALL remaining uncorroborated action letters in MULTI questions across the entire test set,
sorted by probability, along with any mutual exclusion violations against the confirmed pool!
"""
import pandas as pd
from collections import defaultdict, Counter

tr = pd.read_csv("training_qa.csv")
cooccur = defaultdict(Counter)
action_counts = Counter()
for vid_path, grp in tr.groupby('path'):
    acts = set()
    for idx, r in grp.iterrows():
        opts = {l: [a.strip().lower() for a in str(r[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
        for c in str(r['answer']).strip():
            if c in opts:
                for a in opts[c]:
                    acts.add(a)
                    action_counts[a] += 1
    for a1 in acts:
        for a2 in acts:
            if a1 != a2: cooccur[a1][a2] += 1

sub = pd.read_csv("submission_v274_SUMMIT_ULTRA.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

grouped = te.groupby('path')
all_uncorroborated = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    if 'multi' not in cats: continue
        
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
                    for a in opts[char]:
                        corroborated_acts.add(a)
                        
    for l in m_pred:
        act_name = m_opts.get(l, '')
        prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
        if act_name and act_name not in corroborated_acts:
            violates = []
            for v_act in corroborated_acts:
                if action_counts[act_name] >= 10 and action_counts[v_act] >= 10 and cooccur[act_name][v_act] == 0:
                    violates.append(v_act)
            all_uncorroborated.append((qid, m_pred, l, act_name, round(prob, 3), violates, list(corroborated_acts)))

df_un = pd.DataFrame(all_uncorroborated, columns=['qid', 'm_pred', 'letter', 'act_name', 'prob', 'violates', 'pool'])
df_un = df_un.sort_values(by='prob', ascending=True)
print(f"Total remaining uncorroborated multi letters across entire test set: {len(df_un)}")
for idx, r in df_un.iterrows():
    print(f"{r['qid']} (prob={r['prob']}): letter '{r['letter']}' ('{r['act_name']}') | Violations={r['violates']} | Corroborated={r['pool']}")
