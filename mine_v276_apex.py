"""
Mine v276 APEX opportunities over the 0.77192 baseline (submission_v275_MASTER_SUMMIT.csv):
1. Re-evaluate all remaining uncorroborated MULTI action letters (regardless of prob, or up to 0.80).
2. Cross-check COMBINATION and SINGLE questions against our cleaned 0.77192 consensus pool.
3. Check for any logical contradictions in SEQUENCE ordering vs COMBINATION / SINGLE action counts.
"""
import pandas as pd
from collections import defaultdict, Counter

sub = pd.read_csv("submission_v275_MASTER_SUMMIT.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

# Training statistics for contextual awareness
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

grouped = te.groupby('path')
multi_uncorp = []
comb_upgrades = []
single_upgrades = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
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

    # Check remaining uncorroborated in MULTI
    if 'multi' in cats:
        m_row = cats['multi']
        qid = m_row['qa_id']
        m_pred = str(m_row['pred']).strip()
        m_opts = {l: str(m_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        
        for l in m_pred:
            act_name = m_opts.get(l, '')
            prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
            if act_name and act_name not in corroborated_acts:
                multi_uncorp.append((qid, m_pred, l, act_name, round(prob, 3), list(corroborated_acts)))

    # Audit COMBINATION against new 0.77192 clean pool
    if 'combination' in cats:
        c_row = cats['combination']
        qid = c_row['qa_id']
        c_pred = str(c_row['pred']).strip()
        c_opts = {l: set([x.strip().lower() for x in str(c_row[l]).split(',')]) for l in ['A', 'B', 'C', 'D']}
        
        ver_pool = set()
        for cat in ['single', 'multi', 'sequence']:
            if cat in cats:
                r = cats[cat]
                p = str(r['pred']).strip()
                opts = {l: [x.strip().lower() for x in str(r[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
                for char in p:
                    if char in opts:
                        for act in opts[char]:
                            ver_pool.add(act)
                            
        curr_acts = c_opts.get(c_pred, set())
        curr_ov = len(curr_acts & ver_pool)
        curr_unv = len(curr_acts - ver_pool)
        curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{c_pred}', 0.0)
        
        for l, acts in c_opts.items():
            if l == c_pred: continue
            prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
            ov = len(acts & ver_pool)
            unv = len(acts - ver_pool)
            if ov > curr_ov and unv <= curr_unv:
                comb_upgrades.append((qid, c_pred, l, round(curr_prob, 3), round(prob, 3), curr_acts, acts, list(ver_pool)))

    # Audit SINGLE against clean pool
    if 'single' in cats:
        s_row = cats['single']
        qid = s_row['qa_id']
        s_pred = str(s_row['pred']).strip()
        s_opts = {l: str(s_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        curr_act = s_opts.get(s_pred, '')
        curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{s_pred}', 0.0)
        
        ver_pool = set()
        for cat in ['multi', 'combination', 'sequence']:
            if cat in cats:
                r = cats[cat]
                p = str(r['pred']).strip()
                opts = {l: [x.strip().lower() for x in str(r[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
                for char in p:
                    if char in opts:
                        for act in opts[char]:
                            ver_pool.add(act)
                            
        if curr_act not in ver_pool and ver_pool:
            for l, act in s_opts.items():
                if act in ver_pool:
                    prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
                    single_upgrades.append((qid, s_pred, l, curr_act, act, round(curr_prob, 3), round(prob, 3), list(ver_pool)))

print(f"--- 1. REMAINING UNCORROBORATED MULTI INCLUSIONS ({len(multi_uncorp)} found) ---")
for m in sorted(multi_uncorp, key=lambda x: x[4]):
    print(f"[MULTI UNCORROBORATED] {m[0]}: Pred={m[1]} | Letter '{m[2]}' ('{m[3]}') prob={m[4]} | Corroborated={m[5]}")

print(f"\n--- 2. COMBINATION UPGRADES ({len(comb_upgrades)} found) ---")
for u in comb_upgrades:
    print(f"[COMB UPGRADE] {u[0]}: {u[1]} (prob={u[3]}) -> {u[2]} (prob={u[4]}) | Old={u[5]} -> New={u[6]} | Pool={u[7]}")

print(f"\n--- 3. SINGLE UPGRADES ({len(single_upgrades)} found) ---")
for su in single_upgrades:
    print(f"[SINGLE UPGRADE] {su[0]}: {su[1]} ('{su[3]}', prob={su[5]}) -> {su[2]} ('{su[4]}', prob={su[6]}) | Pool={su[7]}")
