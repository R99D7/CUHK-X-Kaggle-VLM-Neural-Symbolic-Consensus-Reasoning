"""
Deep exploratory mining over the 0.74269 baseline (submission_v274_SUMMIT_ULTRA.csv):
1. Uncorroborated MULTI action letters with prob between 0.50 and 0.65 (especially those violating mutual exclusions!).
2. Re-auditing COMBINATION, SINGLE, and SEQUENCE against our newly cleaned 0.74269 consensus pool!
3. Comprehensive scan for ANY remaining mutual exclusion violations across all categories in test predictions!
"""
import pandas as pd
from collections import defaultdict, Counter

# 1. Build training co-occurrence and exclusion baseline
tr = pd.read_csv("training_qa.csv")
cooccur = defaultdict(Counter)
action_counts = Counter()

tr_grouped = tr.groupby('path')
for vid_path, grp in tr_grouped:
    clip_actions = set()
    for idx, r in grp.iterrows():
        ans_chars = str(r['answer']).strip()
        opts = {l: [a.strip().lower() for a in str(r[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
        for c in ans_chars:
            if c in opts:
                for act in opts[c]:
                    clip_actions.add(act)
                    action_counts[act] += 1
    for a1 in clip_actions:
        for a2 in clip_actions:
            if a1 != a2:
                cooccur[a1][a2] += 1

# 2. Analyze v274 baseline (0.74269)
sub = pd.read_csv("submission_v274_SUMMIT_ULTRA.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

grouped = te.groupby('path')
multi_cand_50_65 = []
comb_upgrades = []
single_upgrades = []
remaining_exclusions = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
    # Pool corroborated actions from single, sequence, combination
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

    # All clip actions across all action categories
    all_clip_actions = set(corroborated_acts)
    if 'multi' in cats:
        m_row = cats['multi']
        m_pred = str(m_row['pred']).strip()
        m_opts = {l: [x.strip().lower() for x in str(m_row[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
        for char in m_pred:
            if char in m_opts:
                for act in m_opts[char]:
                    all_clip_actions.add(act)

    # Check for remaining Mutual Exclusions
    clip_list = list(all_clip_actions)
    for i in range(len(clip_list)):
        for j in range(i + 1, len(clip_list)):
            a1, a2 = clip_list[i], clip_list[j]
            if action_counts[a1] >= 10 and action_counts[a2] >= 10 and cooccur[a1][a2] == 0:
                remaining_exclusions.append((vid_path, a1, a2, f"Train freq ({action_counts[a1]}, {action_counts[a2]}) with 0 co-occurrences in train!"))

    # A. Check MULTI for uncorroborated letters with prob in [0.50, 0.65)
    if 'multi' in cats:
        m_row = cats['multi']
        qid = m_row['qa_id']
        m_pred = str(m_row['pred']).strip()
        m_opts = {l: str(m_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        
        for l in m_pred:
            act_name = m_opts.get(l, '')
            prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
            if act_name and act_name not in corroborated_acts and 0.50 <= prob < 0.65:
                # Check if it violates mutual exclusion against ANY corroborated act
                violates = []
                for v_act in corroborated_acts:
                    if action_counts[act_name] >= 10 and action_counts[v_act] >= 10 and cooccur[act_name][v_act] == 0:
                        violates.append(v_act)
                multi_cand_50_65.append((qid, m_pred, l, act_name, round(prob, 3), list(corroborated_acts), violates))

    # B. Re-audit COMBINATION against clean multi+single+sequence pool
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
            if (ov > curr_ov and unv <= curr_unv) or (ov == curr_ov and unv < curr_unv and prob >= curr_prob - 0.05):
                comb_upgrades.append((qid, c_pred, l, round(curr_prob, 3), round(prob, 3), curr_acts, acts, list(ver_pool)))

    # C. Re-audit SINGLE against clean consensus pool
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

print(f"--- 1. UNCORROBORATED MULTI NOISE IN [0.50, 0.65) ({len(multi_cand_50_65)} found) ---")
for idx, m in enumerate(multi_cand_50_65[:25]):
    print(f"[MULTI 0.50-0.65] {m[0]}: Pred={m[1]} | Letter '{m[2]}' ('{m[3]}') prob={m[4]} | Exclusions Violations={m[6]} | Pool={m[5]}")

print(f"\n--- 2. SUPERIOR COMBINATION CANDIDATES OVER 0.74269 ({len(comb_upgrades)} found) ---")
for u in comb_upgrades:
    print(f"[COMB UPGRADE] {u[0]}: {u[1]} (prob={u[3]}) -> {u[2]} (prob={u[4]}) | Old={u[5]} -> New={u[6]} | Pool={u[7]}")

print(f"\n--- 3. SUPERIOR SINGLE CANDIDATES OVER 0.74269 ({len(single_upgrades)} found) ---")
for su in single_upgrades:
    print(f"[SINGLE UPGRADE] {su[0]}: {su[1]} ('{su[3]}', prob={su[5]}) -> {su[2]} ('{su[4]}', prob={su[6]}) | Pool={su[7]}")

print(f"\n--- 4. REMAINING MUTUAL EXCLUSIONS ACROSS ENTIRE TEST SET ({len(remaining_exclusions)} found) ---")
for f in remaining_exclusions[:15]:
    print(f"[REMAINING EXCLUSION] {f[0]}: '{f[1]}' vs '{f[2]}' | {f[3]}")
