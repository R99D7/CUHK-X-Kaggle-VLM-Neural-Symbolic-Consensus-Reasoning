"""
Deep Exploratory Mining over the 0.71929 baseline (submission_v273_FINAL_SUMMIT.csv):
1. Uncorroborated MULTI action letters with prob between 0.35 and 0.50.
2. Mutual Exclusion Laws (action pairs appearing together in test predictions that never appear together in training data).
3. Re-evaluating COMBINATION and SINGLE questions against our newly cleaned 0.71929 multi baseline!
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

# 2. Analyze v273 baseline
sub = pd.read_csv("submission_v273_FINAL_SUMMIT.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

grouped = te.groupby('path')
multi_cand_35_50 = []
comb_upgrades = []
single_upgrades = []
mutual_exclusion_flags = []

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

    # All clip actions across all categories
    all_clip_actions = set(corroborated_acts)
    if 'multi' in cats:
        m_row = cats['multi']
        m_pred = str(m_row['pred']).strip()
        m_opts = {l: [x.strip().lower() for x in str(m_row[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
        for char in m_pred:
            if char in m_opts:
                for act in m_opts[char]:
                    all_clip_actions.add(act)

    # Check for Mutual Exclusions (both actions appear in our test clip, but both appear >= 15 times in training and never together!)
    clip_list = list(all_clip_actions)
    for i in range(len(clip_list)):
        for j in range(i + 1, len(clip_list)):
            a1, a2 = clip_list[i], clip_list[j]
            if action_counts[a1] >= 15 and action_counts[a2] >= 15 and cooccur[a1][a2] == 0:
                mutual_exclusion_flags.append((vid_path, a1, a2, f"Both >15 freq in train ({action_counts[a1]}, {action_counts[a2]}) but 0 co-occurrences!"))

    # A. Check MULTI for uncorroborated letters with prob in [0.35, 0.50)
    if 'multi' in cats:
        m_row = cats['multi']
        qid = m_row['qa_id']
        m_pred = str(m_row['pred']).strip()
        m_opts = {l: str(m_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        
        for l in m_pred:
            act_name = m_opts.get(l, '')
            prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
            if act_name and act_name not in corroborated_acts and 0.35 <= prob < 0.50:
                multi_cand_35_50.append((qid, m_pred, l, act_name, round(prob, 3), list(corroborated_acts)))

    # B. Re-evaluate COMBINATION against clean multi baseline
    if 'combination' in cats:
        c_row = cats['combination']
        qid = c_row['qa_id']
        c_pred = str(c_row['pred']).strip()
        c_opts = {l: set([x.strip().lower() for x in str(c_row[l]).split(',')]) for l in ['A', 'B', 'C', 'D']}
        
        # Build verified actions from multi, sequence, single
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
            # Strict superiority: higher overlap AND equal/lower unverified, or higher prob + higher overlap
            if ov > curr_ov and unv <= curr_unv:
                comb_upgrades.append((qid, c_pred, l, round(curr_prob, 3), round(prob, 3), curr_acts, acts, list(ver_pool)))

print(f"--- 1. UNCORROBORATED MULTI NOISE IN [0.35, 0.50) ({len(multi_cand_35_50)} found) ---")
for idx, m in enumerate(multi_cand_35_50[:20]):
    print(f"[MULTI 0.35-0.50] {m[0]}: Pred={m[1]} | Letter '{m[2]}' ('{m[3]}') has prob={m[4]} | Corroborated={m[5]}")

print(f"\n--- 2. SUPERIOR COMBINATION CANDIDATES OVER 0.71929 ({len(comb_upgrades)} found) ---")
for u in comb_upgrades:
    print(f"[COMB UPGRADE] {u[0]}: {u[1]} (prob={u[3]}) -> {u[2]} (prob={u[4]}) | Old={u[5]} -> New={u[6]} | Verified={u[7]}")

print(f"\n--- 3. MUTUAL EXCLUSION FLAGS ({len(mutual_exclusion_flags)} found) ---")
for f in mutual_exclusion_flags[:15]:
    print(f"[MUTUAL EXCLUSION] {f[0]}: '{f[1]}' vs '{f[2]}' | {f[3]}")
