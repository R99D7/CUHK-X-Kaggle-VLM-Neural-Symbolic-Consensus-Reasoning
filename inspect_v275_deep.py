"""
Inspect the full options and corroborated pool for test_0199, test_0122, and examine
the exact impact of pruning all uncorroborated multi letters that either:
1. Directly violate Mutual Exclusion laws against verified scene actions, OR
2. Have raw model probability < 0.68.
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

# 1. Inspect test_0199 and test_0122 in deep detail
for check_qid in ['test_0199', 'test_0122']:
    row = te[te['qa_id'] == check_qid].iloc[0]
    vid = row['path']
    print(f"\n=================== DEEP INSPECTION: {check_qid} ({vid}) ===================")
    grp = te[te['path'] == vid]
    for idx, r in grp.iterrows():
        opts = {l: str(r[l]).strip() for l in ['A', 'B', 'C', 'D']}
        probs = {l: round(raw_map.get(r['qa_id'], {}).get(f'raw_prob_{l}', 0.0), 3) for l in ['A', 'B', 'C', 'D']}
        print(f"[{r['category'].upper()}] {r['qa_id']} -> Pred: '{r['pred']}' | Options: {opts} | Probs: {probs}")

# 2. Evaluate systemic pruning of uncorroborated items that violate exclusions OR prob < 0.68
print("\n=================== SYSTEMIC PRUNE CANDIDATES (Violations OR Prob < 0.68) ===================")
grouped = te.groupby('path')
candidates = []

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
                        
    noisy = []
    retained = []
    for l in m_pred:
        act_name = m_opts.get(l, '')
        prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
        violates = []
        if act_name not in corroborated_acts:
            for v_act in corroborated_acts:
                if action_counts[act_name] >= 10 and action_counts[v_act] >= 10 and cooccur[act_name][v_act] == 0:
                    violates.append(v_act)
            if len(violates) > 0 or prob < 0.68:
                noisy.append((l, act_name, round(prob, 3), violates))
            else:
                retained.append(l)
        else:
            retained.append(l)
            
    # Also check if any option IN m_opts IS in corroborated_acts but was NOT in m_pred!
    missing = []
    for l, act_name in m_opts.items():
        if act_name in corroborated_acts and l not in m_pred:
            missing.append((l, act_name, round(raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0), 3)))
            
    if noisy or missing:
        new_pred = "".join(sorted([x for x in retained] + [x[0] for x in missing]))
        if len(new_pred) == 0:
            # Check what available options match corroborated acts or have highest prob
            best_l = max(m_opts.keys(), key=lambda x: raw_map.get(qid, {}).get(f'raw_prob_{x}', 0.0))
            new_pred = best_l
            print(f"[RESOLVED EMPTY] {qid}: Old={m_pred}, setting to best available={new_pred} ('{m_opts[new_pred]}')")
        if new_pred != m_pred:
            candidates.append((qid, m_pred, new_pred, noisy, missing, list(corroborated_acts)))

print(f"Total upgrades found: {len(candidates)}")
for c in candidates:
    print(f"[v275 UPGRADE] {c[0]}: {c[1]} -> {c[2]} | Pruned={c[3]} | Added={c[4]} | Pool={c[5]}")
