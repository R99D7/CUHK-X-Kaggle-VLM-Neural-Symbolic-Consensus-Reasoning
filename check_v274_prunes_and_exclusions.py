"""
Verify that pruning uncorroborated multi action letters with probability < 0.50:
1. Never leaves any prediction empty.
2. Resolves mutual exclusion contradictions across the test set!
3. Checks if any single/comb questions themselves contain mutual exclusion contradictions against the confirmed pool!
"""
import pandas as pd
from collections import defaultdict, Counter

sub = pd.read_csv("submission_v273_FINAL_SUMMIT.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

# Load training mutual exclusions
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
prunes_050 = []
empty_warnings = 0
resolved_exclusions = 0

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
                        
    noisy_letters = []
    retained_letters = []
    for l in m_pred:
        act_name = m_opts.get(l, '')
        prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
        # Prune if uncorroborated AND probability < 0.50
        if act_name not in corroborated_acts and prob < 0.50:
            noisy_letters.append((l, act_name, round(prob, 3)))
        else:
            retained_letters.append(l)
            
    if noisy_letters:
        new_pred = "".join(sorted(retained_letters))
        if len(new_pred) == 0:
            empty_warnings += 1
            # If removing all leaves empty, keep the highest probability letter
            best_l = max(m_pred, key=lambda x: raw_map.get(qid, {}).get(f'raw_prob_{x}', 0.0))
            new_pred = best_l
            print(f"[WARNING EMPTY] {qid} would be empty! Keeping highest prob '{best_l}'")
            
        # Check how many mutual exclusions are destroyed by removing these noisy letters!
        for l, act_name, prob in noisy_letters:
            for ver_act in corroborated_acts:
                if action_counts[act_name] >= 15 and action_counts[ver_act] >= 15 and cooccur[act_name][ver_act] == 0:
                    resolved_exclusions += 1
                    
        prunes_050.append({
            'qid': qid, 'old_pred': m_pred, 'new_pred': new_pred,
            'pruned': noisy_letters, 'retained': [m_opts[x] for x in new_pred],
            'pool': list(corroborated_acts)
        })

df_050 = pd.DataFrame(prunes_050)
print(f"Total < 0.50 uncorroborated surgical prunes: {len(df_050)}")
print(f"Empty answer warnings: {empty_warnings}")
print(f"Mutual exclusions directly resolved by this pruning: {resolved_exclusions}")

for idx, r in df_050.iterrows():
    print(f"[v274 SURGICAL PRUNE] {r['qid']}: {r['old_pred']} -> {r['new_pred']} | Pruned: {r['pruned']} | Retained/Corroborated: {r['retained']}")
