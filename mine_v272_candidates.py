"""
Mine all remaining multi-modal contradictions in submission_v271_SUMMIT_PLUS.csv,
evaluate them against training co-occurrence distributions, and score them using
raw transformer prediction probabilities!
"""
import pandas as pd
from collections import defaultdict, Counter

# 1. Build Action Co-occurrence matrix from training data
tr = pd.read_csv("training_qa.csv")
cooccur = defaultdict(Counter) # action -> Counter of co-occurring actions in same clip
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
    for a1 in clip_actions:
        for a2 in clip_actions:
            if a1 != a2:
                cooccur[a1][a2] += 1

print(f"Built training co-occurrence matrix across {len(cooccur)} unique actions.")

# 2. Analyze v271 test predictions and check for contradictions / superior consensus choices
sub = pd.read_csv("submission_v271_SUMMIT_PLUS.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

te_grouped = te.groupby('path')
candidates = []

for vid_path, grp in te_grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
    # Collect consensus actions and their support counts from all categories except emotion/object
    action_support = Counter()
    for cat in ['single', 'multi', 'sequence', 'combination']:
        if cat in cats:
            r = cats[cat]
            p = str(r['pred']).strip()
            opts = {l: [x.strip().lower() for x in str(r[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
            for char in p:
                if char in opts:
                    for act in opts[char]:
                        action_support[act] += 1
                        
    # Check Combination questions for superior options
    if 'combination' in cats:
        c_row = cats['combination']
        qid = c_row['qa_id']
        c_pred = str(c_row['pred']).strip()
        c_opts = {l: set([x.strip().lower() for x in str(c_row[l]).split(',')]) for l in ['A', 'B', 'C', 'D']}
        
        # Actions voted by at least one OTHER category in this clip
        other_actions = set()
        for cat in ['single', 'multi', 'sequence']:
            if cat in cats:
                r = cats[cat]
                p = str(r['pred']).strip()
                opts = {l: [x.strip().lower() for x in str(r[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
                for char in p:
                    if char in opts:
                        for act in opts[char]:
                            other_actions.add(act)
                            
        curr_acts = c_opts.get(c_pred, set())
        curr_ov = len(curr_acts & other_actions)
        curr_unv = len(curr_acts - other_actions)
        
        for l, acts in c_opts.items():
            if l == c_pred: continue
            ov = len(acts & other_actions)
            unv = len(acts - other_actions)
            # If option l has significantly better overlap or eliminates hallucinations
            if ov > curr_ov and unv <= curr_unv:
                prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
                curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{c_pred}', 0.0)
                candidates.append({
                    'qid': qid,
                    'cat': 'combination',
                    'old_pred': c_pred,
                    'new_pred': l,
                    'old_ov': curr_ov, 'old_unv': curr_unv,
                    'new_ov': ov, 'new_unv': unv,
                    'old_prob': round(curr_prob, 3), 'new_prob': round(prob, 3),
                    'desc': f"Old {curr_acts} (ov={curr_ov}, unv={curr_unv}) -> New {acts} (ov={ov}, unv={unv}) | Verified pool: {other_actions}"
                })

    # Check Single questions for remaining contradictions
    if 'single' in cats:
        s_row = cats['single']
        qid = s_row['qa_id']
        s_pred = str(s_row['pred']).strip()
        s_opts = {l: str(s_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        curr_act = s_opts.get(s_pred, '')
        
        other_actions = set()
        for cat in ['multi', 'sequence', 'combination']:
            if cat in cats:
                r = cats[cat]
                p = str(r['pred']).strip()
                opts = {l: [x.strip().lower() for x in str(r[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
                for char in p:
                    if char in opts:
                        for act in opts[char]:
                            other_actions.add(act)
                            
        if curr_act not in other_actions and other_actions:
            for l, act in s_opts.items():
                if act in other_actions:
                    prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
                    curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{s_pred}', 0.0)
                    candidates.append({
                        'qid': qid,
                        'cat': 'single',
                        'old_pred': s_pred,
                        'new_pred': l,
                        'old_ov': 0, 'old_unv': 1,
                        'new_ov': 1, 'new_unv': 0,
                        'old_prob': round(curr_prob, 3), 'new_prob': round(prob, 3),
                        'desc': f"Old single '{curr_act}' not in verified pool {other_actions}. Option {l} ('{act}') is verified!"
                    })

df_cand = pd.DataFrame(candidates)
print(f"Total potential upgrades uncovered: {len(df_cand)}")
for idx, r in df_cand.iterrows():
    print(f"[{r['cat'].upper()}] {r['qid']}: {r['old_pred']} (prob={r['old_prob']}) -> {r['new_pred']} (prob={r['new_prob']}) | {r['desc']}")
