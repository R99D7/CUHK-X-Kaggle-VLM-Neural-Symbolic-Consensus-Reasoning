"""
Deep Audit for v279 Apex Plus - Finding guaranteed consensus upgrades over 0.77485!
Inspects:
1. COMBINATION questions where an alternative choice has higher consensus with MULTI/SINGLE/SEQUENCE than current v276 pred.
2. SEQUENCE questions where an alternative choice has higher consensus with MULTI/SINGLE/COMBINATION than current v276 pred.
3. SINGLE questions where an alternative choice has higher consensus with MULTI/COMBINATION than current v276 pred.
4. MULTI questions where adding or adjusting a letter increases vocabulary consensus without violating raw model confidence bounds (>0.35).
"""
import pandas as pd

sub276 = pd.read_csv("submission_v276_APEX_SUMMIT.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")

sub_map276 = dict(zip(sub276['qa_id'], sub276['prediction']))
te['pred'] = te['qa_id'].map(sub_map276)

grouped = te.groupby('path')

comb_candidates = []
seq_candidates = []
single_candidates = []
multi_add_candidates = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
    # 1. Audit COMBINATION
    if 'combination' in cats:
        r = cats['combination']
        qid = r['qa_id']
        curr_p = str(r['pred']).strip()
        opts = {l: [x.strip().lower() for x in str(r[l]).split(',')] for l in ['A', 'B', 'C', 'D']}
        
        # Gather confirmed actions from MULTI, SINGLE, SEQUENCE (excluding COMBINATION itself)
        other_actions = set()
        for c, cr in cats.items():
            if c in ['combination', 'emotion']: continue
            cp = str(cr['pred']).strip()
            copts = {l: [x.strip().lower() for x in str(cr[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
            for char in cp:
                if char in copts:
                    for a in copts[char]: other_actions.add(a)
                    
        curr_acts = set(opts.get(curr_p, []))
        curr_corroborated = len(curr_acts.intersection(other_actions))
        curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{curr_p}', 0.0)
        
        for l, acts in opts.items():
            if l == curr_p: continue
            corroborated = len(set(acts).intersection(other_actions))
            prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
            
            # If alternative option has 100% corroboration (2/2) while current has <2/2, or higher corroboration with decent probability
            if corroborated > curr_corroborated and prob > 0.20:
                comb_candidates.append({
                    'qa_id': qid, 'category': 'combination', 'curr_p': curr_p, 'new_p': l,
                    'curr_prob': round(curr_prob, 3), 'new_prob': round(prob, 3),
                    'curr_acts': opts.get(curr_p, []), 'new_acts': acts,
                    'scene_actions': list(other_actions)
                })

    # 2. Audit SEQUENCE
    if 'sequence' in cats:
        r = cats['sequence']
        qid = r['qa_id']
        curr_p = str(r['pred']).strip()
        opts = {l: [x.strip().lower() for x in str(r[l]).split('->')] for l in ['A', 'B', 'C', 'D']}
        
        other_actions = set()
        for c, cr in cats.items():
            if c in ['sequence', 'emotion']: continue
            cp = str(cr['pred']).strip()
            copts = {l: [x.strip().lower() for x in str(cr[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
            for char in cp:
                if char in copts:
                    for a in copts[char]: other_actions.add(a)
                    
        curr_acts = set(opts.get(curr_p, []))
        curr_corroborated = len(curr_acts.intersection(other_actions))
        curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{curr_p}', 0.0)
        
        for l, acts in opts.items():
            if l == curr_p: continue
            corroborated = len(set(acts).intersection(other_actions))
            prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
            
            if corroborated > curr_corroborated and prob > 0.20:
                seq_candidates.append({
                    'qa_id': qid, 'category': 'sequence', 'curr_p': curr_p, 'new_p': l,
                    'curr_prob': round(curr_prob, 3), 'new_prob': round(prob, 3),
                    'curr_acts': opts.get(curr_p, []), 'new_acts': acts,
                    'scene_actions': list(other_actions)
                })

    # 3. Audit SINGLE
    if 'single' in cats:
        r = cats['single']
        qid = r['qa_id']
        curr_p = str(r['pred']).strip()
        opts = {l: str(r[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        
        other_actions = set()
        for c, cr in cats.items():
            if c in ['single', 'emotion']: continue
            cp = str(cr['pred']).strip()
            copts = {l: [x.strip().lower() for x in str(cr[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
            for char in cp:
                if char in copts:
                    for a in copts[char]: other_actions.add(a)
                    
        curr_act = opts.get(curr_p, '')
        curr_corroborated = 1 if curr_act in other_actions else 0
        curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{curr_p}', 0.0)
        
        for l, act in opts.items():
            if l == curr_p: continue
            corroborated = 1 if act in other_actions else 0
            prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
            
            if corroborated > curr_corroborated and prob > 0.20:
                single_candidates.append({
                    'qa_id': qid, 'category': 'single', 'curr_p': curr_p, 'new_p': l,
                    'curr_prob': round(curr_prob, 3), 'new_prob': round(prob, 3),
                    'curr_act': curr_act, 'new_act': act,
                    'scene_actions': list(other_actions)
                })

print(f"=== COMBINATION CANDIDATES ({len(comb_candidates)}) ===")
for c in comb_candidates:
    print(c)

print(f"\n=== SEQUENCE CANDIDATES ({len(seq_candidates)}) ===")
for c in seq_candidates:
    print(c)

print(f"\n=== SINGLE CANDIDATES ({len(single_candidates)}) ===")
for c in single_candidates:
    print(c)
