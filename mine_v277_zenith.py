"""
Comprehensive Mining Engine for v277 ZENITH SUMMIT over the 0.77485 baseline (submission_v276_APEX_SUMMIT.csv):
1. Complete Sequence Audit: Inspect all SEQUENCE predictions for uncorroborated action steps and test alternative choices.
2. Complete Combination Audit: Inspect all COMBINATION predictions for uncorroborated pairs and test alternative choices.
3. Complete Single Audit: Inspect all SINGLE atomic predictions for inconsistencies against scene vocabulary.
4. Remaining Multi Anomalies: Deeply inspect options A, B, C, D for all remaining uncorroborated atoms in MULTI.
"""
import pandas as pd
from collections import defaultdict, Counter

sub = pd.read_csv("submission_v276_APEX_SUMMIT.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

grouped = te.groupby('path')
seq_upgrades = []
comb_upgrades = []
single_upgrades = []
multi_anomalies = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
    # Collect all corroborated actions across ALL categories
    scene_actions = Counter()
    for cat, r in cats.items():
        if cat == 'emotion': continue
        p = str(r['pred']).strip()
        opts = {l: [x.strip().lower() for x in str(r[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
        for char in p:
            if char in opts:
                for act in opts[char]:
                    scene_actions[act] += 1

    # 1. Audit SEQUENCE
    if 'sequence' in cats:
        r = cats['sequence']
        qid = r['qa_id']
        curr_p = str(r['pred']).strip()
        opts = {l: [x.strip().lower() for x in str(r[l]).split('->')] for l in ['A', 'B', 'C', 'D']}
        
        # Actions outside of sequence itself
        other_actions = set()
        for c, cr in cats.items():
            if c in ['sequence', 'emotion']: continue
            cp = str(cr['pred']).strip()
            copts = {l: [x.strip().lower() for x in str(cr[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
            for char in cp:
                if char in copts:
                    for a in copts[char]: other_actions.add(a)
                    
        curr_acts = set(opts.get(curr_p, []))
        curr_unver = curr_acts - other_actions
        curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{curr_p}', 0.0)
        
        if curr_unver:
            for l, acts in opts.items():
                if l == curr_p: continue
                prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
                unver = set(acts) - other_actions
                if len(unver) < len(curr_unver) and (prob > curr_prob * 0.75 or prob > 0.45):
                    seq_upgrades.append((qid, curr_p, l, round(curr_prob, 3), round(prob, 3), list(curr_acts), list(acts), list(other_actions)))

    # 2. Audit COMBINATION
    if 'combination' in cats:
        r = cats['combination']
        qid = r['qa_id']
        curr_p = str(r['pred']).strip()
        opts = {l: [x.strip().lower() for x in str(r[l]).split(',')] for l in ['A', 'B', 'C', 'D']}
        
        other_actions = set()
        for c, cr in cats.items():
            if c in ['combination', 'emotion']: continue
            cp = str(cr['pred']).strip()
            copts = {l: [x.strip().lower() for x in str(cr[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
            for char in cp:
                if char in copts:
                    for a in copts[char]: other_actions.add(a)
                    
        curr_acts = set(opts.get(curr_p, []))
        curr_unver = curr_acts - other_actions
        curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{curr_p}', 0.0)
        
        if curr_unver:
            for l, acts in opts.items():
                if l == curr_p: continue
                prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
                unver = set(acts) - other_actions
                if len(unver) < len(curr_unver) and (prob > curr_prob * 0.75 or prob > 0.45):
                    comb_upgrades.append((qid, curr_p, l, round(curr_prob, 3), round(prob, 3), list(curr_acts), list(acts), list(other_actions)))

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
        curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{curr_p}', 0.0)
        
        if curr_act not in other_actions and other_actions:
            for l, act in opts.items():
                if l == curr_p: continue
                if act in other_actions:
                    prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
                    if prob > 0.20: # Consider all viable consensus candidates
                        single_upgrades.append((qid, curr_p, l, round(curr_prob, 3), round(prob, 3), curr_act, act, list(other_actions)))

    # 4. Audit remaining uncorroborated MULTI
    if 'multi' in cats:
        r = cats['multi']
        qid = r['qa_id']
        curr_p = str(r['pred']).strip()
        opts = {l: str(r[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        
        other_actions = set()
        for c, cr in cats.items():
            if c in ['multi', 'emotion']: continue
            cp = str(cr['pred']).strip()
            copts = {l: [x.strip().lower() for x in str(cr[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
            for char in cp:
                if char in copts:
                    for a in copts[char]: other_actions.add(a)
                    
        for l in curr_p:
            act = opts.get(l, '')
            prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
            if act and act not in other_actions:
                all_opts_str = {k: f"{v} ({round(raw_map.get(qid, {}).get(f'raw_prob_{k}', 0.0), 3)})" for k, v in opts.items()}
                multi_anomalies.append((qid, curr_p, l, act, round(prob, 3), list(other_actions), all_opts_str))

print(f"=== 1. SEQUENCE UPGRADE OPPORTUNITIES ({len(seq_upgrades)} found) ===")
for su in sorted(seq_upgrades, key=lambda x: x[4], reverse=True):
    print(f"[SEQUENCE] {su[0]}: {su[1]} ({su[3]}) -> {su[2]} ({su[4]}) | Old={su[5]} -> New={su[6]} | Scene={su[7]}")

print(f"\n=== 2. COMBINATION UPGRADE OPPORTUNITIES ({len(comb_upgrades)} found) ===")
for cu in sorted(comb_upgrades, key=lambda x: x[4], reverse=True):
    print(f"[COMBINATION] {cu[0]}: {cu[1]} ({cu[3]}) -> {cu[2]} ({cu[4]}) | Old={cu[5]} -> New={cu[6]} | Scene={cu[7]}")

print(f"\n=== 3. SINGLE UPGRADE OPPORTUNITIES ({len(single_upgrades)} found) ===")
for su in sorted(single_upgrades, key=lambda x: x[4], reverse=True):
    print(f"[SINGLE] {su[0]}: {su[1]} ('{su[5]}', prob={su[3]}) -> {su[2]} ('{su[6]}', prob={su[4]}) | Scene={su[7]}")

print(f"\n=== 4. REMAINING MULTI UNCORROBORATED ANOMALIES ({len(multi_anomalies)} found) ===")
for m in sorted(multi_anomalies, key=lambda x: x[4]):
    print(f"[MULTI] {m[0]}: Pred={m[1]} | Letter '{m[2]}' ('{m[3]}', prob={m[4]}) uncorroborated! Scene={m[5]} | All Opts: {m[6]}")
