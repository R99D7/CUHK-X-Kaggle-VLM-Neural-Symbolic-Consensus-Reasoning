"""
Exhaustive Deep-Search Audit for v273 Ultimate Summit:
1. Check for Neural Network High-Confidence Overrides (raw model strongly disagrees with current prediction while consensus supports the raw model).
2. Check for Sequence Vocabulary Contradictions (single/comb action predicted is not in the 4 items of a sequence question).
3. Check for ANY remaining consensus opportunities across all categories.
"""
import pandas as pd
from collections import defaultdict, Counter

sub = pd.read_csv("submission_v272_SUMMIT_PRO.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

grouped = te.groupby('path')
findings = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
    # Get sequence 4-action universe if present
    seq_universe = set()
    if 'sequence' in cats:
        sq = cats['sequence']
        seq_universe = set([str(sq[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])

    # Pool verified actions across all categories in clip
    verified_acts = set()
    for cat in ['single', 'multi', 'sequence', 'combination']:
        if cat in cats:
            r = cats[cat]
            p = str(r['pred']).strip()
            opts = {l: [x.strip().lower() for x in str(r[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
            for char in p:
                if char in opts:
                    for act in opts[char]:
                        verified_acts.add(act)
                        
    # 1. Audit SINGLE questions against Sequence Universe and Neural Probabilities
    if 'single' in cats:
        s = cats['single']
        qid = s['qa_id']
        pred = str(s['pred']).strip()
        opts = {l: str(s[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        curr_act = opts.get(pred, '')
        curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{pred}', 0.0)
        
        # Check if another option has much higher neural probability AND matches verified actions / seq universe
        for l, act in opts.items():
            if l == pred: continue
            prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
            # If the candidate action is in sequence universe or verified acts, and current is not, OR candidate prob is significantly higher + verified
            if (act in seq_universe and curr_act not in seq_universe and seq_universe):
                findings.append({
                    'qid': qid, 'cat': 'single (seq universe violation)', 'old': pred, 'new': l,
                    'desc': f"Old '{curr_act}' (prob={curr_prob:.2f}) not in seq universe {seq_universe}. New '{act}' (prob={prob:.2f}) IS in seq universe!"
                })
            elif (act in verified_acts and curr_act not in verified_acts):
                findings.append({
                    'qid': qid, 'cat': 'single (consensus alignment)', 'old': pred, 'new': l,
                    'desc': f"Old '{curr_act}' not in verified consensus {verified_acts}. New '{act}' IS verified!"
                })
            elif (prob - curr_prob >= 0.20 and act in verified_acts):
                findings.append({
                    'qid': qid, 'cat': 'single (high confidence neural override)', 'old': pred, 'new': l,
                    'desc': f"Old '{curr_act}' prob={curr_prob:.2f} vs New '{act}' prob={prob:.2f} (both verified or candidate verified)"
                })

    # 2. Audit COMBINATION questions against Sequence Universe and Neural Probabilities
    if 'combination' in cats:
        c = cats['combination']
        qid = c['qa_id']
        pred = str(c['pred']).strip()
        opts = {l: set([x.strip().lower() for x in str(c[l]).split(',')]) for l in ['A', 'B', 'C', 'D']}
        curr_acts = opts.get(pred, set())
        curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{pred}', 0.0)
        
        for l, acts in opts.items():
            if l == pred: continue
            prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
            ov_curr = len(curr_acts & verified_acts)
            ov_cand = len(acts & verified_acts)
            # Check against sequence universe
            if seq_universe:
                seq_ov_curr = len(curr_acts & seq_universe)
                seq_ov_cand = len(acts & seq_universe)
                if seq_ov_cand > seq_ov_curr and len(acts - seq_universe) <= len(curr_acts - seq_universe):
                    findings.append({
                        'qid': qid, 'cat': 'combination (seq universe upgrade)', 'old': pred, 'new': l,
                        'desc': f"Old {curr_acts} (seq_ov={seq_ov_curr}) -> New {acts} (seq_ov={seq_ov_cand}) | Seq Universe={seq_universe}"
                    })

df_find = pd.DataFrame(findings)
print(f"Total rigorous findings discovered: {len(df_find)}")
if len(df_find) > 0:
    for idx, r in df_find.iterrows():
        print(f"[{r['cat'].upper()}] QID: {r['qid']} | {r['old']} -> {r['new']} | {r['desc']}")
