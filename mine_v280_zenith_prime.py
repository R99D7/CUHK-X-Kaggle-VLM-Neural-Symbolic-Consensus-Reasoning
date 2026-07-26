"""
Mine v280 Zenith Prime:
Find Combination & Sequence questions where an alternative option has 100% agreement with MULTI & SINGLE
while the current prediction has an uncorroborated action!
This is the EXACT rule that boosted v275 (0.77192) to v276 (0.77485)!
"""
import pandas as pd

sub276 = pd.read_csv("submission_v276_APEX_SUMMIT.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")

sub_map276 = dict(zip(sub276['qa_id'], sub276['prediction']))
te['pred'] = te['qa_id'].map(sub_map276)

grouped = te.groupby('path')

comb_matches = []
seq_matches = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
    # 1. Audit COMBINATION
    if 'combination' in cats:
        r = cats['combination']
        qid = r['qa_id']
        curr_p = str(r['pred']).strip()
        opts = {l: [x.strip().lower() for x in str(r[l]).split(',')] for l in ['A', 'B', 'C', 'D']}
        
        # Gather confirmed actions from MULTI & SINGLE ONLY
        confirmed = set()
        for c in ['multi', 'single']:
            if c in cats:
                cr = cats[c]
                cp = str(cr['pred']).strip()
                copts = {l: str(cr[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
                for char in cp:
                    if char in copts:
                        confirmed.add(copts[char])
                        
        curr_acts = set(opts.get(curr_p, []))
        curr_unverified = curr_acts - confirmed
        curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{curr_p}', 0.0)
        
        if len(curr_unverified) > 0:
            for l, acts in opts.items():
                if l == curr_p: continue
                unverified = set(acts) - confirmed
                prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
                if len(unverified) < len(curr_unverified) and (prob > 0.15 or prob > curr_prob * 0.70):
                    comb_matches.append((qid, curr_p, l, round(curr_prob, 3), round(prob, 3), opts.get(curr_p, []), acts, list(confirmed), list(curr_unverified), list(unverified)))

    # 2. Audit SEQUENCE
    if 'sequence' in cats:
        r = cats['sequence']
        qid = r['qa_id']
        curr_p = str(r['pred']).strip()
        opts = {l: [x.strip().lower() for x in str(r[l]).split('->')] for l in ['A', 'B', 'C', 'D']}
        
        confirmed = set()
        for c in ['multi', 'single']:
            if c in cats:
                cr = cats[c]
                cp = str(cr['pred']).strip()
                copts = {l: str(cr[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
                for char in cp:
                    if char in copts:
                        confirmed.add(copts[char])
                        
        curr_acts = set(opts.get(curr_p, []))
        curr_unverified = curr_acts - confirmed
        curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{curr_p}', 0.0)
        
        if len(curr_unverified) > 0:
            for l, acts in opts.items():
                if l == curr_p: continue
                unverified = set(acts) - confirmed
                prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
                if len(unverified) < len(curr_unverified) and (prob > 0.15 or prob > curr_prob * 0.70):
                    seq_matches.append((qid, curr_p, l, round(curr_prob, 3), round(prob, 3), opts.get(curr_p, []), acts, list(confirmed), list(curr_unverified), list(unverified)))

print(f"=== COMBINATION OPPORTUNITIES ({len(comb_matches)} found) ===")
for cm in comb_matches:
    print(f"[COMB] {cm[0]}: Curr={cm[1]} ({cm[3]}) -> New={cm[2]} ({cm[4]}) | OldActs={cm[5]} -> NewActs={cm[6]} | Confirmed={cm[7]} | OldUnver={cm[8]} -> NewUnver={cm[9]}")

print(f"\n=== SEQUENCE OPPORTUNITIES ({len(seq_matches)} found) ===")
for sm in seq_matches:
    print(f"[SEQ] {sm[0]}: Curr={sm[1]} ({sm[3]}) -> New={sm[2]} ({sm[4]}) | OldActs={sm[5]} -> NewActs={sm[6]} | Confirmed={sm[7]} | OldUnver={sm[8]} -> NewUnver={sm[9]}")
