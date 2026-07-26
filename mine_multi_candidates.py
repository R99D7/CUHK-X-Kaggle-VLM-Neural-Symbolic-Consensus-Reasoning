"""
Mine Multi Candidates:
Find Multi questions where adding a letter corroborated by COMBINATION/SINGLE (prob > 0.65)
or pruning an uncorroborated letter (prob < 0.40) creates 100% cross-category alignment!
"""
import pandas as pd

sub276 = pd.read_csv("submission_v276_APEX_SUMMIT.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")

sub_map276 = dict(zip(sub276['qa_id'], sub276['prediction']))
te['pred'] = te['qa_id'].map(sub_map276)

grouped = te.groupby('path')

add_candidates = []
prune_candidates = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
    if 'multi' in cats:
        r = cats['multi']
        qid = r['qa_id']
        curr_p = str(r['pred']).strip()
        opts = {l: str(r[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        probs = {l: raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0) for l in ['A', 'B', 'C', 'D']}
        
        # Confirmed actions in COMBINATION and SINGLE
        confirmed_acts = set()
        for c in ['combination', 'single', 'sequence']:
            if c in cats:
                cr = cats[c]
                cp = str(cr['pred']).strip()
                copts = {l: [x.strip().lower() for x in str(cr[l]).replace('->', ',').split(',')] for l in ['A', 'B', 'C', 'D']}
                for char in cp:
                    if char in copts:
                        for act in copts[char]: confirmed_acts.add(act)
                        
        # Check for missing letter that is CONFIRMED in other tracks and has high probability (>0.60)
        for l, act in opts.items():
            if l not in curr_p:
                if act in confirmed_acts and probs[l] > 0.60:
                    add_candidates.append((qid, curr_p, l, act, round(probs[l], 3), list(confirmed_acts)))

print(f"=== MULTI ADD CANDIDATES ({len(add_candidates)} found) ===")
for ac in add_candidates:
    print(f"[MULTI ADD] {ac[0]}: CurrPred='{ac[1]}' + AddLetter='{ac[2]}' ('{ac[3]}', prob={ac[4]}) | ConfirmedInScene={ac[5]}")
