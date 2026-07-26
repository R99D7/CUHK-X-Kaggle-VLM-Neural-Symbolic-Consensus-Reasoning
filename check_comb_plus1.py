"""
Check if there are any combination questions where switching options increases verified action overlap
without introducing unverified actions.
"""
import pandas as pd

sub = pd.read_csv("submission_v270_TRUE_SUMMIT.csv")
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

grouped = te.groupby('path')
cand_fixes = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    if 'combination' not in cats:
        continue
        
    c_row = cats['combination']
    c_opts = {l: set([x.strip().lower() for x in str(c_row[l]).split(',')]) for l in ['A', 'B', 'C', 'D']}
    c_pred = str(c_row['pred']).strip()
    
    # Pool verified actions from single, multi, sequence
    verified = set()
    for cat in ['single', 'multi', 'sequence']:
        if cat in cats:
            r = cats[cat]
            p = str(r['pred']).strip()
            opts = {l: [x.strip().lower() for x in str(r[l]).split(',')] for l in ['A', 'B', 'C', 'D']}
            for letter in p:
                if letter in opts:
                    for a in opts[letter]:
                        verified.add(a)
                        
    curr_overlap = len(c_opts.get(c_pred, set()) & verified)
    curr_unverified = len(c_opts.get(c_pred, set()) - verified)
    
    for l, acts in c_opts.items():
        if l == c_pred: continue
        ov = len(acts & verified)
        unv = len(acts - verified)
        # If the new option has more verified actions AND fewer or zero unverified actions
        if ov > curr_overlap and unv <= curr_unverified:
            cand_fixes.append((c_row['qa_id'], c_pred, l, f"Old ({c_opts.get(c_pred, set())}) ov={curr_overlap}, unv={curr_unverified} -> New ({acts}) ov={ov}, unv={unv} with verified={verified}"))

print(f"Total superior combination candidates: {len(cand_fixes)}")
for f in cand_fixes:
    print(f"[COMB SUPERIOR] ID: {f[0]} | {f[1]} -> {f[2]} | {f[3]}")
