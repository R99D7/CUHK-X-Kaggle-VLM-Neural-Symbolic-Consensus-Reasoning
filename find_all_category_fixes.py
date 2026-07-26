"""
Investigate all guaranteed logic violations across ALL categories (not just multi!) in submission_v270_TRUE_SUMMIT.csv.
Also explore rules involving object_interaction and emotion, and reverse implications.
"""
import pandas as pd
from collections import defaultdict

# Load our verified 0.69883 summit submission
sub = pd.read_csv("submission_v270_TRUE_SUMMIT.csv")
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

grouped = te.groupby('path')
potential_fixes = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
    def get_acts(q_row):
        pred_letters = str(q_row['pred']).strip()
        opts = {l: str(q_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        actions = set()
        for l in pred_letters:
            if l in opts:
                for act in opts[l].split(','):
                    actions.add(act.strip())
        return actions, opts, pred_letters

    parsed = {c: get_acts(r) for c, r in cats.items()}
    
    # Pool all actions verified by high-precision categories
    all_verified_actions = set()
    for c, (acts, opts, pred) in parsed.items():
        if c in ['single', 'multi', 'sequence', 'combination']:
            all_verified_actions.update(acts)
            
    # Check sequence questions: if a verified action is in sequence options, is it in the prediction?
    if 'sequence' in cats:
        sq_row = cats['sequence']
        sq_pred = str(sq_row['pred']).strip()
        sq_opts = {l: str(sq_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        missing = [l for l, txt in sq_opts.items() if txt in all_verified_actions and l not in sq_pred]
        if missing:
            potential_fixes.append((sq_row['qa_id'], 'sequence', sq_pred, f"Missing verified action letters in sequence: {missing} ({[sq_opts[l] for l in missing]})"))

    # Check single questions: if a single action predicted is NOT in all_verified_actions of COMBINATION or MULTI, why?
    if 'single' in cats and 'combination' in cats:
        s_row = cats['single']
        s_acts, _, s_pred = parsed['single']
        c_acts, _, c_pred = parsed['combination']
        for act in s_acts:
            if act and act not in c_acts:
                potential_fixes.append((s_row['qa_id'], 'single vs combination', f"Single: {s_pred} ({act}) vs Comb: {c_pred} ({c_acts})", f"Single action '{act}' not in Comb!"))

    if 'combination' in cats:
        c_row = cats['combination']
        c_opts = {l: [a.strip().lower() for a in str(c_row[l]).split(',')] for l in ['A', 'B', 'C', 'D']}
        c_pred = str(c_row['pred']).strip()
        # Which option in combination contains the highest number of all_verified_actions without extra unverified actions?
        best_opt = None
        best_score = -1
        for opt_letter, opt_acts in c_opts.items():
            overlap = sum(1 for a in opt_acts if a in all_verified_actions)
            if overlap > best_score:
                best_score = overlap
                best_opt = opt_letter
        if best_opt != c_pred and best_score > sum(1 for a in c_opts.get(c_pred, []) if a in all_verified_actions):
            potential_fixes.append((c_row['qa_id'], 'combination optimal selection', c_pred, f"Comb option {best_opt} ({c_opts[best_opt]}) overlaps better with verified actions ({all_verified_actions}) than current {c_pred} ({c_opts.get(c_pred, [])})"))

print(f"Total cross-category potential fixes found in v270: {len(potential_fixes)}")
for f in potential_fixes[:30]:
    print(f"[{f[1]}] ID: {f[0]} | Current: {f[2]} | Info: {f[3]}")
