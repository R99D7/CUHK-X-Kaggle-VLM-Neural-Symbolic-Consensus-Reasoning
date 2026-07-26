"""
Check how many 100.00% guaranteed logical violations exist in our 0.69590 submission.
"""
import pandas as pd

sub = pd.read_csv("submission_v267_NEWCOMB2MULTI.csv")
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

grouped = te.groupby('path')
violations = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    if 'multi' not in cats:
        continue
        
    multi_q = cats['multi']
    multi_pred = str(multi_q['pred']).strip()
    multi_opts = {l: str(multi_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    multi_acts = set([multi_opts[l] for l in multi_pred if l in multi_opts])
    
    def get_acts(q_row):
        pred_letters = str(q_row['pred']).strip()
        opts = {l: str(q_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        actions = set()
        for l in pred_letters:
            if l in opts:
                for act in opts[l].split(','):
                    actions.add(act.strip())
        return actions

    s_acts = get_acts(cats['single']) if 'single' in cats else set()
    sq_acts = get_acts(cats['sequence']) if 'sequence' in cats else set()
    c_acts = get_acts(cats['combination']) if 'combination' in cats else set()
    
    missing_letters = set()
    for l, txt in multi_opts.items():
        if txt in s_acts or txt in sq_acts or txt in c_acts:
            if l not in multi_pred:
                missing_letters.add(l)
                
    if missing_letters:
        new_pred = "".join(sorted(set(multi_pred) | missing_letters))
        violations.append((multi_q['qa_id'], multi_pred, new_pred, missing_letters))

print(f"Total 100% guaranteed multi-choice logic violations in 0.69590 baseline: {len(violations)}")
for v in violations[:20]:
    print(f"{v[0]}: {v[1]} -> {v[2]} (added {v[3]})")
