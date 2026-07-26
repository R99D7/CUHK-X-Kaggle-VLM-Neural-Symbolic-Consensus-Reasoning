"""
Mine all logical relationship rules between question categories on the training set to find ANY rule with 100% empirical precision.
"""
import pandas as pd
from collections import defaultdict

tr = pd.read_csv("training_qa.csv")
grouped = tr.groupby('path')

rules = defaultdict(lambda: [0, 0]) # [matches, total]

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
    def get_acts(q_row):
        pred = str(q_row['answer']).strip()
        opts = {l: str(q_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        actions = set()
        for l in pred:
            if l in opts:
                for act in opts[l].split(','):
                    actions.add(act.strip())
        return actions, opts, pred

    parsed = {c: get_acts(r) for c, r in cats.items()}
    
    # Test all pairs of categories (cat1, cat2)
    for c1, (acts1, opts1, ans1) in parsed.items():
        for c2, (acts2, opts2, ans2) in parsed.items():
            if c1 == c2: continue
            
            # Test: Every action in c1 answer is present in c2 answer (when it appears in c2 options)
            for act in acts1:
                c2_opt_vals = [x.strip() for v in opts2.values() for x in v.split(',')]
                if act in c2_opt_vals:
                    rules[(f"{c1} action in {c2} options => present in {c2} answer")][1] += 1
                    if act in acts2:
                        rules[(f"{c1} action in {c2} options => present in {c2} answer")][0] += 1

print("--- TRAINING SET EMPIRICAL IMPLICATION RULES (Precision >= 98%) ---")
for r_name, (m, t) in sorted(rules.items(), key=lambda x: -x[1][0]/x[1][1] if x[1][1]>0 else 0):
    prec = m / t if t > 0 else 0
    if prec >= 0.98 and t >= 50:
        print(f"{r_name}: {m}/{t} ({prec:.2%})")
