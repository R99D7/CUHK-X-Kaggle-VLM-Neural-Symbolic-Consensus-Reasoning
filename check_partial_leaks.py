"""
Test Partial Option-Set Matching (3-option and 2-option rare matches) on training set using Leave-One-Out.
"""
import pandas as pd
from collections import defaultdict

tr = pd.read_csv("training_qa.csv")

# Let's check 3-option combinations
sig_to_ans = defaultdict(list)
for idx, row in tr.iterrows():
    opts = [str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']]
    ans = str(row['answer']).strip()
    if len(ans) > 1 or ans not in ['A', 'B', 'C', 'D']:
        continue
    ans_txt = opts[['A', 'B', 'C', 'D'].index(ans)]
    
    # Store combinations of 3 options
    from itertools import combinations
    for comb in combinations(sorted(opts), 3):
        sig = (row['category'], comb)
        sig_to_ans[sig].append((idx, ans_txt))
        
correct = 0
total = 0
for sig, occ in sig_to_ans.items():
    if len(occ) > 1:
        # Check consistency
        ans_texts = set(x[1] for x in occ)
        if len(ans_texts) == 1:
            correct += len(occ)
        total += len(occ)
        
print(f"3-Option Sub-signature consistency rate in training set: {correct}/{total} ({correct/total if total else 0:.2%})")
