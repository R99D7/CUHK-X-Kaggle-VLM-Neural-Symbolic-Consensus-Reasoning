"""
Find logical ordering of actions in training sequences.
"""
import pandas as pd
from collections import defaultdict

tr = pd.read_csv('training_qa.csv')

# Build a directed graph of action precedences
precedence = defaultdict(int)

for idx, row in tr[tr['category'] == 'sequence'].iterrows():
    ans = str(row['answer']).strip()
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    # ans is like 'CDAB'
    if len(ans) != 4: continue
    
    for i in range(len(ans)):
        for j in range(i+1, len(ans)):
            act1 = opts.get(ans[i])
            act2 = opts.get(ans[j])
            if act1 and act2:
                precedence[(act1, act2)] += 1

print("Strong precedences (act1 always happens before act2):")
for (a1, a2), count in precedence.items():
    rev_count = precedence.get((a2, a1), 0)
    if count >= 10 and rev_count == 0:
        print(f"  {a1} -> {a2} (count: {count})")
    elif count >= 10 and count > 10 * rev_count:
        print(f"  {a1} -> {a2} (count: {count}, rev: {rev_count})")
