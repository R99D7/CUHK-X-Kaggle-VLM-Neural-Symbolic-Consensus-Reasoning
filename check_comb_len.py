"""
Check length of correct combinations in training.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
comb_lens = []

for idx, row in tr[tr['category'] == 'combination'].iterrows():
    ans = str(row['answer']).strip()
    opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    acts = [a.strip() for a in opts.get(ans, "").split(',')]
    comb_lens.append(len(acts))

from collections import Counter
print(Counter(comb_lens))
