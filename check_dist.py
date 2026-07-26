"""
Check distribution of predictions vs training.
"""
import pandas as pd
from collections import Counter

tr = pd.read_csv('training_qa.csv')
sub = pd.read_csv('submission.csv')
te = pd.read_csv('test_qa.csv')

def get_dist(df, cat, col):
    if col == 'prediction':
        q_ids = te[te['category'] == cat]['qa_id'].tolist()
        vals = df[df['qa_id'].isin(q_ids)][col].tolist()
    else:
        vals = df[df['category'] == cat][col].tolist()
    return Counter([str(v).strip() for v in vals if len(str(v).strip()) == 1])

for cat in ['single', 'combination', 'emotion', 'object_interaction']:
    tr_dist = get_dist(tr, cat, 'answer')
    sub_dist = get_dist(sub, cat, 'prediction')
    
    tr_total = sum(tr_dist.values())
    sub_total = sum(sub_dist.values())
    
    print(f"Category: {cat}")
    print("  Training:")
    for l in ['A', 'B', 'C', 'D']:
        if tr_total > 0:
            print(f"    {l}: {tr_dist.get(l, 0) / tr_total * 100:.1f}% ({tr_dist.get(l, 0)})")
    
    print("  Predictions:")
    for l in ['A', 'B', 'C', 'D']:
        if sub_total > 0:
            print(f"    {l}: {sub_dist.get(l, 0) / sub_total * 100:.1f}% ({sub_dist.get(l, 0)})")
    print()
