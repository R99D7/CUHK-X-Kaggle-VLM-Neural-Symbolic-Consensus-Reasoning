"""
Check single letter distribution in training set vs our 0.69590 submission.
"""
import pandas as pd
from collections import Counter

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['prediction'] = te['qa_id'].map(sub_map)

print("--- TRAINING SET SINGLE CHOICE (single, object_interaction, emotion) ---")
tr_single = tr[tr['category'].isin(['single', 'object_interaction', 'emotion'])]
tr_ans = [str(x).strip() for x in tr_single['answer'] if len(str(x).strip()) == 1]
tr_counts = Counter(tr_ans)
total_tr = sum(tr_counts.values())
for k in ['A', 'B', 'C', 'D']:
    print(f"{k}: {tr_counts[k]} ({tr_counts[k]/total_tr:.2%})")

print("\n--- TEST SET 0.69590 PREDICTIONS (single, object_interaction, emotion) ---")
te_single = te[te['category'].isin(['single', 'object_interaction', 'emotion'])]
te_ans = [str(x).strip() for x in te_single['prediction'] if len(str(x).strip()) == 1]
te_counts = Counter(te_ans)
total_te = sum(te_counts.values())
for k in ['A', 'B', 'C', 'D']:
    print(f"{k}: {te_counts[k]} ({te_counts[k]/total_te:.2%})")
