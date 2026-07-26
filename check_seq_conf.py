"""
Check confidence of sequence predictions.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')
raw = pd.read_csv('transformer_fixed_raw_predictions.csv')

seq_probs = []
for idx, row in te[te['category'] == 'sequence'].iterrows():
    pred = sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]
    r = raw[raw['qa_id'] == row['qa_id']].iloc[0]
    p = r[f'raw_prob_{pred}']
    seq_probs.append(p)

import numpy as np
print(f"Sequence confidence:")
print(f"Mean: {np.mean(seq_probs):.4f}")
print(f"Median: {np.median(seq_probs):.4f}")
print(f"Min: {np.min(seq_probs):.4f}")
print(f"Max: {np.max(seq_probs):.4f}")
print(f"< 0.30: {sum(1 for p in seq_probs if p < 0.3)}")
