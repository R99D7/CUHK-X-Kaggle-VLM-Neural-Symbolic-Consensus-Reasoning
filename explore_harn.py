"""
The HARn source is a different dataset. Let's check those separately.
For the HARn questions, maybe there are different structural patterns.
Also: let's look at what sequence questions currently DISAGREE with v237 baseline
and see if any obvious errors remain.
"""
import pandas as pd
from collections import defaultdict
from itertools import permutations

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')
v237 = pd.read_csv('submission_v237_SELF_LEAK.csv')

# Check how many v246 sequence predictions agree with v237
te_seq = te[te['category'] == 'sequence']
sub_seq = sub[sub['qa_id'].isin(te_seq['qa_id'])]
v237_seq = v237[v237['qa_id'].isin(te_seq['qa_id'])]

merged = sub_seq.merge(v237_seq, on='qa_id', suffixes=('_v248', '_v237'))
diffs = merged[merged['prediction_v248'] != merged['prediction_v237']]
print(f"Sequence questions where v248 differs from v237: {len(diffs)}")
print(diffs[['qa_id', 'prediction_v237', 'prediction_v248']].to_string())

# For HARn source, check if the structure is different
print("\n\nHARn sequence questions:")
harn_seq = te[(te['source'] == 'HARn') & (te['category'] == 'sequence')]
print(f"Count: {len(harn_seq)}")
for idx, row in harn_seq.iterrows():
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    print(f"  {row['qa_id']}: A={row['A'][:20]}, B={row['B'][:20]}, C={row['C'][:20]}, D={row['D'][:20]} -> {pred}")
