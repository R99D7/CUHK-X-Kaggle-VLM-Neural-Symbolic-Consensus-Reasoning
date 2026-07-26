"""
Let me verify the accuracy of this approach more carefully.
The 73.4% comes from TRAINING with the TRUE combination answer.
But our combination predictions may be wrong ~14-20% of the time.

Let me simulate: on training, if combination is wrong 15% of the time,
what is the accuracy of comb->multi?

Actually, the key question is: is 73.4% (from high-conf combo) >> current multi accuracy?
Let me check our current submission accuracy vs v257 (before these multi changes).
"""
import pandas as pd

v257 = pd.read_csv('submission_v257_CROSS3.csv')
v259 = pd.read_csv('submission.csv')
te = pd.read_csv('test_qa.csv')

# Compare v257 vs v259 for multi questions
te_multi = te[te['category'] == 'multi']
v257_multi = v257[v257['qa_id'].isin(te_multi['qa_id'])]
v259_multi = v259[v259['qa_id'].isin(te_multi['qa_id'])]

diffs = v257_multi.merge(v259_multi, on='qa_id', suffixes=('_257', '_259'))
changed = diffs[diffs['prediction_257'] != diffs['prediction_259']]
print(f"Multi changes from v257 to v259: {len(changed)}")

# Check all changes from our BEST submission (v255 = 0.57602)
v255 = pd.read_csv('submission_v255_CROSS_HIGH_CONF.csv')
diffs_from_255 = v255.merge(v259, on='qa_id', suffixes=('_255', '_259'))
all_changed = diffs_from_255[diffs_from_255['prediction_255'] != diffs_from_255['prediction_259']]
print(f"Total changes from v255 (best, 0.57602): {len(all_changed)}")
print(all_changed.merge(te[['qa_id', 'category']], on='qa_id')['category'].value_counts())

# Show summary of current submission
print("\nCurrent submission stats:")
current = v259['prediction'].value_counts().head(20)
print(current)
