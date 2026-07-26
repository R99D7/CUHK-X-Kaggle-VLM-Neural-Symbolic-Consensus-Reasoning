"""
FINAL BLEND: Combine fixed Depth_Color Moondream predictions with baseline 0.403.
Rules learned from baseline analysis:
- single/emotion/combination/object_interaction/sequence: use fixed Moondream (Depth_Color)
- multi: keep baseline (it already matches training distribution well)
- For any missing/invalid fixed predictions: fall back to baseline

Also correct known biases:
- combination: baseline under-predicts C. If fixed pred = C, trust it.
- emotion: baseline over-predicts B. If fixed pred != B, trust it slightly more.
"""
import pandas as pd
import numpy as np

test = pd.read_csv('test_qa.csv')
baseline = pd.read_csv('submission_ultimate_v12.csv')
fixed = pd.read_csv('submission_depthcolor.csv')

base_dict = dict(zip(baseline['qa_id'], baseline['prediction']))
fixed_dict = dict(zip(fixed['qa_id'], fixed['prediction']))

letters = set('ABCD')
final_preds = []

for _, row in test.iterrows():
    qa_id = row['qa_id']
    cat = row['category']
    base_pred = str(base_dict.get(qa_id, 'A'))
    fixed_pred = str(fixed_dict.get(qa_id, ''))

    final_pred = base_pred  # default: keep baseline

    if cat == 'multi':
        # Keep baseline for multi — it already models multi-label well
        final_pred = base_pred

    elif cat in ['single', 'emotion', 'combination', 'object_interaction']:
        # Use fixed Moondream (Depth_Color) if it produced a valid single letter
        if fixed_pred in ['A', 'B', 'C', 'D']:
            final_pred = fixed_pred
        else:
            final_pred = base_pred

    elif cat == 'sequence':
        # Use fixed if it's a valid 4-letter permutation
        if len(fixed_pred) == 4 and set(fixed_pred) == set('ABCD'):
            final_pred = fixed_pred
        else:
            final_pred = base_pred

    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

out = pd.DataFrame(final_preds)
out.to_csv('submission_v13_depthcolor.csv', index=False)

# Stats
merged = out.merge(baseline.rename(columns={'prediction': 'base'}), on='qa_id')
changed = (merged['prediction'] != merged['base']).sum()
print(f"Total: {len(out)} rows")
print(f"Changed from baseline: {changed}/682")

for cat in test['category'].unique():
    ids = test[test['category'] == cat]['qa_id'].tolist()
    c = merged[merged['qa_id'].isin(ids)]
    ch = (c['prediction'] != c['base']).sum()
    covered = sum(1 for i in ids if fixed_dict.get(i,'') != '')
    print(f"  {cat}: {ch}/{len(ids)} changed ({covered} covered by fixed run)")

print("\nDone! -> submission_v13_depthcolor.csv")
