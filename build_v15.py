"""
Analyze ALL available prediction files.
Find which ones are most diverse and strongest per category.
Build optimal majority-vote ensemble from the BEST prediction files only.
"""
import pandas as pd
import numpy as np
import os

test = pd.read_csv('test_qa.csv')
train = pd.read_csv('training_qa.csv')

# Load all valid submission CSVs
csv_files = [
    'submission_ultimate_v12.csv',   # 0.403 - our BEST
    'submission_ml_v9.csv',
    'submission_ensemble_v10.csv',
    'submission_ensemble_v12.csv',
    'submission_ml_v11.csv',
    'submission_majority.csv',
    'submission_ensemble_v13.csv',
    'submission_ensemble_ultimate.csv',
    'submission_smart_length.csv',
    'submission_super_fixed.csv',
    'submission_new_freq_fixed.csv',
    'submission_ultimate.csv',
    'submission_ml_v7.csv',
    'submission_ensemble_v8.csv',
    'submission_ultimate_v8.csv',
    'submission_ultimate_v9.csv',
]

preds = {}
for f in csv_files:
    if os.path.exists(f):
        try:
            df = pd.read_csv(f)
            if 'qa_id' in df.columns and 'prediction' in df.columns and len(df) >= 682:
                preds[f] = dict(zip(df['qa_id'], df['prediction']))
                print(f"Loaded: {f} ({len(df)} rows)")
        except Exception as e:
            print(f"Skip {f}: {e}")

print(f"\nTotal valid files: {len(preds)}")

# For each test question, count votes per option
final_preds = []
base = preds.get('submission_ultimate_v12.csv', {})

for _, row in test.iterrows():
    qa_id = row['qa_id']
    cat = row['category']
    base_pred = str(base.get(qa_id, 'A'))

    # Collect all predictions for this qa_id
    all_preds = [str(d.get(qa_id, '')) for d in preds.values()]
    all_preds = [p for p in all_preds if p != '']

    if cat == 'single' or cat == 'emotion' or cat == 'combination' or cat == 'object_interaction':
        # Simple majority vote among single-letter predictions
        valid = [p for p in all_preds if p in ['A', 'B', 'C', 'D']]
        if valid:
            from collections import Counter
            votes = Counter(valid)
            top_pred, top_count = votes.most_common(1)[0]
            # Only use majority vote if it wins by margin
            if top_count >= len(valid) * 0.5:
                final_pred = top_pred
            else:
                final_pred = base_pred
        else:
            final_pred = base_pred

    elif cat == 'multi':
        # Multi: use base_pred (proven most reliable)
        final_pred = base_pred

    elif cat == 'sequence':
        # Sequence: use base_pred (too hard without video)
        final_pred = base_pred

    else:
        final_pred = base_pred

    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

out = pd.DataFrame(final_preds)
out.to_csv('submission_v15_ensemble.csv', index=False)

# Stats
merged = out.merge(pd.read_csv('submission_ultimate_v12.csv').rename(columns={'prediction':'base'}), on='qa_id')
changed = (merged['prediction'] != merged['base']).sum()
merged2 = merged.merge(test[['qa_id','category']], on='qa_id')
print(f"\nTotal changed from 0.403 baseline: {changed}/682")
for cat in test['category'].unique():
    s = merged2[merged2['category']==cat]
    ch = (s['prediction'] != s['base']).sum()
    print(f"  {cat}: {ch}/{len(s)} changed")

print("\nDone -> submission_v15_ensemble.csv")
