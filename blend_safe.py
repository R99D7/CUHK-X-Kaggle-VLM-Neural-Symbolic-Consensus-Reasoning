"""
CORRECTED BLEND — lessons learned:
- Partial submission (75 single changes only) = 0.40350 IMPROVED
- Full submission v13 (changed multi too!) = 0.267 DESTROYED

Root cause: blend_depthcolor.py had a bug - it changed multi despite saying it wouldn't.
Depth_Color Moondream descriptions for combination/emotion categories are still too noisy.

Safe strategy proven by results:
- single: use Moondream Depth_Color (partial test confirmed improvement)  
- multi: STRICT baseline only (132 changes KILLED the score)
- combination: baseline only (too ambiguous for Moondream)
- emotion: baseline only (adverb matching unreliable)
- sequence: baseline only (permutations too specific)
- object_interaction: baseline only (too few samples to risk)
"""
import pandas as pd

test = pd.read_csv('test_qa.csv')
baseline = pd.read_csv('submission_ultimate_v12.csv')
dc = pd.read_csv('submission_depthcolor.csv')

base_dict = dict(zip(baseline['qa_id'], baseline['prediction']))
dc_dict = dict(zip(dc['qa_id'], dc['prediction']))

final_preds = []
for _, row in test.iterrows():
    qa_id = row['qa_id']
    cat = row['category']
    base_pred = str(base_dict.get(qa_id, 'A'))
    dc_pred = str(dc_dict.get(qa_id, ''))

    if cat == 'single':
        # ONLY category proven to improve with Depth_Color Moondream
        final_pred = dc_pred if dc_pred in ['A','B','C','D'] else base_pred
    else:
        # ALL other categories: strictly keep baseline
        final_pred = base_pred

    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

out = pd.DataFrame(final_preds)
out.to_csv('submission_v14_safe.csv', index=False)

# Verify
merged = out.merge(baseline.rename(columns={'prediction':'base'}), on='qa_id')
changed = (merged['prediction'] != merged['base']).sum()
print(f'Total changed from baseline: {changed}/682')

merged2 = merged.merge(test[['qa_id','category']], on='qa_id')
for cat in test['category'].unique():
    s = merged2[merged2['category']==cat]
    ch = (s['prediction'] != s['base']).sum()
    print(f'  {cat}: {ch}/{len(s)} changed')

print('\nDone -> submission_v14_safe.csv')
