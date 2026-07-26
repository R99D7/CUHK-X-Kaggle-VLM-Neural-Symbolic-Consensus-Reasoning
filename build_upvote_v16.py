import pandas as pd
from collections import Counter
import glob

# Identify the best files based on previous analysis
best_files = [
    'submission_ultimate_v12.csv',
    'submission_smart_length.csv',
    'submission_ensemble_v10.csv',
    'submission_majority.csv',
    'submission_ensemble_ultimate.csv',
    'submission_ultimate_v8.csv',
    'submission_super_fixed.csv',
    'submission_ensemble_v12.csv',
    'submission_ultimate.csv',
    'submission_ultimate_v9.csv',
    'submission_new_freq_fixed.csv',
    'submission_ensemble_v8.csv'
]

test = pd.read_csv('test_qa.csv')
preds = {}
for f in best_files:
    try:
        df = pd.read_csv(f)
        preds[f] = dict(zip(df['qa_id'], df['prediction'].astype(str)))
    except Exception as e:
        print(f"Skipped {f}: {e}")

base_preds = preds['submission_ultimate_v12.csv']

final_preds = []
changed_count = 0

for _, row in test.iterrows():
    qa_id = row['qa_id']
    cat = row['category']
    base = base_preds.get(qa_id, 'A')
    
    votes = []
    # Give the v12 baseline a weight of 3 to anchor it
    votes.extend([base] * 3)
    
    for f in best_files:
        if f != 'submission_ultimate_v12.csv':
            pred = preds[f].get(qa_id, '')
            if pred:
                votes.append(pred)
                
    c = Counter(votes)
    top_pred, top_count = c.most_common(1)[0]
    
    final_pred = top_pred
    
    # If tie, fallback to base
    if len(c) > 1 and c.most_common(2)[0][1] == c.most_common(2)[1][1]:
        final_pred = base
        
    if final_pred != base:
        changed_count += 1
        
    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

out = pd.DataFrame(final_preds)
out.to_csv('submission_upvote_v16.csv', index=False)

print(f"Total changed from 0.403 baseline: {changed_count}/682")

# Category breakdown of changes
merged = out.merge(pd.read_csv('submission_ultimate_v12.csv').rename(columns={'prediction':'base'}), on='qa_id')
merged2 = merged.merge(test[['qa_id','category']], on='qa_id')
for cat in test['category'].unique():
    s = merged2[merged2['category']==cat]
    ch = (s['prediction'] != s['base']).sum()
    print(f"  {cat}: {ch}/{len(s)} changed")
