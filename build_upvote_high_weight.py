import pandas as pd
from collections import Counter

# We will use the very best files
best_files = [
    'submission_ultimate_v12.csv',       # 0.403
    'submission_smart_length.csv',       # Very strong NLP
    'submission_ensemble_v10.csv',       # High score
    'submission_majority.csv',           # Good ensemble
    'submission_ensemble_ultimate.csv',
    'submission_super_fixed.csv',
    'submission_ultimate_v8.csv',
    'submission_ensemble_v12.csv',
    'submission_ultimate.csv',
    'submission_ultimate_v9.csv',
    'submission_new_freq_fixed.csv',
    'submission_ml_v9.csv',
    'submission_ml_v11.csv'
]

test = pd.read_csv('test_qa.csv')
preds = {}
for f in best_files:
    try:
        df = pd.read_csv(f)
        preds[f] = dict(zip(df['qa_id'], df['prediction'].astype(str)))
    except Exception as e:
        pass

base_preds = preds['submission_ultimate_v12.csv']
final_preds = []
changed_count = 0

for _, row in test.iterrows():
    qa_id = row['qa_id']
    base = base_preds.get(qa_id, 'A')
    
    votes = []
    
    # GIVE A MUCH HIGHER UPVOTE WEIGHT TO THE 0.403 BASELINE
    # Weight = 6 means 6 of the 12 other models must unanimously agree to override it!
    votes.extend([base] * 6)
    
    for f in best_files:
        if f != 'submission_ultimate_v12.csv':
            pred = preds[f].get(qa_id, '')
            if pred:
                votes.append(pred)
                
    c = Counter(votes)
    top_pred = c.most_common(1)[0][0]
    
    final_pred = top_pred
    
    # In case of tie or close vote, stick to base
    if len(c) > 1 and c.most_common(2)[0][1] == c.most_common(2)[1][1]:
        final_pred = base
        
    if final_pred != base:
        changed_count += 1
        
    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

out = pd.DataFrame(final_preds)
out.to_csv('submission_upvote_high_weight.csv', index=False)

print(f"Total changed from 0.403 baseline (v12): {changed_count}/682")

merged = out.merge(pd.read_csv('submission_ultimate_v12.csv').rename(columns={'prediction':'base'}), on='qa_id')
merged2 = merged.merge(test[['qa_id','category']], on='qa_id')
for cat in test['category'].unique():
    s = merged2[merged2['category']==cat]
    ch = (s['prediction'] != s['base']).sum()
    print(f"  {cat}: {ch}/{len(s)} changed")
