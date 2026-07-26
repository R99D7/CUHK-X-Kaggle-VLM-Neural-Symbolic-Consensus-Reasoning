import pandas as pd
from collections import Counter

# Use a diverse set of good submissions for the ensemble
best_files = [
    'submission_ultimate_v12.csv',
    'submission_friend.csv',
    'submission_majority.csv',
    'submission_new.csv',
    'submission_super.csv',
    'submission_smart_length.csv',
    'submission_hybrid_trust_new.csv'
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
    for f in best_files:
        pred = preds[f].get(qa_id, '')
        if pred:
            # We can give v12 a weight of 2 just to break ties safely
            if f == 'submission_ultimate_v12.csv':
                votes.extend([pred, pred])
            else:
                votes.append(pred)
            
    c = Counter(votes)
    top_pred = c.most_common(1)[0][0]
    
    final_pred = top_pred
    
    if final_pred != base:
        changed_count += 1
        
    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

out = pd.DataFrame(final_preds)
out.to_csv('submission_upvote_v18_diverse.csv', index=False)

print(f"Total changed from 0.403 baseline (v12): {changed_count}/682")

merged = out.merge(pd.read_csv('submission_ultimate_v12.csv').rename(columns={'prediction':'base'}), on='qa_id')
merged2 = merged.merge(test[['qa_id','category']], on='qa_id')
for cat in test['category'].unique():
    s = merged2[merged2['category']==cat]
    ch = (s['prediction'] != s['base']).sum()
    print(f"  {cat}: {ch}/{len(s)} changed")
