import pandas as pd
from collections import Counter

files = [
    'submission_ensemble_v12.csv', # 0.377
    'submission_ensemble_v8.csv',  # 0.374
    'submission_ensemble_v4.csv',  # 0.362
    'submission_ml_v11.csv',       # 0.353
    'submission_ml_v3.csv'         # 0.342
]

dfs = [pd.read_csv(f) for f in files]
qa_ids = dfs[0]['qa_id']

final_preds = []

for i, qa_id in enumerate(qa_ids):
    preds = [df.iloc[i]['prediction'] for df in dfs]
    
    # Simple majority voting
    vote_counts = Counter(preds)
    best_pred = vote_counts.most_common(1)[0][0]
    
    final_preds.append({'qa_id': qa_id, 'prediction': best_pred})

pd.DataFrame(final_preds).to_csv('submission_majority.csv', index=False)
print("Saved submission_majority.csv!")
