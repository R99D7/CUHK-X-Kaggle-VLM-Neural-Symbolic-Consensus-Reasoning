import pandas as pd
from collections import Counter

friend = pd.read_csv('submission_friend.csv')
v12 = pd.read_csv('submission_ensemble_v12.csv')
v8 = pd.read_csv('submission_ensemble_v8.csv')
v11 = pd.read_csv('submission_ml_v11.csv')

friend_dict = dict(zip(friend['qa_id'], friend['prediction']))
v12_dict = dict(zip(v12['qa_id'], v12['prediction']))
v8_dict = dict(zip(v8['qa_id'], v8['prediction']))
v11_dict = dict(zip(v11['qa_id'], v11['prediction']))

final_preds = []

for qa_id in friend_dict:
    preds = [
        friend_dict.get(qa_id, 'A'), 
        friend_dict.get(qa_id, 'A'), # Double weight for the friend's 0.38011
        v12_dict.get(qa_id, 'A'),    # 0.37719
        v8_dict.get(qa_id, 'A'),     # 0.37426
        v11_dict.get(qa_id, 'A')     # 0.35380
    ]
    
    vote_counts = Counter(preds)
    final_pred = vote_counts.most_common(1)[0][0]
        
    if pd.isna(final_pred) or final_pred == '':
        final_pred = 'A'
        
    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

ensemble_df = pd.DataFrame(final_preds)
ensemble_df.to_csv('submission_super.csv', index=False)
print("Super Ensemble saved to submission_super.csv!")
