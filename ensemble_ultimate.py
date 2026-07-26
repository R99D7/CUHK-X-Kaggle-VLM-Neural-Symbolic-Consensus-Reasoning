import pandas as pd
from collections import Counter

test_df = pd.read_csv('test_qa.csv')
cat_dict = dict(zip(test_df['qa_id'], test_df['category']))

moondream = pd.read_csv('submission_final.csv')
v11 = pd.read_csv('submission_ml_v11.csv')
v7 = pd.read_csv('submission_ml_v7.csv')
v5 = pd.read_csv('submission_ml_v5.csv')
v3 = pd.read_csv('submission_ml_v3.csv')

md_dict = dict(zip(moondream['qa_id'], moondream['prediction']))
v11_dict = dict(zip(v11['qa_id'], v11['prediction']))
v7_dict = dict(zip(v7['qa_id'], v7['prediction']))
v5_dict = dict(zip(v5['qa_id'], v5['prediction']))
v3_dict = dict(zip(v3['qa_id'], v3['prediction']))

final_preds = []

for qa_id in cat_dict:
    cat = cat_dict[qa_id]
    
    if cat in ['emotion', 'object_interaction', 'single']:
        # Trust vision model for strictly visual tasks
        final_pred = md_dict.get(qa_id, 'A')
    else:
        # Trust a MAJORITY VOTE of the best Text ML models for structural logic
        preds = [
            v11_dict.get(qa_id, 'A'),
            v7_dict.get(qa_id, 'A'),
            v5_dict.get(qa_id, 'A'),
            v3_dict.get(qa_id, 'A'),
            v11_dict.get(qa_id, 'A') # Give V11 (the best ML model) a double-weight in the vote
        ]
        vote_counts = Counter(preds)
        final_pred = vote_counts.most_common(1)[0][0]
        
    if pd.isna(final_pred) or final_pred == '':
        final_pred = 'A'
        
    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

ensemble_df = pd.DataFrame(final_preds)
ensemble_df.to_csv('submission_ensemble_ultimate.csv', index=False)
print("Ultimate Ensemble saved to submission_ensemble_ultimate.csv!")
