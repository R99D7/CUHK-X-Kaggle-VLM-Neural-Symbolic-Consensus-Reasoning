import pandas as pd

new_df = pd.read_csv('submission_new.csv')
super_df = pd.read_csv('submission_super.csv')
friend_df = pd.read_csv('submission.csv')

new_dict = dict(zip(new_df['qa_id'], new_df['prediction']))
super_dict = dict(zip(super_df['qa_id'], super_df['prediction']))
friend_dict = dict(zip(friend_df['qa_id'], friend_df['prediction']))

final_preds = []
for qa_id in new_dict:
    n_pred = str(new_dict.get(qa_id, 'A'))
    s_pred = str(super_dict.get(qa_id, 'A'))
    f_pred = str(friend_dict.get(qa_id, 'A'))
    
    if n_pred == 'nan': n_pred = 'A'
    if s_pred == 'nan': s_pred = 'A'
    if f_pred == 'nan': f_pred = 'A'
    
    votes = {n_pred: 1.5, s_pred: 1.2, f_pred: 1.0}
    # Actually, we need to accumulate votes
    scores = {}
    for pred, weight in [(n_pred, 1.5), (s_pred, 1.2), (f_pred, 1.0)]:
        scores[pred] = scores.get(pred, 0) + weight
        
    best_pred = max(scores, key=scores.get)
    final_preds.append({'qa_id': qa_id, 'prediction': best_pred})
    
pd.DataFrame(final_preds).to_csv('submission_ultimate.csv', index=False)
print("Created ultimate ensemble!")
