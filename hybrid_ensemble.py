import pandas as pd

ult_df = pd.read_csv('submission_ultimate.csv')
new_df = pd.read_csv('submission_new.csv')

new_dict = dict(zip(new_df['qa_id'], new_df['prediction']))

final_preds = []
for idx, row in ult_df.iterrows():
    qa_id = row['qa_id']
    ult_pred = str(row['prediction'])
    new_pred = str(new_dict.get(qa_id, 'A'))
    
    if len(ult_pred) != len(new_pred):
        # The crowd tried to change the length of our best model! Revert to the best model.
        final_preds.append({'qa_id': qa_id, 'prediction': new_pred})
    else:
        # The crowd agrees on length, we trust the crowd's letters.
        final_preds.append({'qa_id': qa_id, 'prediction': ult_pred})
        
pd.DataFrame(final_preds).to_csv('submission_hybrid_trust_new.csv', index=False)
print("Created hybrid submission!")
