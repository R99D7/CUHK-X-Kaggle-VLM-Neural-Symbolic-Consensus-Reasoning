import pandas as pd

test_df = pd.read_csv('test_qa.csv')
cat_dict = dict(zip(test_df['qa_id'], test_df['category']))

v11 = pd.read_csv('submission_ml_v11.csv')
moondream = pd.read_csv('submission_final.csv')

v11_dict = dict(zip(v11['qa_id'], v11['prediction']))
md_dict = dict(zip(moondream['qa_id'], moondream['prediction']))

final_preds = []

for qa_id in cat_dict:
    cat = cat_dict[qa_id]
    pred_v11 = v11_dict.get(qa_id, 'A')
    pred_md = md_dict.get(qa_id, 'A')
    
    # Moondream2 is terrible overall (30.1%) but has visual grounding.
    # We should ONLY use it for strictly visual tasks like emotion and object_interaction.
    # Text ML V11 (35.3%) is much stronger overall, so we will use it for 'single', 'sequence', 'multi', 'combination'.
    
    if cat in ['emotion', 'object_interaction']:
        final_pred = pred_md
    else:
        final_pred = pred_v11
        
    if pd.isna(final_pred) or final_pred == '':
        final_pred = pred_v11
        
    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

ensemble_df = pd.DataFrame(final_preds)
ensemble_df.to_csv('submission_ensemble_v13.csv', index=False)
print("Ensemble saved to submission_ensemble_v13.csv!")
