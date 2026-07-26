import pandas as pd

# Load test data to know the categories
test_df = pd.read_csv('test_qa.csv')
cat_dict = dict(zip(test_df['qa_id'], test_df['category']))

# Load the two submissions
ml_v3 = pd.read_csv('submission_ml_v3.csv')
moondream = pd.read_csv('submission_final.csv')

ml_dict = dict(zip(ml_v3['qa_id'], ml_v3['prediction']))
md_dict = dict(zip(moondream['qa_id'], moondream['prediction']))

final_preds = []

for qa_id in cat_dict:
    cat = cat_dict[qa_id]
    pred_ml = ml_dict.get(qa_id, 'A')
    pred_md = md_dict.get(qa_id, 'A')
    
    # Heuristic for ensembling:
    # Moondream2 actually saw the frame, so it's much better at visual tasks like emotion/object interaction.
    # Text ML V3 is a strong structural prior (34% accuracy!) and is much better at complex logic (sequence, combination, multi).
    
    if cat in ['emotion', 'object_interaction', 'single']:
        # Trust vision model for pure visual tasks
        final_pred = pred_md
    else:
        # Trust ML language prior for sequence/multi/combination
        final_pred = pred_ml
        
    # Edge case fallback: if one is invalid length, use the other
    if pd.isna(final_pred) or final_pred == '':
        final_pred = pred_ml
        
    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

# Save the ensemble
ensemble_df = pd.DataFrame(final_preds)
ensemble_df.to_csv('submission_ensemble_v4.csv', index=False)
print("Ensemble saved to submission_ensemble_v4.csv!")
