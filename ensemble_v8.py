import pandas as pd

# Load the test data for categories
test_df = pd.read_csv('test_qa.csv')
cat_dict = dict(zip(test_df['qa_id'], test_df['category']))

# Load the two submissions
v7 = pd.read_csv('submission_ml_v7.csv')
moondream = pd.read_csv('submission_final.csv')

v7_dict = dict(zip(v7['qa_id'], v7['prediction']))
md_dict = dict(zip(moondream['qa_id'], moondream['prediction']))

final_preds = []

for qa_id in cat_dict:
    cat = cat_dict[qa_id]
    pred_v7 = v7_dict.get(qa_id, 'A')
    pred_md = md_dict.get(qa_id, 'A')
    
    # Moondream2 is much better at visual tasks like emotion/object interaction.
    # Text ML V7 is heavily optimized for structural logic (sequence, combination, multi).
    
    if cat in ['emotion', 'object_interaction', 'single']:
        # Trust vision model for visual tasks
        final_pred = pred_md
    else:
        # Trust ML language prior for sequence/multi/combination
        final_pred = pred_v7
        
    # Edge case fallback
    if pd.isna(final_pred) or final_pred == '':
        final_pred = pred_v7
        
    final_preds.append({'qa_id': qa_id, 'prediction': final_pred})

# Save the ensemble
ensemble_df = pd.DataFrame(final_preds)
ensemble_df.to_csv('submission_ensemble_v8.csv', index=False)
print("Ensemble saved to submission_ensemble_v8.csv!")
