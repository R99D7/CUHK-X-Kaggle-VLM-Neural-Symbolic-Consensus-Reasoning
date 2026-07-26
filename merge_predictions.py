import pandas as pd

# Load files
test_df = pd.read_csv('test_qa.csv')
sample_df = pd.read_csv('sample_submission.csv')
v3_df = pd.read_csv('submission_ml_v3.csv')

# Create a final predictions dataframe
final_preds = []

for idx, row in test_df.iterrows():
    cat = row['category']
    # If the category is complex (multi or sequence), we CANNOT just output a single letter.
    # The sample_submission.csv already has the correct formatting (e.g. 'DCBA' or 'BCD').
    # Since we can't do VLM here, and our model only predicts 1 option, we will fallback
    # to the sample_submission value for these complex categories.
    if cat in ['multi', 'sequence']:
        pred = sample_df.iloc[idx]['prediction']
    else:
        # For single, emotion, combination, object_interaction, our semantic model works best!
        pred = v3_df.iloc[idx]['answer']
        
    final_preds.append(pred)

# Save final submission
final_df = pd.DataFrame({
    'qa_id': test_df['id'],
    'prediction': final_preds
})

final_df.to_csv('submission_final.csv', index=False)
print("Saved best-effort merged predictions to submission_final.csv!")
