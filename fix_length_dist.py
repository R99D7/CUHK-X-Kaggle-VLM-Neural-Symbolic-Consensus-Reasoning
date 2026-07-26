import pandas as pd
import numpy as np

sub = pd.read_csv('submission.csv')
te = pd.read_csv('test_qa.csv')
probs = pd.read_csv('transformer_fixed_raw_predictions.csv')

sub = sub.merge(te[['qa_id', 'category']], on='qa_id')
sub = sub.merge(probs, on='qa_id')

multi_mask = sub['category'] == 'multi'

# Calculate drop confidences for length 3 predictions
len3_mask = multi_mask & (sub['prediction'].astype(str).str.strip().str.len() == 3)
len3_indices = sub[len3_mask].index

drop_info = []
for idx in len3_indices:
    pred = str(sub.loc[idx, 'prediction']).strip()
    # Get probabilities for the predicted letters
    letter_probs = {l: sub.loc[idx, f'raw_prob_{l}'] for l in pred}
    
    # Find the weakest letter
    weakest_letter = min(letter_probs, key=letter_probs.get)
    weakest_prob = letter_probs[weakest_letter]
    
    # The new prediction would be the other two letters
    new_pred = "".join(sorted([l for l in pred if l != weakest_letter]))
    
    drop_info.append({
        'index': idx,
        'qa_id': sub.loc[idx, 'qa_id'],
        'old_pred': pred,
        'weakest_letter': weakest_letter,
        'weakest_prob': weakest_prob,
        'new_pred': new_pred
    })

drop_df = pd.DataFrame(drop_info)
drop_df = drop_df.sort_values('weakest_prob', ascending=True)

# We want to drop exactly 14 to hit the target distribution
TARGET_DROPS = 14

print(f'Dropping {TARGET_DROPS} weakest length 3 predictions to length 2:')
for i in range(min(TARGET_DROPS, len(drop_df))):
    row = drop_df.iloc[i]
    print(f"{row['qa_id']}: {row['old_pred']} -> {row['new_pred']} (dropped {row['weakest_letter']} with prob {row['weakest_prob']:.4f})")
    
    sub.loc[row['index'], 'prediction'] = row['new_pred']

# Let's also check if we need to drop length 2 to length 1!
# Train len 1 = 38.56%. 38.56% of 144 = 55.5 questions.
# Currently Test len 1 = 37.50% (54 questions).
# We are extremely close to the target! No need to touch length 2.

sub[['qa_id', 'prediction']].to_csv('submission_v242_LENGTH_FIX.csv', index=False)
sub[['qa_id', 'prediction']].to_csv('submission.csv', index=False)
print('Saved to submission.csv')
