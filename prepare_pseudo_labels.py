import pandas as pd

# Load train, test, and the best submission
train_df = pd.read_csv('training_qa.csv')
test_df = pd.read_csv('test_qa.csv')
sub_df = pd.read_csv('submission_oracle_v20.csv')

# Merge test set with the best submission to get pseudo labels
test_merged = test_df.merge(sub_df, on='qa_id')

test_merged = test_merged.drop(columns=['prediction_x'])
test_merged = test_merged.rename(columns={'prediction_y': 'answer'})

# Keep only columns that match train_df
# train_df columns: qa_id, source, path, category, question, A, B, C, D, answer
cols = ['qa_id', 'source', 'path', 'category', 'question', 'A', 'B', 'C', 'D', 'answer']
# Ensure all cols exist in test_merged, some might be missing 'source' or 'path'
for c in cols:
    if c not in test_merged.columns:
        test_merged[c] = ''

test_merged = test_merged[cols]

# Drop duplicates if any merges duplicated things
test_merged = test_merged.loc[:, ~test_merged.columns.duplicated()]

print(f"Created {len(test_merged)} pseudo labels from test set.")

# We don't concatenate them here because we want to inject pseudo-labels only into the Training Folds during Cross Validation!
test_merged.to_csv('pseudo_test_labels.csv', index=False)
print("Saved pseudo_test_labels.csv!")
