"""
For HARn test single questions with D=nan (3 options only),
the model predicted D for test_0497 which is NaN! This is a definite error.

Also: the LM_test paths have numbers. Let's check if consecutive LM_test IDs
that share the same options have consistent answers in training.

More importantly: for HARn test single with 3 options, the action name is in 
the HARn training path. Let's try a text-matching approach:
- For each HARn test option, match it to HARn training action labels
- The action label that matches a training action category IS the answer
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

harn_te = te[te['source'] == 'HARn']
harn_te_single = harn_te[harn_te['category'] == 'single']

# Fix obvious error: test_0497 predicted D but D is nan
print("test_0497 prediction:", sub[sub['qa_id'] == 'test_0497']['prediction'].values)

# What options are available?
row = harn_te_single[harn_te_single['qa_id'] == 'test_0497'].iloc[0]
print(f"Options: A={row['A']}, B={row['B']}, C={row['C']}, D={row['D']}")
print("Correct prediction should be A, B, or C")

# Fix: if predicted D but D is nan, pick highest prob from raw probs
probs = pd.read_csv('transformer_fixed_raw_predictions.csv')
p = probs[probs['qa_id'] == 'test_0497'].iloc[0]
print(f"Raw probs: A={p['raw_prob_A']:.4f}, B={p['raw_prob_B']:.4f}, C={p['raw_prob_C']:.4f}")
best_non_d = max(['A', 'B', 'C'], key=lambda l: p[f'raw_prob_{l}'])
print(f"Best non-D: {best_non_d}")

# Apply fix
sub.loc[sub['qa_id'] == 'test_0497', 'prediction'] = best_non_d

# Also check all HARn test single where predicted D
harn_te_single_preds = sub[sub['qa_id'].isin(harn_te_single['qa_id'])]
for idx, row in harn_te_single.iterrows():
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    d_val = str(row['D']).strip()
    if pred == 'D' and (d_val == 'nan' or not d_val):
        print(f"ERROR: {row['qa_id']} predicted D but D is nan!")

sub.to_csv('submission_v249_HARN_FIX.csv', index=False)
sub.to_csv('submission.csv', index=False)
print("\nSaved to submission.csv")
