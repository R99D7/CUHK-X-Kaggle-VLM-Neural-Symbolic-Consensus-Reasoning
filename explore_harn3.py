"""
The HARn training paths contain the ACTION NAME in the path!
e.g. 'HARn/0_Wash_face/user16/1-1-2'

This means for HARn training, we know exactly which action is being performed!
But the test HARn paths don't show the action... BUT:
The LM_test numbers might correspond to specific actions.
Let's check if we can match HARn test single questions to training by the action
category (the action label is embedded in the training path).
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

harn_tr = tr[tr['source'] == 'HARn']
harn_te = te[te['source'] == 'HARn']

# Extract action from training path
harn_tr = harn_tr.copy()
harn_tr['action'] = harn_tr['path'].apply(lambda x: x.split('/')[1] if 'HARn/' in x else '')

print("HARn training actions:")
print(harn_tr['action'].value_counts())

# For HARn single questions - the question asks "which action?"
# The training answer IS the action that appears in the path!
# So for HARn test: the option that matches the path's action label is the answer

# For test: we don't have the action in the path.
# But: if the EXACT SAME options appear in training, we can use that!

# Check test HARn single questions against train
harn_tr_single = harn_tr[harn_tr['category'] == 'single']
harn_te_single = harn_te[harn_te['category'] == 'single']

print(f"\nHARn train single: {len(harn_tr_single)}")
print(f"HARn test single: {len(harn_te_single)}")

# Show first few test HARn single questions
for idx, row in harn_te_single.head(5).iterrows():
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
    print(f"\n{row['qa_id']} path={row['path']}")
    print(f"  A:{row['A']} B:{row['B']} C:{row['C']} D:{row['D']} -> pred={pred}")
