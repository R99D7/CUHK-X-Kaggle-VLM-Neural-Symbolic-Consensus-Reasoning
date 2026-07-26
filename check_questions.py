"""
Deep analysis: For every test SINGLE question, use the question text itself as a hint.
The question may specify WHICH action is being asked about!
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')  # v246

# Look at the question text for single category
te_single = te[te['category'] == 'single']
for idx, row in te_single.head(20).iterrows():
    print(f"{row['qa_id']}: {row['question']}")
    print(f"  A: {row['A']}  B: {row['B']}  C: {row['C']}  D: {row['D']}")
    pred = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0])
    print(f"  Predicted: {pred}\n")
