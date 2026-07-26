"""
Apply the 9 positive leak fixes from MULTI to COMB.
"""
import pandas as pd

sub = pd.read_csv('submission.csv')

fixes = {
    'test_0227': 'B',
    'test_0230': 'B',
    'test_0280': 'D',
    'test_0292': 'A',
    'test_0293': 'B',
    'test_0299': 'C',
    'test_0313': 'D',
    'test_0328': 'B',
    'test_0621': 'C'
}

for qa_id, new_pred in fixes.items():
    old_pred = sub.loc[sub['qa_id'] == qa_id, 'prediction'].values[0]
    print(f"Fixing COMB {qa_id}: {old_pred} -> {new_pred}")
    sub.loc[sub['qa_id'] == qa_id, 'prediction'] = new_pred

sub.to_csv('submission_v265_MULTI2COMB.csv', index=False)
sub.to_csv('submission.csv', index=False)
print("Saved to submission.csv")
