"""
Apply the cascading SINGLE fixes.
"""
import pandas as pd

sub = pd.read_csv('submission.csv')
sub.loc[sub['qa_id'] == 'test_0074', 'prediction'] = 'C'
sub.loc[sub['qa_id'] == 'test_0075', 'prediction'] = 'D'
sub.to_csv('submission.csv', index=False)
print("Applied cascading SINGLE fixes for test_0074 and test_0075.")
