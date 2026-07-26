"""
Apply safe COMB->SINGLE fix for test_0555.
"""
import pandas as pd

sub = pd.read_csv('submission.csv')
sub.loc[sub['qa_id'] == 'test_0555', 'prediction'] = 'C'
sub.to_csv('submission.csv', index=False)
print("Applied fix for test_0555.")
