"""
Check if the 2 unordered multi leaks are already in the submission.
"""
import pandas as pd

sub = pd.read_csv('submission.csv') # this is 0.69590
q1 = str(sub.loc[sub['qa_id'] == 'test_0114', 'prediction'].values[0])
q2 = str(sub.loc[sub['qa_id'] == 'test_0176', 'prediction'].values[0])

print(f"test_0114 prediction is: {q1}")
print(f"test_0176 prediction is: {q2}")
