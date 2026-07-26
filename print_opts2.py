"""
Check single vs multi options for test_0064.
"""
import pandas as pd
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')
vid = 'large_model_track_test/LM_test_0160/Depth/Depth.mp4'

sin_q = te[(te['path'] == vid) & (te['category'] == 'single')].iloc[0]
mul_q = te[(te['path'] == vid) & (te['category'] == 'multi')].iloc[0]

print('SINGLE options:')
for l in ['A', 'B', 'C', 'D']:
    print(f'  {l}: {sin_q[l]}')
print(f'SINGLE pred: {sub[sub["qa_id"] == sin_q["qa_id"]]["prediction"].values[0]}')

print('\nMULTI options:')
for l in ['A', 'B', 'C', 'D']:
    print(f'  {l}: {mul_q[l]}')
print(f'MULTI pred: {sub[sub["qa_id"] == mul_q["qa_id"]]["prediction"].values[0]}')
