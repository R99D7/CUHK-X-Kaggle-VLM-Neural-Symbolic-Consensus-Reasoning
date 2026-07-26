"""
Print COMB pred for LM_test_0101.
"""
import pandas as pd
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')
vid = 'large_model_track_test/LM_test_0101/Depth/Depth.mp4'
comb_q = te[(te['path']==vid) & (te['category']=='combination')].iloc[0]
print(f"COMB qa_id: {comb_q['qa_id']}")
print('COMB options:')
for l in ['A','B','C','D']: print(f"  {l}: {comb_q[l]}")
print('COMB pred:', sub[sub['qa_id'] == comb_q['qa_id']]['prediction'].values[0])
