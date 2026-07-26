"""
Check if combination actions for the 4 videos are in multi prediction.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

vids = [
    'large_model_track_test/LM_test_0098/Depth/Depth.mp4',
    'large_model_track_test/LM_test_0170/Depth/Depth.mp4',
    'large_model_track_test/LM_test_0176/Depth/Depth.mp4',
    'large_model_track_test/LM_test_0204/Depth/Depth.mp4'
]

for vid in vids:
    print(f"\n--- {vid} ---")
    comb_q = te[(te['path'] == vid) & (te['category'] == 'combination')].iloc[0]
    multi_q = te[(te['path'] == vid) & (te['category'] == 'multi')].iloc[0]
    
    comb_pred = str(sub[sub['qa_id'] == comb_q['qa_id']]['prediction'].values[0])
    comb_opts = {l: str(comb_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    comb_acts = set([a.strip() for a in comb_opts.get(comb_pred, '').split(',')])
    
    multi_pred = str(sub[sub['qa_id'] == multi_q['qa_id']]['prediction'].values[0])
    multi_opts = {l: str(multi_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    
    print(f"COMB pred: {comb_pred} -> {comb_acts}")
    print(f"MULTI pred: {multi_pred}")
    for l, txt in multi_opts.items():
        if txt in comb_acts:
            status = 'YES' if l in multi_pred else 'MISSING!'
            print(f"  {l} ({txt}) -> {status}")
