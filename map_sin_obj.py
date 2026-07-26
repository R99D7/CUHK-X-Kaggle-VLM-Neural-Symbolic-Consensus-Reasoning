"""
Map single and object_interaction for the 8 overlapping videos.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

vids = [
    'large_model_track_test/LM_test_0013/Depth/Depth.mp4',
    'large_model_track_test/LM_test_0023/Depth/Depth.mp4',
    'large_model_track_test/LM_test_0029/Depth/Depth.mp4',
    'large_model_track_test/LM_test_0032/Depth/Depth.mp4',
    'large_model_track_test/LM_test_0033/Depth/Depth.mp4',
    'large_model_track_test/LM_test_0035/Depth/Depth.mp4',
    'large_model_track_test/LM_test_0055/Depth/Depth.mp4',
    'large_model_track_test/LM_test_0061/Depth/Depth.mp4'
]

te_for_vids = te[te['path'].isin(vids)]

for vid in vids:
    print(f"\n--- {vid} ---")
    
    sin_q = te_for_vids[(te_for_vids['path'] == vid) & (te_for_vids['category'] == 'single')].iloc[0]
    obj_q = te_for_vids[(te_for_vids['path'] == vid) & (te_for_vids['category'] == 'object_interaction')].iloc[0]
    
    sin_opts = {l: str(sin_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D'] if str(sin_q[l]) != 'nan'}
    obj_opts = {l: str(obj_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D'] if str(obj_q[l]) != 'nan'}
    
    sin_pred = str(sub[sub['qa_id'] == sin_q['qa_id']]['prediction'].values[0]).strip()
    obj_pred = str(sub[sub['qa_id'] == obj_q['qa_id']]['prediction'].values[0]).strip()
    
    print(f"SINGLE ({sin_q['qa_id']}): Pred={sin_pred} ({sin_opts.get(sin_pred, '')})")
    for l, txt in sin_opts.items():
        print(f"  {l}: {txt}")
        
    print(f"OBJECT ({obj_q['qa_id']}): Pred={obj_pred} ({obj_opts.get(obj_pred, '')})")
    for l, txt in obj_opts.items():
        print(f"  {l}: {txt}")
