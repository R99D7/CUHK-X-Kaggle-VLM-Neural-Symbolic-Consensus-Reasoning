"""
Verify SINGLE predictions using COMBINATION before adding to MULTI.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

vids = [
    'large_model_track_test/LM_test_0090/Depth/Depth.mp4', # test_0017
    'large_model_track_test/LM_test_0091/Depth/Depth.mp4', # test_0018
    'large_model_track_test/LM_test_0154/Depth/Depth.mp4', # test_0060
    'large_model_track_test/LM_test_0166/Depth/Depth.mp4', # test_0069
    'large_model_track_test/LM_test_0174/Depth/Depth.mp4', # test_0077
    'large_model_track_test/LM_test_0175/Depth/Depth.mp4'  # test_0078
]

for vid in vids:
    single_q = te[(te['path'] == vid) & (te['category'] == 'single')].iloc[0]
    multi_q = te[(te['path'] == vid) & (te['category'] == 'multi')].iloc[0]
    
    comb_qs = te[(te['path'] == vid) & (te['category'] == 'combination')]
    if comb_qs.empty: continue
    comb_q = comb_qs.iloc[0]
    
    single_pred = str(sub[sub['qa_id'] == single_q['qa_id']]['prediction'].values[0]).strip()
    single_act = str(single_q[single_pred]).strip().lower()
    
    comb_pred = str(sub[sub['qa_id'] == comb_q['qa_id']]['prediction'].values[0])
    comb_opts = {l: str(comb_q[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
    comb_acts = set([a.strip() for a in comb_opts.get(comb_pred, '').split(',')])
    
    if single_act in comb_acts:
        print(f"VERIFIED! {vid} SINGLE {single_act} is in COMB {comb_q['qa_id']}")
    else:
        print(f"NOT VERIFIED. {vid} SINGLE {single_act} is NOT in COMB {comb_q['qa_id']} ({comb_acts})")
