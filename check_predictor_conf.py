"""
Check confidence of the multi and comb predictions causing the fixes.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')
raw = pd.read_csv('transformer_fixed_raw_predictions.csv')

def check_conf(qa_id, pred_label):
    if len(pred_label) > 1: return None
    r = raw[raw['qa_id'] == qa_id]
    if r.empty: return None
    return r.iloc[0][f'raw_prob_{pred_label}']

print("Fixes from MULTI:")
print(f"test_0064 MULTI predictor: test_0175 (vid: LM_test_0160)")
mul_q = te[(te['path'] == 'large_model_track_test/LM_test_0160/Depth/Depth.mp4') & (te['category'] == 'multi')].iloc[0]
mul_pred = sub[sub['qa_id'] == mul_q['qa_id']]['prediction'].values[0]
print(f"  MULTI pred: {mul_pred}, conf: {check_conf(mul_q['qa_id'], mul_pred)}")

print("\nFixes from COMBINATION:")
vids = {
    'test_0021': 'large_model_track_test/LM_test_0098/Depth/Depth.mp4',
    'test_0073': 'large_model_track_test/LM_test_0170/Depth/Depth.mp4',
    'test_0079': 'large_model_track_test/LM_test_0176/Depth/Depth.mp4',
    'test_0107': 'large_model_track_test/LM_test_0204/Depth/Depth.mp4',
}

for single_id, vid in vids.items():
    comb_q = te[(te['path'] == vid) & (te['category'] == 'combination')].iloc[0]
    comb_pred = sub[sub['qa_id'] == comb_q['qa_id']]['prediction'].values[0]
    print(f"{single_id} COMB predictor: {comb_q['qa_id']}")
    print(f"  COMB pred: {comb_pred}, conf: {check_conf(comb_q['qa_id'], comb_pred)}")

