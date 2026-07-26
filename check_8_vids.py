"""
Print the single question predictions for the 8 videos with both single and object_interaction.
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
    vid_rows = te_for_vids[te_for_vids['path'] == vid]
    for idx, row in vid_rows.iterrows():
        opts = {l: str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        pred_l = str(sub[sub['qa_id'] == row['qa_id']]['prediction'].values[0]).strip()
        print(f"{row['category']}: {row['qa_id']}")
        print(f"  Options: {opts}")
        if len(pred_l) == 1:
            print(f"  Pred: {pred_l} ({opts.get(pred_l, '')})")
        else:
            print(f"  Pred: {pred_l}")

