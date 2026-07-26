"""
Inspect the 4 single atomic candidates (test_0111, test_0558, test_0507, test_0089) in context of their full video clips.
"""
import pandas as pd

sub276 = pd.read_csv("submission_v276_APEX_SUMMIT.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))
te = pd.read_csv("test_qa.csv")
sub_map276 = dict(zip(sub276['qa_id'], sub276['prediction']))
te['pred'] = te['qa_id'].map(sub_map276)

grouped = te.groupby('path')

target_qids = ['test_0111', 'test_0558', 'test_0507', 'test_0089']

for qid in target_qids:
    row = te[te['qa_id'] == qid].iloc[0]
    path = row['path']
    grp = te[te['path'] == path]
    print(f"\n==================== QA ID: {qid} (Path: {path}) ====================")
    for idx, r in grp.iterrows():
        q_id = r['qa_id']
        cat = r['category']
        p = r['pred']
        opts = {l: r[l] for l in ['A', 'B', 'C', 'D']}
        probs = {l: round(raw_map.get(q_id, {}).get(f'raw_prob_{l}', 0.0), 3) for l in ['A', 'B', 'C', 'D']}
        print(f"  [{cat.upper()}] {q_id}: Pred='{p}' | Opts={opts} | Probs={probs}")
