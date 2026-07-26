import pandas as pd
from collections import Counter

files = {
    'v117': ('submission_v117_ultimate_multimodal.csv', 3.0),
    'v118': ('submission_v118_ultimate_multimodal_055.csv', 2.0),
    'v116': ('submission_v116_ultimate_safe_threshold_040.csv', 2.0),
    'v114': ('submission_v114_ultimate_safe_dual_agreement.csv', 1.5),
    'v113': ('submission_v113_dual_agreement_override.csv', 1.5),
    'v112': ('submission_v112_ultimate_crossencoder_tfidf_blend.csv', 1.0),
    'v110': ('submission_v110_real_pytorch_override.csv', 1.0),
    'v105': ('submission_v105_ultimate_soft_ensemble.csv', 1.0),
}

test_df = pd.read_csv('test_qa.csv').set_index('qa_id')
dfs = {k: (pd.read_csv(v).set_index('qa_id'), w) for k, (v, w) in files.items()}
ref = dfs['v117'][0]
new_sub = pd.read_csv('submission_v132_weighted_ensemble.csv').set_index('qa_id')

print("=== 19 Overrides from v117 ===")
for qa_id in ref.index:
    v117_pred = str(ref.loc[qa_id, 'prediction'])
    new_pred = str(new_sub.loc[qa_id, 'prediction'])
    if v117_pred != new_pred:
        cat = test_df.loc[qa_id, 'category'] if qa_id in test_df.index else '?'
        vote = Counter()
        for k, (df, w) in dfs.items():
            vote[str(df.loc[qa_id, 'prediction'])] += w
        print(f"{qa_id} | cat={cat} | v117={v117_pred} -> new={new_pred} | votes={dict(vote)}")
