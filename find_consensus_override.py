"""
Find questions where submission.csv strongly disagrees with the consensus of other top models.
"""
import pandas as pd
from collections import Counter

sub = pd.read_csv('submission.csv')
models = [
    'submission_v138_deep_ensemble.csv',
    'submission_v213_MEGA_ENSEMBLE.csv',
    'submission_v144_ilp_perfect.csv',
    'submission_v203_VISION_TIEBREAKER.csv',
    'submission_v229_CV_ENSEMBLE.csv'
]

# Track exactly matched leaks so we don't touch them
# We ran check_all_leaks.py before and found 31 total exact leaks
tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
def get_sig(row):
    opts = frozenset([str(row['A']).strip().lower(), str(row['B']).strip().lower(), 
                      str(row['C']).strip().lower(), str(row['D']).strip().lower()])
    return (row['category'], opts)

tr_sigs = {}
for idx, row in tr.iterrows():
    ans = str(row['answer'])
    if len(ans) > 1: continue
    sig = get_sig(row)
    if sig not in tr_sigs: tr_sigs[sig] = []
    ans_text = str(row[ans]).strip().lower()
    tr_sigs[sig].append(ans_text)

exact_leaks = set()
for idx, row in te.iterrows():
    sig = get_sig(row)
    if sig in tr_sigs:
        train_ans_texts = set(tr_sigs[sig])
        if len(train_ans_texts) == 1:
            exact_leaks.add(row['qa_id'])
            
print(f"Protected {len(exact_leaks)} exact leaks.")

model_dfs = []
for m in models:
    try:
        df = pd.read_csv(m)
        model_dfs.append(dict(zip(df['qa_id'], df['prediction'])))
    except:
        print(f"Could not read {m}")

overrides = 0
for idx, row in sub.iterrows():
    qid = row['qa_id']
    if qid in exact_leaks: continue
    
    current_pred = str(row['prediction'])
    
    # Collect predictions from other models
    other_preds = []
    for m_dict in model_dfs:
        if qid in m_dict:
            other_preds.append(str(m_dict[qid]))
            
    if not other_preds: continue
    
    # Check if there is a strong consensus against the current pred
    counts = Counter(other_preds)
    top_pred, count = counts.most_common(1)[0]
    
    # If 4 or 5 out of 5 models agree on something else!
    if top_pred != current_pred and count >= 4:
        print(f"Override {qid}: {current_pred} -> {top_pred} (Consensus: {count}/{len(model_dfs)})")
        sub.at[idx, 'prediction'] = top_pred
        overrides += 1

print(f"Total overrides: {overrides}")
sub.to_csv('submission_v268_CONSENSUS.csv', index=False)
print("Saved to submission_v268_CONSENSUS.csv")

