import pandas as pd
import numpy as np

print("Loading PyTorch VideoQA Raw Predictions...")
try:
    raw = pd.read_csv('videoqa_raw_predictions.csv')
except FileNotFoundError:
    print("Cannot build v29. videoqa_raw_predictions.csv not found.")
    exit(1)

print("Loading test categories and sample submission...")
test = pd.read_csv('test_qa.csv')
sample = pd.read_csv('sample_submission.csv')

raw = raw.merge(test[['qa_id', 'category']], on='qa_id').merge(sample[['qa_id', 'prediction']], on='qa_id', suffixes=('', '_sample'))

print("Loading Best Baseline (0.42397)...")
v20 = pd.read_csv('submission_oracle_v20.csv')
v20_dict = dict(zip(v20['qa_id'], v20['prediction']))

final_preds = []
changed_from_baseline = 0

for _, row in raw.iterrows():
    qid = row['qa_id']
    cat = row['category']
    baseline_pred = v20_dict[qid]
    
    # Expected length
    exp_len = len(str(row['prediction_sample']))
    if exp_len == 0 or str(row['prediction_sample']) == 'nan': exp_len = 1
    
    sorted_letters = row['sorted_letters'] # e.g. "CABD"
    
    # 1. Truncate to expected length
    dl_pred = sorted_letters[:exp_len]
    
    # 2. Sort letters alphabetically for combinations (e.g. CA -> AC)
    if cat in ['multi', 'combination'] and exp_len > 1:
        dl_pred = "".join(sorted(list(dl_pred)))
        
    final_pred = baseline_pred
    
    # We will ONLY trust the Deep Learning model for the 'emotion' and 'combination' categories
    # because these rely heavily on motion/heat distributions that Text-only ML struggles with.
    if cat in ['emotion', 'combination']:
        if dl_pred != baseline_pred:
            final_pred = dl_pred
            changed_from_baseline += 1
            
    final_preds.append({'qa_id': qid, 'prediction': final_pred})
    
out = pd.DataFrame(final_preds)
out.to_csv('submission_v29_videoqa.csv', index=False)

print(f"Built v29 VideoQA Submission!")
print(f"Changed {changed_from_baseline} predictions from the 0.42397 baseline (targeting Emotion & Combination).")
