import pandas as pd

print("Loading V32 Multi-Modal CV Ensembled Predictions...")
try:
    raw = pd.read_csv('v32_multimodal_cv_predictions.csv')
except FileNotFoundError:
    print("Cannot build v32. v32_multimodal_cv_predictions.csv not found.")
    exit(1)

test = pd.read_csv('test_qa.csv')
sample = pd.read_csv('sample_submission.csv')

raw = raw.merge(test[['qa_id', 'category']], on='qa_id').merge(sample[['qa_id', 'prediction']], on='qa_id')

print("Loading Best Baseline (0.42397)...")
v20 = pd.read_csv('submission_oracle_v20.csv')
v20_dict = dict(zip(v20['qa_id'], v20['prediction']))

final_preds = []
changed_from_baseline = 0

for _, row in raw.iterrows():
    qid = row['qa_id']
    cat = row['category']
    baseline_pred = v20_dict[qid]
    
    exp_len = len(str(row['prediction']))
    if exp_len == 0 or str(row['prediction']) == 'nan': exp_len = 1
    
    sorted_letters = row['sorted_letters']
    dl_pred = sorted_letters[:exp_len]
    
    if cat in ['multi', 'combination'] and exp_len > 1:
        dl_pred = "".join(sorted(list(dl_pred)))
        
    final_pred = baseline_pred
    
    if cat in ['emotion', 'combination', 'single', 'multi']:
        if dl_pred != baseline_pred:
            final_pred = dl_pred
            changed_from_baseline += 1
            
    final_preds.append({'qa_id': qid, 'prediction': final_pred})
    
out = pd.DataFrame(final_preds)
out.to_csv('submission_v32_ultimate_blend.csv', index=False)

print(f"Built v32 Ultimate Multi-Modal Submission!")
print(f"Changed {changed_from_baseline} predictions from the baseline utilizing the robust 5-Fold Cross Validated model!")
