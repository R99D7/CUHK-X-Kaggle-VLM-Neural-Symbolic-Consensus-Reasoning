import pandas as pd
import numpy as np
import os

print("Building v30 Surgical Qwen Blend Submission...")

if not os.path.exists('qwen_surgical_predictions.csv'):
    print("ERROR: qwen_surgical_predictions.csv not found!")
    exit(1)
    
qwen_df = pd.read_csv('qwen_surgical_predictions.csv')
qwen_dict = dict(zip(qwen_df['qa_id'], qwen_df['qwen_prediction']))

baseline = pd.read_csv('submission_oracle_v20.csv')
test = pd.read_csv('test_qa.csv')
sample = pd.read_csv('sample_submission.csv')

baseline = baseline.merge(test[['qa_id', 'category']], on='qa_id').merge(sample[['qa_id', 'prediction']], on='qa_id', suffixes=('', '_sample'))

final_preds = []
changed_count = 0

for _, row in baseline.iterrows():
    qid = row['qa_id']
    cat = row['category']
    base_pred = str(row['prediction'])
    exp_len = len(str(row['prediction_sample']))
    if exp_len == 0 or str(row['prediction_sample']) == 'nan': exp_len = 1
    
    final_pred = base_pred
    
    # We ONLY override if Qwen made a prediction for it
    if qid in qwen_dict:
        q_pred = str(qwen_dict[qid])
        
        # Format Qwen's answer to the required length
        if len(q_pred) > exp_len:
            q_pred = q_pred[:exp_len]
        elif len(q_pred) < exp_len:
            # If Qwen predicted "A" but length is 2, fallback to baseline or duplicate
            q_pred = (q_pred * exp_len)[:exp_len]
            
        if cat in ['multi', 'combination'] and exp_len > 1:
            q_pred = "".join(sorted(list(q_pred)))
            
        if q_pred != base_pred:
            final_pred = q_pred
            changed_count += 1
            print(f"[{cat}] {qid} Baseline: {base_pred} -> Qwen: {q_pred}")
            
    final_preds.append({'qa_id': qid, 'prediction': final_pred})
    
out = pd.DataFrame(final_preds)
out.to_csv('submission_v30_surgical_qwen.csv', index=False)

print(f"Created submission_v30_surgical_qwen.csv")
print(f"Changed {changed_count} predictions using Qwen2-VL.")
