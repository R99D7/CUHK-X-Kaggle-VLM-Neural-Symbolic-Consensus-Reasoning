import pandas as pd
import numpy as np

cnn = pd.read_csv('transformer_raw_predictions.csv')
v46 = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')
sample = pd.read_csv('sample_submission.csv')
test = pd.read_csv('test_qa.csv')

diffs = 0
cnn_preds = []
for idx, row in cnn.iterrows():
    qa_id = row['qa_id']
    v46_pred = str(v46[v46['qa_id'] == qa_id]['prediction'].iloc[0])
    
    probs = {'A': row['raw_prob_A'], 'B': row['raw_prob_B'], 'C': row['raw_prob_C'], 'D': row['raw_prob_D']}
    sorted_letters = sorted(probs, key=probs.get, reverse=True)
    
    expected_len = len(str(sample[sample['qa_id'] == qa_id]['prediction'].iloc[0]))
    if expected_len == 0: expected_len = 1
    
    cat = test[test['qa_id'] == qa_id]['category'].iloc[0]
    
    pred_letters = sorted_letters[:expected_len]
    if cat != 'sequence':
        pred_letters.sort()
    
    pred = "".join(pred_letters)
    cnn_preds.append({'qa_id': qa_id, 'prediction': pred})
    
    if pred != v46_pred:
        diffs += 1

print('Pure CNN-Transformer differences from v46:', diffs)
pd.DataFrame(cnn_preds).to_csv('submission_v107_pure_cnn.csv', index=False)
