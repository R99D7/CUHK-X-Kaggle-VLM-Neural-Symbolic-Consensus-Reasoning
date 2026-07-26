import pandas as pd
import numpy as np

def logit(p):
    # Cliping probabilities to avoid infinity
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x, axis=1, keepdims=True)) / temp)
    return e_x / e_x.sum(axis=1, keepdims=True)

def fuse():
    # Load raw probabilities
    print("Loading probabilities...")
    v11 = pd.read_csv('v11_raw_probs.csv').set_index('qa_id')
    cv = pd.read_csv('crossencoder_cv_raw_predictions.csv').set_index('qa_id')
    test_df = pd.read_csv('test_qa.csv')
    sample = pd.read_csv('sample_submission.csv')
    
    expected_lengths = dict(zip(sample['qa_id'], sample['prediction'].apply(lambda x: max(1, len(str(x))))))
    
    # We apply inverse sigmoid (logit) to get the original raw model scores
    # v11 and cv both use independent binary classifications
    cv_logits_A = logit(cv['raw_prob_A'].values)
    cv_logits_B = logit(cv['raw_prob_B'].values)
    cv_logits_C = logit(cv['raw_prob_C'].values)
    cv_logits_D = logit(cv['raw_prob_D'].values)
    
    v11_logits_A = logit(v11['prob_A'].values)
    v11_logits_B = logit(v11['prob_B'].values)
    v11_logits_C = logit(v11['prob_C'].values)
    v11_logits_D = logit(v11['prob_D'].values)
    
    cv_logits = np.column_stack([cv_logits_A, cv_logits_B, cv_logits_C, cv_logits_D])
    v11_logits = np.column_stack([v11_logits_A, v11_logits_B, v11_logits_C, v11_logits_D])
    
    # We weight the models. v11 is very strong (0.47+). Crossencoder CV should be highly regularized.
    # Let's use 60% v11 and 40% CV. 
    W_V11 = 0.6
    W_CV = 0.4
    
    fused_logits = W_V11 * v11_logits + W_CV * cv_logits
    
    # Apply softmax to get proper normalized probabilities
    # (Optional, but since we pick top-K, ordering by logits or softmax is identical for independent choices)
    fused_probs = softmax(fused_logits, temp=1.0)
    
    letters = ['A', 'B', 'C', 'D']
    final_preds = []
    
    for idx, row in test_df.iterrows():
        qa_id = row['qa_id']
        category = row['category']
        expected_len = expected_lengths.get(qa_id, 1)
        
        # Get the row index in the dataframes
        row_idx = v11.index.get_loc(qa_id)
        
        # Scores for A, B, C, D
        scores = list(zip(fused_probs[row_idx], letters))
        scores.sort(key=lambda x: x[0], reverse=True)
        
        pred_letters = [l for p, l in scores[:expected_len]]
        
        if category == 'sequence':
            pred_str = "".join(pred_letters)
        else:
            pred_letters.sort()
            pred_str = "".join(pred_letters)
            
        final_preds.append({'qa_id': qa_id, 'prediction': pred_str})
        
    out = pd.DataFrame(final_preds)
    out.to_csv('submission_v229_CV_ENSEMBLE.csv', index=False)
    print("Generated submission_v229_CV_ENSEMBLE.csv!")

if __name__ == '__main__':
    fuse()
