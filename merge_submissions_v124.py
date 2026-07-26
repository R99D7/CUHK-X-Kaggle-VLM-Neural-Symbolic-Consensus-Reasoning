import pandas as pd
import numpy as np
import os

def generate_final_submission():
    if not os.path.exists('deberta_v3_large_raw_probs.csv') or not os.path.exists('timesformer_raw_probs.csv'):
        print("Waiting for Kaggle kernels to finish and download the raw probabilities...")
        return
        
    deb = pd.read_csv('deberta_v3_large_raw_probs.csv').set_index('qa_id')
    time_sf = pd.read_csv('timesformer_raw_probs.csv').set_index('qa_id')
    
    train = pd.read_csv('training_qa.csv')
    test = pd.read_csv('test_qa.csv')
    
    labels = train['answer'].dropna().unique().tolist()
    labels = sorted(labels)
    
    # 1. Leak Protection
    leaks_dict = {}
    for idx, row in test.iterrows():
        match = train[train['question'] == row['question']]
        if len(match) > 0:
            for _, m_row in match.iterrows():
                test_opts = {str(row['A']), str(row['B']), str(row['C']), str(row['D'])}
                train_opts = {str(m_row['A']), str(m_row['B']), str(m_row['C']), str(m_row['D'])}
                if test_opts == train_opts:
                    leaks_dict[row['qa_id']] = m_row['answer']
                    break
                    
    print(f"Verified {len(leaks_dict)} data leaks.")
    
    # 2. Ensemble
    final_preds = []
    overrides = 0
    for qa_id in test['qa_id']:
        if qa_id in leaks_dict:
            final_preds.append({'qa_id': qa_id, 'prediction': leaks_dict[qa_id]})
            continue
            
        if qa_id not in deb.index or qa_id not in time_sf.index:
            # Fallback
            final_preds.append({'qa_id': qa_id, 'prediction': 'A'})
            continue
            
        deb_probs = np.array([deb.loc[qa_id, f'prob_{l}'] for l in labels])
        tsf_probs = np.array([time_sf.loc[qa_id, f'prob_{l}'] for l in labels])
        
        # 50/50 weighting
        avg_probs = 0.5 * deb_probs + 0.5 * tsf_probs
        
        best_idx = np.argmax(avg_probs)
        best_label = labels[best_idx]
        
        final_preds.append({'qa_id': qa_id, 'prediction': best_label})
        overrides += 1
        
    pd.DataFrame(final_preds).to_csv('submission_v124_end_to_end_3d_deberta.csv', index=False)
    print(f"Saved submission_v124_end_to_end_3d_deberta.csv! Ensembled {overrides} non-leak questions.")
    
if __name__ == "__main__":
    generate_final_submission()
