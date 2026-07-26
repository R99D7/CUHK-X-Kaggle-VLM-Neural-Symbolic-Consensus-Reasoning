import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from train_v32_multimodal_cv import MultiModalQADataset, CrossAttentionMultiModalFusion, precompute_text_embeddings

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading Test Data...")
test_df = pd.read_csv('test_qa.csv')
text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

# We just need to precompute embeddings for test set for inference
all_embeddings = precompute_text_embeddings([test_df], text_encoder)

test_dataset = MultiModalQADataset(test_df, 'video_features_resnet', all_embeddings)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

test_predictions = []

for fold in range(2):
    model = CrossAttentionMultiModalFusion().to(DEVICE)
    model.load_state_dict(torch.load(f'model_fold{fold}.pth', weights_only=True))
    model.eval()
    
    fold_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Predicting Test Fold {fold+1}/2"):
            qa_ids = batch['qa_id']
            vid = batch['vid_feat_stack'].to(DEVICE)
            txt = batch['text_feats'].to(DEVICE)
            scores = model(vid, txt)
            probs = torch.sigmoid(scores)
            
            for i in range(len(qa_ids)):
                fold_preds.append({
                    'qa_id': qa_ids[i],
                    f'prob_A_f{fold}': probs[i][0].item(),
                    f'prob_B_f{fold}': probs[i][1].item(),
                    f'prob_C_f{fold}': probs[i][2].item(),
                    f'prob_D_f{fold}': probs[i][3].item(),
                })
    test_predictions.append(pd.DataFrame(fold_preds))
    
final_pred_df = test_predictions[0]
final_pred_df = final_pred_df.merge(test_predictions[1], on='qa_id')
    
final_pred_df['prob_A'] = final_pred_df[[f'prob_A_f{i}' for i in range(2)]].mean(axis=1)
final_pred_df['prob_B'] = final_pred_df[[f'prob_B_f{i}' for i in range(2)]].mean(axis=1)
final_pred_df['prob_C'] = final_pred_df[[f'prob_C_f{i}' for i in range(2)]].mean(axis=1)
final_pred_df['prob_D'] = final_pred_df[[f'prob_D_f{i}' for i in range(2)]].mean(axis=1)

sample_df = pd.read_csv('sample_submission.csv')
sample_lens = {row['qa_id']: len(str(row['prediction'])) for _, row in sample_df.iterrows()}

final_res = []
for _, row in final_pred_df.iterrows():
    qa_id = row['qa_id']
    expected_len = sample_lens.get(qa_id, 1)
    
    p = [row['prob_A'], row['prob_B'], row['prob_C'], row['prob_D']]
    sorted_idx = np.argsort(p)[::-1]
    letters = ['A', 'B', 'C', 'D']
    
    if expected_len == 4:
        # Full permutation ordered by confidence
        pred = "".join([letters[idx] for idx in sorted_idx])
    elif expected_len == 1:
        # Single best letter
        pred = letters[sorted_idx[0]]
    else:
        # Multiple choice (e.g. 2 or 3 answers) - take top K and sort alphabetically
        top_k = [letters[idx] for idx in sorted_idx[:expected_len]]
        pred = "".join(sorted(top_k))
        
    final_res.append({'qa_id': qa_id, 'prediction': pred})
    
pd.DataFrame(final_res).to_csv('submission_v32_2fold_safety.csv', index=False)
print("Saved 2-Fold Ensembled Predictions to submission_v32_2fold_safety.csv")
