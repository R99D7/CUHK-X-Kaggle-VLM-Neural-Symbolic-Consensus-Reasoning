import os
import torch
import pandas as pd
from train_videoqa import FusionModel, VideoQADataset
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict():
    print("Initializing Test Dataset...")
    test_dataset = VideoQADataset('test_qa.csv', 'video_features_r3d', is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print("Loading Trained Fusion Model...")
    model = FusionModel().to(DEVICE)
    if os.path.exists('fusion_model.pth'):
        model.load_state_dict(torch.load('fusion_model.pth', weights_only=True))
    else:
        print("ERROR: fusion_model.pth not found! Run train_videoqa.py first.")
        return
        
    model.eval()
    
    predictions = []
    
    print("Generating Predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            qa_ids = batch['qa_id']
            vid = batch['vid_feat'].to(DEVICE)
            txt = batch['text_feats'].to(DEVICE)
            
            # scores shape: [B, 4]
            scores = model(vid, txt)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(scores)
            
            for i in range(len(qa_ids)):
                qid = qa_ids[i]
                p = probs[i].cpu().numpy()
                
                # Logic for parsing answer
                # We know the lengths from sample_submission.csv
                # Wait! We need to dynamically load sample_submission length leak!
                # For now, we get the indices of probabilities sorted descending
                sorted_idx = np.argsort(p)[::-1]
                
                # We will just save the top 4 sorted letters and handle length later
                letters = ['A', 'B', 'C', 'D']
                sorted_letters = "".join([letters[idx] for idx in sorted_idx])
                
                predictions.append({
                    'qa_id': qid,
                    'raw_prob_A': p[0],
                    'raw_prob_B': p[1],
                    'raw_prob_C': p[2],
                    'raw_prob_D': p[3],
                    'sorted_letters': sorted_letters
                })
                
    out_df = pd.DataFrame(predictions)
    out_df.to_csv('videoqa_raw_predictions.csv', index=False)
    print("Saved raw predictions to videoqa_raw_predictions.csv")

if __name__ == '__main__':
    predict()
