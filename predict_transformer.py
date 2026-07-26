import os
import torch
import pandas as pd
import numpy as np
from train_transformer_videoqa import CNNTransformerFusionModel, VideoQATransformerDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict():
    print("Initializing CNN-Transformer Test Dataset...")
    test_dataset = VideoQATransformerDataset('test_qa.csv', 'video_features_resnet', is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("Loading Trained CNN-Transformer Fusion Model...")
    model = CNNTransformerFusionModel().to(DEVICE)
    if os.path.exists('cnn_transformer_model.pth'):
        model.load_state_dict(torch.load('cnn_transformer_model.pth', map_location=DEVICE, weights_only=True))
    else:
        print("ERROR: cnn_transformer_model.pth not found! Run train_transformer_videoqa.py first.")
        return
        
    model.eval()
    
    predictions = []
    
    print("Generating Predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            qa_ids = batch['qa_id']
            vid = batch['vid_feat'].to(DEVICE)
            txt = batch['text_feats'].to(DEVICE)
            
            scores = model(vid, txt)
            probs = torch.sigmoid(scores)
            
            for i in range(len(qa_ids)):
                qid = qa_ids[i]
                p = probs[i].cpu().numpy()
                
                sorted_idx = np.argsort(p)[::-1]
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
    out_df.to_csv('transformer_raw_predictions.csv', index=False)
    print("Saved raw predictions to transformer_raw_predictions.csv")

if __name__ == '__main__':
    predict()
