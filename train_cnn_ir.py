import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BATCH_SIZE = 32
EPOCHS = 8
LR = 5e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class VideoQATransformerDataset(Dataset):
    def __init__(self, csv_file, video_feature_dir, text_model_name='all-MiniLM-L6-v2', is_test=False):
        self.data = pd.read_csv(csv_file)
        self.video_feature_dir = video_feature_dir
        self.is_test = is_test
        
        print(f"Loading Text Encoder: {text_model_name}...")
        self.text_encoder = SentenceTransformer(text_model_name, device=DEVICE)
        
        self.samples = []
        self._prepare_data()

    def _prepare_data(self):
        print("Pre-embedding Text Features...")
        options = ['A', 'B', 'C', 'D']
        
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            qa_id = row['qa_id']
            
            vid_id = qa_id.split('_')[0] if '_' in qa_id and not qa_id.startswith('test') and not qa_id.startswith('training') else qa_id
            if len(vid_id) > 12 and qa_id.startswith('LM_test'):
                vid_id = '_'.join(qa_id.split('_')[:3])
            elif qa_id.startswith('test_'):
                vid_id = qa_id
            elif qa_id.startswith('training_'):
                vid_id = qa_id
                
            # THE FIX: Add _IR.pt
            vid_feat_path = os.path.join(self.video_feature_dir, f"{vid_id}_IR.pt")
            
            text_strings = [f"Category: {row['category']}. Question: {row['question']} Option: {row[opt]}" for opt in options]
            text_emb = self.text_encoder.encode(text_strings, convert_to_tensor=True, show_progress_bar=False).cpu()
            
            label = -1
            if not self.is_test and 'answer' in row:
                label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                ans = str(row['answer'])
                label = torch.zeros(4)
                for char in ans:
                    if char in label_map:
                        label[label_map[char]] = 1.0
            
            self.samples.append({
                'qa_id': row['qa_id'],
                'vid_feat_path': vid_feat_path,
                'text_features': text_emb,
                'label': label
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if os.path.exists(sample['vid_feat_path']):
            vid_feat = torch.load(sample['vid_feat_path'], weights_only=True)
        else:
            vid_feat = torch.zeros((16, 2048)) # Fallback if missing
            
        return {
            'qa_id': sample['qa_id'],
            'vid_feat': vid_feat,      # [16, 2048]
            'text_feats': sample['text_features'], # [4, 384]
            'label': sample['label']   # [4]
        }

class CNNTransformerFusionModel(nn.Module):
    def __init__(self, vid_dim=2048, txt_dim=384, hidden_dim=512, num_layers=2, nhead=8):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vid_dim, 
            nhead=nhead, 
            dim_feedforward=vid_dim, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlp = nn.Sequential(
            nn.Linear(vid_dim + txt_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, vid_seq, text_feats):
        encoded_vid_seq = self.transformer(vid_seq)
        global_vid_feat = encoded_vid_seq.mean(dim=1) 
        vid_expanded = global_vid_feat.unsqueeze(1).expand(-1, 4, -1)
        fused = torch.cat([vid_expanded, text_feats], dim=-1)
        scores = self.mlp(fused).squeeze(-1)
        return scores

def train():
    print("Initializing CNN-Transformer Training Dataset...")
    train_dataset = VideoQATransformerDataset('training_qa.csv', 'video_features_resnet', is_test=False)
    
    valid_samples = [s for s in train_dataset.samples if os.path.exists(s['vid_feat_path'])]
    print(f"Found {len(valid_samples)} valid training videos with extracted features out of {len(train_dataset.samples)}.")
    train_dataset.samples = valid_samples
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = CNNTransformerFusionModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            vid = batch['vid_feat'].to(DEVICE)
            txt = batch['text_feats'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            scores = model(vid, txt)
            loss = criterion(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), 'cnn_ir.pth')
    print("Model saved to cnn_ir.pth")
    
    print("Generating Test Predictions...")
    test_dataset = VideoQATransformerDataset('test_qa.csv', 'video_features_resnet', is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            vid = batch['vid_feat'].to(DEVICE)
            txt = batch['text_feats'].to(DEVICE)
            qa_ids = batch['qa_id']
            
            scores = model(vid, txt)
            probs = torch.sigmoid(scores).cpu().numpy()
            
            for i, qa_id in enumerate(qa_ids):
                all_preds.append({
                    'qa_id': qa_id,
                    'raw_prob_A': probs[i][0],
                    'raw_prob_B': probs[i][1],
                    'raw_prob_C': probs[i][2],
                    'raw_prob_D': probs[i][3]
                })
                
    pd.DataFrame(all_preds).to_csv('cnn_ir_raw_predictions.csv', index=False)
    print("Saved cnn_ir_raw_predictions.csv!")

if __name__ == '__main__':
    train()
