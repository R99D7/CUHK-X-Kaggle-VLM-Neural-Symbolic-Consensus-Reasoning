import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configurations
BATCH_SIZE = 32
EPOCHS = 10
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
            vid_id = row['qa_id'].split('_')[0] if '_' in row['qa_id'] and not row['qa_id'].startswith('test') and not row['qa_id'].startswith('train') else row['qa_id']
            if len(vid_id) > 12:
                vid_id = '_'.join(row['qa_id'].split('_')[:3])
                
            vid_feat_path = os.path.join(self.video_feature_dir, f"{vid_id}.pt")
            
            text_strings = [f"Question: {row['question']} Option: {row[opt]}" for opt in options]
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
        
        # Load Video Feature Sequence (Shape: [16, 2048])
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
        
        # Transformer Encoder for Temporal Modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vid_dim, 
            nhead=nhead, 
            dim_feedforward=vid_dim, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP for Multi-Modal Late Fusion
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
        # vid_seq: [B, 16, 2048]
        # text_feats: [B, 4, 384]
        
        B = vid_seq.size(0)
        
        # 1. Temporal Modeling
        # Pass sequence through Transformer: [B, 16, 2048]
        encoded_vid_seq = self.transformer(vid_seq)
        
        # Global Average Pooling over time to get a single vector per video
        # [B, 2048]
        global_vid_feat = encoded_vid_seq.mean(dim=1) 
        
        # 2. Cross-Modal Fusion
        # Expand video features for each of the 4 text options
        vid_expanded = global_vid_feat.unsqueeze(1).expand(-1, 4, -1) # [B, 4, 2048]
        
        fused = torch.cat([vid_expanded, text_feats], dim=-1) # [B, 4, 2432]
        
        # 3. Scoring
        scores = self.mlp(fused).squeeze(-1) # [B, 4]
        return scores

def train():
    print("Initializing CNN-Transformer Training Dataset...")
    train_dataset = VideoQATransformerDataset('training_qa.csv', 'video_features_resnet', is_test=False)
    
    # We filter out missing videos for training
    valid_samples = [s for s in train_dataset.samples if os.path.exists(s['vid_feat_path'])]
    print(f"Found {len(valid_samples)} valid training videos with extracted features.")
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
            
            vid = batch['vid_feat'].to(DEVICE) # [B, 16, 2048]
            txt = batch['text_feats'].to(DEVICE) # [B, 4, 384]
            labels = batch['label'].to(DEVICE) # [B, 4]
            
            scores = model(vid, txt)
            loss = criterion(scores, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), 'cnn_transformer_model.pth')
    print("Model saved to cnn_transformer_model.pth")

if __name__ == '__main__':
    train()
