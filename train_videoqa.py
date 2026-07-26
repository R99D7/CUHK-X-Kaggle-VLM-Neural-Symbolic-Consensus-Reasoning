import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configurations
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Dataset Definition
class VideoQADataset(Dataset):
    def __init__(self, csv_file, video_feature_dir, text_model_name='all-MiniLM-L6-v2', is_test=False):
        self.data = pd.read_csv(csv_file)
        self.video_feature_dir = video_feature_dir
        self.is_test = is_test
        
        # Load SentenceTransformer for text embedding
        print(f"Loading Text Encoder: {text_model_name}...")
        self.text_encoder = SentenceTransformer(text_model_name, device=DEVICE)
        
        self.samples = []
        self._prepare_data()

    def _prepare_data(self):
        print("Pre-embedding Text Features...")
        
        # We need to process each question and its 4 options
        options = ['A', 'B', 'C', 'D']
        
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            vid_id = row['qa_id'].split('_')[0] if '_' in row['qa_id'] and not row['qa_id'].startswith('test') and not row['qa_id'].startswith('train') else row['qa_id']
            # Fallback for naming conventions (e.g. 'LM_test_0001_1' -> 'LM_test_0001')
            if len(vid_id) > 12: # LM_test_0001 is 12 chars
                vid_id = '_'.join(row['qa_id'].split('_')[:3])
                
            vid_feat_path = os.path.join(self.video_feature_dir, f"{vid_id}.pt")
            
            # Combine Question + Option into 4 strings
            text_strings = [f"Question: {row['question']} Option: {row[opt]}" for opt in options]
            
            # Embed text (Shape: [4, 384])
            text_emb = self.text_encoder.encode(text_strings, convert_to_tensor=True, show_progress_bar=False).cpu()
            
            label = -1
            if not self.is_test and 'answer' in row:
                label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                # Handle cases where answer might be multi-choice (e.g., 'ABC')
                # For categorical CE, we need a single target. If multi, we might skip or treat differently.
                # For simplicity, if it's multi-choice, we take the first char or skip.
                # Since the model predicts a single choice score, we will use BCEWithLogitsLoss for multi-label!
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
        
        # Load Video Feature (Shape: [512])
        if os.path.exists(sample['vid_feat_path']):
            vid_feat = torch.load(sample['vid_feat_path'], weights_only=True)
        else:
            vid_feat = torch.zeros(512) # Fallback if missing
            
        return {
            'qa_id': sample['qa_id'],
            'vid_feat': vid_feat, # [512]
            'text_feats': sample['text_features'], # [4, 384]
            'label': sample['label'] # [4]
        }

# 2. Dual-Encoder Fusion Model
class FusionModel(nn.Module):
    def __init__(self, vid_dim=512, txt_dim=384, hidden_dim=256):
        super().__init__()
        # Projects concatenated (Video + Text) into a single score
        self.mlp = nn.Sequential(
            nn.Linear(vid_dim + txt_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1) # Outputs a scalar score for the Option
        )

    def forward(self, vid_feat, text_feats):
        # vid_feat: [B, 512]
        # text_feats: [B, 4, 384]
        
        B = vid_feat.size(0)
        # Expand vid_feat to match the 4 options: [B, 4, 512]
        vid_expanded = vid_feat.unsqueeze(1).expand(-1, 4, -1)
        
        # Concatenate: [B, 4, 896]
        fused = torch.cat([vid_expanded, text_feats], dim=-1)
        
        # Score each option: [B, 4, 1] -> [B, 4]
        scores = self.mlp(fused).squeeze(-1)
        return scores

# 3. Training Loop
def train():
    print("Initializing Training Dataset...")
    train_dataset = VideoQADataset('training_qa.csv', 'video_features_r3d', is_test=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = FusionModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # We use BCEWithLogitsLoss because some answers are 'multi' (e.g. 'AB')
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting Training Loop...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            vid = batch['vid_feat'].to(DEVICE)
            txt = batch['text_feats'].to(DEVICE)
            labels = batch['label'].to(DEVICE) # [B, 4]
            
            scores = model(vid, txt) # [B, 4]
            
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), 'fusion_model.pth')
    print("Model saved to fusion_model.pth")

if __name__ == '__main__':
    train()
