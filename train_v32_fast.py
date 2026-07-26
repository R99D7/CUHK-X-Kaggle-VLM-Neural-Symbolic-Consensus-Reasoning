import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.model_selection import KFold

BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FastQADataset(Dataset):
    def __init__(self, data_df, video_feature_dir, text_encoder=None):
        self.data = data_df
        self.video_feature_dir = video_feature_dir
        self.text_encoder = text_encoder if text_encoder else SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
        self.samples = []
        self._prepare_data()

    def _prepare_data(self):
        options = ['A', 'B', 'C', 'D']
        for _, row in self.data.iterrows():
            qa_id = row['qa_id']
            vid_id = qa_id.split('_')[0] if '_' in qa_id and not qa_id.startswith('test') and not qa_id.startswith('train') else qa_id
            if len(vid_id) > 12 and qa_id.startswith('LM_test'):
                vid_id = '_'.join(qa_id.split('_')[:3])
                
            vid_feat = os.path.join(self.video_feature_dir, f"{vid_id}.pt")
            
            text_strings = [f"Question: {row['question']} Option: {row[opt]}" for opt in options]
            text_emb = self.text_encoder.encode(text_strings, convert_to_tensor=True, show_progress_bar=False).cpu()
            
            label = torch.zeros(4)
            if 'answer' in row and not pd.isna(row['answer']):
                label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                for char in str(row['answer']):
                    if char in label_map: label[label_map[char]] = 1.0
            
            self.samples.append({
                'qa_id': qa_id,
                'vid_feat_path': vid_feat,
                'text_features': text_emb,
                'label': label
            })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        vid_feat = torch.load(s['vid_feat_path'], weights_only=True) if os.path.exists(s['vid_feat_path']) else torch.zeros((16, 2048))
        return {'qa_id': s['qa_id'], 'vid': vid_feat, 'txt': s['text_features'], 'label': s['label']}

class FastFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True), num_layers=2)
        self.mlp = nn.Sequential(
            nn.Linear(2048 + 384, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, vid, txt):
        enc = self.transformer(vid).mean(dim=1)
        fused = torch.cat([enc.unsqueeze(1).expand(-1, 4, -1), txt], dim=-1)
        return self.mlp(fused).squeeze(-1)

def train_fast_cv():
    train_df = pd.read_csv('training_qa.csv')
    pseudo_df = pd.read_csv('pseudo_test_labels.csv')
    test_df = pd.read_csv('test_qa.csv')
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    test_dataset = FastQADataset(test_df, 'video_features_resnet', text_encoder)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"--- FOLD {fold+1}/5 ---")
        fold_train = pd.concat([train_df.iloc[train_idx], pseudo_df], ignore_index=True)
        fold_val = train_df.iloc[val_idx]
        
        train_loader = DataLoader(FastQADataset(fold_train, 'video_features_resnet', text_encoder), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(FastQADataset(fold_val, 'video_features_resnet', text_encoder), batch_size=BATCH_SIZE, shuffle=False)
        
        model = FastFusionModel().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val = float('inf')
        for epoch in range(EPOCHS):
            model.train()
            for b in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(b['vid'].to(DEVICE), b['txt'].to(DEVICE)), b['label'].to(DEVICE))
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for b in val_loader:
                    val_loss += criterion(model(b['vid'].to(DEVICE), b['txt'].to(DEVICE)), b['label'].to(DEVICE)).item()
            val_loss /= len(val_loader)
            print(f"E{epoch+1} Val Loss: {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), f'fast_fold{fold}.pth')
                
        model.load_state_dict(torch.load(f'fast_fold{fold}.pth', weights_only=True))
        model.eval()
        f_preds = []
        with torch.no_grad():
            for b in test_loader:
                probs = torch.sigmoid(model(b['vid'].to(DEVICE), b['txt'].to(DEVICE)))
                for i, qid in enumerate(b['qa_id']):
                    f_preds.append({
                        'qa_id': qid,
                        f'prob_A_f{fold}': probs[i][0].item(), f'prob_B_f{fold}': probs[i][1].item(),
                        f'prob_C_f{fold}': probs[i][2].item(), f'prob_D_f{fold}': probs[i][3].item()
                    })
        test_preds.append(pd.DataFrame(f_preds))
        
    final = test_preds[0]
    for i in range(1, 5): final = final.merge(test_preds[i], on='qa_id')
    
    res = []
    for _, row in final.iterrows():
        p = [row[[f'prob_A_f{i}' for i in range(5)]].mean(), row[[f'prob_B_f{i}' for i in range(5)]].mean(), 
             row[[f'prob_C_f{i}' for i in range(5)]].mean(), row[[f'prob_D_f{i}' for i in range(5)]].mean()]
        res.append({'qa_id': row['qa_id'], 'sorted_letters': "".join(['A','B','C','D'][idx] for idx in np.argsort(p)[::-1])})
        
    pd.DataFrame(res).to_csv('v32_fast_predictions.csv', index=False)

if __name__ == '__main__': train_fast_cv()
