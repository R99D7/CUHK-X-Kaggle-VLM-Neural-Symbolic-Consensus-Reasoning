import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.model_selection import KFold

# Configurations
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODALITIES = ['Depth_Color', 'Depth', 'IR', 'Thermal']

class MultiModalQADataset(Dataset):
    def __init__(self, data_df, video_feature_dir, text_embeddings):
        self.data = data_df
        self.video_feature_dir = video_feature_dir
        self.text_embeddings = text_embeddings
        self.samples = []
        self._prepare_data()

    def _prepare_data(self):
        for idx, row in self.data.iterrows():
            qa_id = row['qa_id']
            vid_id = qa_id.split('_')[0] if '_' in qa_id and not qa_id.startswith('test') and not qa_id.startswith('train') else qa_id
            if len(vid_id) > 12 and qa_id.startswith('LM_test'):
                vid_id = '_'.join(qa_id.split('_')[:3])
                
            vid_feats = {}
            for mod in MODALITIES:
                vid_feats[mod] = os.path.join(self.video_feature_dir, f"{vid_id}_{mod}.pt")
                # Fallback for train videos that didn't have suffix if we only extracted Depth_Color initially
                if mod == 'Depth_Color' and not os.path.exists(vid_feats[mod]):
                    vid_feats[mod] = os.path.join(self.video_feature_dir, f"{vid_id}.pt")
            
            text_emb = self.text_embeddings[qa_id]
            
            label = torch.zeros(4)
            if 'answer' in row and not pd.isna(row['answer']):
                label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                ans = str(row['answer'])
                for char in ans:
                    if char in label_map:
                        label[label_map[char]] = 1.0
            
            self.samples.append({
                'qa_id': qa_id,
                'vid_feats': vid_feats,
                'text_features': text_emb,
                'label': label
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        mod_tensors = []
        for mod in MODALITIES:
            path = sample['vid_feats'][mod]
            if os.path.exists(path):
                mod_tensors.append(torch.load(path, weights_only=True))
            else:
                mod_tensors.append(torch.zeros((16, 2048)))
                
        # Stack all modalities [4, 16, 2048]
        vid_feat_stack = torch.stack(mod_tensors, dim=0)
            
        return {
            'qa_id': sample['qa_id'],
            'vid_feat_stack': vid_feat_stack, # [4, 16, 2048]
            'text_feats': sample['text_features'], # [4, 384]
            'label': sample['label']   # [4]
        }

class CrossAttentionMultiModalFusion(nn.Module):
    def __init__(self, vid_dim=2048, txt_dim=384, hidden_dim=512, num_layers=2):
        super().__init__()
        
        # 4 Independent Transformer Encoders for Temporal Modeling (1 per modality)
        encoder_layer = nn.TransformerEncoderLayer(d_model=vid_dim, nhead=8, dim_feedforward=vid_dim, dropout=0.1, batch_first=True)
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, num_layers=1) for _ in range(4)
        ])
        
        # Cross-Attention over Modalities
        self.modality_attn = nn.MultiheadAttention(embed_dim=vid_dim, num_heads=4, batch_first=True)
        
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

    def forward(self, vid_feat_stack, text_feats):
        # vid_feat_stack: [B, 4, 16, 2048]
        B = vid_feat_stack.size(0)
        
        global_mods = []
        for i in range(4):
            seq = vid_feat_stack[:, i, :, :] # [B, 16, 2048]
            enc = self.transformers[i](seq) # [B, 16, 2048]
            global_mods.append(enc.mean(dim=1)) # [B, 2048]
            
        # [B, 4, 2048]
        mods_stacked = torch.stack(global_mods, dim=1) 
        
        # Cross Attention: query=mods, key=mods, value=mods (Self-Attention across modalities)
        attn_out, _ = self.modality_attn(mods_stacked, mods_stacked, mods_stacked)
        
        # Final Modality Fusion (Mean over the 4 attended modalities)
        fused_vid = attn_out.mean(dim=1) # [B, 2048]
        
        # Expand for text options
        vid_expanded = fused_vid.unsqueeze(1).expand(-1, 4, -1) # [B, 4, 2048]
        fused_final = torch.cat([vid_expanded, text_feats], dim=-1) # [B, 4, 2432]
        
        scores = self.mlp(fused_final).squeeze(-1) # [B, 4]
        return scores

def precompute_text_embeddings(df_list, text_encoder):
    embeddings_dict = {}
    options = ['A', 'B', 'C', 'D']
    print("Precomputing text embeddings...")
    for df in df_list:
        text_strings = []
        qa_ids = []
        for idx, row in df.iterrows():
            qa_ids.append(row['qa_id'])
            for opt in options:
                text_strings.append(f"Question: {row['question']} Option: {row[opt]}")
        
        # Batch encode in chunks
        chunk_size = 512
        embs = []
        for i in tqdm(range(0, len(text_strings), chunk_size)):
            chunk = text_strings[i:i+chunk_size]
            with torch.no_grad():
                emb = text_encoder.encode(chunk, convert_to_tensor=True, show_progress_bar=False).cpu()
            embs.append(emb)
        embs = torch.cat(embs, dim=0) # [N*4, 384]
        
        # Reshape to [N, 4, 384]
        embs = embs.view(-1, 4, 384)
        for i, qa_id in enumerate(qa_ids):
            embeddings_dict[qa_id] = embs[i]
            
    return embeddings_dict

def train_and_cv():
    print("Loading Dataframes...")
    train_df = pd.read_csv('training_qa.csv')
    pseudo_df = pd.read_csv('pseudo_test_labels.csv')
    test_df = pd.read_csv('test_qa.csv')
    
    # Init text encoder once to save memory
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    # Precompute all text embeddings once!
    all_embeddings = precompute_text_embeddings([train_df, pseudo_df, test_df], text_encoder)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    test_dataset = MultiModalQADataset(test_df, 'video_features_resnet', all_embeddings)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    test_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n--- FOLD {fold+1}/5 ---")
        
        fold_train = train_df.iloc[train_idx].copy()
        fold_val = train_df.iloc[val_idx].copy()
        
        # INJECT PSEUDO LABELS ONLY INTO THE TRAIN FOLD
        fold_train = pd.concat([fold_train, pseudo_df], ignore_index=True)
        
        train_dataset = MultiModalQADataset(fold_train, 'video_features_resnet', all_embeddings)
        val_dataset = MultiModalQADataset(fold_val, 'video_features_resnet', all_embeddings)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        
        model = CrossAttentionMultiModalFusion().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                vid = batch['vid_feat_stack'].to(DEVICE)
                txt = batch['text_feats'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                scores = model(vid, txt)
                loss = criterion(scores, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    vid = batch['vid_feat_stack'].to(DEVICE)
                    txt = batch['text_feats'].to(DEVICE)
                    labels = batch['label'].to(DEVICE)
                    scores = model(vid, txt)
                    loss = criterion(scores, labels)
                    val_loss += loss.item()
                    
            val_loss /= len(val_loader)
            train_loss /= len(train_loader)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'model_fold{fold}.pth')
                
        # Load best model for test prediction
        model.load_state_dict(torch.load(f'model_fold{fold}.pth', weights_only=True))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predicting Test Fold {fold+1}"):
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
        
        # CLEAR MEMORY TO PREVENT OOM DEADLOCK ON NEXT FOLD
        del model
        del optimizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        if fold == 1:
            print("Stopping after Fold 2 (2-Fold mode) to save time...")
            break

    # Aggregate Predictions
    num_folds_run = len(test_predictions)
    final_pred_df = test_predictions[0]
    for i in range(1, num_folds_run):
        final_pred_df = final_pred_df.merge(test_predictions[i], on='qa_id')
        
    final_pred_df['prob_A'] = final_pred_df[[f'prob_A_f{i}' for i in range(num_folds_run)]].mean(axis=1)
    final_pred_df['prob_B'] = final_pred_df[[f'prob_B_f{i}' for i in range(num_folds_run)]].mean(axis=1)
    final_pred_df['prob_C'] = final_pred_df[[f'prob_C_f{i}' for i in range(num_folds_run)]].mean(axis=1)
    final_pred_df['prob_D'] = final_pred_df[[f'prob_D_f{i}' for i in range(num_folds_run)]].mean(axis=1)
    
    # Generate Sorted Letters
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
            pred = "".join([letters[idx] for idx in sorted_idx])
        elif expected_len == 1:
            pred = letters[sorted_idx[0]]
        else:
            top_k = [letters[idx] for idx in sorted_idx[:expected_len]]
            pred = "".join(sorted(top_k))
            
        final_res.append({'qa_id': qa_id, 'prediction': pred})
        
    pd.DataFrame(final_res).to_csv('submission_v32_multimodal_cv.csv', index=False)
    print("Saved 2-Fold Ensembled Predictions to submission_v32_multimodal_cv.csv")

if __name__ == '__main__':
    train_and_cv()
