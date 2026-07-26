import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.model_selection import KFold

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 8
EPOCHS = 3
LR = 3e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FOLDS = 5

class QADataset(Dataset):
    def __init__(self, data_df, tokenizer, max_length=128, is_test=False):
        self.data = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.options = ['A', 'B', 'C', 'D']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = f"Category: {row['category']}. Question: {row['question']}"
        
        input_ids = []
        attention_masks = []
        
        for opt in self.options:
            text = str(row[opt])
            enc = self.tokenizer(
                question,
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids.append(enc['input_ids'].squeeze(0))
            attention_masks.append(enc['attention_mask'].squeeze(0))
            
        input_ids = torch.stack(input_ids) # [4, seq_len]
        attention_masks = torch.stack(attention_masks) # [4, seq_len]
        
        label = -1
        if not self.is_test and 'answer' in row:
            label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            ans = str(row['answer'])
            label = torch.zeros(4)
            for char in ans:
                if char in label_map:
                    label[label_map[char]] = 1.0
                    
        return {
            'qa_id': row['qa_id'],
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'label': label
        }

class CrossEncoderMultipleChoice(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        B, num_choices, seq_len = input_ids.shape
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # [B*4, hidden]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # [B*4, 1]
        
        logits = logits.view(B, num_choices) # [B, 4]
        return logits

def train():
    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_df = pd.read_csv('training_qa.csv')
    test_df = pd.read_csv('test_qa.csv')
    
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    test_dataset = QADataset(test_df, tokenizer, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_fold_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n=== Fold {fold+1}/{FOLDS} ===")
        train_sub = train_df.iloc[train_idx]
        
        train_dataset = QADataset(train_sub, tokenizer, is_test=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        model = CrossEncoderMultipleChoice(MODEL_NAME).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, leave=False):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(train_loader):.4f}")
            
        print("Generating Test Predictions for Fold...")
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                qa_ids = batch['qa_id']
                
                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                for i, qa_id in enumerate(qa_ids):
                    fold_preds.append({
                        'qa_id': qa_id,
                        'raw_prob_A': probs[i][0],
                        'raw_prob_B': probs[i][1],
                        'raw_prob_C': probs[i][2],
                        'raw_prob_D': probs[i][3]
                    })
        fold_df = pd.DataFrame(fold_preds).set_index('qa_id')
        all_fold_preds.append(fold_df)
        
        del model
        torch.cuda.empty_cache()
        
    print("\nAveraging predictions...")
    avg_preds = sum(all_fold_preds) / FOLDS
    avg_preds.reset_index().to_csv('crossencoder_cv_raw_predictions.csv', index=False)
    print("Saved crossencoder_cv_raw_predictions.csv!")

if __name__ == '__main__':
    train()
