import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME = 'cross-encoder/nli-deberta-v3-base'
BATCH_SIZE = 8
EPOCHS = 2
LR = 3e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class QADataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128, is_test=False):
        self.data = pd.read_csv(csv_file)
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
        # input_ids: [B, 4, seq_len]
        B, num_choices, seq_len = input_ids.shape
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        if getattr(outputs, 'pooler_output', None) is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # [B*4, 1]
        
        logits = logits.view(B, num_choices) # [B, 4]
        return logits

def train():
    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = QADataset('training_qa.csv', tokenizer, is_test=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = CrossEncoderMultipleChoice(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader)
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")
            
        print(f"Epoch {epoch+1}/{EPOCHS} Average Loss: {total_loss/len(train_loader):.4f}")
        
    print("Generating Test Predictions...")
    test_dataset = QADataset('test_qa.csv', tokenizer, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            qa_ids = batch['qa_id']
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            for i, qa_id in enumerate(qa_ids):
                all_preds.append({
                    'qa_id': qa_id,
                    'raw_prob_A': probs[i][0],
                    'raw_prob_B': probs[i][1],
                    'raw_prob_C': probs[i][2],
                    'raw_prob_D': probs[i][3]
                })
                
    pd.DataFrame(all_preds).to_csv('deberta_nli_raw_predictions.csv', index=False)
    print("Saved deberta_nli_raw_predictions.csv!")

if __name__ == '__main__':
    train()
