import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import os

def run_deberta():
    model_name = "microsoft/deberta-v3-base"
    print(f"Loading Tokenizer {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    train_df = pd.read_csv('training_qa.csv')
    test_df = pd.read_csv('test_qa.csv')
    
    # Fill NaN
    for col in ['question', 'A', 'B', 'C', 'D']:
        train_df[col] = train_df[col].fillna('')
        test_df[col] = test_df[col].fillna('')
        
    labels = train_df['answer'].dropna().unique().tolist()
    labels = sorted(labels)  # Ensure deterministic order
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    
    print(f"Mapped {len(labels)} unique labels.", flush=True)
    
    def format_df(df, is_train=True):
        records = []
        for _, row in df.iterrows():
            prompt = f"Question: {row['question']} [SEP] Option A: {row['A']} [SEP] Option B: {row['B']} [SEP] Option C: {row['C']} [SEP] Option D: {row['D']}"
            record = {'text': prompt, 'qa_id': row.get('qa_id', '')}
            if is_train:
                ans = str(row['answer']).strip()
                if ans in label2id:
                    record['label'] = label2id[ans]
                else:
                    continue
            records.append(record)
        return Dataset.from_pandas(pd.DataFrame(records))

    train_ds = format_df(train_df, is_train=True)
    test_ds = format_df(test_df, is_train=False)
    
    def tokenize_func(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)
        
    print("Tokenizing...", flush=True)
    tokenized_train = train_ds.map(tokenize_func, batched=True)
    tokenized_test = test_ds.map(tokenize_func, batched=True)
    
    print("Loading Model...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)
    
    training_args = TrainingArguments(
        output_dir="./deberta_v3_large_out",
        evaluation_strategy="no",
        save_strategy="no",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
    )
    
    print("Starting Training...", flush=True)
    trainer.train()
    
    print("Predicting on Test Set...", flush=True)
    predictions = trainer.predict(tokenized_test)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    
    final_sub = []
    raw_probs = []
    
    for i in range(len(test_ds)):
        qa_id = test_ds[i]['qa_id']
        pred_label = id2label[preds[i]]
        final_sub.append({'qa_id': qa_id, 'prediction': pred_label})
        
        prob_dict = {'qa_id': qa_id}
        for j, l in enumerate(labels):
            prob_dict[f'prob_{l}'] = float(probs[i][j])
        raw_probs.append(prob_dict)
        
    pd.DataFrame(final_sub).to_csv("submission_v122_deberta_v3_large.csv", index=False)
    pd.DataFrame(raw_probs).to_csv("deberta_v3_large_raw_probs.csv", index=False)
    print("Saved submission_v122_deberta_v3_large.csv and deberta_v3_large_raw_probs.csv!", flush=True)

if __name__ == "__main__":
    run_deberta()
