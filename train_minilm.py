import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from datasets import Dataset
import os
import time

def prepare_data():
    print("Loading datasets...", flush=True)
    train_df = pd.read_csv('training_qa.csv')
    test_df = pd.read_csv('test_qa.csv')
    
    # We need a pseudo-label for the test set to format it, but we won't use it for evaluation
    pseudo = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')
    test_df = test_df.merge(pseudo[['qa_id', 'prediction']], on='qa_id')
    test_df.rename(columns={'prediction': 'answer'}, inplace=True)
    
    # Clean NaNs
    for col in ['question', 'A', 'B', 'C', 'D']:
        train_df[col] = train_df[col].fillna('')
        test_df[col] = test_df[col].fillna('')
        
    def format_df(df, is_train=True):
        records = []
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        for _, row in df.iterrows():
            ans = row['answer']
            if ans not in label_map:
                continue
                
            records.append({
                'prompt': row['question'],
                'A': row['A'],
                'B': row['B'],
                'C': row['C'],
                'D': row['D'],
                'label': label_map[ans],
                'qa_id': row.get('qa_id', '')
            })
        return Dataset.from_pandas(pd.DataFrame(records))

    train_ds = format_df(train_df)
    test_ds = format_df(test_df, is_train=False)
    
    return train_ds, test_ds

def run_deberta():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading Tokenizer {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    train_ds, test_ds = prepare_data()
    
    def preprocess_function(examples):
        prompts = [[context] * 4 for context in examples["prompt"]]
        choices = []
        for i in range(len(examples['prompt'])):
            choices.append([examples['A'][i], examples['B'][i], examples['C'][i], examples['D'][i]])
            
        prompts = sum(prompts, [])
        choices = sum(choices, [])
        
        tokenized_examples = tokenizer(prompts, choices, truncation=True, max_length=64)
        
        return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        
    print("Tokenizing datasets...", flush=True)
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_test = test_ds.map(preprocess_function, batched=True)
    
    # We only train on a highly confident subset to save time and prevent overfitting to noise
    # Wait, the dataset is highly reliable, let's train on 15,000 random samples to keep it under 15 mins
    print("Subsampling training set for speed...", flush=True)
    tokenized_train = tokenized_train.shuffle(seed=42)
    
    from transformers import DataCollatorForMultipleChoice
    import torch
    
    class DataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            
        def __call__(self, features):
            label_name = "label"
            labels = [feature.pop(label_name) for feature in features]
            batch_size = len(features)
            num_choices = len(features[0]["input_ids"])
            flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
            flattened_features = sum(flattened_features, [])
            
            batch = self.tokenizer.pad(
                flattened_features,
                padding=True,
                max_length=64,
                return_tensors="pt",
            )
            batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
            return batch

    print("Loading Model...", flush=True)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir="./deberta_output",
        evaluation_strategy="no",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=DataCollator(tokenizer),
    )
    
    print("Starting Training (This will take ~10-15 mins)...", flush=True)
    trainer.train()
    
    print("Predicting on Test Set...", flush=True)
    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=1)
    
    letters = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    final_sub = []
    
    for i in range(len(test_ds)):
        qa_id = test_ds[i]['qa_id']
        final_sub.append({'qa_id': qa_id, 'prediction': letters[preds[i]]})
        
    out = pd.DataFrame(final_sub)
    out.to_csv("submission_v72_minilm.csv", index=False)
    print("Saved submission_v72_minilm.csv!", flush=True)
    
if __name__ == "__main__":
    run_deberta()
