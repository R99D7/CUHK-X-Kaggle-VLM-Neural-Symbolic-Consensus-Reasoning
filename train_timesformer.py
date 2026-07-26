import os
import glob
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification, TrainingArguments, Trainer
from transformers import default_data_collator

def get_frames(video_path, num_frames=8):
    if not os.path.exists(video_path):
        # Return empty frames if missing (should not happen on Kaggle)
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames
        
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
    cap.release()
    return frames

class VideoQADataset(Dataset):
    def __init__(self, df, video_dir, image_processor, label2id=None, is_train=True):
        self.df = df
        self.video_dir = video_dir
        self.image_processor = image_processor
        self.label2id = label2id
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_id = str(row['video_id'])
        video_path = os.path.join(self.video_dir, f"{vid_id}.mp4")
        
        frames = get_frames(video_path, num_frames=8)
        
        # Processor expects a list of frames, each frame is HxWxC numpy array
        inputs = self.image_processor(list(frames), return_tensors="pt")
        
        # Return a dict
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        
        if self.is_train:
            ans = str(row['answer']).strip()
            item['labels'] = torch.tensor(self.label2id[ans], dtype=torch.long)
            
        return item

def run_timesformer():
    print("Loading datasets...", flush=True)
    train_path = glob.glob('/kaggle/input/**/training_qa.csv', recursive=True)[0]
    test_path = glob.glob('/kaggle/input/**/test_qa.csv', recursive=True)[0]
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Kaggle directory structure
    train_video_dir = os.path.join(os.path.dirname(train_path), 'train_video')
    test_video_dir = os.path.join(os.path.dirname(test_path), 'test_video')
    
    # We might have test video in a slightly different path if testing/training are separated
    # Let's just do a glob search for one test video to be safe
    test_vid_sample = test_df.iloc[0]['video_id']
    test_vid_glob = glob.glob(f'/kaggle/input/**/{test_vid_sample}.mp4', recursive=True)
    if len(test_vid_glob) > 0:
        test_video_dir = os.path.dirname(test_vid_glob[0])
        
    train_vid_sample = train_df.iloc[0]['video_id']
    train_vid_glob = glob.glob(f'/kaggle/input/**/{train_vid_sample}.mp4', recursive=True)
    if len(train_vid_glob) > 0:
        train_video_dir = os.path.dirname(train_vid_glob[0])
    
    # Ensure deterministic label order
    labels = train_df['answer'].dropna().unique().tolist()
    labels = sorted(labels)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    
    model_name = "facebook/timesformer-base-finetuned-k400"
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    
    train_dataset = VideoQADataset(train_df, train_video_dir, image_processor, label2id, is_train=True)
    test_dataset = VideoQADataset(test_df, test_video_dir, image_processor, label2id, is_train=False)
    
    model = TimesformerForVideoClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    training_args = TrainingArguments(
        output_dir="./timesformer_out",
        evaluation_strategy="no",
        save_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    
    print("Starting Training...", flush=True)
    trainer.train()
    
    print("Predicting on Test Set...", flush=True)
    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    
    final_sub = []
    raw_probs = []
    
    for i in range(len(test_df)):
        qa_id = test_df.iloc[i]['qa_id']
        pred_label = id2label[preds[i]]
        final_sub.append({'qa_id': qa_id, 'prediction': pred_label})
        
        prob_dict = {'qa_id': qa_id}
        for j, l in enumerate(labels):
            prob_dict[f'prob_{l}'] = float(probs[i][j])
        raw_probs.append(prob_dict)
        
    pd.DataFrame(final_sub).to_csv("submission_v123_timesformer.csv", index=False)
    pd.DataFrame(raw_probs).to_csv("timesformer_raw_probs.csv", index=False)
    print("Saved submission_v123_timesformer.csv!", flush=True)

if __name__ == "__main__":
    run_timesformer()
