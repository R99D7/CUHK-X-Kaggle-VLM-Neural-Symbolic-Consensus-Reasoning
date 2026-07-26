import os
import zipfile
import pandas as pd
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import time

def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def run_clip():
    print("Checking test videos...", flush=True)
    if not os.path.exists("large_model_track_test"):
        print("Unzipping large_model_track_test.zip...", flush=True)
        with zipfile.ZipFile("large_model_track_test.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
            
    print("Loading CLIP Model...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    test_df = pd.read_csv('test_qa.csv')
    
    results = []
    
    start_time = time.time()
    for idx, row in test_df.iterrows():
        qa_id = row['qa_id']
        video_path = row['path']
        
        if not os.path.exists(video_path):
            # Try to fix the path if it's nested
            alt_path = os.path.join("large_model_track_test", video_path.split("/")[-1])
            if os.path.exists(alt_path):
                video_path = alt_path
            
        frame = extract_middle_frame(video_path)
        
        choices = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
        letters = ['A', 'B', 'C', 'D']
        
        if frame is None:
            results.append({'qa_id': qa_id, 'prediction': 'A', 'prob_A': 0, 'prob_B': 0, 'prob_C': 0, 'prob_D': 0})
            continue
            
        inputs = processor(text=choices, images=frame, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
        best_idx = np.argmax(probs)
        pred = letters[best_idx]
        
        results.append({
            'qa_id': qa_id, 
            'prediction': pred,
            'prob_A': probs[0],
            'prob_B': probs[1],
            'prob_C': probs[2],
            'prob_D': probs[3]
        })
        
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(test_df)} videos...", flush=True)
            
    print(f"Finished in {time.time() - start_time:.2f} seconds.", flush=True)
    
    out_df = pd.DataFrame(results)
    out_df.to_csv("clip_zero_shot_raw.csv", index=False)
    print("Saved clip_zero_shot_raw.csv", flush=True)

if __name__ == '__main__':
    run_clip()
