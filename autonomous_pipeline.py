import os
import time
import zipfile
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ZIP_PATH = 'large_model_track_test.zip'

print("2. Unzipping videos...")
os.makedirs("videos", exist_ok=True)
try:
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("videos")
except Exception as e:
    print(f"Error unzipping: {e}")

print("3. Loading Moondream2 on CPU (This will take a while to download weights)...")
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

print("4. Running Inference...")
test_df = pd.read_csv("test_qa.csv")
predictions = []

for idx, row in test_df.iterrows():
    video_path = f"videos/{row['path']}"
    if not os.path.exists(video_path):
        video_path = f"videos/large_model_track_test/{row['path']}" # Just in case it unzipped into a subfolder
        
    question = row['question']
    options_text = f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}"
    
    cat = row['category']
    if cat in ['multi', 'sequence']:
        prompt = f"Answer the multiple choice question with ONLY the correct combination of letters (e.g. BCD, DCBA).\nQuestion: {question}\n{options_text}"
    else:
        prompt = f"Answer the multiple choice question with ONLY a single letter A, B, C, or D.\nQuestion: {question}\n{options_text}"

    clean_resp = "A" # Default fallback
    try:
        # Extract middle frame
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                enc_image = model.encode_image(image)
                answer = model.answer_question(enc_image, prompt, tokenizer)
                
                res = ''.join([c for c in answer if c in 'ABCD'])
                if res: clean_resp = res
    except Exception as e:
        print(f"Error processing {row['path']}: {e}")
        
    predictions.append(clean_resp)
    if idx % 10 == 0:
        print(f"Processed {idx}/682 videos.", flush=True)
        # Save checkpoints just in case
        pd.DataFrame({'qa_id': test_df['qa_id'][:len(predictions)], 'prediction': predictions}).to_csv('submission_checkpoint.csv', index=False)

print("5. Saving Final Submission...")
sub_df = pd.DataFrame({'qa_id': test_df['qa_id'], 'prediction': predictions})
sub_df.to_csv('submission_final.csv', index=False)
print("COMPLETED!", flush=True)
