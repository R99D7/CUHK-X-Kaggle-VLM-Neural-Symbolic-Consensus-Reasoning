import os
import torch
import cv2
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print('Loading LOCAL Moondream2...')
MODEL_PATH = 'moondream2'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16, device_map='cuda'
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

test_df = pd.read_csv('test_qa.csv')
sample = pd.read_csv('sample_submission.csv')
base = pd.read_csv('submission_oracle_v20.csv')

base_preds = dict(zip(base['qa_id'], base['prediction']))
sample_lengths = dict(zip(sample['qa_id'], sample['prediction'].apply(lambda x: len(str(x)))))

final_preds = []
failed_count = 0

def create_grid(frames):
    w, h = frames[0].size
    grid = Image.new('RGB', (w*2, h*2))
    grid.paste(frames[0], (0, 0))
    if len(frames) > 1: grid.paste(frames[1], (w, 0))
    if len(frames) > 2: grid.paste(frames[2], (0, h))
    if len(frames) > 3: grid.paste(frames[3], (w, h))
    return grid

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    qa_id = row['qa_id']
    num = qa_id.split('_')[1]
    
    modalities = ['Depth_Color', 'Depth', 'IR', 'Thermal']
    video_path = None
    
    for mod in modalities:
        p = os.path.join('videos', 'large_model_track_test', f'LM_test_{num}', mod, f'{mod}.mp4')
        if os.path.exists(p):
            video_path = p
            break
            
    expected_len = sample_lengths[qa_id]
    safe_fallback = base_preds[qa_id]
    
    if video_path is None:
        final_preds.append({'qa_id': qa_id, 'prediction': safe_fallback})
        failed_count += 1
        continue
        
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            final_preds.append({'qa_id': qa_id, 'prediction': safe_fallback})
            failed_count += 1
            cap.release()
            continue
            
        frames = []
        for pct in [0.1, 0.4, 0.7, 0.95]:
            frame_idx = int(total_frames * pct)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total_frames - 1))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to 378x378 for optimal moondream processing
                resized = cv2.resize(frame_rgb, (378, 378))
                frames.append(Image.fromarray(resized))
        cap.release()
        
        if len(frames) == 0:
            final_preds.append({'qa_id': qa_id, 'prediction': safe_fallback})
            failed_count += 1
            continue
            
        grid_image = create_grid(frames)
        enc_image = model.encode_image(grid_image)
        
        question = row['question']
        if expected_len == 1:
            q_prompt = f"""This is a 2x2 grid of sequential frames from a video (top-left is first, bottom-right is last).
Question: {question}
A) {row['a']}
B) {row['b']}
C) {row['c']}
D) {row['d']}
Answer strictly with one letter (A, B, C, or D)."""
        elif expected_len > 1 and row['category'] == 'multi':
            q_prompt = f"""This is a 2x2 grid of sequential video frames.
Question: {question}
A) {row['a']}
B) {row['b']}
C) {row['c']}
D) {row['d']}
This is a multiple choice question with EXACTLY {expected_len} correct options. Answer strictly with {expected_len} letters."""
        else:
            q_prompt = f"""This is a 2x2 grid of sequential video frames.
Question: {question}
A) {row['a']}
B) {row['b']}
C) {row['c']}
D) {row['d']}
Determine the correct chronological sequence. Answer strictly with exactly {expected_len} letters in order."""
            
        answer = model.answer_question(enc_image, q_prompt, tokenizer)
        
        pred = ''.join([c for c in answer.upper() if c in 'ABCD'])
        if len(pred) > expected_len: pred = pred[:expected_len]
        if len(pred) != expected_len: pred = safe_fallback
            
        final_preds.append({'qa_id': qa_id, 'prediction': pred})
        
    except Exception as e:
        final_preds.append({'qa_id': qa_id, 'prediction': safe_fallback})
        failed_count += 1

out = pd.DataFrame(final_preds)
out.to_csv('submission_v40_moondream_grid.csv', index=False)
print(f'Done! Saved submission_v40_moondream_grid.csv. Fallbacks used: {failed_count}')
