import os
import torch
import cv2
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_OPT_IN"] = "1"

print('Loading Moondream2...')
MODEL_PATH = 'vikhyatk/moondream2'
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

test_df = pd.read_csv('test_qa.csv')
sample = pd.read_csv('sample_submission.csv')
base = pd.read_csv('submission_oracle_v20.csv')

# Pre-load base predictions as safety fallback
base_preds = dict(zip(base['qa_id'], base['prediction']))
sample_lengths = dict(zip(sample['qa_id'], sample['prediction'].apply(lambda x: len(str(x)))))

final_preds = []
failed_count = 0

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    qa_id = row['qa_id']
    num = qa_id.split('_')[1]
    
    # Try all possible visual modalities. Fallback logic:
    # Depth_Color -> Depth -> IR -> Thermal
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
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            final_preds.append({'qa_id': qa_id, 'prediction': safe_fallback})
            failed_count += 1
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Craft precise prompt depending on category
        question = row['question']
        
        if expected_len == 1:
            q_prompt = f"Question: {question}\nA) {row['a']}\nB) {row['b']}\nC) {row['c']}\nD) {row['d']}\nAnalyze the image and select the SINGLE best option. Answer STRICTLY with just one letter: A, B, C, or D."
        elif expected_len > 1 and row['category'] == 'multi':
            q_prompt = f"Question: {question}\nA) {row['a']}\nB) {row['b']}\nC) {row['c']}\nD) {row['d']}\nAnalyze the image. This is a multiple choice question with EXACTLY {expected_len} correct options.\nAnswer STRICTLY with exactly {expected_len} letters from A, B, C, and D."
        else: # sequence/permutation
            q_prompt = f"Question: {question}\nA) {row['a']}\nB) {row['b']}\nC) {row['c']}\nD) {row['d']}\nAnalyze the image and determine the correct chronological sequence of events.\nAnswer STRICTLY with exactly {expected_len} letters in the correct order."
            
        enc_image = model.encode_image(image)
        answer = model.answer_question(enc_image, q_prompt, tokenizer)
        
        pred = ''.join([c for c in answer.upper() if c in 'ABCD'])
        
        # Clean up prediction
        if len(pred) > expected_len:
            pred = pred[:expected_len]
        
        # If it failed to answer properly or length mismatch, FALLBACK!
        if len(pred) != expected_len:
            pred = safe_fallback
            failed_count += 1
            
        final_preds.append({'qa_id': qa_id, 'prediction': pred})
        
    except Exception as e:
        final_preds.append({'qa_id': qa_id, 'prediction': safe_fallback})
        failed_count += 1

out = pd.DataFrame(final_preds)
out.to_csv('submission_v37_moondream_ultimate.csv', index=False)
print(f'Done! Saved submission_v37_moondream_ultimate.csv. Fallbacks used: {failed_count}')
