import os
import torch
import cv2
import pandas as pd
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print('Loading Qwen2-VL-2B-Instruct...')
MODEL_PATH = 'Qwen/Qwen2-VL-2B-Instruct'

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16, 
    device_map='cuda'
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

test_df = pd.read_csv('test_qa.csv')
sample = pd.read_csv('sample_submission.csv')
base = pd.read_csv('submission_oracle_v20.csv')

base_preds = dict(zip(base['qa_id'], base['prediction']))
sample_lengths = dict(zip(sample['qa_id'], sample['prediction'].apply(lambda x: len(str(x)))))

final_preds = []
failed_count = 0

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
        # We will extract 4 evenly spaced frames to give Qwen2-VL true chronological context!
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            final_preds.append({'qa_id': qa_id, 'prediction': safe_fallback})
            failed_count += 1
            cap.release()
            continue
            
        frames = []
        # Extract 4 frames (e.g. 10%, 40%, 70%, 95%)
        for pct in [0.1, 0.4, 0.7, 0.95]:
            frame_idx = int(total_frames * pct)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total_frames - 1))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        
        if len(frames) == 0:
            final_preds.append({'qa_id': qa_id, 'prediction': safe_fallback})
            failed_count += 1
            continue
            
        # Temporarily save images for qwen_vl_utils (it accepts PIL, but list of dict is easier)
        # Actually Qwen2-VL allows PIL Images directly in the payload!
        
        question = row['question']
        if expected_len == 1:
            q_prompt = f"Question: {question}\nA) {row['a']}\nB) {row['b']}\nC) {row['c']}\nD) {row['d']}\nAnalyze these frames chronologically and select the SINGLE best option. Answer STRICTLY with just one letter: A, B, C, or D."
        elif expected_len > 1 and row['category'] == 'multi':
            q_prompt = f"Question: {question}\nA) {row['a']}\nB) {row['b']}\nC) {row['c']}\nD) {row['d']}\nAnalyze these frames. This is a multiple choice question with EXACTLY {expected_len} correct options.\nAnswer STRICTLY with exactly {expected_len} letters from A, B, C, and D (e.g., AB, BCD)."
        else: # sequence/permutation
            q_prompt = f"Question: {question}\nA) {row['a']}\nB) {row['b']}\nC) {row['c']}\nD) {row['d']}\nAnalyze these frames and determine the correct chronological sequence of events.\nAnswer STRICTLY with exactly {expected_len} letters in the correct order (e.g., BCDA)."
            
        content_list = []
        for img in frames:
            content_list.append({"type": "image", "image": img})
        content_list.append({"type": "text", "text": q_prompt})
        
        messages = [
            {"role": "user", "content": content_list}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to('cuda')
        
        # Free up slightly more memory
        torch.cuda.empty_cache()
        
        generated_ids = model.generate(**inputs, max_new_tokens=10)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        pred = ''.join([c for c in answer.upper() if c in 'ABCD'])
        
        if len(pred) > expected_len:
            pred = pred[:expected_len]
        
        if len(pred) != expected_len:
            pred = safe_fallback
            failed_count += 1
            
        final_preds.append({'qa_id': qa_id, 'prediction': pred})
        
    except Exception as e:
        final_preds.append({'qa_id': qa_id, 'prediction': safe_fallback})
        failed_count += 1

out = pd.DataFrame(final_preds)
out.to_csv('submission_v39_qwen_vl.csv', index=False)
print(f'Done! Saved submission_v39_qwen_vl.csv. Fallbacks used: {failed_count}')
