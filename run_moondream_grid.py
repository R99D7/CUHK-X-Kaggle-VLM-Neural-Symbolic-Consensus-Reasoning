import os
import torch
import cv2
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

print("Starting Moondream2 Multi-Frame Grid Inference!")

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"

print("Loading moondream2 model...")
# Ensure it loads smoothly into 6GB VRAM
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,
    torch_dtype=torch.float16, device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

test_qa_path = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\test_qa.csv"
test_df = pd.read_csv(test_qa_path)

# Run exactly 50 percent of the dataset
target_df = test_df.head(341)
print(f"Evaluating {len(target_df)} questions...")

final_preds = []

video_base_dir = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\videos"
if not os.path.exists(video_base_dir):
    video_base_dir = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\test_video"

for idx, row in target_df.iterrows():
    vid_path = os.path.join(video_base_dir, row['path'])
    if not os.path.exists(vid_path):
        vid_path = os.path.join(r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\videos", row['path'])
    if not os.path.exists(vid_path):
        vid_path = os.path.join(r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\test_video", row['path'])
    if not os.path.exists(vid_path):
        vid_path = os.path.join(r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK", row['path'].replace('large_model_track_test/', 'test_video/large_model_track_test/'))

    default_pred = 'ABCD' if row['category'] == 'sequence' else 'AB' if row['category'] == 'multi' else 'A'
    
    if not os.path.exists(vid_path):
        final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})
        continue
        
    try:
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract 4 frames (start, 1/3, 2/3, end)
        frame_indices = [
            int(total_frames * 0.1),
            int(total_frames * 0.35),
            int(total_frames * 0.65),
            int(total_frames * 0.9)
        ]
        
        frames = []
        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(f_idx, total_frames - 1))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize each frame to be small
                frame = cv2.resize(frame, (256, 256))
                frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})
            continue
            
        # If we didn't get 4 frames, duplicate the last one
        while len(frames) < 4:
            frames.append(frames[-1])
            
        # Stitch frames into a 2x2 grid (chronological reading order)
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        grid_img = np.vstack((top_row, bottom_row))
        pil_img = Image.fromarray(grid_img)
        
        enc_image = model.encode_image(pil_img)
        
        instruction = "Based on this grid of 4 chronological video frames, strictly output only the correct option letter (A, B, C, or D)."
        if row['category'] == 'sequence':
            instruction = "Based on this grid of 4 chronological video frames, strictly output the correct sequence as a permutation of A, B, C, and D (e.g. ABCD, BADC). You MUST output all 4 letters in the correct chronological order."
        elif row['category'] == 'multi':
            instruction = "Based on this grid of 4 chronological video frames, strictly output all correct option letters (e.g. AB, ACD)."

        question_prompt = (
            f"Question: {row['question']}\n"
            f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}\n"
            f"Category: {row['category']}\n"
            f"{instruction}"
        )
        
        answer = model.answer_question(enc_image, question_prompt, tokenizer)
        
        pred = ''.join([c for c in answer.upper() if c in 'ABCD'])
        
        if row['category'] == 'sequence':
            pred = ''.join(dict.fromkeys(pred))
            missing = [c for c in 'ABCD' if c not in pred]
            pred = pred + ''.join(missing)
            if len(pred) != 4 or sorted(pred) != ['A', 'B', 'C', 'D']:
                pred = 'ABCD'
        elif row['category'] == 'multi':
            pred = ''.join(sorted(dict.fromkeys(pred)))
            if len(pred) == 0:
                pred = 'AB'
        else:
            if len(pred) == 0: pred = 'A'
            pred = pred[0]
            
        final_preds.append({'qa_id': row['qa_id'], 'prediction': pred})
        print(f"Processed {idx+1}/{len(target_df)}: {row['qa_id']} -> {pred}")
        
    except Exception as e:
        print(f"Error on {row['qa_id']}: {e}")
        final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})

df_moon = pd.DataFrame(final_preds)
df_moon.to_csv('submission_moondream_grid.csv', index=False)
print("Finished Moondream Grid local inference!")

# Blend with ultimate_v9
df_base = pd.read_csv('submission_ultimate_v9.csv')
df_moon_dict = dict(zip(df_moon['qa_id'], df_moon['prediction']))
df_base['prediction'] = df_base.apply(
    lambda row: df_moon_dict.get(row['qa_id'], row['prediction']),
    axis=1
)

df_base.to_csv('submission_ultimate_v11.csv', index=False)
print("Finished creating final blended submission_ultimate_v11.csv!")
