import os
import torch
from PIL import Image
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.phi.configuration_phi import PhiConfig
import cv2
import glob

print("Starting Local GPU Inference!")

# Fix for PhiConfig error
PhiConfig.pad_token_id = None
if not hasattr(PhiConfig, 'pad_token_id'):
    setattr(PhiConfig, 'pad_token_id', None)
    
from transformers.modeling_utils import PreTrainedModel
PreTrainedModel.all_tied_weights_keys = property(lambda self: {})

MODEL_PATH = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\moondream2"

print(f"Loading model from {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, device_map={"": "cuda"}
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

test_qa_path = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\test_qa.csv"
test_df = pd.read_csv(test_qa_path)

# Only sequence and multi
target_df = test_df[test_df['category'].isin(['sequence', 'multi'])]
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

    if not os.path.exists(vid_path):
        print(f"[{row['qa_id']}] Video not found: {vid_path}")
        default_pred = 'ABCD' if row['category'] == 'sequence' else 'AB'
        final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})
        continue
        
    try:
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            default_pred = 'ABCD' if row['category'] == 'sequence' else 'AB'
            final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        instruction = "Based on the image, strictly output all correct option letters (e.g. AB, ACD)."
        if row['category'] == 'sequence':
            instruction = "Based on the image, strictly output the correct sequence as a permutation of A, B, C, and D (e.g. ABCD, BADC). You MUST output all 4 letters in the correct order."

        question_prompt = (
            f"Question: {row['question']}\n"
            f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}\n"
            f"Category: {row['category']}\n"
            f"{instruction}"
        )
        
        enc_image = model.encode_image(image)
        answer = model.answer_question(enc_image, question_prompt, tokenizer)
        
        # Clean answer to only A B C D
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
        default_pred = 'ABCD' if row['category'] == 'sequence' else 'AB'
        final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})

df_moon = pd.DataFrame(final_preds)
df_moon.to_csv('submission_moondream_local.csv', index=False)
print("Finished Moondream local inference!")

# Blend
df_base = pd.read_csv('submission_ultimate_v3.csv')
df_moon_dict = dict(zip(df_moon['qa_id'], df_moon['prediction']))
df_base['prediction'] = df_base.apply(
    lambda row: df_moon_dict.get(row['qa_id'], row['prediction']),
    axis=1
)

df_base.to_csv('submission_ultimate_v7.csv', index=False)
print("Finished creating final blended submission_ultimate_v7.csv!")
