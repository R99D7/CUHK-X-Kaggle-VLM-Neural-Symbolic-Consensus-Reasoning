import os
import gc
import json
import torch
from PIL import Image
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

print("Loading Moondream2 directly from HuggingFace with updated transformers...")
from transformers.modeling_utils import PreTrainedModel
PreTrainedModel.all_tied_weights_keys = property(lambda self: {})
MODEL_ID = "vikhyatk/moondream2"
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
except Exception as e:
    print("Failed to load model:", e)
    raise

test_df = pd.read_csv('/kaggle/input/cuhk-sysu-llm-competition/test_qa.csv')
video_dir = '/kaggle/input/cuhk-sysu-llm-competition/test_video'

target_df = test_df[test_df['category'].isin(['multi', 'sequence'])]
print(f"Evaluating {len(target_df)} questions (multi and sequence)...")

final_preds = []
for idx, row in target_df.iterrows():
    video_path = os.path.join(video_dir, str(row['video_id']) + '.mp4')
    if not os.path.exists(video_path):
        final_preds.append({'qa_id': row['qa_id'], 'prediction': 'A'})
        continue
        
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            final_preds.append({'qa_id': row['qa_id'], 'prediction': 'A'})
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        question_prompt = (
            f"Question: {row['question']}\n"
            f"A) {row['a']}\nB) {row['b']}\nC) {row['c']}\nD) {row['d']}\n"
            f"Category: {row['category']}\n"
            "Based on the image, strictly output ONLY the correct option letter(s) (A, B, C, or D)."
        )
        
        enc_image = model.encode_image(image)
        answer = model.answer_question(enc_image, question_prompt, tokenizer)
        
        pred = ''.join([c for c in answer.upper() if c in 'ABCD'])
        if len(pred) == 0: pred = 'A'
        
        final_preds.append({'qa_id': row['qa_id'], 'prediction': pred})
    except Exception as e:
        print(f"Error on {row['qa_id']}: {e}")
        final_preds.append({'qa_id': row['qa_id'], 'prediction': 'A'})
        
pd.DataFrame(final_preds).to_csv('submission_moondream_gpu.csv', index=False)
print("Finished creating submission_moondream_gpu.csv")
