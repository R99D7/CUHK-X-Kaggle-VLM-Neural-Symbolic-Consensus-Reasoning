import os
import torch
from PIL import Image
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Use the fully downloaded snapshot directory
MODEL_PATH = r"C:\Users\MUTHURAMANRAMANATHAN\.cache\huggingface\hub\models--vikhyatk--moondream2\snapshots\6b714b26eea5cbd9f31e4edb2541c170afa935ba"

def run_inference():
    print(f"Loading Moondream from {MODEL_PATH}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print("Error loading model:", e)
        return

    test_df = pd.read_csv('test_qa.csv')
    video_dir = 'test_video'

    # Filter for ONLY multi and sequence categories (exactly 183 questions ~ 25% of 682)
    target_df = test_df[test_df['category'].isin(['multi', 'sequence'])]
    print(f"Found {len(target_df)} questions to evaluate (approx 25% of test set).")

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
            print(f"Processed {row['qa_id']}: {pred}")
        except Exception as e:
            print(f"Error on {row['qa_id']}: {e}")
            final_preds.append({'qa_id': row['qa_id'], 'prediction': 'A'})
            
    df_out = pd.DataFrame(final_preds)
    df_out.to_csv('submission_moondream_local_25percent.csv', index=False)
    print("Saved submission_moondream_local_25percent.csv successfully.")

if __name__ == '__main__':
    run_inference()
