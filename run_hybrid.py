import os
import torch
import cv2
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np

print("Starting Moondream2 + MiniLM Hybrid Inference!")

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"

print("Loading moondream2 model...")
moondream_path = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\moondream2"
model = AutoModelForCausalLM.from_pretrained(
    moondream_path, trust_remote_code=True, revision=revision,
    torch_dtype=torch.float16, device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(moondream_path, revision=revision)

print("Loading SentenceTransformer NLP model...")
nlp_model = SentenceTransformer('all-MiniLM-L6-v2')

test_qa_path = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\test_qa.csv"
test_df = pd.read_csv(test_qa_path)

final_preds = []
processed_qa_ids = set()
csv_file = 'submission_hybrid_moondream.csv'
if os.path.exists(csv_file):
    try:
        existing_df = pd.read_csv(csv_file)
        for _, r in existing_df.iterrows():
            final_preds.append({'qa_id': r['qa_id'], 'prediction': r['prediction']})
            processed_qa_ids.add(r['qa_id'])
        print(f"Resuming from {len(processed_qa_ids)} previously processed videos!")
    except Exception as e:
        print(f"Could not load existing CSV: {e}")

video_base_dir = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\videos"

for idx, row in test_df.iterrows():
    if row['qa_id'] in processed_qa_ids:
        continue
        
    vid_path = os.path.join(video_base_dir, row['path'])
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
                frame = cv2.resize(frame, (256, 256))
                frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})
            continue
            
        while len(frames) < 4:
            frames.append(frames[-1])
            
        # Stitch frames into a 2x2 grid
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        grid_img = np.vstack((top_row, bottom_row))
        pil_img = Image.fromarray(grid_img)
        
        enc_image = model.encode_image(pil_img)
        
        # Step 1: Generate Visual Description
        question = row['question']
        desc_prompt = f"Describe exactly what is happening in this video sequence (grid of 4 chronological frames). The context is: {question}. Provide a detailed description of the actions, objects, and emotions."
        visual_description = model.answer_question(enc_image, desc_prompt, tokenizer)
        
        # Step 2: Semantic Matching with Options
        options = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
        letters = ['A', 'B', 'C', 'D']
        
        # We append the question to the description to give the NLP model full context
        full_context = f"Question: {question}. Video Description: {visual_description}"
        
        context_emb = nlp_model.encode(full_context, convert_to_tensor=True)
        option_embs = nlp_model.encode(options, convert_to_tensor=True)
        
        similarities = util.pytorch_cos_sim(context_emb, option_embs)[0].cpu().numpy()
        
        pred = 'A'
        
        if row['category'] == 'single':
            best_idx = np.argmax(similarities)
            pred = letters[best_idx]
            
        elif row['category'] == 'sequence':
            sorted_indices = np.argsort(similarities)[::-1]
            pred = "".join([letters[i] for i in sorted_indices])
            
        elif row['category'] == 'multi':
            mean_sim = np.mean(similarities)
            valid_indices = np.where(similarities > mean_sim)[0]
            if len(valid_indices) == 0:
                valid_indices = [np.argmax(similarities)]
            pred = "".join([letters[i] for i in valid_indices])
            
        else:
            best_idx = np.argmax(similarities)
            pred = letters[best_idx]
            
        final_preds.append({'qa_id': row['qa_id'], 'prediction': pred})
        print(f"Processed {idx+1}/{len(test_df)}: {pred}", flush=True)
            
    except Exception as e:
        print(f"Error on {row['qa_id']}: {e}", flush=True)
        final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})
        
    # Always save incrementally
    pd.DataFrame(final_preds).to_csv('submission_hybrid_moondream.csv', index=False)

df_moon = pd.DataFrame(final_preds)
df_moon.to_csv('submission_hybrid_moondream.csv', index=False)
print("Finished Hybrid Moondream + NLP Inference!")
