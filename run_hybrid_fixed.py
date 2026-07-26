"""
FIXED: Use Depth_Color videos instead of Depth videos.
Depth = raw depth maps (dark, nearly black - Moondream sees nothing)
Depth_Color = colorized depth (looks like real video - Moondream works!)

This is why score dropped to 0.237 - fix it by using Depth_Color.
"""
import os
import torch
import cv2
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np

print("Starting FIXED Hybrid Inference with Depth_Color videos!")

moondream_path = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\moondream2"
print("Loading moondream2...")
model = AutoModelForCausalLM.from_pretrained(
    moondream_path, trust_remote_code=True, revision="2024-05-20",
    torch_dtype=torch.float16, device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(moondream_path, revision="2024-05-20")

print("Loading NLP model...")
nlp = SentenceTransformer('all-MiniLM-L6-v2')

test_df = pd.read_csv('test_qa.csv')
print(f"Evaluating {len(test_df)} questions using Depth_Color videos!")

final_preds = []
processed_ids = set()
csv_file = 'submission_depthcolor.csv'
if os.path.exists(csv_file):
    ex = pd.read_csv(csv_file)
    for _, r in ex.iterrows():
        final_preds.append({'qa_id': r['qa_id'], 'prediction': r['prediction']})
        processed_ids.add(r['qa_id'])
    print(f"Resuming from {len(processed_ids)} done.")

base_dir = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\videos"

for idx, row in test_df.iterrows():
    if row['qa_id'] in processed_ids:
        continue

    # KEY FIX: replace Depth/Depth.mp4 with Depth_Color/Depth_Color.mp4
    depth_path = row['path']
    color_path = depth_path.replace('Depth/Depth.mp4', 'Depth_Color/Depth_Color.mp4')
    vid_path = os.path.join(base_dir, color_path)

    opts = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
    letters = ['A', 'B', 'C', 'D']
    cat = row['category']
    default_pred = 'A'

    if not os.path.exists(vid_path):
        final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})
        pd.DataFrame(final_preds).to_csv(csv_file, index=False)
        continue

    try:
        cap = cv2.VideoCapture(vid_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [int(total*0.1), int(total*0.35), int(total*0.65), int(total*0.9)]
        frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(fi, total-1))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (256, 256))
                frames.append(frame)
        cap.release()

        while len(frames) < 4:
            frames.append(frames[-1] if frames else np.zeros((256,256,3), dtype=np.uint8))

        grid = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])])
        pil_img = Image.fromarray(grid)
        enc = model.encode_image(pil_img)

        question = row['question']
        desc = model.answer_question(enc,
            f"Describe in detail what actions and activities are happening in this video. Context: {question}",
            tokenizer)

        full_ctx = f"Question: {question}. Video: {desc}"
        ctx_emb = nlp.encode(full_ctx, convert_to_tensor=True)
        opt_embs = nlp.encode(opts, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(ctx_emb, opt_embs)[0].cpu().numpy()

        if cat == 'single' or cat == 'emotion' or cat == 'object_interaction' or cat == 'combination':
            pred = letters[np.argmax(sims)]
        elif cat == 'sequence':
            si = np.argsort(sims)[::-1]
            pred = ''.join(letters[i] for i in si)
        elif cat == 'multi':
            mean_s = np.mean(sims)
            vi = [i for i, s in enumerate(sims) if s > mean_s]
            if not vi:
                vi = [np.argmax(sims)]
            pred = ''.join(letters[i] for i in vi)
        else:
            pred = letters[np.argmax(sims)]

        final_preds.append({'qa_id': row['qa_id'], 'prediction': pred})
        print(f"[{idx+1}/{len(test_df)}] {cat}: {pred}  | {desc[:60]}", flush=True)

    except Exception as e:
        print(f"Error {row['qa_id']}: {e}", flush=True)
        final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})

    pd.DataFrame(final_preds).to_csv(csv_file, index=False)

pd.DataFrame(final_preds).to_csv(csv_file, index=False)
print(f"\nDone! Saved {len(final_preds)} predictions to {csv_file}")
