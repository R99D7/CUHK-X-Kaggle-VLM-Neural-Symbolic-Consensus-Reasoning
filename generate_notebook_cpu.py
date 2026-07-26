import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10"
    }
}

code_cells = [
    """!pip install -q einops timm==0.9.16""",
    
    """import os
import glob
import torch
from PIL import Image
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.phi.configuration_phi import PhiConfig
import cv2

print("Running on CPU!")""",
    
    """MODEL_ID = "vikhyatk/moondream2"

from transformers.modeling_utils import PreTrainedModel
PreTrainedModel.all_tied_weights_keys = property(lambda self: {})

PhiConfig.pad_token_id = None
if not hasattr(PhiConfig, 'pad_token_id'):
    setattr(PhiConfig, 'pad_token_id', None)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)""",
    
    """# Find test_qa.csv dynamically
test_qa_paths = glob.glob('/kaggle/input/**/test_qa.csv', recursive=True)
if not test_qa_paths:
    raise FileNotFoundError("Could not find test_qa.csv in /kaggle/input!")
test_qa_path = test_qa_paths[0]
print(f"Found test_qa.csv at {test_qa_path}")

test_df = pd.read_csv(test_qa_path)
target_df = test_df[test_df['category'].isin(['multi', 'sequence'])]
print(f"Evaluating {len(target_df)} questions...")

# Find all mp4 files to map them robustly
all_mp4s = glob.glob('/kaggle/input/**/*.mp4', recursive=True)
print(f"Found {len(all_mp4s)} mp4 files in /kaggle/input")

final_preds = []

for idx, row in target_df.iterrows():
    video_path = None
    
    # Try to resolve path using row['path'] or row['video_id']
    if 'path' in row:
        row_p = str(row['path'])
        for p in all_mp4s:
            if row_p in p or row_p.split('/')[-2] + '/' + row_p.split('/')[-1] in p:
                video_path = p
                break
    elif 'video_id' in row:
        vid = str(row['video_id']) + '.mp4'
        for p in all_mp4s:
            if vid in p:
                video_path = p
                break

    if video_path is None or not os.path.exists(video_path):
        print(f"Could not find video for qa_id {row['qa_id']}")
        default_pred = 'ABCD' if row['category'] == 'sequence' else 'AB'
        final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})
        continue
        
    try:
        cap = cv2.VideoCapture(video_path)
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
            f"Question: {row['question']}\\n"
            f"A) {row['a']}\\nB) {row['b']}\\nC) {row['c']}\\nD) {row['d']}\\n"
            f"Category: {row['category']}\\n"
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
df_moon.to_csv('submission_moondream_cpu.csv', index=False)
print("Finished Moondream inference!")""",
    
    """# Blend the Moondream predictions with the baseline
base_csv_paths = glob.glob('/kaggle/input/**/submission_ultimate_v3.csv', recursive=True)
if not base_csv_paths:
    print("Could not find submission_ultimate_v3.csv, saving just moondream preds")
    df_base = df_moon
else:
    df_base = pd.read_csv(base_csv_paths[0])
    df_moon_dict = dict(zip(df_moon['qa_id'], df_moon['prediction']))
    df_base['prediction'] = df_base.apply(
        lambda row: df_moon_dict.get(row['qa_id'], row['prediction']),
        axis=1
    )

df_base.to_csv('submission_ultimate_v8.csv', index=False)
print("Finished creating final blended submission_ultimate_v8.csv!")"""
]

nb['cells'] = [nbf.v4.new_code_cell(code) for code in code_cells]

with open('cuhk-moondream-cpu.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Successfully generated notebook with proper prediction formats!")
