import nbformat as nbf

nb = nbf.v4.new_notebook()

code_cells = [
    """!pip install -q einops
!pip install -q timm==0.9.16
!pip install -q transformers==4.39.3""",
    
    """import os
import torch
from PIL import Image
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import cv2

print("PyTorch GPU Available:", torch.cuda.is_available())""",
    
    """MODEL_ID = "vikhyatk/moondream2"

# Monkey patch for compatibility with Kaggle's older transformers version
from transformers.modeling_utils import PreTrainedModel
PreTrainedModel.all_tied_weights_keys = property(lambda self: {})

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)""",
    
    """test_df = pd.read_csv('/kaggle/input/cuhk-x-large-model-track/Large-Model-Track/Testing/data/test_qa.csv')
video_dir = '/kaggle/input/cuhk-x-large-model-track/Large-Model-Track/Testing/data/test_video'

# The user requested 25% of the files: we use the 'multi' and 'sequence' categories (183 / 729 ~ 25%)
target_df = test_df[test_df['category'].isin(['multi', 'sequence'])]
print(f"Evaluating {len(target_df)} questions...")

final_preds = []

for idx, row in target_df.iterrows():
    video_path = os.path.join(video_dir, str(row['video_id']) + '.mp4')
    if not os.path.exists(video_path):
        final_preds.append({'qa_id': row['qa_id'], 'prediction': 'A'})
        continue
        
    try:
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
            f"Question: {row['question']}\\n"
            f"A) {row['a']}\\nB) {row['b']}\\nC) {row['c']}\\nD) {row['d']}\\n"
            f"Category: {row['category']}\\n"
            "Based on the image, strictly output ONLY the correct option letter(s) (A, B, C, or D)."
        )
        
        enc_image = model.encode_image(image)
        answer = model.answer_question(enc_image, question_prompt, tokenizer)
        
        pred = ''.join([c for c in answer.upper() if c in 'ABCD'])
        if len(pred) == 0: pred = 'A'
        
        final_preds.append({'qa_id': row['qa_id'], 'prediction': pred})
        print(f"Processed {idx+1}/{len(target_df)}: {row['qa_id']} -> {pred}")
    except Exception as e:
        print(f"Error on {row['qa_id']}: {e}")
        final_preds.append({'qa_id': row['qa_id'], 'prediction': 'A'})
        
df_moon = pd.DataFrame(final_preds)
df_moon.to_csv('submission_moondream_gpu.csv', index=False)
print("Finished Moondream inference!")""",
    
    """# Blend the Moondream predictions with the baseline
df_base = pd.read_csv('/kaggle/input/cuhk-x-large-model-track/Large-Model-Track/Testing/data/submission_ultimate_v3.csv')
# (If your submission_ultimate_v3.csv is uploaded as a Kaggle dataset, adjust the path above as needed! 
# You might need to add it as a dataset to your Kaggle notebook first.)

# Update baseline with our Moondream predictions
df_moon_dict = dict(zip(df_moon['qa_id'], df_moon['prediction']))
df_base['prediction'] = df_base.apply(
    lambda row: df_moon_dict.get(row['qa_id'], row['prediction']),
    axis=1
)

df_base.to_csv('submission_ultimate_v6.csv', index=False)
print("Finished creating final blended submission_ultimate_v6.csv!")"""
]

nb['cells'] = [nbf.v4.new_code_cell(code) for code in code_cells]

with open('cuhk-moondream-gpu.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Successfully generated notebook!")
