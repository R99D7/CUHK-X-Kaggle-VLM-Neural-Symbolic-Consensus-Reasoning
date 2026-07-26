# ==============================================================================
# CUHK-X LARGE MODEL TRACK - 1ST PLACE VLM SCRIPT
# INSTRUCTIONS:
# 1. Create a New Notebook on Kaggle.
# 2. Set the Accelerator to GPU T4 x2 (or better).
# 3. Add your HuggingFace Token in Kaggle Secrets (name it 'HF_TOKEN').
# 4. Paste this code and Run All.
# ==============================================================================

import os
import pandas as pd
from kaggle_secrets import UserSecretsClient
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Step 1: Authenticating with HuggingFace...")
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")

print("Step 2: Downloading the Gated Test Videos (2GB)...")
dataset_path = snapshot_download(
    repo_id="Kevin-Pal/CUHK-X_Large_Model_Track",
    repo_type="dataset",
    token=hf_token,
    allow_patterns=["Large-Model-Track/Testing/data/*"]
)

# Unzip the videos
os.system(f"unzip -q {dataset_path}/Large-Model-Track/Testing/data/large_model_track_test.zip -d ./videos")

print("Step 3: Loading Vision-Language Model (Qwen-VL)...")
# Using Qwen-VL-Chat as it is lightweight enough for T4 GPUs and excellent at Video QA
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()

print("Step 4: Running Inference on all 682 Test Videos...")
test_df = pd.read_csv("test_qa.csv")
predictions = []

for idx, row in test_df.iterrows():
    video_path = f"./videos/{row['path']}.mp4"
    question = row['question']
    
    # Format options
    options_text = f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}"
    
    # Prompt engineering based on category
    cat = row['category']
    if cat in ['multi', 'sequence']:
        prompt = f"Watch the video and answer the question. This is a {cat} question. Output ONLY the combination or sequence of letters (e.g. BCD or DCBA).\nQuestion: {question}\n{options_text}"
    else:
        prompt = f"Watch the video and answer the question. Output ONLY a single letter (A, B, C, or D).\nQuestion: {question}\n{options_text}"
        
    query = tokenizer.from_list_format([
        {'video': video_path},
        {'text': prompt},
    ])
    
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)
    
    pred = model.generate(**inputs, max_new_tokens=10)
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    
    # Clean the response to ensure it only contains the letters
    clean_resp = ''.join([c for c in response if c in 'ABCD'])
    predictions.append(clean_resp if clean_resp else "A")

print("Step 5: Saving Final Submission...")
sub_df = pd.DataFrame({
    'qa_id': test_df['id'],
    'prediction': predictions
})
sub_df.to_csv('submission.csv', index=False)
print("Done! Submit 'submission.csv' to the leaderboard.")
