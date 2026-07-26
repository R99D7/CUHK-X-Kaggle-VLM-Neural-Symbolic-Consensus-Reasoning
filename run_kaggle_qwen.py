import pandas as pd
import json
import os
import subprocess
import time

notebook_code = """
import pandas as pd
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys
import os
import cv2

# Ensure we use 4-bit quantization
from transformers import BitsAndBytesConfig

# Paths
test_csv = "/kaggle/input/cuhk-x-large-model-track/test_qa.csv"
video_dir = "/kaggle/input/cuhk-x-large-model-track/test_video/test_video"

print("Loading Qwen2-VL-2B-Instruct...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

df = pd.read_csv(test_csv)
results = []

print("Starting inference...")
for idx, row in df.iterrows():
    qa_id = row['qa_id']
    question = row['question']
    ops = [row['a0'], row['a1'], row['a2'], row['a3']]
    
    vid_path = os.path.join(video_dir, f"{row['video_id']}.mp4")
    if not os.path.exists(vid_path):
        results.append({'qa_id': qa_id, 'prediction': 'A'})
        continue
        
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": vid_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": f"Question: {question}\\nOptions:\\nA: {ops[0]}\\nB: {ops[1]}\\nC: {ops[2]}\\nD: {ops[3]}\\nAnswer exactly with a single letter (A, B, C, or D)."},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=5, pad_token_id=processor.tokenizer.pad_token_id)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        # Extract A, B, C, or D
        pred = 'A'
        for letter in ['A', 'B', 'C', 'D']:
            if letter in output_text.upper():
                pred = letter
                break
                
        results.append({'qa_id': qa_id, 'prediction': pred})
    except Exception as e:
        print(f"Error on {qa_id}: {e}")
        results.append({'qa_id': qa_id, 'prediction': 'A'})
        
    if idx % 10 == 0:
        print(f"Processed {idx}/{len(df)}")
        pd.DataFrame(results).to_csv("submission.csv", index=False)

pd.DataFrame(results).to_csv("submission.csv", index=False)
print("Saved submission.csv")
"""

os.makedirs("kaggle_qwen2_submission", exist_ok=True)
with open("kaggle_qwen2_submission/kernel.py", "w") as f:
    f.write(notebook_code)

kernel_meta = {
  "id": "kevinpal/cuhk-qwen2-vl-inference",
  "title": "cuhk-qwen2-vl-inference",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": ["kevinpal/cuhk-x-large-model-track"],
  "competition_sources": [],
  "kernel_sources": []
}

with open("kaggle_qwen2_submission/kernel-metadata.json", "w") as f:
    json.dump(kernel_meta, f, indent=2)

print("Pushing notebook to Kaggle...")
subprocess.run(["kaggle", "kernels", "push", "-p", "kaggle_qwen2_submission"], check=True)
print("Pushed! Kaggle will run Qwen2-VL 4-bit inference in the cloud in 10-15 mins!")
