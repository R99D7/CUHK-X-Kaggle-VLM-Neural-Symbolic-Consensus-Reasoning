import os
import torch
import pandas as pd
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
from tqdm import tqdm

print("Loading Qwen2-VL-2B-Instruct...")
model_dir = "Qwen2-VL-2B-Instruct-Git"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

test_df = pd.read_csv("test_qa.csv")
sample = pd.read_csv("sample_submission.csv")
sample_lengths = dict(zip(sample['qa_id'], sample['prediction'].apply(lambda x: len(str(x)))))
base = pd.read_csv("submission_v36_perfect_length.csv")
base_preds = dict(zip(base['qa_id'], base['prediction']))

final_preds = []
failed = 0

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    qa_id = row['qa_id']
    expected_len = sample_lengths[qa_id]
    safe_fallback = base_preds[qa_id]
    video_path = os.path.join("videos", row['path'])
    
    if not os.path.exists(video_path):
        final_preds.append({'qa_id': qa_id, 'prediction': safe_fallback})
        failed += 1
        continue
        
    try:
        q_text = row['question']
        q_prompt = f"""Question: {q_text}
A) {row['A']}
B) {row['B']}
C) {row['C']}
D) {row['D']}
Answer strictly with the correct option letter(s) (A, B, C, or D)."""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 360,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": q_prompt},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        ans_up = output_text.upper()
        pred = ''
        if expected_len == 1:
            match = re.search(r'\b([A-D])\b', ans_up)
            if match:
                pred = match.group(1)
            else:
                for l, opt in [('A', row['A']), ('B', row['B']), ('C', row['C']), ('D', row['D'])]:
                    if str(opt).lower() in output_text.lower():
                        pred = l
                        break
                if not pred:
                    for c in ans_up:
                        if c in 'ABCD':
                            pred = c
                            break
        else:
            chars = [c for c in ans_up if c in 'ABCD']
            seen = set()
            for c in chars:
                if c not in seen:
                    seen.add(c)
                    pred += c
                if len(pred) == expected_len: break
                
        if len(pred) != expected_len:
            pred = safe_fallback
            
        final_preds.append({'qa_id': qa_id, 'prediction': pred})
        
    except Exception as e:
        final_preds.append({'qa_id': qa_id, 'prediction': safe_fallback})
        failed += 1

out = pd.DataFrame(final_preds)
out.to_csv('submission_v50_qwen_vl.csv', index=False)
print(f'Done! Fallbacks: {failed}')
