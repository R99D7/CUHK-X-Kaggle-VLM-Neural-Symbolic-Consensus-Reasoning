import os
import gc
import torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def extract_valid_prediction(raw_text, category):
    import re
    matches = re.findall(r'[A-D]', raw_text)
    if category in ['single', 'emotion', 'object_interaction', 'combination']:
        return matches[0] if matches else 'C'
    elif category == 'multi':
        unique_matches = list(dict.fromkeys(matches))
        return ''.join(sorted(unique_matches)) if unique_matches else 'B'
    elif category == 'sequence':
        unique_matches = list(dict.fromkeys(matches))
        missing = [x for x in ['A', 'B', 'C', 'D'] if x not in unique_matches]
        ans = unique_matches + missing
        return ''.join(ans[:4])
    return 'C'

print('Loading model...')
model = Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', torch_dtype=torch.bfloat16, device_map='auto')
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')

test_df = pd.read_csv('test_qa.csv').head(2)

print('Running 2 test samples...')
for idx, row in test_df.iterrows():
    qa_id = row['qa_id']
    video_path = f"test_video/{row['video_id']}.mp4"
    prompt = f"{row['question']}\nA: {row['A']}\nB: {row['B']}\nC: {row['C']}\nD: {row['D']}\nAnswer strictly with the option letter(s)."
    
    messages = [{'role': 'user', 'content': [{'type': 'video', 'video': video_path, 'max_pixels': 360 * 420, 'fps': 1.0}, {'type': 'text', 'text': prompt}]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt').to('cuda')
    
    generated_ids = model.generate(**inputs, max_new_tokens=15)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print(f'{qa_id} RAW:', output_text, '| PARSED:', extract_valid_prediction(output_text, row['category']))
    
    torch.cuda.empty_cache()
    gc.collect()

print('Dry run successful!')
