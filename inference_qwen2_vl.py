import os
import gc
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def extract_valid_prediction(raw_text, category):
    import re
    # Find all occurrences of A, B, C, D
    matches = re.findall(r'[A-D]', raw_text)
    
    if category in ['single', 'emotion', 'object_interaction', 'combination']:
        if matches:
            return matches[0]
        else:
            return 'C' # Fallback
            
    elif category == 'multi':
        # 1 to 3 characters, alphabetical, no duplicates
        unique_matches = list(dict.fromkeys(matches))
        if not unique_matches:
            return 'B'
        if len(unique_matches) > 3:
            unique_matches = unique_matches[:3]
        return ''.join(sorted(unique_matches))
        
    elif category == 'sequence':
        # exactly 4 unique characters
        unique_matches = list(dict.fromkeys(matches))
        missing = [x for x in ['A', 'B', 'C', 'D'] if x not in unique_matches]
        ans = unique_matches + missing
        return ''.join(ans[:4])
        
    return 'C'

def main():
    print("Loading model and processor...")
    # Load the model in bfloat16 to fit in 6GB VRAM
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    test_df = pd.read_csv('test_qa.csv')
    
    results = []
    
    print("Starting inference on test videos...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        qa_id = row['qa_id']
        video_path = f"test_video/{row['video_id']}.mp4"
        question = row['question']
        opts = f"A: {row['A']}\nB: {row['B']}\nC: {row['C']}\nD: {row['D']}"
        category = row['category']
        
        prompt = f"{question}\n{opts}\nAnswer strictly with the option letter(s)."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        try:
            # Preparation for inference
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # Inference
            generated_ids = model.generate(**inputs, max_new_tokens=15)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            final_ans = extract_valid_prediction(output_text, category)
        except Exception as e:
            print(f"Error on {qa_id}: {e}")
            final_ans = extract_valid_prediction("C", category)
            
        results.append({'qa_id': qa_id, 'prediction': final_ans})
        
        # Free memory aggressively 
        torch.cuda.empty_cache()
        gc.collect()

    pd.DataFrame(results).to_csv('submission_v132_qwen2_vl.csv', index=False)
    print("Done! Saved submission_v132_qwen2_vl.csv")

if __name__ == "__main__":
    main()
