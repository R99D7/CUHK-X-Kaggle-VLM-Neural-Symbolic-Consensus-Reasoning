import os
import gc
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def run_local_surgical():
    print('Loading Qwen2-VL-2B-Instruct locally from cache...')
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct', 
        torch_dtype=torch.bfloat16, 
        device_map='cuda',
        local_files_only=True
    )
    processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
    
    test_df = pd.read_csv('test_qa.csv')
    contentious = ['test_0112', 'test_0113', 'test_0114', 'test_0116', 'test_0117', 'test_0119', 'test_0120', 'test_0123', 'test_0124', 'test_0125']
    test_df = test_df[test_df['qa_id'].isin(contentious)]
    print(f'Loaded {len(test_df)} contentious videos.')
    
    predictions = {}
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        qa_id = row['qa_id']
        video_path = f"test_videos/{row['path']}"
        if not os.path.exists(video_path):
            video_path = f"test_videos/large_model_track_test/{row['path']}"
            
        question = row['question']
        A, B, C, D = row['A'], row['B'], row['C'], row['D']
        
        prompt = (
            f"Question: {question}\n"
            f"A: {A}\n"
            f"B: {B}\n"
            f"C: {C}\n"
            f"D: {D}\n"
            "Based on the video, answer the question with ONLY the correct letter (A, B, C, or D). Do not explain."
        )
        
        messages = [{'role': 'user', 'content': [{'type': 'video', 'video': video_path, 'max_pixels': 256*256, 'nframes': 4}, {'type': 'text', 'text': prompt}]}]
        
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors='pt',
            ).to('cuda')
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=10)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            
            ans = 'A'
            for letter in ['A', 'B', 'C', 'D']:
                if letter in output_text.upper():
                    ans = letter; break
            predictions[qa_id] = ans
            print(f"{qa_id}: {ans}")
        except Exception as e:
            print(f"Error processing {qa_id}: {e}")
            
        gc.collect()
        torch.cuda.empty_cache()
        
    print("Done generating true-vision answers. Injecting into best submission...")
    
    best_df = pd.read_csv('submission_ultimate.csv')
    final_preds = []
    changes = 0
    
    for idx, row in best_df.iterrows():
        qa_id = row['qa_id']
        pred = row['prediction']
        
        if qa_id in predictions:
            new_pred = predictions[qa_id]
            if str(pred) != str(new_pred):
                print(f"OVERRIDE: {qa_id} changed from {pred} to {new_pred}")
                pred = new_pred
                changes += 1
                
        final_preds.append({'qa_id': qa_id, 'prediction': pred})
        
    pd.DataFrame(final_preds).to_csv('submission_local_qwen_override.csv', index=False)
    print(f"Made {changes} true-vision overrides! Saved to submission_local_qwen_override.csv")

if __name__ == "__main__":
    run_local_surgical()
