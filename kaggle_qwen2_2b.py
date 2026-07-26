import os
import gc
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def run_qwen_kaggle():
    print("Loading Qwen2-VL-2B-Instruct...")
    # device_map="auto" will use TPU/CPU
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    dataset_dir = "/kaggle/input/cuhk-action-understanding" 
    test_csv_path = os.path.join(dataset_dir, "test_qa.csv")
    
    if not os.path.exists(test_csv_path):
        import glob
        found_paths = glob.glob("/kaggle/input/**/test_qa.csv", recursive=True)
        if found_paths:
            test_csv_path = found_paths[0]
            dataset_dir = os.path.dirname(test_csv_path)

    test_df = pd.read_csv(test_csv_path)
    print(f"Loaded test_qa.csv with {len(test_df)} rows.")
    
    predictions = {}
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        qa_id = row['qa_id']
        video_path = os.path.join(dataset_dir, "videos", row['path'])
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
        
        messages = [{"role": "user", "content": [{"type": "video", "video": video_path, "nframes": 4}, {"type": "text", "text": prompt}]}]
        
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            
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
        except Exception as e:
            print(f"Error processing {qa_id}: {e}")
            predictions[qa_id] = 'A'
            
        gc.collect()

    final_preds = []
    sample_df = pd.read_csv(os.path.join(dataset_dir, 'sample_submission.csv'))
    sample_dict = dict(zip(sample_df['qa_id'], sample_df['prediction']))
    cat_dict = dict(zip(test_df['qa_id'], test_df['category']))
    
    for qa_id, cat in cat_dict.items():
        pred_letter = predictions.get(qa_id, 'A')
        expected = str(sample_dict.get(qa_id, 'A'))
        final_pred = pred_letter * max(1, len(expected)) if cat == 'sequence' else pred_letter
        final_preds.append({'qa_id': qa_id, 'prediction': final_pred})
        
    pd.DataFrame(final_preds).to_csv('/kaggle/working/submission_qwen_2b.csv', index=False)
    print("Finished! Saved to /kaggle/working/submission_qwen_2b.csv")

if __name__ == '__main__':
    run_qwen_kaggle()
