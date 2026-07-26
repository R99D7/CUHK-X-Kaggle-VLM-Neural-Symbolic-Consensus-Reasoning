import os
import gc
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def run_qwen_local():
    print("Loading Qwen2-VL-2B-Instruct in bfloat16 on GPU...")
    # Load model in bfloat16 to fit perfectly inside the 6GB RTX 3050 VRAM
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    test_df = pd.read_csv("test_qa.csv")
    print(f"Loaded test_qa.csv with {len(test_df)} rows.")
    
    output_file = "qwen_local_preds.json"
    predictions = {}
    
    # Resume from checkpoint if exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            predictions = json.load(f)
        print(f"Resumed from {output_file}, {len(predictions)} already processed.")
        
    save_interval = 5
    
    # We will process 1 video at a time to keep VRAM low
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        qa_id = row['qa_id']
        
        if qa_id in predictions:
            continue
            
        video_path = f"videos/{row['path']}"
        question = row['question']
        A, B, C, D = row['A'], row['B'], row['C'], row['D']
        
        # Check if the video file exists locally
        if not os.path.exists(video_path):
            # Try to fix path just in case
            if "Depth.mp4" in video_path and not os.path.exists(video_path):
                # We saw Depth_Color.mp4 earlier. Let's fallback if needed.
                pass
        
        prompt = (
            f"Question: {question}\n"
            f"A: {A}\n"
            f"B: {B}\n"
            f"C: {C}\n"
            f"D: {D}\n"
            "Based on the video, answer the question with ONLY the correct letter (A, B, C, or D). Do not explain."
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "nframes": 2, # Extract just 2 frames to make it blazingly fast and save VRAM
                    },
                    {"type": "text", "text": prompt},
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
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=10)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Parse answer
            ans = 'A'
            for letter in ['A', 'B', 'C', 'D']:
                if letter in output_text.upper():
                    ans = letter
                    break
                    
            predictions[qa_id] = ans
            
        except Exception as e:
            print(f"Error processing {qa_id}: {e}")
            predictions[qa_id] = 'A' # Fallback
            
        if len(predictions) % save_interval == 0:
            with open(output_file, 'w') as f:
                json.dump(predictions, f)
                
        # Free memory aggressively to prevent GPU OOM
        torch.cuda.empty_cache()
        gc.collect()

    # Final save
    with open(output_file, 'w') as f:
        json.dump(predictions, f)
        
    # Convert to submission CSV format
    final_preds = []
    sample_df = pd.read_csv('sample_submission.csv')
    sample_dict = dict(zip(sample_df['qa_id'], sample_df['prediction']))
    cat_dict = dict(zip(test_df['qa_id'], test_df['category']))
    
    for qa_id, cat in cat_dict.items():
        pred_letter = predictions.get(qa_id, 'A')
        expected = str(sample_dict.get(qa_id, 'A'))
        
        if cat == 'sequence':
            final_pred = pred_letter * max(1, len(expected))
        else:
            final_pred = pred_letter
            
        final_preds.append({'qa_id': qa_id, 'prediction': final_pred})
        
    pd.DataFrame(final_preds).to_csv('submission_qwen_local.csv', index=False)
    print("Finished! Saved to submission_qwen_local.csv")

if __name__ == '__main__':
    run_qwen_local()
