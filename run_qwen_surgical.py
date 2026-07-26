import os
import torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

print("Starting Qwen2-VL Surgical Inference (Emotion & Combination Categories Only)!")

MODEL_ID = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\Qwen2-VL-2B-Instruct"
TEST_QA_PATH = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\test_qa.csv"
VIDEOS_DIR = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\videos\large_model_track_test"

print(f"Loading {MODEL_ID} in 4-bit precision...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    device_map="auto",
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

df = pd.read_csv(TEST_QA_PATH)
# SURGICAL TARGETING: We only run the massive VLM on visual-heavy categories that text ML fails at
target_df = df[df['category'].isin(['emotion', 'combination'])].reset_index(drop=True)
print(f"Found {len(target_df)} target questions (emotion & combination) out of {len(df)} total.")

results = []

for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
    qa_id = row['qa_id']
    question = row['question']
    opts = f"A: {row['A']}, B: {row['B']}, C: {row['C']}, D: {row['D']}"
    
    # Locate the video
    vid_id = qa_id.split('_')[0] if '_' in qa_id else qa_id
    if len(vid_id) > 12:
        vid_id = '_'.join(qa_id.split('_')[:3])
        
    vid_path = os.path.join(VIDEOS_DIR, vid_id, "Depth", "Depth.mp4")
    if not os.path.exists(vid_path):
        vid_path = os.path.join(VIDEOS_DIR, vid_id, "Depth_Color", "Depth_Color.mp4")
        if not os.path.exists(vid_path):
            print(f"WARNING: Video not found for {vid_id}")
            continue

    prompt = (
        f"You are an expert action and emotion recognition system analyzing depth videos. "
        f"Watch the video carefully and answer the multiple-choice question. "
        f"Question: {question}\nOptions: {opts}\n"
        f"Answer with ONLY the single letter of the correct option (A, B, C, or D). Do not provide explanations."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": vid_path, "fps": 1.0}, # Extract 1 frame per second to save VRAM
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    try:
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        generated_ids = model.generate(**inputs, max_new_tokens=4, temperature=0.1)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Clean the output to get just the letter
        clean_ans = ""
        for char in output_text.upper():
            if char in ['A', 'B', 'C', 'D']:
                clean_ans += char
                
        if len(clean_ans) == 0:
            clean_ans = "A" # Fallback
            
        results.append({"qa_id": qa_id, "qwen_prediction": clean_ans[0]})
        
    except Exception as e:
        print(f"Error on {qa_id}: {e}")
        results.append({"qa_id": qa_id, "qwen_prediction": "A"}) # Fallback
        
out_df = pd.DataFrame(results)
out_df.to_csv('qwen_surgical_predictions.csv', index=False)
print("Finished saving Qwen predictions to qwen_surgical_predictions.csv")
