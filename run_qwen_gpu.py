import os
import torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import gc

print("Starting Qwen2-VL Local Inference with 4-bit Quantization!")

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print(f"Loading {MODEL_ID} in 4-bit precision to fit comfortably in 6GB VRAM...")
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

test_qa_path = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\test_qa.csv"
test_df = pd.read_csv(test_qa_path)

# 50 percent of the total data
target_df = test_df.head(341)
print(f"Evaluating {len(target_df)} questions...")

final_preds = []

video_base_dir = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\videos"
if not os.path.exists(video_base_dir):
    video_base_dir = r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\test_video"

for idx, row in target_df.iterrows():
    vid_path = os.path.join(video_base_dir, row['path'])
    if not os.path.exists(vid_path):
        vid_path = os.path.join(r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\videos", row['path'])
    if not os.path.exists(vid_path):
        vid_path = os.path.join(r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK\test_video", row['path'])
    if not os.path.exists(vid_path):
        vid_path = os.path.join(r"C:\Users\MUTHURAMANRAMANATHAN\Downloads\CUHK", row['path'].replace('large_model_track_test/', 'test_video/large_model_track_test/'))

    default_pred = 'ABCD' if row['category'] == 'sequence' else 'AB' if row['category'] == 'multi' else 'A'
    
    if not os.path.exists(vid_path):
        print(f"[{row['qa_id']}] Video not found: {vid_path}")
        final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})
        continue
        
    try:
        instruction = "Based on the video, strictly output only the correct option letter (A, B, C, or D)."
        if row['category'] == 'sequence':
            instruction = "Based on the video, strictly output the correct sequence as a permutation of A, B, C, and D (e.g. ABCD). You MUST output all 4 letters in the correct chronological order."
        elif row['category'] == 'multi':
            instruction = "Based on the video, strictly output all correct option letters (e.g. AB, ACD)."

        question_prompt = (
            f"Question: {row['question']}\n"
            f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}\n"
            f"Category: {row['category']}\n"
            f"{instruction}"
        )
        
        # Extremely conservative video parsing to prevent OOM during forward pass
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": vid_path,
                        "max_pixels": 65536,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": question_prompt},
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Free up CPU memory
        del image_inputs, video_inputs
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        
        # Clean answer
        pred = ''.join([c for c in answer.upper() if c in 'ABCD'])
        
        if row['category'] == 'sequence':
            pred = ''.join(dict.fromkeys(pred))
            missing = [c for c in 'ABCD' if c not in pred]
            pred = pred + ''.join(missing)
            if len(pred) != 4 or sorted(pred) != ['A', 'B', 'C', 'D']:
                pred = 'ABCD'
        elif row['category'] == 'multi':
            pred = ''.join(sorted(dict.fromkeys(pred)))
            if len(pred) == 0:
                pred = 'AB'
        else:
            if len(pred) == 0: pred = 'A'
            pred = pred[0]
            
        final_preds.append({'qa_id': row['qa_id'], 'prediction': pred})
        print(f"Processed {idx+1}/{len(target_df)}: {row['qa_id']} -> {pred}")
        
    except Exception as e:
        print(f"Error on {row['qa_id']}: {e}")
        final_preds.append({'qa_id': row['qa_id'], 'prediction': default_pred})

df_qwen = pd.DataFrame(final_preds)
df_qwen.to_csv('submission_qwen_local.csv', index=False)
print("Finished Qwen local inference!")

# Blend with ultimate_v9
df_base = pd.read_csv('submission_ultimate_v9.csv')
df_qwen_dict = dict(zip(df_qwen['qa_id'], df_qwen['prediction']))
df_base['prediction'] = df_base.apply(
    lambda row: df_qwen_dict.get(row['qa_id'], row['prediction']),
    axis=1
)

df_base.to_csv('submission_ultimate_v10.csv', index=False)
print("Finished creating final blended submission_ultimate_v10.csv!")
