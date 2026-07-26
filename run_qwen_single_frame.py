import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import pandas as pd
import cv2
import os
from tqdm import tqdm

def extract_middle_frame(video_path, output_path):
    if os.path.exists(output_path):
        return True
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    cap.release()
    return ret

os.makedirs('temp_frames', exist_ok=True)
df = pd.read_csv('test_qa.csv')
submission = []

print('Loading model...')
model = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto',
    load_in_4bit=True
)
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')

for idx, row in tqdm(df.iterrows(), total=len(df)):
    vid_path = 'large_model_track_test/' + row['path'].replace('large_model_track_test/', '')
    frame_path = f'temp_frames/{row["qa_id"]}.jpg'
    
    if not extract_middle_frame(vid_path, frame_path):
        submission.append({'qa_id': row['qa_id'], 'prediction': 'A'})
        continue
        
    try:
        img = Image.open(frame_path)
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': img},
                    {'type': 'text', 'text': 'You are a visual assistant. Read the question and options in the image. Answer the question with just the single correct letter (A, B, C, or D).'}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = processor.image_processor(images=[img], videos=None, return_tensors='pt')
        
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
        ans = 'A'
        for letter in ['A', 'B', 'C', 'D']:
            if letter in output_text.upper():
                ans = letter
                break
        submission.append({'qa_id': row['qa_id'], 'prediction': ans})
    except Exception as e:
        submission.append({'qa_id': row['qa_id'], 'prediction': 'A'})

pd.DataFrame(submission).to_csv('submission_qwen_single_frame.csv', index=False)
print('Done!')
