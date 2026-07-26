import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import pandas as pd
import cv2
import os
from tqdm import tqdm

df = pd.read_csv('test_qa.csv')
os.makedirs('temp_frames_all', exist_ok=True)

print('Loading Qwen2-VL-2B-Instruct...')
model = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto',
    load_in_4bit=True
)
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')

predictions = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    qa_id = row['qa_id']
    question = row['question']
    is_seq = row['source'] == 'HARn'
    
    # Extract frame
    vid_path = 'large_model_track_test/' + row['path'].replace('large_model_track_test/', '')
    frame_path = f'temp_frames_all/{qa_id}.jpg'
    
    if not os.path.exists(frame_path):
        cap = cv2.VideoCapture(vid_path)
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(frame_path, frame)
        cap.release()
        
    if not os.path.exists(frame_path):
        predictions.append({'qa_id': qa_id, 'prediction': 'A'})
        continue
        
    try:
        img = Image.open(frame_path).convert('RGB')
        
        # Build prompt
        text_prompt = f"Question: {question}\nOptions:\nA) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}\n"
        if is_seq:
            text_prompt += "Output the correct sequence of letters (e.g. ABCD or BDA). No other text."
        else:
            text_prompt += "Output just the single correct letter (A, B, C, or D). No other text."
            
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': img},
                    {'type': 'text', 'text': text_prompt}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
        ans = ''.join([c for c in output_text.upper() if c in 'ABCD'])
        if not ans: ans = 'A'
        
        predictions.append({'qa_id': qa_id, 'prediction': ans})
    except Exception as e:
        predictions.append({'qa_id': qa_id, 'prediction': 'A'})
        
    # Checkpoint every 50
    if idx % 50 == 0:
        pd.DataFrame(predictions).to_csv('submission_qwen_local_raw.csv', index=False)

pd.DataFrame(predictions).to_csv('submission_qwen_local_raw.csv', index=False)
print('Done processing all images!')
