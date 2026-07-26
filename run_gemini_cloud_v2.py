import os
import sys
import time
import pandas as pd
import google.generativeai as genai

API_KEY = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-3.5-flash')

df = pd.read_csv('test_qa.csv')
predictions = []

# Load existing progress if any
csv_path = 'submission_gemini_cloud_raw.csv'
if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)
    existing_ids = set(existing_df['qa_id'])
    predictions = existing_df.to_dict('records')
    print(f"Resuming... found {len(existing_ids)} completed predictions.", flush=True)
else:
    existing_ids = set()
    print(f"Starting Cloud Inference for {len(df)} videos...", flush=True)

for idx, row in df.iterrows():
    qa_id = row['qa_id']
    if qa_id in existing_ids:
        continue
        
    vid_path = 'large_model_track_test/' + row['path'].replace('large_model_track_test/', '')
    if not os.path.exists(vid_path):
        pred = 'A'
        predictions.append({'qa_id': qa_id, 'prediction': pred})
        pd.DataFrame(predictions).to_csv(csv_path, index=False)
        print(f'{qa_id}: {pred} (Missing Video)', flush=True)
        continue
        
    try:
        video_file = genai.upload_file(path=vid_path)
        while video_file.state.name == 'PROCESSING':
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
            
        if video_file.state.name == 'FAILED':
            pred = 'A'
        else:
            prompt = f"""
Watch the video and answer the multiple-choice question.
Question: {row['question']}
A) {row['A']}
B) {row['B']}
C) {row['C']}
D) {row['D']}

You must output ONLY the correct option letter(s) (e.g., 'A', 'BC'). No other text.
"""
            response = model.generate_content([video_file, prompt])
            ans = response.text.strip().upper()
            pred = ''.join([c for c in ans if c in 'ABCD'])
            if not pred: pred = 'A'
            print(f'{qa_id}: {pred} (Raw: {ans})', flush=True)
            genai.delete_file(video_file.name)
            
        predictions.append({'qa_id': qa_id, 'prediction': pred})
        pd.DataFrame(predictions).to_csv(csv_path, index=False)
        
        time.sleep(4)
    except Exception as e:
        print(f'Error on {qa_id}: {e}', flush=True)
        # We don't append to predictions so it retries later, or maybe we append A.
        predictions.append({'qa_id': qa_id, 'prediction': 'A'})
        pd.DataFrame(predictions).to_csv(csv_path, index=False)
        time.sleep(10) # Backoff

# Build Final Blended CSV
df_pred = pd.read_csv(csv_path)
df_base = pd.read_csv('submission_v163_final_trust.csv')
df_pred_dict = dict(zip(df_pred['qa_id'], df_pred['prediction']))
df_base['prediction'] = df_base.apply(lambda r: df_pred_dict.get(r['qa_id'], r['prediction']), axis=1)

df_base.to_csv('submission_gemini_cloud_final.csv', index=False)
print('Finished! Saved to submission_gemini_cloud_final.csv', flush=True)
