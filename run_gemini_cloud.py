import os
import time
import pandas as pd
import google.generativeai as genai

# Setup your API key
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    print('Please set the GEMINI_API_KEY environment variable. e.g., set GEMINI_API_KEY=your_key')
    exit(1)

genai.configure(api_key=API_KEY)

# Use Gemini 1.5 Flash for speed (approx. 0.70 score), or Gemini 1.5 Pro for max accuracy (approx. 0.85+ score)
model = genai.GenerativeModel('gemini-1.5-flash')

df = pd.read_csv('test_qa.csv')
predictions = []

print(f'Starting Cloud Inference for {len(df)} videos...')

for idx, row in df.iterrows():
    qa_id = row['qa_id']
    vid_path = 'large_model_track_test/' + row['path'].replace('large_model_track_test/', '')
    
    if not os.path.exists(vid_path):
        predictions.append({'qa_id': qa_id, 'prediction': 'A'})
        continue
        
    try:
        # Upload to Gemini Cloud
        video_file = genai.upload_file(path=vid_path)
        
        # Wait for processing
        while video_file.state.name == 'PROCESSING':
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
            
        if video_file.state.name == 'FAILED':
            predictions.append({'qa_id': qa_id, 'prediction': 'A'})
            continue
            
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
        
        # Clean answer
        pred = ''.join([c for c in ans if c in 'ABCD'])
        if not pred: pred = 'A'
        
        print(f'{qa_id}: {pred} (Raw: {ans})')
        predictions.append({'qa_id': qa_id, 'prediction': pred})
        
        # Clean up cloud storage
        genai.delete_file(video_file.name)
        
        # Rate limit (15 RPM for free tier)
        time.sleep(4)
        
    except Exception as e:
        print(f'Error on {qa_id}: {e}')
        predictions.append({'qa_id': qa_id, 'prediction': 'A'})

df_pred = pd.DataFrame(predictions)

# Blend with your best local model for any API failures
df_base = pd.read_csv('submission_v163_final_trust.csv')
df_pred_dict = dict(zip(df_pred['qa_id'], df_pred['prediction']))
df_base['prediction'] = df_base.apply(lambda r: df_pred_dict.get(r['qa_id'], r['prediction']), axis=1)

df_base.to_csv('submission_gemini_cloud.csv', index=False)
print('Finished! Saved to submission_gemini_cloud.csv')
