import os
import pandas as pd
df = pd.read_csv('test_qa.csv')
vid = df['path'].iloc[0]
print(f"First video path in CSV: {vid}")
print(f"Exists in current dir?: {os.path.exists(vid)}")
print(f"Exists in large_model_track_test?: {os.path.exists('large_model_track_test/' + vid.split('/')[-1])}")
print(f"Exists in videos/?: {os.path.exists('videos/' + vid.split('/')[-1])}")

try:
    import qwen_vl_utils
    import transformers
    print("Transformers and qwen_vl_utils are installed.")
except ImportError as e:
    print(f"Import error: {e}")
