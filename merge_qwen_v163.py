import pandas as pd
import numpy as np

v163_df = pd.read_csv('submission_v163_final_trust.csv')
try:
    qwen_df = pd.read_csv('submission_qwen_single_frame.csv')
    
    # Merge
    merged = pd.merge(v163_df, qwen_df, on='qa_id', suffixes=('_v163', '_qwen'))
    
    # Replace valid qwen answers
    valid_answers = ['A', 'B', 'C', 'D']
    merged['prediction'] = merged.apply(
        lambda row: row['prediction_qwen'] if row['prediction_qwen'] in valid_answers else row['prediction_v163'],
        axis=1
    )
    
    final_df = merged[['qa_id', 'prediction']]
    final_df.to_csv('submission_final_qwen_blend.csv', index=False)
    print('Merged successfully to submission_final_qwen_blend.csv!')
except Exception as e:
    print('Error merging:', e)
