import pandas as pd
import os

def generate_ultimate_hybrid():
    print("Loading base models...")
    df_46 = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv') # 0.44152
    
    if not os.path.exists('submission_v50_qwen_vl.csv'):
        print("ERROR: Qwen2-VL has not finished yet! Waiting for file...")
        return
        
    df_qwen = pd.read_csv('submission_v50_qwen_vl.csv')
    df_32 = pd.read_csv('submission_v32_multimodal_cv.csv')
    
    preds = []
    qwen_overrides = 0
    
    for idx, row in df_46.iterrows():
        qa_id = row['qa_id']
        p_46 = str(row['prediction'])
        p_qwen = str(df_qwen[df_qwen['qa_id'] == qa_id]['prediction'].values[0])
        p_32 = str(df_32[df_32['qa_id'] == qa_id]['prediction'].values[0])
        
        # Logic: 0.44152 is our absolute base.
        # If Qwen2-VL (2B parameter Vision-Language) strongly disagrees with it, 
        # AND Qwen2-VL's answer is supported by our other robust model (v32), we override.
        if p_46 == p_qwen:
            final_pred = p_46
        else:
            if p_qwen == p_32:
                final_pred = p_qwen
                qwen_overrides += 1
            else:
                final_pred = p_46 # Default back to 0.44152 for safety
                
        preds.append({'qa_id': qa_id, 'prediction': final_pred})
        
    print(f"Done! Qwen2-VL successfully corrected {qwen_overrides} answers from the 0.44152 submission.")
    
    out = pd.DataFrame(preds)
    out.to_csv('submission_v54_ultimate_qwen_hybrid.csv', index=False)
    print("Saved as: submission_v54_ultimate_qwen_hybrid.csv")

if __name__ == '__main__':
    generate_ultimate_hybrid()
