import pandas as pd
import os
import time
import subprocess

def fetch_and_merge():
    # Try to download the output from Kaggle
    print("Downloading surgical output from Kaggle...")
    subprocess.run("kaggle kernels output muthuramanraman7/cuhk-qwen2-vl-surgical -p .", shell=True)
    
    if not os.path.exists("submission.csv"):
        print("Error: surgical submission.csv not found yet. It might still be running!")
        return
        
    surg_df = pd.read_csv('submission.csv')
    best_df = pd.read_csv('submission_hybrid_trust_new.csv')
    
    surg_dict = dict(zip(surg_df['qa_id'], surg_df['prediction']))
    
    # Merge
    final_preds = []
    changes = 0
    for idx, row in best_df.iterrows():
        qa_id = row['qa_id']
        pred = row['prediction']
        
        if qa_id in surg_dict:
            new_pred = str(surg_dict[qa_id])
            if str(pred) != new_pred:
                print(f"Surgical override for {qa_id}: {pred} -> {new_pred}")
                pred = new_pred
                changes += 1
                
        final_preds.append({'qa_id': qa_id, 'prediction': pred})
        
    pd.DataFrame(final_preds).to_csv('submission_final_surgical.csv', index=False)
    print(f"Successfully merged! Made {changes} true-vision overrides.")
    print("Output saved to submission_final_surgical.csv")
    
if __name__ == "__main__":
    fetch_and_merge()
