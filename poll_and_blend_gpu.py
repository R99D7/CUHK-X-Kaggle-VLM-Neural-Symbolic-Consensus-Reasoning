import time
import subprocess
import os
import pandas as pd

print("Polling Kaggle GPU notebook...")
while True:
    try:
        out = subprocess.check_output("kaggle kernels status muthuramanraman7/cuhk-moondream-gpu", shell=True).decode()
        print(out.strip())
        if 'complete' in out.lower() or 'error' in out.lower():
            break
    except:
        pass
    time.sleep(30)

print("Downloading output from Kaggle (ignoring emoji encoding errors)...")
# Redirecting to NUL avoids cp1252 charmap encoding crash on Windows console
subprocess.run("kaggle kernels output muthuramanraman7/cuhk-moondream-gpu -p . > NUL 2>&1", shell=True)

if os.path.exists('submission_moondream_gpu.csv'):
    print("Found GPU output. Blending...")
    sf = pd.read_csv('submission_ultimate_v3.csv', keep_default_na=False)
    sm = pd.read_csv('submission_moondream_gpu.csv', keep_default_na=False)
    test = pd.read_csv('test_qa.csv')
    
    sf_dict = dict(zip(sf['qa_id'], sf['prediction']))
    sm_dict = dict(zip(sm['qa_id'], sm['prediction']))
    
    final_preds = []
    changes = 0
    
    for idx, row in test.iterrows():
        qa_id = row['qa_id']
        cat = row['category']
        
        f_pred = str(sf_dict.get(qa_id, 'A')).upper()
        m_pred = str(sm_dict.get(qa_id, f_pred)).upper()
        
        if f_pred in ['NA', 'NAN', '']:
            final_pred = m_pred
            changes += 1
        elif cat == 'multi' and qa_id in sm_dict:
            combined = set(f_pred) | set(m_pred)
            combined_pred = "".join(sorted([c for c in combined if c in 'ABCD']))
            if len(combined_pred) == 0: combined_pred = 'A'
            final_pred = combined_pred
            if final_pred != f_pred: changes += 1
        elif cat == 'sequence' and qa_id in sm_dict:
            final_pred = m_pred
            if final_pred != f_pred: changes += 1
        else:
            final_pred = f_pred
            
        final_preds.append({'qa_id': qa_id, 'prediction': final_pred})
        
    df_out = pd.DataFrame(final_preds)
    df_out.to_csv('submission_ultimate_v6.csv', index=False)
    print(f"Created submission_ultimate_v6.csv with {changes} strategic overrides from Moondream!")
else:
    print("Error: submission_moondream_gpu.csv was not found.")
