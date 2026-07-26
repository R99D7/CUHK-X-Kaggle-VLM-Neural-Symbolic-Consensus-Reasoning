"""
Analyze all existing Gemini and high-tier VLM submission files in the directory.
"""
import pandas as pd
import os

files_to_check = [
    'submission_v200_GEMINI_VISION_ULTIMATE.csv',
    'submission_v201_GEMINI_FIXED.csv',
    'submission_gemini_almost_complete.csv',
    'submission_gemini_cloud_final.csv',
    'submission_gemini_cloud_raw.csv',
    'submission_v202_QWEN_LOCAL_HYBRID.csv',
    'submission_v203_VISION_TIEBREAKER.csv',
    'submission.csv'
]

te = pd.read_csv('test_qa.csv')
print(f"Test QA count: {len(te)}")

sub_ref = pd.read_csv('submission.csv')
sub_ref_map = dict(zip(sub_ref['qa_id'], sub_ref['prediction']))

for fname in files_to_check:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        n_rows = len(df)
        df_map = dict(zip(df['qa_id'], df['prediction']))
        
        # Calculate match rate with our current best submission (0.69590)
        matches = 0
        common_ids = 0
        for qid, pred in sub_ref_map.items():
            if qid in df_map:
                common_ids += 1
                if str(df_map[qid]).strip().upper() == str(pred).strip().upper():
                    matches += 1
        
        match_rate = matches / common_ids if common_ids > 0 else 0
        print(f"[{fname}] - Rows: {n_rows} - Common IDs: {common_ids} - Match with 0.69590: {matches}/{common_ids} ({match_rate:.2%})")
    else:
        print(f"[{fname}] - NOT FOUND")
