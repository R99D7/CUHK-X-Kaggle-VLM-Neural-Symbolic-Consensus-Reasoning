"""
Check the results of the 6-frame VLM run and compare against our verified 0.69590 baseline.
"""
import pandas as pd
import json

with open("qwen2vl_6frame_preds.json", "r") as f:
    preds = json.load(f)
    
df_base = pd.read_csv("submission_v267_NEWCOMB2MULTI.csv") # The 0.69590 baseline
base_map = dict(zip(df_base['qa_id'], df_base['prediction']))

df_new = pd.read_csv("submission_vlm_ultimate_6frame.csv")
new_map = dict(zip(df_new['qa_id'], df_new['prediction']))

matches = 0
total = 0
fallback_count = 0

for qid, pred in base_map.items():
    total += 1
    new_p = new_map.get(qid, '')
    if str(pred).strip().upper() == str(new_p).strip().upper():
        matches += 1
        
print(f"Total predictions: {total}")
print(f"Agreement rate with 0.69590 baseline: {matches}/{total} ({matches/total:.2%})")

# Let's check value counts of predictions in the new VLM run
print("\nValue counts in new VLM submission:")
print(df_new['prediction'].value_counts().head(10))
