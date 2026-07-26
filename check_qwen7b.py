import pandas as pd
import os

p = 'kernel_qwen7b_out/submission.csv'
if os.path.exists(p):
    df = pd.read_csv(p)
    print(f"Loaded {p} with {len(df)} rows.")
    print("Value counts:")
    print(df['prediction'].value_counts().head(10))
    
    # Compare with 0.69590 baseline (submission_v267_NEWCOMB2MULTI.csv)
    base_df = pd.read_csv("submission_v267_NEWCOMB2MULTI.csv")
    base_map = dict(zip(base_df['qa_id'], base_df['prediction']))
    matches = 0
    total = 0
    for idx, row in df.iterrows():
        qid = row['qa_id']
        pred = str(row['prediction']).strip()
        base_p = str(base_map.get(qid, '')).strip()
        if pred.upper() == base_p.upper():
            matches += 1
        total += 1
    print(f"Agreement rate with 0.69590 baseline: {matches}/{total} ({matches/total:.2%})")
else:
    print(f"File {p} does not exist.")
