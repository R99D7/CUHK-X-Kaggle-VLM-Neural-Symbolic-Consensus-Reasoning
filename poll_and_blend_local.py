import time
import os
import pandas as pd

log_file = 'local_inference.log'
csv_file = 'submission_moondream_gpu.csv'

print("Waiting for local inference to complete...")
while True:
    if os.path.exists(csv_file):
        print(f"Found {csv_file}!")
        break
    time.sleep(5)

print("Inference complete. Blending results...")
df_moon = pd.read_csv(csv_file)
df_base = pd.read_csv('submission_ultimate_v3.csv')

df_moon_dict = dict(zip(df_moon['qa_id'], df_moon['prediction']))
df_base['prediction'] = df_base.apply(
    lambda row: df_moon_dict.get(row['qa_id'], row['prediction']),
    axis=1
)

df_base.to_csv('submission_ultimate_v6.csv', index=False)
print("Finished creating submission_ultimate_v6.csv!")
