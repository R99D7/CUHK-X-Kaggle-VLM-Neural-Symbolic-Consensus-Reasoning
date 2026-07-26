import pandas as pd
import os

print("Loading DeBERTa and v46 predictions...")
deberta = pd.read_csv('submission_v72_deberta.csv')
v46 = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')

# Merge
merged = pd.merge(v46, deberta, on='qa_id', suffixes=('_v46', '_deb'))

diffs = sum(merged['prediction_v46'] != merged['prediction_deb'])
print(f"Total differences between DeBERTa and v46: {diffs}")

# Because we don't have logits for DeBERTa, we will just give DeBERTa 100% control,
# or we can do a fallback strategy.
# But actually, the top Kaggle submissions are pure DeBERTa.
# For safety, let's just use DeBERTa purely since DeBERTa fundamentally understands text better than TF-IDF.
# This script is a placeholder in case we wanted to blend.
