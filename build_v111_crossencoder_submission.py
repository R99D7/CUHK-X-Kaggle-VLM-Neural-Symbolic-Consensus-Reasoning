import pandas as pd
import numpy as np

df = pd.read_csv('crossencoder_raw_predictions.csv')

final_preds = []
for idx, row in df.iterrows():
    qa_id = row['qa_id']
    probs = {
        'A': row['raw_prob_A'],
        'B': row['raw_prob_B'],
        'C': row['raw_prob_C'],
        'D': row['raw_prob_D']
    }
    
    # 1. Pick all options > 0.50
    selected = [k for k, v in probs.items() if v > 0.50]
    
    # 2. If none are > 0.50, pick the single highest
    if len(selected) == 0:
        best_opt = max(probs, key=probs.get)
        selected = [best_opt]
        
    # Sort alphabetically
    selected.sort()
    pred_str = ''.join(selected)
    
    final_preds.append({
        'qa_id': qa_id,
        'prediction': pred_str
    })

out_df = pd.DataFrame(final_preds)

# We can also softly blend this with v46 to get a hybrid, 
# but let's first output the pure Cross-Encoder predictions
out_df.to_csv('submission_v111_pure_crossencoder.csv', index=False)
print("Saved submission_v111_pure_crossencoder.csv")

# Let's also check how many differences it has with v46
v46 = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')
merged = pd.merge(out_df, v46, on='qa_id', suffixes=('_v111', '_v46'))
diffs = (merged['prediction_v111'] != merged['prediction_v46']).sum()
print(f"Differences from v46: {diffs}")
