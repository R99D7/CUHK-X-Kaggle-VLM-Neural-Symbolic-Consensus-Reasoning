import pandas as pd
import numpy as np

test = pd.read_csv('test_qa.csv')
v12 = pd.read_csv('submission_ultimate_v12.csv')
moondream = pd.read_csv('submission_depthcolor.csv')

v12_dict = dict(zip(v12['qa_id'], v12['prediction']))
md_dict = dict(zip(moondream['qa_id'], moondream['prediction']))

final_preds = []
changed = 0

for _, row in test.iterrows():
    qa_id = row['qa_id']
    cat = row['category']
    base = v12_dict.get(qa_id, 'A')
    md = md_dict.get(qa_id, '')
    
    final = base
    
    # Correcting combination bias: v12 underpredicts C (18.7% vs 26.6%), overpredicts A (25.9% vs 22.3%) and B (30.2% vs 27.2%)
    if cat == 'combination':
        if base in ['A', 'B'] and md == 'C':
            final = 'C'
            
    # Correcting emotion bias: v12 overpredicts B (30.6% vs 25.3%), underpredicts A (25.0% vs 28.3%)
    elif cat == 'emotion':
        if base == 'B' and md == 'A':
            final = 'A'
            
    if final != base:
        changed += 1
        
    final_preds.append({'qa_id': qa_id, 'prediction': final})

out = pd.DataFrame(final_preds)
out.to_csv('submission_bias_corrected.csv', index=False)

print(f"Total changed from 0.403 baseline (v12): {changed}/682")

merged = out.merge(v12.rename(columns={'prediction':'base'}), on='qa_id')
merged2 = merged.merge(test[['qa_id','category']], on='qa_id')
for cat in test['category'].unique():
    s = merged2[merged2['category']==cat]
    ch = (s['prediction'] != s['base']).sum()
    print(f"  {cat}: {ch}/{len(s)} changed")
