"""
Check single and object interaction overlap.
"""
import pandas as pd

te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission.csv')

obj_q = te[te['category'] == 'object_interaction']
single_q = te[te['category'] == 'single']

merged = pd.merge(obj_q, single_q, on='path', suffixes=('_obj', '_single'))
print(f"Total overlapping videos: {len(merged)}")

for idx, row in merged.iterrows():
    vid = row['path']
    o_qa = row['qa_id_obj']
    s_qa = row['qa_id_single']
    
    o_pred = str(sub[sub['qa_id'] == o_qa]['prediction'].values[0])
    s_pred = str(sub[sub['qa_id'] == s_qa]['prediction'].values[0])
    
    o_opt = str(row[o_pred + '_obj']).strip().lower()
    s_opt = str(row[s_pred + '_single']).strip().lower()
    
    print(f"Vid: {vid}")
    print(f"  Object predicted: {o_opt} (Option {o_pred})")
    print(f"  Single predicted: {s_opt} (Option {s_pred})")
    print(f"  Single options: A: {row['A_single']}, B: {row['B_single']}, C: {row['C_single']}, D: {row['D_single']}")
    print("-" * 50)
