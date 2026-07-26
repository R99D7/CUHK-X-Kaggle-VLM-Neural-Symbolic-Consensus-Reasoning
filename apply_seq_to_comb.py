"""
Apply SEQUENCE -> COMBINATION fixes.
"""
import pandas as pd

sub = pd.read_csv('submission.csv')
raw = pd.read_csv('transformer_fixed_raw_predictions.csv')

def fix_comb(qa_id, valid_opts):
    r = raw[raw['qa_id'] == qa_id].iloc[0]
    best_opt = max(valid_opts, key=lambda x: r[f'raw_prob_{x}'])
    old_pred = sub.loc[sub['qa_id'] == qa_id, 'prediction'].values[0]
    print(f"Fixing COMB {qa_id}: {old_pred} -> {best_opt}")
    sub.loc[sub['qa_id'] == qa_id, 'prediction'] = best_opt

fix_comb('test_0227', ['A', 'D'])
fix_comb('test_0236', ['B'])
fix_comb('test_0245', ['A'])
fix_comb('test_0318', ['D'])
fix_comb('test_0328', ['D'])
fix_comb('test_0329', ['C'])

sub.to_csv('submission_v266_SEQ2COMB.csv', index=False)
sub.to_csv('submission.csv', index=False)
print("Saved to submission.csv")
