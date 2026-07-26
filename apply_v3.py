import pandas as pd

df = pd.read_csv('submission_ultimate_v2.csv')

changes = {
    'test_0116': 'ABD',
    'test_0139': 'ABC',
    'test_0152': 'ACD',
    'test_0171': 'ACD',
    'test_0191': 'ABD',
    'test_0202': 'ABD',
    'test_0606': 'ACD'
}

for qa_id, new_ans in changes.items():
    old_ans = df.loc[df['qa_id'] == qa_id, 'prediction'].values[0]
    df.loc[df['qa_id'] == qa_id, 'prediction'] = new_ans
    print(f"Changed {qa_id} from {old_ans} to {new_ans}")

df.to_csv('submission_ultimate_v3.csv', index=False)
print("Saved to submission_ultimate_v3.csv")
