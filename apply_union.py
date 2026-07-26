import pandas as pd

df = pd.read_csv('submission_ultimate.csv')

changes = {
    'test_0144': 'AB',
    'test_0162': 'AB',
    'test_0183': 'BD',
    'test_0185': 'AB',
    'test_0598': 'AD'
}

for qa_id, new_ans in changes.items():
    old_ans = df.loc[df['qa_id'] == qa_id, 'prediction'].values[0]
    df.loc[df['qa_id'] == qa_id, 'prediction'] = new_ans
    print(f"Changed {qa_id} from {old_ans} to {new_ans}")

df.to_csv('submission_ultimate_v2.csv', index=False)
print("Saved to submission_ultimate_v2.csv")
