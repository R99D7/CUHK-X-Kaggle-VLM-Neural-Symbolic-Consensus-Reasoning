import pandas as pd

df = pd.read_csv('submission_ultimate_v3.csv')

changes = {
    # 2v2 Overlaps
    'test_0127': 'ABD',
    'test_0147': 'BCD',
    'test_0148': 'BCD',
    'test_0164': 'BCD',
    'test_0177': 'ABD',
    'test_0189': 'ACD',
    'test_0585': 'ABD',
    'test_0602': 'ACD',
    # 1v3 Overlaps (where ultimate was single)
    'test_0120': 'ABC',
    'test_0150': 'BCD',
    'test_0161': 'ABC',
    'test_0165': 'ABD',
    'test_0204': 'ABC',
    'test_0208': 'BCD',
    'test_0576': 'BCD',
    'test_0578': 'BCD'
}

for qa_id, new_ans in changes.items():
    old_ans = df.loc[df['qa_id'] == qa_id, 'prediction'].values[0]
    df.loc[df['qa_id'] == qa_id, 'prediction'] = new_ans
    print(f"Changed {qa_id} from {old_ans} to {new_ans}")

df.to_csv('submission_ultimate_v4.csv', index=False)
print("Saved to submission_ultimate_v4.csv")
