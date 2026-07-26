import pandas as pd

df1 = pd.read_csv('submission_ultimate.csv')
df2 = pd.read_csv(r'C:\Users\MUTHURAMANRAMANATHAN\Downloads\submission (8).csv')
test = pd.read_csv('test_qa.csv')
merged = pd.merge(df1, df2, on='qa_id', suffixes=('_ult', '_8'))
merged = pd.merge(merged, test[['qa_id', 'category']], on='qa_id')
diff = merged[merged['prediction_ult'] != merged['prediction_8']]
multi_diff = diff[diff['category'] == 'multi']

def is_disjoint(ans1, ans2):
    return len(set(ans1).intersection(set(ans2))) == 0

disjoint_cases = []
for idx, row in multi_diff.iterrows():
    if is_disjoint(str(row['prediction_ult']), str(row['prediction_8'])):
        disjoint_cases.append(row)

print(f'Total disjoint cases: {len(disjoint_cases)}')
for row in disjoint_cases:
    union = ''.join(sorted(set(str(row['prediction_ult'])).union(set(str(row['prediction_8'])))))
    print(f"{row['qa_id']}: ult={row['prediction_ult']} 8={row['prediction_8']} -> UNION={union}")
