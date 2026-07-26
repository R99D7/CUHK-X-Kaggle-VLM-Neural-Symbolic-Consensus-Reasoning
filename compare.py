import pandas as pd
df1 = pd.read_csv('submission_ultimate.csv')
df2 = pd.read_csv(r'C:\Users\MUTHURAMANRAMANATHAN\Downloads\submission (8).csv')
test = pd.read_csv('test_qa.csv')
merged = pd.merge(df1, df2, on='qa_id', suffixes=('_ult', '_8'))
merged = pd.merge(merged, test[['qa_id', 'category']], on='qa_id')
diff = merged[merged['prediction_ult'] != merged['prediction_8']]
multi_diff = diff[diff['category'] == 'multi']
single_single = multi_diff[(multi_diff['prediction_ult'].str.len() == 1) & (multi_diff['prediction_8'].str.len() == 1)]
print(f'Single vs Single multi differences: {len(single_single)}')
for idx, row in single_single.iterrows():
    print(f"{row['qa_id']}: ult={row['prediction_ult']} 8={row['prediction_8']}")
