import pandas as pd
df = pd.read_csv('submission_qwen_single_frame.csv')
print(df.head())
print("Nulls:", df['prediction'].isnull().sum())
print("Value counts:", df['prediction'].value_counts())
