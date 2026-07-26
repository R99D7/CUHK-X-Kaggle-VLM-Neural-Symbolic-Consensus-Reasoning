"""
Analyze emotion category in training data.
"""
import pandas as pd

tr = pd.read_csv('training_qa.csv')
emo = tr[tr['category'] == 'emotion']

print(f"Total emotion questions: {len(emo)}")
print("Answers distribution:")
for ans in emo['answer'].unique():
    # count how many times this option is correct
    cnt = len(emo[emo['answer'] == ans])
    print(f"  {ans}: {cnt}")

# Let's see some examples
for idx, row in emo.head(5).iterrows():
    print(f"\n{row['question']}")
    for l in ['A', 'B', 'C', 'D']:
        print(f"  {l}: {row[l]}")
    print(f"Answer: {row['answer']}")
