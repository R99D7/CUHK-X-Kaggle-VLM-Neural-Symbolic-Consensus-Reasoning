import pandas as pd

train = pd.read_csv('training_qa.csv')
test = pd.read_csv('test_qa.csv')

for cat in ['emotion', 'combination', 'single', 'multi']:
    print(f'\n=== {cat.upper()} SAMPLES ===')
    for _, r in train[train['category']==cat].head(4).iterrows():
        print(f"  Q: {r['question']}")
        print(f"  A:{r['A']} | B:{r['B']} | C:{r['C']} | D:{r['D']} -> Answer:{r['answer']}")
        print()

print('\n=== TEST QUESTIONS SAMPLE ===')
for cat in ['emotion', 'combination']:
    print(f'\n-- {cat} --')
    for _, r in test[test['category']==cat].head(3).iterrows():
        print(f"  Q: {r['question']}")
        print(f"  A:{r['A']} | B:{r['B']} | C:{r['C']} | D:{r['D']}")
        print()
