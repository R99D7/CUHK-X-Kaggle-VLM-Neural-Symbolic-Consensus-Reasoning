import pandas as pd
train = pd.read_csv('training_qa.csv')
test = pd.read_csv('test_qa.csv')
t_row = test[test['qa_id'] == 'test_0341'].iloc[0]
tr_row = train[train['qa_id'] == 'training_2660'].iloc[0]
print('TEST 0341')
print(f"A: {t_row['A']}")
print(f"B: {t_row['B']}")
print(f"C: {t_row['C']}")
print(f"D: {t_row['D']}")
print('TRAIN 2660')
print(f"A: {tr_row['A']}")
print(f"B: {tr_row['B']}")
print(f"C: {tr_row['C']}")
print(f"D: {tr_row['D']}")
print(f"Answer: {tr_row['answer']}")
