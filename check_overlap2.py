import pandas as pd

train = pd.read_csv('training_qa.csv')
test = pd.read_csv('test_qa.csv')

match = 0
for _, t_row in test.iterrows():
    t_opts = set([str(t_row['A']), str(t_row['B']), str(t_row['C']), str(t_row['D'])])
    tr_rows = train[train['question'] == t_row['question']]
    for _, tr_row in tr_rows.iterrows():
        tr_opts = set([str(tr_row['A']), str(tr_row['B']), str(tr_row['C']), str(tr_row['D'])])
        if t_opts == tr_opts:
            match += 1
            break

print("Matched questions with exact options set:", match)
