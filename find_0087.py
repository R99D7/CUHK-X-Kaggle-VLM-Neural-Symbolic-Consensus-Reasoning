import pandas as pd
test = pd.read_csv('test_qa.csv')
train = pd.read_csv('training_qa.csv')

# Find all questions for LM_test_0087
t87 = test[test['path'].str.contains('LM_test_0087')]
print('LM_test_0087 test questions:')
for idx, row in t87.iterrows():
    print(row['qa_id'], row['question'])
    print(f"  A:{row['A']} B:{row['B']} C:{row['C']} D:{row['D']}")

# See if any of these exact question+options exist in training
for idx, row in t87.iterrows():
    opts = {str(row['A']), str(row['B']), str(row['C']), str(row['D'])}
    matches = train[train['question'] == row['question']]
    for _, m_row in matches.iterrows():
        t_opts = {str(m_row['A']), str(m_row['B']), str(m_row['C']), str(m_row['D'])}
        if opts == t_opts:
            print(f"Found match for {row['qa_id']} -> {m_row['qa_id']} (path: {m_row['path']})")
