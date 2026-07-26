import pandas as pd

# Load the best submission and the datasets
best_sub = pd.read_csv('submission_v117_ultimate_multimodal.csv').set_index('qa_id')
train = pd.read_csv('training_qa.csv')
test = pd.read_csv('test_qa.csv')

# Find exact match leaks
leaks_dict = {}
for idx, row in test.iterrows():
    match = train[train['question'] == row['question']]
    if len(match) > 0:
        for _, m_row in match.iterrows():
            test_opts = {str(row['A']), str(row['B']), str(row['C']), str(row['D'])}
            train_opts = {str(m_row['A']), str(m_row['B']), str(m_row['C']), str(m_row['D'])}
            if test_opts == train_opts:
                leaks_dict[row['qa_id']] = m_row['answer']
                break

print(f'Verified {len(leaks_dict)} data leaks.')

# Apply leaks
results = []
changed = 0
correct_already = 0
for qa_id in test['qa_id']:
    pred = best_sub.loc[qa_id, 'prediction']
    if qa_id in leaks_dict:
        true_ans = leaks_dict[qa_id]
        if str(pred) != str(true_ans):
            pred = true_ans
            changed += 1
        else:
            correct_already += 1
            
    results.append({'qa_id': qa_id, 'prediction': pred})

pd.DataFrame(results).to_csv('submission_v140_v117_leaks.csv', index=False)
print(f'Saved submission_v140_v117_leaks.csv. Changed {changed} answers, {correct_already} were already correct.')
