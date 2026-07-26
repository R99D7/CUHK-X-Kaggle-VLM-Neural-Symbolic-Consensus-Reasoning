import pandas as pd
train = pd.read_csv('training_qa.csv')
test = pd.read_csv('test_qa.csv')
v46 = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')

leaks = 0
for idx, row in test.iterrows():
    match = train[train['question'] == row['question']]
    if len(match) > 0:
        for _, m_row in match.iterrows():
            test_opts = {str(row['A']), str(row['B']), str(row['C']), str(row['D'])}
            train_opts = {str(m_row['A']), str(m_row['B']), str(m_row['C']), str(m_row['D'])}
            
            if test_opts == train_opts:
                true_ans_chars = str(m_row['answer'])
                true_texts = []
                for char in true_ans_chars:
                    if char in ['A', 'B', 'C', 'D']:
                        true_texts.append(str(m_row[char]))
                
                test_ans_chars = []
                for t_char in ['A', 'B', 'C', 'D']:
                    if str(row[t_char]) in true_texts:
                        test_ans_chars.append(t_char)
                
                expected_ans = ''.join(sorted(test_ans_chars))
                actual_v46 = str(v46.loc[v46['qa_id'] == row['qa_id'], 'prediction'].values[0])
                
                if expected_ans != actual_v46:
                    print(f"LEAK MISMATCH! {row['qa_id']}: Expected {expected_ans}, v46 has {actual_v46}")
                else:
                    print(f"LEAK OK: {row['qa_id']} is correctly {expected_ans}")
                leaks += 1
                break

print(f'Total leaks found: {leaks}')
