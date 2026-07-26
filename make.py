with open('train_ml_v33_visual_bagging.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'print("Extracting numeric and text features...")' in line:
        new_lines.append('''    print('Loading 0.42982 submission as pseudo labels...')
    pseudo_df = pd.read_csv('submission_v36_perfect_length.csv')
    pseudo_map = dict(zip(pseudo_df['qa_id'], pseudo_df['prediction']))
    test_df['answer'] = test_df['qa_id'].map(pseudo_map).fillna('A')\n''')
        new_lines.append(line)
    elif 'X_text_train, X_num_train, y_train, train_qa_ids, train_letters, train_sentences = extract_features(train_df, is_train=True)' in line:
        new_lines.append('    X_text_train_raw, X_num_train_raw, y_train_raw, train_qa_ids, train_letters, train_sentences_raw = extract_features(train_df, is_train=True)\n')
    elif 'X_text_test, X_num_test, _, test_qa_ids, test_letters, test_sentences = extract_features(test_df, is_train=False)' in line:
        new_lines.append('''    X_text_test, X_num_test, y_test_pseudo, test_qa_ids, test_letters, test_sentences = extract_features(test_df, is_train=True)
    
    print('Merging Train + Pseudo-Labeled Test...')
    X_text_train = X_text_train_raw + X_text_test
    X_num_train = np.vstack([X_num_train_raw, X_num_test])
    y_train = y_train_raw + y_test_pseudo
    train_sentences = train_sentences_raw + test_sentences\n''')
    elif 'submission_ml_v33_visual_bagging.csv' in line:
        new_lines.append(line.replace('submission_ml_v33_visual_bagging.csv', 'submission_v38_pseudo_label.csv'))
    else:
        new_lines.append(line)

with open('train_v38_pseudo_label.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Created!')
