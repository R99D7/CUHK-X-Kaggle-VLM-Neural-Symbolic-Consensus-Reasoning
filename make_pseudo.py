with open('train_ml_v33_visual_bagging.py', 'r', encoding='utf-8') as f:
    code = f.read()

target1 = "    test_df['answer'] = 'A' # dummy for test"
replacement1 = '''    pseudo_df = pd.read_csv('submission_v36_perfect_length.csv')
    pseudo_map = dict(zip(pseudo_df['qa_id'], pseudo_df['prediction']))
    test_df['answer'] = test_df['qa_id'].map(pseudo_map).fillna('A')'''
code = code.replace(target1, replacement1)

target2 = '''    print("Extracting numeric and text features...")
    X_text_train, X_num_train, y_train, train_qa_ids, train_letters, train_sentences = extract_features(train_df, is_train=True)
    X_text_test, X_num_test, _, test_qa_ids, test_letters, test_sentences = extract_features(test_df, is_train=False)'''

replacement2 = '''    print("Extracting numeric and text features...")
    X_text_train_raw, X_num_train_raw, y_train_raw, train_qa_ids, train_letters, train_sentences_raw = extract_features(train_df, is_train=True)
    X_text_test, X_num_test, y_test_pseudo, test_qa_ids, test_letters, test_sentences = extract_features(test_df, is_train=True)
    
    print("Merging Train + Pseudo-Labeled Test...")
    X_text_train = X_text_train_raw + X_text_test
    X_num_train = np.vstack([X_num_train_raw, X_num_test])
    y_train = y_train_raw + y_test_pseudo
    train_sentences = train_sentences_raw + test_sentences'''

code = code.replace(target2, replacement2)
code = code.replace('submission_ml_v33_visual_bagging.csv', 'submission_v38_pseudo_label.csv')

with open('train_v38_pseudo_label.py', 'w', encoding='utf-8') as f:
    f.write(code)
