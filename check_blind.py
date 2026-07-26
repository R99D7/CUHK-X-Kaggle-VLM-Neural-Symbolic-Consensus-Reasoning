import pandas as pd
probs = pd.read_csv('transformer_fixed_raw_predictions.csv')
te = pd.read_csv('test_qa.csv')
probs = probs.merge(te[['qa_id', 'category']], on='qa_id')

blind = 0
for idx, row in probs.iterrows():
    p_max = max(row['raw_prob_A'], row['raw_prob_B'], row['raw_prob_C'], row['raw_prob_D'])
    if p_max < 0.28:
        print(f"{row['qa_id']} ({row['category']}): max prob {p_max:.4f}")
        blind += 1
print(f'Total completely blind predictions (max prob < 0.28): {blind}')
