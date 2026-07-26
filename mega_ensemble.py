import glob
import pandas as pd
from collections import Counter

v46 = pd.read_csv('submission_v46_aggressive_hybrid_v20.csv')
v46_preds = v46['prediction'].values

votes = {qa_id: [] for qa_id in v46['qa_id']}
valid_files = 0

for file in glob.glob('submission*.csv'):
    try:
        df = pd.read_csv(file)
        if len(df) == 682 and 'prediction' in df.columns:
            preds = df['prediction'].values
            diffs = sum(preds != v46_preds)
            if diffs <= 100:
                valid_files += 1
                for idx, row in df.iterrows():
                    votes[row['qa_id']].append(str(row['prediction']))
    except Exception as e:
        pass

print(f'Voting across {valid_files} models...')

final_preds = []
diff_from_v46 = 0
for idx, row in v46.iterrows():
    qa_id = row['qa_id']
    v46_pred = str(row['prediction'])
    c = Counter(votes[qa_id])
    best_pred = c.most_common(1)[0][0]
    final_preds.append({'qa_id': qa_id, 'prediction': best_pred})
    if best_pred != v46_pred:
        diff_from_v46 += 1
        print(f'{qa_id}: v46={v46_pred}, ensemble={best_pred}')

out_df = pd.DataFrame(final_preds)
out_df.to_csv('submission_v106_mega_ensemble.csv', index=False)
print(f'Total differences from v46: {diff_from_v46}')
