import pandas as pd
df1 = pd.read_csv('submission_v117_ultimate_multimodal.csv')
df2 = pd.read_csv('submission_v53_aggressive_hybrid_v46_v28.csv')
df3 = pd.read_csv('kernel_output/submission.csv')
diffs = (df1['prediction'] != df2['prediction']).sum()
print(f'Diffs between v117 and v53: {diffs}')
df_vote = df1.copy()
for i in range(len(df1)):
    p1 = df1.loc[i, 'prediction']
    p2 = df2.loc[i, 'prediction']
    p3 = df3.loc[i, 'prediction']
    votes = [p1, p2, p3]
    if votes.count(p1) >= 2:
        df_vote.loc[i, 'prediction'] = p1
    elif votes.count(p2) >= 2:
        df_vote.loc[i, 'prediction'] = p2
    elif votes.count(p3) >= 2:
        df_vote.loc[i, 'prediction'] = p3
    else:
        df_vote.loc[i, 'prediction'] = p1
changes = (df1['prediction'] != df_vote['prediction']).sum()
print(f'Changes from v117 in 3-way vote: {changes}')
df_vote.to_csv('submission_v143_qwen_tiebreaker.csv', index=False)
