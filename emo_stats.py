import pandas as pd

tr = pd.read_csv('training_qa.csv')
te = pd.read_csv('test_qa.csv')
sub = pd.read_csv('submission_v245_PURE_MARKOV.csv')

# Get train video actions and emotions
tr_acts = {}
tr_emos = {}
for idx, row in tr.iterrows():
    vid = row['path']
    ans_letters = str(row['answer']).strip()
    try:
        if row['category'] == 'sequence': 
            acts = set([str(row[l]).strip().lower() for l in ans_letters])
            tr_acts[vid] = acts
        elif row['category'] == 'emotion':
            tr_emos[vid] = str(row[ans_letters]).strip().lower()
    except: pass

changes = 0
for idx, row in te[te['category'] == 'sequence'].iterrows():
    vid = row['path']
    opts = set([str(row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']])
    
    emo_row = te[(te['path'] == vid) & (te['category'] == 'emotion')]
    if len(emo_row) == 0:
        continue
        
    emo_qa_id = emo_row.iloc[0]['qa_id']
    emo_opts = {str(emo_row.iloc[0][l]).strip().lower(): l for l in ['A', 'B', 'C', 'D']}
    
    votes = {}
    for t_vid, t_acts in tr_acts.items():
        if len(opts.intersection(t_acts)) >= 3:
            if t_vid in tr_emos:
                emo = tr_emos[t_vid]
                if emo in emo_opts:
                    votes[emo_opts[emo]] = votes.get(emo_opts[emo], 0) + 1
                    
    if len(votes) > 0:
        best_l = max(votes, key=votes.get)
        pred = str(sub[sub['qa_id'] == emo_qa_id]['prediction'].values[0]).strip()
        if pred != best_l:
            print(f"{emo_qa_id}: predicted={pred}, stats={best_l} (votes={votes})")
            sub.loc[sub['qa_id'] == emo_qa_id, 'prediction'] = best_l
            changes += 1

print(f'Applied {changes} statistical emotion inferences.')
