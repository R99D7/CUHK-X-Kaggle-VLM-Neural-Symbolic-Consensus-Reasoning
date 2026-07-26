import pandas as pd
from collections import Counter

test_df = pd.read_csv('test_qa.csv').set_index('qa_id')
v117 = pd.read_csv('submission_v117_ultimate_multimodal.csv').set_index('qa_id')
cross = pd.read_csv('crossencoder_raw_predictions.csv').set_index('qa_id')

all_models = {
    'v117': pd.read_csv('submission_v117_ultimate_multimodal.csv').set_index('qa_id'),
    'v118': pd.read_csv('submission_v118_ultimate_multimodal_055.csv').set_index('qa_id'),
    'v116': pd.read_csv('submission_v116_ultimate_safe_threshold_040.csv').set_index('qa_id'),
    'v114': pd.read_csv('submission_v114_ultimate_safe_dual_agreement.csv').set_index('qa_id'),
    'v113': pd.read_csv('submission_v113_dual_agreement_override.csv').set_index('qa_id'),
    'v112': pd.read_csv('submission_v112_ultimate_crossencoder_tfidf_blend.csv').set_index('qa_id'),
}

weights = {'v117': 3.0, 'v118': 2.0, 'v116': 2.0, 'v114': 1.5, 'v113': 1.5, 'v112': 1.0}

results = []
agreed, overridden = 0, 0

for qa_id in v117.index:
    v117_pred = str(v117.loc[qa_id, 'prediction'])
    
    # Get crossencoder prediction
    crow = cross.loc[qa_id]
    cross_pred = max('ABCD', key=lambda k: crow['raw_prob_' + k])
    cross_conf = max(crow['raw_prob_A'], crow['raw_prob_B'], crow['raw_prob_C'], crow['raw_prob_D'])
    
    # If v117 and crossencoder agree AND crossencoder is confident -> keep it
    if v117_pred == cross_pred and cross_conf > 0.50:
        results.append({'qa_id': qa_id, 'prediction': v117_pred})
        agreed += 1
    else:
        # Use weighted majority vote from all models
        vote = Counter()
        for k, df in all_models.items():
            vote[str(df.loc[qa_id, 'prediction'])] += weights[k]
        winner = vote.most_common(1)[0][0]
        results.append({'qa_id': qa_id, 'prediction': winner})
        overridden += 1

out = pd.DataFrame(results)
out.to_csv('submission_v133_crossencoder_anchor.csv', index=False)

v133 = out.set_index('qa_id')
diffs = sum(v133['prediction'].astype(str) != v117['prediction'].astype(str))
print('v117+crossencoder agreed and kept:', agreed)
print('Fell back to weighted vote:', overridden)
print('v133 vs v117 diffs:', diffs)
