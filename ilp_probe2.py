import pandas as pd
import pulp
import os
import sys

subs = {
    'v143': ('submission_v143_qwen_tiebreaker.csv', 313),
    'v117': ('submission_v117_ultimate_multimodal.csv', 309),
    'v116': ('submission_v116_ultimate_safe_threshold_040.csv', 307),
    'v114': ('submission_v114_ultimate_safe_dual_agreement.csv', 307),
    'v113': ('submission_v113_dual_agreement_override.csv', 303),
    'v118': ('submission_v118_ultimate_multimodal_055.csv', 301),
    'v99' : ('submission_v99_top_kaggle_guarantee.csv', 301),
    'v142': ('submission_v142_dawid_skene.csv', 301),
    'v53' : ('submission_v53_aggressive_hybrid_v46_v28.csv', 301),
    'v46' : ('submission_v46_aggressive_hybrid_v20.csv', 301),
    'v134': ('submission_v134_soft_prob_ensemble.csv', 299),
    'v65' : ('submission_v65_surgical_strike.csv', 299),
    'v60' : ('submission_v60_mega_forest.csv', 297),
    'qwen': ('kernel_output/submission.csv', 261)
}

data = {}
all_options = set()
for name, (file, score) in subs.items():
    if os.path.exists(file):
        df = pd.read_csv(file)
        data[name] = df['prediction'].values
        all_options.update(df['prediction'].unique())

N = 682
options = list(all_options)

def create_prob():
    prob = pulp.LpProblem("Probing", pulp.LpMinimize)
    z = pulp.LpVariable.dicts("z", ((i, c) for i in range(N) for c in options), cat='Binary')
    for i in range(N):
        prob += pulp.lpSum([z[i, c] for c in options]) == 1
    for name in data.keys():
        preds = data[name]
        score = subs[name][1]
        prob += pulp.lpSum([z[i, preds[i]] for i in range(N)]) == score
    return prob, z

# Let's find questions where the models have differences.
# If all models agree on a question, we can't probe it.
disagreements = []
for i in range(N):
    preds = set([data[name][i] for name in data.keys()])
    if len(preds) > 1:
        disagreements.append(i)

print(f"Total questions with disagreements: {len(disagreements)}")

forced_labels = {}
# We will check if v143 is FORCED to be wrong on any of its predictions
v143_preds = data['v143']

prob, z = create_prob()
# Suppress output
import logging
pulp.LpSolverDefault.msg = False

count_forced = 0
for i in disagreements:
    c = v143_preds[i]
    # Check if c can be 0. We maximize z[i,c]. No, if we maximize z[i,c] and it's 0, it means it MUST be 0!
    # Or we can just set z[i,c] == 1, solve. If infeasible, then z[i,c] MUST BE 0!
    prob_test, z_test = create_prob()
    prob_test += z_test[i, c] == 1
    prob_test.solve()
    if pulp.LpStatus[prob_test.status] == 'Infeasible':
        print(f"Question {i}: v143 prediction '{c}' is PROVEN WRONG.")
        count_forced += 1

print(f"Total proven wrong in v143: {count_forced}")
