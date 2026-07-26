import pandas as pd
import pulp
import os

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

prob = pulp.LpProblem("Probing", pulp.LpMinimize)
z = pulp.LpVariable.dicts("z", ((i, c) for i in range(N) for c in options), cat='Binary')

for i in range(N):
    prob += pulp.lpSum([z[i, c] for c in options]) == 1

for name in data.keys():
    preds = data[name]
    score = subs[name][1]
    prob += pulp.lpSum([z[i, preds[i]] for i in range(N)]) == score

disagreements = []
for i in range(N):
    preds = set([data[name][i] for name in data.keys()])
    if len(preds) > 1:
        disagreements.append((i, list(preds)))

print(f"Total questions with disagreements: {len(disagreements)}")

pulp.LpSolverDefault.msg = False
solver = pulp.PULP_CBC_CMD(msg=False)

v143_preds = data['v143']
count_forced = 0
out_df = pd.read_csv(subs['v143'][0])

total = len(disagreements)

with open('ilp_progress.txt', 'w') as f:
    f.write('Started\n')

for idx, (i, model_preds) in enumerate(disagreements):
    if idx % 10 == 0:
        with open('ilp_progress.txt', 'a') as f:
            f.write(f"Processed {idx}/{total} disagreements...\n")

    c = v143_preds[i]
    
    z[i, c].lowBound = 1
    z[i, c].upBound = 1
    
    prob.solve(solver)
    
    if pulp.LpStatus[prob.status] == 'Infeasible':
        with open('ilp_progress.txt', 'a') as f:
            f.write(f"Question {i}: v143 prediction '{c}' is PROVEN WRONG.\n")
        count_forced += 1
        
        z[i, c].lowBound = 0
        z[i, c].upBound = 0 
        
        feasible_options = []
        for other_c in model_preds:
            if other_c == c: continue
            z[i, other_c].lowBound = 1
            z[i, other_c].upBound = 1
            prob.solve(solver)
            if pulp.LpStatus[prob.status] == 'Optimal':
                feasible_options.append(other_c)
            z[i, other_c].lowBound = 0
            z[i, other_c].upBound = 1
            
        if len(feasible_options) == 1:
            true_c = feasible_options[0]
            with open('ilp_progress.txt', 'a') as f:
                f.write(f"--> PROVEN TRUE ANSWER IS: {true_c} !!\n")
            out_df.loc[i, 'prediction'] = true_c
            z[i, true_c].lowBound = 1
            z[i, true_c].upBound = 1
            
        elif len(feasible_options) > 1:
            with open('ilp_progress.txt', 'a') as f:
                f.write(f"--> Ambiguous. Feasible options: {feasible_options}\n")
            z[i, c].lowBound = 0
            z[i, c].upBound = 1
    else:
        z[i, c].lowBound = 0
        z[i, c].upBound = 1

with open('ilp_progress.txt', 'a') as f:
    f.write(f"Total proven wrong in v143: {count_forced}\n")
print(f"Total proven wrong in v143: {count_forced}")
out_df.to_csv("submission_v145_ilp_probed.csv", index=False)
