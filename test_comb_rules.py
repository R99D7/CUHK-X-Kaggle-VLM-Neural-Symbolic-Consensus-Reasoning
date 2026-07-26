"""
Verify rules between COMBINATION and MULTI/SINGLE/SEQUENCE on TRAINING set ground truth!
"""
import pandas as pd

tr = pd.read_csv("training_qa.csv")
grouped = tr.groupby('path')

single_in_comb_matches = 0
single_in_comb_total = 0

multi_in_comb_matches = 0
multi_in_comb_total = 0

comb_in_multi_matches = 0
comb_in_multi_total = 0

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
    def get_ans_texts(q_row):
        ans = str(q_row['answer']).strip()
        opts = {l: str(q_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        actions = set()
        for l in ans:
            if l in opts:
                for act in opts[l].split(','):
                    actions.add(act.strip())
        return actions, opts, ans

    s_acts, s_opts, s_ans = get_ans_texts(cats['single']) if 'single' in cats else (set(), {}, '')
    m_acts, m_opts, m_ans = get_ans_texts(cats['multi']) if 'multi' in cats else (set(), {}, '')
    c_acts, c_opts, c_ans = get_ans_texts(cats['combination']) if 'combination' in cats else (set(), {}, '')

    # Test Single in Comb answer
    if 'single' in cats and 'combination' in cats:
        for act in s_acts:
            single_in_comb_total += 1
            if act in c_acts:
                single_in_comb_matches += 1

    # Test Multi answer in Comb answer
    if 'multi' in cats and 'combination' in cats:
        for act in m_acts:
            multi_in_comb_total += 1
            if act in c_acts:
                multi_in_comb_matches += 1

    # Test Comb answer in Multi answer (when available in Multi options)
    if 'combination' in cats and 'multi' in cats:
        for act in c_acts:
            multi_opt_vals = list(m_opts.values())
            if act in multi_opt_vals:
                comb_in_multi_total += 1
                if act in m_acts:
                    comb_in_multi_matches += 1

print(f"Single action present in Comb answer: {single_in_comb_matches}/{single_in_comb_total} ({single_in_comb_matches/single_in_comb_total if single_in_comb_total else 0:.2%})")
print(f"Multi action present in Comb answer: {multi_in_comb_matches}/{multi_in_comb_total} ({multi_in_comb_matches/multi_in_comb_total if multi_in_comb_total else 0:.2%})")
print(f"Comb action present in Multi answer (when in options): {comb_in_multi_matches}/{comb_in_multi_total} ({comb_in_multi_matches/comb_in_multi_total if comb_in_multi_total else 0:.2%})")
