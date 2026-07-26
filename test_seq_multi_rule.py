"""
Verify the logical relationships between single, sequence, combination, and multi on TRAINING set ground truth!
"""
import pandas as pd

tr = pd.read_csv("training_qa.csv")
grouped = tr.groupby('path')

single_in_multi_matches = 0
single_in_multi_total = 0

seq_in_multi_matches = 0
seq_in_multi_total = 0

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
    sq_acts, sq_opts, sq_ans = get_ans_texts(cats['sequence']) if 'sequence' in cats else (set(), {}, '')

    # Test Single in Multi
    if 'single' in cats and 'multi' in cats:
        for act in s_acts:
            # Check if act was an available option in multi
            multi_opt_vals = list(m_opts.values())
            if act in multi_opt_vals:
                single_in_multi_total += 1
                if act in m_acts:
                    single_in_multi_matches += 1
                else:
                    print(f"[TR VIOLATION] Single act '{act}' in multi options but not in multi answer! Vid: {vid_path}")

    # Test Sequence in Multi
    if 'sequence' in cats and 'multi' in cats:
        for act in sq_acts:
            multi_opt_vals = list(m_opts.values())
            if act in multi_opt_vals:
                seq_in_multi_total += 1
                if act in m_acts:
                    seq_in_multi_matches += 1

print(f"Single action present in Multi answer (when available in options): {single_in_multi_matches}/{single_in_multi_total} ({single_in_multi_matches/single_in_multi_total if single_in_multi_total else 0:.2%})")
print(f"Sequence action present in Multi answer (when available in options): {seq_in_multi_matches}/{seq_in_multi_total} ({seq_in_multi_matches/seq_in_multi_total if seq_in_multi_total else 0:.2%})")
