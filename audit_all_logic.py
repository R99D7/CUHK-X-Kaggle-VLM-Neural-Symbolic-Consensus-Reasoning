"""
Exhaustive logic audit on submission_v267_NEWCOMB2MULTI.csv.
Check for EVERY possible logical relationship and discrepancy across categories for the exact same video path.
"""
import pandas as pd

sub = pd.read_csv("submission_v267_NEWCOMB2MULTI.csv")
te = pd.read_csv("test_qa.csv")
sub_map = dict(zip(sub['qa_id'], sub['prediction']))
te['pred'] = te['qa_id'].map(sub_map)

# Group by video path
grouped = te.groupby('path')

discrepancies = []

for vid_path, grp in grouped:
    cats = {row['category']: row for idx, row in grp.iterrows()}
    
    # Helper to get action text for a predicted letter in a question
    def get_action_texts(q_row):
        pred_letters = str(q_row['pred']).strip()
        opts = {l: str(q_row[l]).strip().lower() for l in ['A', 'B', 'C', 'D']}
        actions = set()
        for l in pred_letters:
            if l in opts:
                for act in opts[l].split(','):
                    actions.add(act.strip())
        return actions, opts, pred_letters

    single_acts, s_opts, s_pred = get_action_texts(cats['single']) if 'single' in cats else (set(), {}, '')
    comb_acts, c_opts, c_pred = get_action_texts(cats['combination']) if 'combination' in cats else (set(), {}, '')
    multi_acts, m_opts, m_pred = get_action_texts(cats['multi']) if 'multi' in cats else (set(), {}, '')
    seq_acts, sq_opts, sq_pred = get_action_texts(cats['sequence']) if 'sequence' in cats else (set(), {}, '')

    # Rule 1: SINGLE action MUST be present in COMBINATION actions and MULTI actions
    if 'single' in cats and 'combination' in cats:
        for act in single_acts:
            if act and act not in comb_acts:
                discrepancies.append((vid_path, 'SINGLE in COMB', f"Single action '{act}' missing from Comb ({c_pred}): {comb_acts}"))

    if 'single' in cats and 'multi' in cats:
        for act in single_acts:
            if act and act not in multi_acts:
                discrepancies.append((vid_path, 'SINGLE in MULTI', f"Single action '{act}' missing from Multi ({m_pred}): {multi_acts}"))

    # Rule 2: COMBINATION actions MUST be identical to or a subset/superset of MULTI actions
    if 'combination' in cats and 'multi' in cats:
        # Check if comb has actions missing in multi
        for act in comb_acts:
            if act and act not in multi_acts:
                discrepancies.append((vid_path, 'COMB in MULTI', f"Comb action '{act}' missing in Multi ({m_pred}): {multi_acts}"))
        # Check if multi has actions missing in comb
        for act in multi_acts:
            if act and act not in comb_acts:
                discrepancies.append((vid_path, 'MULTI in COMB', f"Multi action '{act}' missing in Comb ({c_pred}): {comb_acts}"))

    # Rule 3: SEQUENCE actions should match MULTI/COMBINATION actions
    if 'sequence' in cats and 'combination' in cats:
        for act in seq_acts:
            if act and act not in comb_acts:
                discrepancies.append((vid_path, 'SEQ in COMB', f"Seq action '{act}' missing in Comb ({c_pred}): {comb_acts}"))

print(f"Total logic discrepancies found in v267: {len(discrepancies)}")
for d in discrepancies[:30]:
    print(f"[{d[0]}] {d[1]}: {d[2]}")
