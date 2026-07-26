"""
Action Routine Ground Truth Match Engine (Excluding Emotion Adverbs):
Filters 4,087 training questions for exact action routine matches in SINGLE, COMBINATION, MULTI, and SEQUENCE tracks!
"""
import pandas as pd

train = pd.read_csv("training_qa.csv")
test = pd.read_csv("test_qa.csv")
raw = pd.read_csv("transformer_fixed_raw_predictions.csv")
raw_map = dict(zip(raw['qa_id'], raw.to_dict('records')))

v276 = pd.read_csv("submission_v276_APEX_SUMMIT.csv")
v276_map = dict(zip(v276['qa_id'], v276['prediction']))

# Build GT lookup for non-emotion categories
gt_action_map = {}
for idx, r in train.iterrows():
    cat = str(r['category']).strip().lower()
    if cat == 'emotion': continue
    ans = str(r['answer']).strip()
    if ans in ['A', 'B', 'C', 'D']:
        ans_text = str(r[ans]).strip().lower()
        gt_action_map[(cat, ans_text)] = ans

print(f"Built non-emotion Ground Truth Action Map with {len(gt_action_map)} unique entries.")

action_overrides = []
for idx, r in test.iterrows():
    qid = r['qa_id']
    cat = str(r['category']).strip().lower()
    if cat == 'emotion': continue
    curr_pred = v276_map.get(qid, '')
    
    opts = {l: str(r[l]).strip().lower() for l in ['A', 'B', 'C', 'D'] if pd.notna(r[l])}
    
    for l, opt_text in opts.items():
        if (cat, opt_text) in gt_action_map:
            # Check if this choice l matches training GT, but is different from current prediction
            if l != curr_pred and len(curr_pred) == 1:
                prob = raw_map.get(qid, {}).get(f'raw_prob_{l}', 0.0)
                curr_prob = raw_map.get(qid, {}).get(f'raw_prob_{curr_pred}', 0.0)
                if prob > 0.25: #Viable probability candidate
                    action_overrides.append((qid, cat, curr_pred, l, round(curr_prob, 3), round(prob, 3), opt_text))

print(f"\nTotal Action Routine GT Overrides found: {len(action_overrides)}")
for ao in action_overrides:
    print(f"[ACTION GT OVERRIDE] {ao[0]} ({ao[1]}): CurrPred='{ao[2]}' ({ao[4]}) -> GTMatch='{ao[3]}' ({ao[5]}) | Routine='{ao[6]}'")
